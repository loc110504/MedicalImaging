import argparse
import logging
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets
from dataloader.mscmr import MSCMRDataSets
from dataloader.wavelet_scribble import WaveletRandomGenerator, WaveletTrainingWrapper
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.xnetv2_wavelet import build_wavelet_batch_from_tensor
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=None, help="dataset root")
parser.add_argument("--exp", type=str, default="Wavelet_UNet_Scribble", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", choices=["ACDC", "MSCMR"], help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold for ACDC")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument(
    "--model",
    type=str,
    default="unet_hl",
    choices=["unet", "unet_hl"],
    help="shared-weight backbone",
)
parser.add_argument("--num_classes", type=int, default=4, help="number of segmentation classes")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=16, help="batch size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256], help="input patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

parser.add_argument("--in_chns", type=int, default=1, help="input channels")
parser.add_argument("--wavelet_type", type=str, default="haar", help="wavelet basis")
parser.add_argument("--train_alpha", nargs=2, type=float, default=[0.0, 0.4], help="training alpha range")
parser.add_argument("--train_beta", nargs=2, type=float, default=[0.0, 0.4], help="training beta range")
parser.add_argument("--val_alpha", nargs=2, type=float, default=[0.2, 0.2], help="validation alpha range")
parser.add_argument("--val_beta", nargs=2, type=float, default=[0.2, 0.2], help="validation beta range")

parser.add_argument("--ce_weight", type=float, default=1.0, help="scribble CE weight per view")
parser.add_argument("--dice_weight", type=float, default=1.0, help="scribble Dice weight per view")
parser.add_argument(
    "--aux_sup_weight",
    type=float,
    default=0.0,
    help="weight for direct scribble supervision on LF/HF views",
)
parser.add_argument("--unsup_weight", type=float, default=1.0, help="pseudo-label loss multiplier")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="pseudo-label ramp-up in epochs")
parser.add_argument("--warmup_iterations", type=int, default=1000, help="iterations before pseudo-labeling starts")
parser.add_argument(
    "--pseudo_reliable_thresh",
    type=float,
    default=0.6,
    help="minimum confidence of fused pseudo-labels",
)
parser.add_argument(
    "--pseudo_mask_mode",
    type=str,
    default="unlabeled",
    choices=["unlabeled", "all"],
    help="where to apply wavelet pseudo supervision",
)
parser.add_argument(
    "--pseudo_loss",
    type=str,
    default="soft_ce",
    choices=["soft_ce", "pce"],
    help="loss for fused pseudo-label supervision",
)
parser.add_argument(
    "--val_mode",
    type=str,
    default="main",
    choices=["main", "low", "high", "mean"],
    help="view used during validation",
)
parser.add_argument("--eval_interval", type=int, default=400, help="validation interval")
parser.add_argument("--save_interval", type=int, default=3000, help="checkpoint interval")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


DATA_ROOTS = {
    "ACDC": "../../data/ACDC",
    "MSCMR": "../../data/MSCMR",
}


def get_current_consistency_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob.detach() * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()
    mask = mask.float()
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    return (ce_map * mask).sum() / (mask.sum() + eps)


def masked_partial_ce_loss(logits, target, mask, ignore_index):
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    mask = mask.bool()
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    target = target.clone().long()
    target[~mask] = ignore_index
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def resolve_root_path(train_args):
    if train_args.root_path is not None:
        return train_args.root_path
    return DATA_ROOTS[train_args.data]


def identity_transform(sample):
    return sample


def create_model(train_args):
    return net_factory(
        net_type=train_args.model,
        in_chns=train_args.in_chns,
        class_num=train_args.num_classes,
    )


def minmax_normalize_batch(batch, eps=1e-8):
    flat = batch.view(batch.size(0), -1)
    min_value = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_value = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (batch - min_value) / (max_value - min_value + eps)


def extract_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        raise ValueError("Unsupported dict output keys: {}".format(list(output.keys())))
    return output


def forward_three_views(model, x_main, x_low, x_high):
    x_main = minmax_normalize_batch(x_main)
    x_low = minmax_normalize_batch(x_low)
    x_high = minmax_normalize_batch(x_high)
    logits_main = extract_logits(model(x_main))
    logits_low = extract_logits(model(x_low))
    logits_high = extract_logits(model(x_high))
    return logits_main, logits_low, logits_high


class SharedUNetInferenceWrapper(torch.nn.Module):
    def __init__(self, model, wavelet_type, alpha, beta, mode="main"):
        super().__init__()
        self.model = model
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    def forward(self, x):
        x = minmax_normalize_batch(x)
        x_low, x_high = build_wavelet_batch_from_tensor(
            x,
            wavelet_type=self.wavelet_type,
            alpha=self.alpha,
            beta=self.beta,
        )
        logits_main, logits_low, logits_high = forward_three_views(self.model, x, x_low, x_high)
        if self.mode == "main":
            return logits_main
        if self.mode == "low":
            return logits_low
        if self.mode == "high":
            return logits_high
        if self.mode == "mean":
            prob = (
                torch.softmax(logits_main, dim=1)
                + torch.softmax(logits_low, dim=1)
                + torch.softmax(logits_high, dim=1)
            ) / 3.0
            return torch.log(prob + 1e-8)
        raise ValueError("Unsupported validation mode: {}".format(self.mode))


def build_train_dataset(train_args):
    if train_args.data == "ACDC":
        base_dataset = ACDCDataSets(
            base_dir=train_args.root_path,
            split="train",
            transform=identity_transform,
            fold=train_args.fold,
            sup_type=train_args.sup_type,
            return_full_label=True,
        )
    else:
        base_dataset = MSCMRDataSets(
            base_dir=train_args.root_path,
            split="train",
            transform=identity_transform,
            sup_type=train_args.sup_type,
            return_full_label=True,
        )
    wavelet_transform = WaveletRandomGenerator(
        output_size=train_args.patch_size,
        wavelet_type=train_args.wavelet_type,
        alpha=train_args.train_alpha,
        beta=train_args.train_beta,
        ignore_index=train_args.num_classes,
    )
    return WaveletTrainingWrapper(base_dataset, wavelet_transform)


def build_val_dataset(train_args):
    if train_args.data == "ACDC":
        return ACDCDataSets(
            base_dir=train_args.root_path,
            fold=train_args.fold,
            split="val",
        )
    return MSCMRDataSets(
        base_dir=train_args.root_path,
        split="val",
    )


@torch.no_grad()
def validate(model, valloader, db_val, num_classes, writer, iter_num, train_args):
    metric_list = 0.0
    model.eval()
    inference_model = SharedUNetInferenceWrapper(
        model=model,
        wavelet_type=train_args.wavelet_type,
        alpha=train_args.val_alpha,
        beta=train_args.val_beta,
        mode=train_args.val_mode,
    )

    for sampled_val in valloader:
        metric_i = test_single_volume(
            sampled_val["image"],
            sampled_val["label"],
            inference_model,
            classes=num_classes,
            patch_size=train_args.patch_size,
        )
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_val)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    for class_i in range(num_classes - 1):
        writer.add_scalar("info/val_{}_dice".format(class_i + 1), metric_list[class_i, 0], iter_num)
        writer.add_scalar("info/val_{}_hd95".format(class_i + 1), metric_list[class_i, 1], iter_num)
    writer.add_scalar("info/val_mean_dice", performance, iter_num)
    writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

    model.train()
    return performance, mean_hd95


def build_pseudo_mask(label_batch, ignore_index, mode):
    if mode == "unlabeled":
        return label_batch == ignore_index
    if mode == "all":
        return torch.ones_like(label_batch, dtype=torch.bool)
    raise ValueError("Unsupported pseudo_mask_mode: {}".format(mode))


def normalize_prob(prob, eps=1e-8):
    return prob / (prob.sum(dim=1, keepdim=True) + eps)


def build_wavelet_fusion_pseudo_label(
    prob_main,
    prob_low,
    prob_high,
    label,
    ignore_index,
    pseudo_mask_mode="unlabeled",
    reliable_thresh=0.6,
    eps=1e-8,
):
    prob_main = prob_main.detach()
    prob_low = prob_low.detach()
    prob_high = prob_high.detach()

    pseudo_low = normalize_prob(0.5 * (prob_main + prob_high), eps)
    pseudo_high = normalize_prob(0.5 * (prob_main + prob_low), eps)

    conf_low = torch.max(pseudo_low, dim=1).values
    conf_high = torch.max(pseudo_high, dim=1).values
    candidate_mask = build_pseudo_mask(label, ignore_index, pseudo_mask_mode)

    reliable_low = candidate_mask
    reliable_high = candidate_mask
    if reliable_thresh > 0:
        reliable_low = reliable_low & (conf_low >= reliable_thresh)
        reliable_high = reliable_high & (conf_high >= reliable_thresh)

    return {
        "pseudo_low": pseudo_low.detach(),
        "pseudo_high": pseudo_high.detach(),
        "reliable_mask_low": reliable_low.float().unsqueeze(1).detach(),
        "reliable_mask_high": reliable_high.float().unsqueeze(1).detach(),
        "candidate_ratio": candidate_mask.float().mean().detach(),
        "reliable_ratio_low": reliable_low.float().mean().detach(),
        "reliable_ratio_high": reliable_high.float().mean().detach(),
        "pseudo_conf_low": conf_low.mean().detach(),
        "pseudo_conf_high": conf_high.mean().detach(),
    }


def compute_pseudo_loss(logits, soft_target, mask, ignore_index, loss_type):
    if loss_type == "soft_ce":
        return masked_soft_ce_loss(logits, soft_target, mask)
    if loss_type == "pce":
        return masked_partial_ce_loss(logits, torch.argmax(soft_target, dim=1), mask, ignore_index)
    raise ValueError("Unsupported pseudo_loss: {}".format(loss_type))


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    model = create_model(train_args)
    db_train = build_train_dataset(train_args)
    db_val = build_val_dataset(train_args)

    def worker_init_fn(worker_id):
        random.seed(train_args.seed + worker_id)
        np.random.seed(train_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    dice_loss = losses.pDLoss(num_classes, ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()

    for _ in iterator:
        for sampled_batch in trainloader:
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda().long()
            low_batch = sampled_batch["wavelet_l"].cuda()
            high_batch = sampled_batch["wavelet_h"].cuda()

            logits_main, logits_low, logits_high = forward_three_views(model, image_batch, low_batch, high_batch)
            prob_main = torch.softmax(logits_main, dim=1)
            prob_low = torch.softmax(logits_low, dim=1)
            prob_high = torch.softmax(logits_high, dim=1)

            loss_main_ce = ce_loss(logits_main, label_batch)
            loss_low_ce = ce_loss(logits_low, label_batch)
            loss_high_ce = ce_loss(logits_high, label_batch)

            loss_main_dice = dice_loss(prob_main, label_batch.unsqueeze(1))
            loss_low_dice = dice_loss(prob_low, label_batch.unsqueeze(1))
            loss_high_dice = dice_loss(prob_high, label_batch.unsqueeze(1))

            loss_sup = (
                train_args.ce_weight * loss_main_ce
                + train_args.dice_weight * loss_main_dice
            )

            pseudo_candidate_mask = build_pseudo_mask(label_batch, num_classes, train_args.pseudo_mask_mode)
            if iter_num >= train_args.warmup_iterations and bool(pseudo_candidate_mask.any().item()):
                pseudo_info = build_wavelet_fusion_pseudo_label(
                    prob_main=prob_main,
                    prob_low=prob_low,
                    prob_high=prob_high,
                    label=label_batch,
                    ignore_index=num_classes,
                    pseudo_mask_mode=train_args.pseudo_mask_mode,
                    reliable_thresh=train_args.pseudo_reliable_thresh,
                )
                loss_pseudo_low = compute_pseudo_loss(
                    logits_low,
                    pseudo_info["pseudo_low"],
                    pseudo_info["reliable_mask_low"],
                    num_classes,
                    train_args.pseudo_loss,
                )
                loss_pseudo_high = compute_pseudo_loss(
                    logits_high,
                    pseudo_info["pseudo_high"],
                    pseudo_info["reliable_mask_high"],
                    num_classes,
                    train_args.pseudo_loss,
                )
                loss_unsup = loss_pseudo_low + loss_pseudo_high
            else:
                pseudo_info = {
                    "candidate_ratio": logits_main.new_tensor(0.0),
                    "reliable_ratio_low": logits_main.new_tensor(0.0),
                    "reliable_ratio_high": logits_main.new_tensor(0.0),
                    "pseudo_conf_low": logits_main.new_tensor(0.0),
                    "pseudo_conf_high": logits_main.new_tensor(0.0),
                }
                loss_pseudo_low = logits_main.new_tensor(0.0)
                loss_pseudo_high = logits_main.new_tensor(0.0)
                loss_unsup = logits_main.new_tensor(0.0)

            pseudo_weight = (
                train_args.unsup_weight
                * get_current_consistency_weight(iter_num // len(trainloader), train_args)
                if iter_num >= train_args.warmup_iterations
                else 0.0
            )
            loss = loss_sup + loss_unsup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss_sup", loss_sup.item(), iter_num)
            writer.add_scalar("info/loss_pseudo", loss_unsup.item(), iter_num)
            writer.add_scalar("info/pseudo_weight", pseudo_weight, iter_num)
            writer.add_scalar("info/loss_main_ce", loss_main_ce.item(), iter_num)
            writer.add_scalar("info/loss_low_ce", loss_low_ce.item(), iter_num)
            writer.add_scalar("info/loss_high_ce", loss_high_ce.item(), iter_num)
            writer.add_scalar("info/loss_main_dice", loss_main_dice.item(), iter_num)
            writer.add_scalar("info/loss_low_dice", loss_low_dice.item(), iter_num)
            writer.add_scalar("info/loss_high_dice", loss_high_dice.item(), iter_num)
            writer.add_scalar("info/loss_pseudo_low", loss_pseudo_low.item(), iter_num)
            writer.add_scalar("info/loss_pseudo_high", loss_pseudo_high.item(), iter_num)
            writer.add_scalar("pseudo/candidate_ratio", pseudo_info["candidate_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/reliable_ratio_low", pseudo_info["reliable_ratio_low"].item(), iter_num)
            writer.add_scalar("pseudo/reliable_ratio_high", pseudo_info["reliable_ratio_high"].item(), iter_num)
            writer.add_scalar("pseudo/conf_low", pseudo_info["pseudo_conf_low"].item(), iter_num)
            writer.add_scalar("pseudo/conf_high", pseudo_info["pseudo_conf_high"].item(), iter_num)
            writer.add_scalar("threshold/pseudo_reliable", train_args.pseudo_reliable_thresh, iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss=%f, sup=%f, pseudo=%f, pw=%f, r_low=%f, r_high=%f",
                    iter_num,
                    loss.item(),
                    loss_sup.item(),
                    loss_unsup.item(),
                    pseudo_weight,
                    pseudo_info["reliable_ratio_low"].item(),
                    pseudo_info["reliable_ratio_high"].item(),
                )

            if iter_num % train_args.eval_interval == 0:
                performance, mean_hd95 = validate(
                    model=model,
                    valloader=valloader,
                    db_val=db_val,
                    num_classes=num_classes,
                    writer=writer,
                    iter_num=iter_num,
                    train_args=train_args,
                )

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "iter_{}_dice_{:.4f}.pth".format(iter_num, best_performance),
                    )
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(train_args.model))
                    checkpoint = {
                        "iter_num": iter_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_performance": best_performance,
                        "args": vars(train_args),
                    }
                    torch.save(checkpoint, save_mode_path)
                    torch.save(checkpoint, save_best)

                logging.info(
                    "iteration %d : mean_dice=%f mean_hd95=%f",
                    iter_num,
                    performance,
                    mean_hd95,
                )

            if iter_num % train_args.save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, "iter_{}.pth".format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to %s", save_mode_path)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    final_model_path = os.path.join(snapshot_path, "{}_final_model.pth".format(train_args.model))
    torch.save(model.state_dict(), final_model_path)
    logging.info("save final model to %s", final_model_path)
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    args.root_path = resolve_root_path(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../../checkpoints/{}_{}".format(args.data, args.exp)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
