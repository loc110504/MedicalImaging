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
from networks.xnetv2 import XNetv2
from utils import losses, ramps
from utils.xnetv2_wavelet import build_wavelet_batch_from_tensor
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=None, help="dataset root")
parser.add_argument("--exp", type=str, default="XNetv2_Scribble", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", choices=["ACDC", "MSCMR"], help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold for ACDC")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--model", type=str, default="xnetv2", help="model name")
parser.add_argument("--num_classes", type=int, default=4, help="number of segmentation classes")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=4, help="batch size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256], help="input patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

parser.add_argument("--in_chns", type=int, default=1, help="input channels")
parser.add_argument("--base_channels", type=int, default=32, help="base channels of XNetv2")
parser.add_argument("--wavelet_type", type=str, default="haar", help="wavelet basis")
parser.add_argument("--train_alpha", nargs=2, type=float, default=[0.0, 0.4], help="training alpha range")
parser.add_argument("--train_beta", nargs=2, type=float, default=[0.0, 0.4], help="training beta range")
parser.add_argument("--val_alpha", nargs=2, type=float, default=[0.2, 0.2], help="validation alpha range")
parser.add_argument("--val_beta", nargs=2, type=float, default=[0.2, 0.2], help="validation beta range")

parser.add_argument("--ce_weight", type=float, default=1.0, help="scribble CE weight per branch")
parser.add_argument("--dice_weight", type=float, default=1.0, help="scribble Dice weight per branch")
parser.add_argument("--unsup_weight", type=float, default=1.0, help="pseudo-label loss multiplier")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="pseudo-label ramp-up in epochs")
parser.add_argument("--warmup_iterations", type=int, default=1000, help="iterations before pseudo-labeling starts")
parser.add_argument("--pseudo_agree_thresh", type=float, default=0.70, help="confidence threshold for agreement pixels")
parser.add_argument("--pseudo_disagree_thresh", type=float, default=0.85, help="confidence threshold for disagreement pixels")
parser.add_argument("--pseudo_margin_thresh", type=float, default=0.10, help="confidence gap threshold for disagreement pixels")
parser.add_argument(
    "--pseudo_mask_mode",
    type=str,
    default="unlabeled",
    choices=["unlabeled", "all"],
    help="where to apply cross-branch pseudo supervision",
)
parser.add_argument(
    "--val_mode",
    type=str,
    default="main",
    choices=["main", "low", "high", "mean"],
    help="branch used during validation",
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
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    return (ce_map * mask).sum() / (mask.sum() + eps)


def resolve_root_path(train_args):
    if train_args.root_path is not None:
        return train_args.root_path
    return DATA_ROOTS[train_args.data]


def identity_transform(sample):
    return sample


def create_model(train_args):
    return XNetv2(
        in_channels=train_args.in_chns,
        num_classes=train_args.num_classes,
        base_channels=train_args.base_channels,
    ).cuda()


def unpack_logits(output):
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise ValueError("XNetv2 is expected to return (main, low, high) logits.")
    return output[0], output[1], output[2]


class XNetv2InferenceWrapper(torch.nn.Module):
    def __init__(self, model, wavelet_type, alpha, beta, mode="main"):
        super().__init__()
        self.model = model
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    def forward(self, x):
        x_low, x_high = build_wavelet_batch_from_tensor(
            x,
            wavelet_type=self.wavelet_type,
            alpha=self.alpha,
            beta=self.beta,
        )
        logits_main, logits_low, logits_high = self.model(x, x_low, x_high)
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
    inference_model = XNetv2InferenceWrapper(
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


def build_xnetv2_confidence_pseudo_label(
    prob_main,
    prob_low,
    prob_high,
    label,
    ignore_index,
    pseudo_mask_mode="unlabeled",
    agree_thresh=0.70,
    disagree_thresh=0.85,
    margin_thresh=0.10,
    eps=1e-8,
):
    prob_main = prob_main.detach()
    prob_low = prob_low.detach()
    prob_high = prob_high.detach()

    conf_main, pred_main = torch.max(prob_main, dim=1)
    conf_low, pred_low = torch.max(prob_low, dim=1)
    conf_high, pred_high = torch.max(prob_high, dim=1)

    candidate_mask = build_pseudo_mask(label, ignore_index, pseudo_mask_mode)

    all_agree = (pred_main == pred_low) & (pred_main == pred_high)
    all_agree = all_agree & (
        torch.minimum(torch.minimum(conf_main, conf_low), conf_high) >= agree_thresh
    ) & candidate_mask

    main_low_agree = (
        (pred_main == pred_low)
        & (pred_main != pred_high)
        & (torch.minimum(conf_main, conf_low) >= agree_thresh)
        & candidate_mask
    )
    main_high_agree = (
        (pred_main == pred_high)
        & (pred_main != pred_low)
        & (torch.minimum(conf_main, conf_high) >= agree_thresh)
        & candidate_mask
    )
    low_high_agree = (
        (pred_low == pred_high)
        & (pred_low != pred_main)
        & (torch.minimum(conf_low, conf_high) >= agree_thresh)
        & candidate_mask
    )

    any_reliable_agree = all_agree | main_low_agree | main_high_agree | low_high_agree

    conf_stack = torch.stack([conf_main, conf_low, conf_high], dim=1)
    prob_stack = torch.stack([prob_main, prob_low, prob_high], dim=1)
    max_conf, max_index = torch.max(conf_stack, dim=1)
    sorted_conf, _ = torch.sort(conf_stack, dim=1, descending=True)
    conf_margin = sorted_conf[:, 0] - sorted_conf[:, 1]

    best_prob = torch.gather(
        prob_stack,
        1,
        max_index[:, None, None, :, :].expand(-1, 1, prob_stack.size(2), prob_stack.size(3), prob_stack.size(4)),
    ).squeeze(1)

    reliable_disagree = (
        (~any_reliable_agree)
        & (max_conf >= disagree_thresh)
        & (conf_margin >= margin_thresh)
        & candidate_mask
    )

    mean_all = (prob_main + prob_low + prob_high) / 3.0
    mean_main_low = 0.5 * (prob_main + prob_low)
    mean_main_high = 0.5 * (prob_main + prob_high)
    mean_low_high = 0.5 * (prob_low + prob_high)

    soft_pseudo_label = mean_all
    soft_pseudo_label = torch.where(main_low_agree.unsqueeze(1), mean_main_low, soft_pseudo_label)
    soft_pseudo_label = torch.where(main_high_agree.unsqueeze(1), mean_main_high, soft_pseudo_label)
    soft_pseudo_label = torch.where(low_high_agree.unsqueeze(1), mean_low_high, soft_pseudo_label)
    soft_pseudo_label = torch.where(reliable_disagree.unsqueeze(1), best_prob, soft_pseudo_label)
    soft_pseudo_label = soft_pseudo_label / (soft_pseudo_label.sum(dim=1, keepdim=True) + eps)

    reliable_agree = any_reliable_agree
    reliable_mask = (reliable_agree | reliable_disagree).float().unsqueeze(1)

    return {
        "soft_pseudo_label": soft_pseudo_label.detach(),
        "reliable_mask": reliable_mask.detach(),
        "reliable_ratio": reliable_mask.mean().detach(),
        "reliable_agree": reliable_agree.detach(),
        "reliable_disagree": reliable_disagree.detach(),
        "agreement_ratio": reliable_agree.float().mean().detach(),
        "disagreement_ratio": reliable_disagree.float().mean().detach(),
        "all_agree_ratio": all_agree.float().mean().detach(),
        "main_low_agree_ratio": main_low_agree.float().mean().detach(),
        "main_high_agree_ratio": main_high_agree.float().mean().detach(),
        "low_high_agree_ratio": low_high_agree.float().mean().detach(),
        "pseudo_conf": max_conf.unsqueeze(1).detach(),
        "conf_margin": conf_margin.unsqueeze(1).detach(),
    }


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

            logits_main, logits_low, logits_high = unpack_logits(model(image_batch, low_batch, high_batch))
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
                train_args.ce_weight * (loss_main_ce + loss_low_ce + loss_high_ce)
                + train_args.dice_weight * (loss_main_dice + loss_low_dice + loss_high_dice)
            )

            pseudo_candidate_mask = build_pseudo_mask(label_batch, num_classes, train_args.pseudo_mask_mode)
            if iter_num >= train_args.warmup_iterations and bool(pseudo_candidate_mask.any().item()):
                pseudo_info = build_xnetv2_confidence_pseudo_label(
                    prob_main=prob_main,
                    prob_low=prob_low,
                    prob_high=prob_high,
                    label=label_batch,
                    ignore_index=num_classes,
                    pseudo_mask_mode=train_args.pseudo_mask_mode,
                    agree_thresh=train_args.pseudo_agree_thresh,
                    disagree_thresh=train_args.pseudo_disagree_thresh,
                    margin_thresh=train_args.pseudo_margin_thresh,
                )
                loss_pseudo_main = masked_soft_ce_loss(
                    logits_main,
                    pseudo_info["soft_pseudo_label"],
                    pseudo_info["reliable_mask"],
                )
                loss_pseudo_low = masked_soft_ce_loss(
                    logits_low,
                    pseudo_info["soft_pseudo_label"],
                    pseudo_info["reliable_mask"],
                )
                loss_pseudo_high = masked_soft_ce_loss(
                    logits_high,
                    pseudo_info["soft_pseudo_label"],
                    pseudo_info["reliable_mask"],
                )
                loss_unsup = loss_pseudo_main + loss_pseudo_low + loss_pseudo_high
            else:
                pseudo_info = {
                    "reliable_ratio": logits_main.new_tensor(0.0),
                    "agreement_ratio": logits_main.new_tensor(0.0),
                    "disagreement_ratio": logits_main.new_tensor(0.0),
                    "all_agree_ratio": logits_main.new_tensor(0.0),
                    "main_low_agree_ratio": logits_main.new_tensor(0.0),
                    "main_high_agree_ratio": logits_main.new_tensor(0.0),
                    "low_high_agree_ratio": logits_main.new_tensor(0.0),
                    "pseudo_conf": logits_main.new_zeros((1, 1, 1, 1)),
                    "conf_margin": logits_main.new_zeros((1, 1, 1, 1)),
                }
                loss_pseudo_main = logits_main.new_tensor(0.0)
                loss_pseudo_low = logits_main.new_tensor(0.0)
                loss_pseudo_high = logits_main.new_tensor(0.0)
                loss_unsup = logits_main.new_tensor(0.0)

            pseudo_weight = (
                train_args.unsup_weight
                * get_current_consistency_weight(iter_num // len(trainloader), train_args)
                if iter_num >= train_args.warmup_iterations
                else 0.0
            )
            loss = loss_sup + pseudo_weight * loss_unsup

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
            writer.add_scalar("info/loss_pseudo_main", loss_pseudo_main.item(), iter_num)
            writer.add_scalar("info/loss_pseudo_low", loss_pseudo_low.item(), iter_num)
            writer.add_scalar("info/loss_pseudo_high", loss_pseudo_high.item(), iter_num)
            writer.add_scalar("threshold/agree", train_args.pseudo_agree_thresh, iter_num)
            writer.add_scalar("threshold/disagree", train_args.pseudo_disagree_thresh, iter_num)
            writer.add_scalar("threshold/margin", train_args.pseudo_margin_thresh, iter_num)
            writer.add_scalar("pseudo/pseudo_mask_ratio", pseudo_candidate_mask.float().mean().item(), iter_num)
            writer.add_scalar("pseudo/reliable_ratio", pseudo_info["reliable_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/agreement_ratio", pseudo_info["agreement_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/disagreement_ratio", pseudo_info["disagreement_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/all_agree_ratio", pseudo_info["all_agree_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/main_low_agree_ratio", pseudo_info["main_low_agree_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/main_high_agree_ratio", pseudo_info["main_high_agree_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/low_high_agree_ratio", pseudo_info["low_high_agree_ratio"].item(), iter_num)
            writer.add_scalar("pseudo/pseudo_conf", pseudo_info["pseudo_conf"].mean().item(), iter_num)
            writer.add_scalar("pseudo/conf_margin", pseudo_info["conf_margin"].mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss=%f, sup=%f, pseudo=%f, pw=%f, reliable=%f, agree=%f, disagree=%f",
                    iter_num,
                    loss.item(),
                    loss_sup.item(),
                    loss_unsup.item(),
                    pseudo_weight,
                    pseudo_info["reliable_ratio"].item(),
                    pseudo_info["agreement_ratio"].item(),
                    pseudo_info["disagreement_ratio"].item(),
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
