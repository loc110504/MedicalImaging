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
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.boundary_utils import boundary_likelihood
from utils.entropy_utils import build_uncertain_mask, normalized_entropy, teacher_confidence_mask
from utils.region_switch import region_wise_teacher_selection
from utils.weighted_losses import (
    masked_symmetric_kl_loss,
    weighted_soft_ce_loss,
)
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="LED_SDT_noHICO", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", help="dataset id")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--model", type=str, default="unet_lgdt", help="network name")
parser.add_argument("--num_classes", type=int, default=4, help="output classes")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=8, help="batch size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="input patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency_rampup", type=float, default=40)

parser.add_argument("--uncertain_mode", type=str, default="quantile", choices=["quantile", "fixed"])
parser.add_argument("--uncertain_top_ratio", type=float, default=0.35)
parser.add_argument("--uncertain_threshold", type=float, default=0.5)
parser.add_argument("--teacher_conf_threshold", type=float, default=0.55)
parser.add_argument("--boundary_gamma", type=float, default=0.5)
parser.add_argument("--boundary_lambda_image", type=float, default=1.0)

parser.add_argument("--lambda_pseudo", type=float, default=0.5)
parser.add_argument("--lambda_aux", type=float, default=0.4)
parser.add_argument("--lambda_consensus", type=float, default=0.1)

parser.add_argument("--pseudo_warmup", type=int, default=3000)
parser.add_argument("--debug_shapes", action="store_true")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_current_consistency_weight(epoch, cfg):
    return ramps.sigmoid_rampup(epoch, cfg.consistency_rampup)


def train(cfg, snapshot_path):
    base_lr = cfg.base_lr
    num_classes = cfg.num_classes
    batch_size = cfg.batch_size
    max_iterations = cfg.max_iterations

    def worker_init_fn(worker_id):
        random.seed(cfg.seed + worker_id)

    model = net_factory(net_type=cfg.model, in_chns=1, class_num=num_classes)
    model.cuda()
    model.train()

    db_train = ACDCDataSets(
        base_dir=cfg.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(cfg.patch_size)]),
        fold=cfg.fold,
        sup_type=cfg.sup_type,
    )
    db_val = ACDCDataSets(
        base_dir=cfg.root_path,
        fold=cfg.fold,
        split="val"
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled in enumerate(trainloader):
            image = sampled["image"].cuda()
            scrib = sampled["label"].cuda().long()

            out = model(image, return_all=True)

            logits_s = out["logits_s"]
            logits_l = out["logits_l"]
            logits_g = out["logits_g"]

            probs_s = torch.softmax(logits_s, dim=1)
            probs_l = torch.softmax(logits_l, dim=1)
            probs_g = torch.softmax(logits_g, dim=1)

            if cfg.debug_shapes and iter_num == 0:
                logging.info(
                    "Shapes logits_s=%s logits_l=%s logits_g=%s",
                    tuple(logits_s.shape),
                    tuple(logits_l.shape),
                    tuple(logits_g.shape),
                )

            # =========================
            # 1. Scribble supervised loss
            # =========================
            loss_ce_s = ce_loss(logits_s, scrib)
            loss_dice_s = dice_loss(probs_s, scrib.unsqueeze(1))
            loss_scrib = loss_ce_s + loss_dice_s

            # =========================
            # 2. Auxiliary supervised loss for two teacher branches
            # =========================
            loss_aux = 0.5 * (
                ce_loss(logits_l, scrib)
                + ce_loss(logits_g, scrib)
            )

            # =========================
            # 3. Student uncertainty mask
            # =========================
            entropy_s = normalized_entropy(probs_s)

            uncertain_mask = build_uncertain_mask(
                entropy=entropy_s,
                scribble=scrib,
                mode=cfg.uncertain_mode,
                top_ratio=cfg.uncertain_top_ratio,
                threshold=cfg.uncertain_threshold,
                ignore_index=4,
            )

            # =========================
            # 4. Boundary-aware teacher selection
            # =========================
            boundary = boundary_likelihood(
                image=image,
                probs_s=probs_s,
                lambda_image=cfg.boundary_lambda_image,
            )

            switch = region_wise_teacher_selection(
                probs_l=probs_l,
                probs_g=probs_g,
                boundary=boundary,
                gamma=cfg.boundary_gamma,
            )

            selected_probs = switch["selected_probs"]
            selected_weight = switch["selected_weight"]
            select_local = switch["select_local"]

            # =========================
            # 5. Pseudo-label loss
            # =========================
            conf_mask = teacher_confidence_mask(
                selected_probs,
                threshold=cfg.teacher_conf_threshold
            )

            pseudo_mask = uncertain_mask & conf_mask

            selected_probs_detached = selected_probs.detach()
            selected_weight_detached = selected_weight.detach()

            if iter_num >= cfg.pseudo_warmup:
                loss_pseudo = weighted_soft_ce_loss(
                    logits=logits_s,
                    soft_targets=selected_probs_detached,
                    mask=pseudo_mask,
                    weight=selected_weight_detached,
                )
            else:
                loss_pseudo = logits_s.sum() * 0.0

            # =========================
            # 6. Easy-region consistency between local/global teachers
            # =========================
            easy_mask = (entropy_s < 0.2) & (scrib.unsqueeze(1) == 4)

            loss_consensus = masked_symmetric_kl_loss(
                probs_a=probs_l,
                probs_b=probs_g,
                mask=easy_mask
            )

            # =========================
            # 7. Ramp-up weights
            # =========================
            consistency_weight = get_current_consistency_weight(
                iter_num // 300,
                cfg
            )

            pseudo_weight = cfg.lambda_pseudo * consistency_weight
            cons_weight = cfg.lambda_consensus * consistency_weight

            # =========================
            # 8. Final loss
            # =========================
            loss = (
                loss_scrib
                + cfg.lambda_aux * loss_aux
                + pseudo_weight * loss_pseudo
                + cons_weight * loss_consensus
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =========================
            # 9. Poly learning rate decay
            # =========================
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # =========================
            # 10. TensorBoard logging
            # =========================
            writer.add_scalar("train/loss_total", loss.item(), iter_num)
            writer.add_scalar("train/loss_scrib", loss_scrib.item(), iter_num)
            writer.add_scalar("train/loss_aux", loss_aux.item(), iter_num)
            writer.add_scalar("train/loss_pseudo", loss_pseudo.item(), iter_num)
            writer.add_scalar("train/loss_consensus", loss_consensus.item(), iter_num)

            writer.add_scalar("train/pseudo_weight", pseudo_weight, iter_num)
            writer.add_scalar("train/cons_weight", cons_weight, iter_num)

            writer.add_scalar("train/entropy_s_mean", entropy_s.mean().item(), iter_num)
            writer.add_scalar("train/uncertain_ratio", uncertain_mask.float().mean().item(), iter_num)
            writer.add_scalar("train/pseudo_ratio", pseudo_mask.float().mean().item(), iter_num)
            writer.add_scalar("train/select_local_ratio", select_local.float().mean().item(), iter_num)
            writer.add_scalar("train/boundary_mean", boundary.mean().item(), iter_num)

            if iter_num % 400 == 0:
                logging.info(
                    "iteration %d : loss %.6f pseudo %.6f consensus %.6f",
                    iter_num,
                    loss.item(),
                    loss_pseudo.item(),
                    loss_consensus.item(),
                )

            # =========================
            # 11. Validation
            # =========================
            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                metric_list = 0.0

                for sampled_batch in valloader:
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        f"info/val_{class_i + 1}_dice",
                        metric_list[class_i, 0],
                        iter_num
                    )
                    writer.add_scalar(
                        f"info/val_{class_i + 1}_hd95",
                        metric_list[class_i, 1],
                        iter_num
                    )

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance

                    save_mode_path = os.path.join(
                        snapshot_path,
                        f"iter_{iter_num}_dice_{round(best_performance, 4)}.pth"
                    )

                    save_best = os.path.join(
                        snapshot_path,
                        f"{cfg.model}_best_model.pth"
                    )

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    "iteration %d : mean_dice %.6f mean_hd95 %.6f",
                    iter_num,
                    performance,
                    mean_hd95
                )

                model.train()

            # =========================
            # 12. Periodic checkpoint
            # =========================
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path,
                    f"iter_{iter_num}.pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to %s", save_mode_path)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../../checkpoints/{}_{}".format(
        args.data,
        args.exp
    )

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)