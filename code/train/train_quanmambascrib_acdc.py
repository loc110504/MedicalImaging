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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.mamba_unet_2d import MambaUnet
from utils import ramps
from utils.quan_mamba_losses import linear_schedule, masked_soft_ce_loss, partial_ce_loss
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="QuanMambaScrib", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--num_classes", type=int, default=4, help="number of segmentation classes")
parser.add_argument("--max_iterations", type=int, default=40000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=16, help="batch size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256], help="input patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

parser.add_argument("--model", type=str, default="quanmambascrib", help="model name")
parser.add_argument("--unet_type", type=str, default="unet_hl", help="compatibility arg from old script")
parser.add_argument("--mamba_variant", type=str, default="vmunet", help="compatibility arg from old script")
parser.add_argument("--in_chns", type=int, default=1, help="input channels")

parser.add_argument("--qpim_backend", type=str, default="torch_angle_fidelity", help="compatibility arg from old script")
parser.add_argument("--qpim_dim", type=int, default=8, help="compatibility arg from old script")
parser.add_argument("--qpim_tau", type=float, default=0.5, help="compatibility arg from old script")
parser.add_argument("--qpim_momentum", type=float, default=0.99, help="compatibility arg from old script")
parser.add_argument("--qpim_normalize_z", type=int, default=1, help="compatibility arg from old script")
parser.add_argument("--qpim_detach_prototypes", type=int, default=0, help="compatibility arg from old script")

parser.add_argument("--lambda_q", type=float, default=1.0, help="compatibility arg from old script")
parser.add_argument("--lambda_cps", type=float, default=1.0, help="weight for cross pseudo supervision")
parser.add_argument("--pseudo_loss_weight", type=float, default=8.0, help="pseudo-label ramped weight")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="ramp-up length in epochs")

parser.add_argument("--warmup_iterations", type=int, default=5000, help="start pseudo-labeling after this iteration")
parser.add_argument("--agree_thresh_start", type=float, default=0.80, help="initial agreement threshold")
parser.add_argument("--agree_thresh_end", type=float, default=0.70, help="final agreement threshold")
parser.add_argument("--disagree_thresh_start", type=float, default=0.90, help="initial disagreement threshold")
parser.add_argument("--disagree_thresh_end", type=float, default=0.80, help="final disagreement threshold")
parser.add_argument("--margin_thresh", type=float, default=0.10, help="confidence margin threshold")
parser.add_argument(
    "--pseudo_mask_mode",
    type=str,
    default="unlabeled",
    choices=["unlabeled", "all"],
    help="candidate region for pseudo-label supervision",
)
parser.add_argument("--tau_u_start", type=float, default=0.95, help="compatibility arg from old script")
parser.add_argument("--tau_u_end", type=float, default=0.75, help="compatibility arg from old script")
parser.add_argument("--tau_m_start", type=float, default=0.95, help="compatibility arg from old script")
parser.add_argument("--tau_m_end", type=float, default=0.75, help="compatibility arg from old script")
parser.add_argument("--tau_q_start", type=float, default=0.90, help="compatibility arg from old script")
parser.add_argument("--tau_q_end", type=float, default=0.70, help="compatibility arg from old script")
parser.add_argument("--threshold_schedule", type=str, default="linear", choices=["linear"], help="compatibility arg from old script")

parser.add_argument("--pseudo_metric_interval", type=int, default=200, help="dense-label debug interval")
parser.add_argument("--dense_label_key", type=str, default="gt_label", help="dense label key")
parser.add_argument("--save_boundary_interval", type=int, default=400, help="compatibility arg from old script")
parser.add_argument("--save_boundary_num_images", type=int, default=10, help="compatibility arg from old script")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class AgreementMambaScrib(torch.nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()
        self.mamba_unet = MambaUnet(in_chns=in_chns, class_num=class_num)

    def forward(self, x, scribble_label=None, update_memory=False, return_q=False):
        del scribble_label, update_memory, return_q
        logits_u, logits_m = self.mamba_unet(x)
        prob_u = torch.softmax(logits_u, dim=1)
        prob_m = torch.softmax(logits_m, dim=1)
        return {
            "logits_u": logits_u,
            "logits_m": logits_m,
            "prob_u": prob_u,
            "prob_m": prob_m,
            "prob_ensemble": 0.5 * (prob_u + prob_m),
        }


def create_model(num_classes, train_args):
    return AgreementMambaScrib(
        in_chns=train_args.in_chns,
        class_num=num_classes,
    ).cuda()


def get_current_consistency_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def safe_scalar(value):
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def safe_masked_mean(values, mask):
    if mask.sum() < 1:
        return np.nan
    return float(values[mask].mean().item())


def masked_label_accuracy(pred, target, mask):
    if mask.sum() < 1:
        return np.nan
    return float((pred[mask] == target[mask]).float().mean().item())


def build_dual_branch_pseudo_label(
    prob_u,
    prob_m,
    label,
    ignore_index,
    agree_thresh=0.70,
    disagree_thresh=0.80,
    margin_thresh=0.10,
    pseudo_mask_mode="unlabeled",
    eps=1e-8,
):
    prob_u = prob_u.detach()
    prob_m = prob_m.detach()

    if label.dim() == 4:
        label = label.squeeze(1)

    if pseudo_mask_mode == "unlabeled":
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == "all":
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError("Unsupported pseudo_mask_mode: {}".format(pseudo_mask_mode))

    conf_u, pred_u = torch.max(prob_u, dim=1)
    conf_m, pred_m = torch.max(prob_m, dim=1)

    same_pred = pred_u == pred_m
    diff_pred = pred_u != pred_m

    min_conf = torch.minimum(conf_u, conf_m)
    max_conf = torch.maximum(conf_u, conf_m)
    margin = torch.abs(conf_u - conf_m)

    reliable_agree = same_pred & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = diff_pred & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & candidate_mask
    reliable_mask = reliable_agree | reliable_disagree

    mean_pseudo = 0.5 * (prob_u + prob_m)
    choose_u = (conf_u > conf_m).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_u, prob_u, prob_m)

    soft_pseudo = torch.where(
        reliable_disagree.unsqueeze(1),
        high_conf_pseudo,
        mean_pseudo,
    )
    soft_pseudo = soft_pseudo / (soft_pseudo.sum(dim=1, keepdim=True) + eps)
    pseudo = torch.argmax(soft_pseudo, dim=1)
    pseudo_conf = torch.max(soft_pseudo, dim=1)[0]

    return {
        "soft_pseudo_u": soft_pseudo.detach(),
        "soft_pseudo_m": soft_pseudo.detach(),
        "pseudo_u": pseudo.detach(),
        "pseudo_m": pseudo.detach(),
        "R_u": reliable_mask.detach(),
        "R_m": reliable_mask.detach(),
        "reliable_agree": reliable_agree.detach(),
        "reliable_disagree": reliable_disagree.detach(),
        "branch_reliable": reliable_mask.detach(),
        "candidate_mask": candidate_mask.detach(),
        "conf_u": conf_u.detach(),
        "conf_m": conf_m.detach(),
        "margin": margin.detach(),
        "pseudo_conf": pseudo_conf.detach(),
        "agreement_u": same_pred.detach(),
        "agreement_m": same_pred.detach(),
    }


def build_empty_pseudo_info(prob_u, prob_m, label):
    conf_u, pred_u = torch.max(prob_u.detach(), dim=1)
    conf_m, pred_m = torch.max(prob_m.detach(), dim=1)
    mean_pseudo = 0.5 * (prob_u.detach() + prob_m.detach())
    pseudo = torch.argmax(mean_pseudo, dim=1)
    return {
        "soft_pseudo_u": mean_pseudo,
        "soft_pseudo_m": mean_pseudo,
        "pseudo_u": pseudo,
        "pseudo_m": pseudo,
        "R_u": torch.zeros_like(label, dtype=torch.bool),
        "R_m": torch.zeros_like(label, dtype=torch.bool),
        "reliable_agree": torch.zeros_like(label, dtype=torch.bool),
        "reliable_disagree": torch.zeros_like(label, dtype=torch.bool),
        "branch_reliable": torch.zeros_like(label, dtype=torch.bool),
        "candidate_mask": torch.ones_like(label, dtype=torch.bool),
        "conf_u": conf_u,
        "conf_m": conf_m,
        "margin": torch.abs(conf_u - conf_m),
        "pseudo_conf": torch.max(mean_pseudo, dim=1)[0],
        "agreement_u": (pred_u == pred_m),
        "agreement_m": (pred_u == pred_m),
    }


def summarize_reliable_masks(pseudo_info):
    r_u = pseudo_info["R_u"]
    r_m = pseudo_info["R_m"]
    stats = {
        "reliable_ratio_u": r_u.float().mean(),
        "reliable_ratio_m": r_m.float().mean(),
        "agreement_ratio_u": pseudo_info["agreement_u"].float().mean(),
        "agreement_ratio_m": pseudo_info["agreement_m"].float().mean(),
        "conf_u_mean": safe_masked_mean(pseudo_info["conf_u"], r_u),
        "conf_m_mean": safe_masked_mean(pseudo_info["conf_m"], r_m),
    }
    if "reliable_agree" in pseudo_info:
        stats["reliable_agree_ratio"] = pseudo_info["reliable_agree"].float().mean()
    if "reliable_disagree" in pseudo_info:
        stats["reliable_disagree_ratio"] = pseudo_info["reliable_disagree"].float().mean()
    if "branch_reliable" in pseudo_info:
        stats["branch_reliable_ratio"] = pseudo_info["branch_reliable"].float().mean()
    if "pseudo_conf" in pseudo_info:
        stats["pseudo_conf_u_mean"] = safe_masked_mean(pseudo_info["pseudo_conf"], r_u)
        stats["pseudo_conf_m_mean"] = safe_masked_mean(pseudo_info["pseudo_conf"], r_m)
    if "margin" in pseudo_info:
        stats["margin_u_mean"] = safe_masked_mean(pseudo_info["margin"], r_u)
        stats["margin_m_mean"] = safe_masked_mean(pseudo_info["margin"], r_m)
    return stats


class EnsembleInferenceWrapper(torch.nn.Module):
    def __init__(self, model, mode="ensemble"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        out = self.model(x, scribble_label=None, update_memory=False, return_q=False)
        if self.mode == "ensemble":
            return torch.log(out["prob_ensemble"] + 1e-8)
        if self.mode == "mamba":
            return out["logits_m"]
        if self.mode == "unet":
            return out["logits_u"]
        raise ValueError("Unsupported inference mode: {}".format(self.mode))


def validate(model, valloader, db_val, num_classes, writer, iter_num):
    metric_list_ens = 0.0
    metric_list_m = 0.0
    model.eval()

    ensemble_model = EnsembleInferenceWrapper(model, mode="ensemble")
    mamba_model = EnsembleInferenceWrapper(model, mode="mamba")

    for sampled_val in valloader:
        metric_ens = test_single_volume(
            sampled_val["image"],
            sampled_val["label"],
            ensemble_model,
            classes=num_classes,
        )
        metric_m = test_single_volume(
            sampled_val["image"],
            sampled_val["label"],
            mamba_model,
            classes=num_classes,
        )
        metric_list_ens += np.array(metric_ens)
        metric_list_m += np.array(metric_m)

    metric_list_ens = metric_list_ens / len(db_val)
    metric_list_m = metric_list_m / len(db_val)

    performance_ens = np.mean(metric_list_ens, axis=0)[0]
    mean_hd95_ens = np.mean(metric_list_ens, axis=0)[1]
    performance_m = np.mean(metric_list_m, axis=0)[0]
    mean_hd95_m = np.mean(metric_list_m, axis=0)[1]

    writer.add_scalar("info/val_mean_dice_ensemble", performance_ens, iter_num)
    writer.add_scalar("info/val_mean_hd95_ensemble", mean_hd95_ens, iter_num)
    writer.add_scalar("info/val_mean_dice_mamba", performance_m, iter_num)
    writer.add_scalar("info/val_mean_hd95_mamba", mean_hd95_m, iter_num)

    model.train()
    return performance_ens, mean_hd95_ens, performance_m, mean_hd95_m


def log_dense_debug(writer, pseudo_info, sampled_batch, iter_num, dense_label_key, num_classes):
    if dense_label_key not in sampled_batch:
        return
    gt_label = sampled_batch[dense_label_key].cuda()
    writer.add_scalar(
        "debug/pseudo_u_acc",
        safe_scalar(masked_label_accuracy(pseudo_info["pseudo_u"], gt_label, pseudo_info["R_u"])),
        iter_num,
    )
    writer.add_scalar(
        "debug/pseudo_m_acc",
        safe_scalar(masked_label_accuracy(pseudo_info["pseudo_m"], gt_label, pseudo_info["R_m"])),
        iter_num,
    )
    ensemble_pred = torch.argmax(
        0.5
        * (
            torch.nn.functional.one_hot(pseudo_info["pseudo_u"], num_classes=num_classes).permute(0, 3, 1, 2).float()
            + torch.nn.functional.one_hot(pseudo_info["pseudo_m"], num_classes=num_classes).permute(0, 3, 1, 2).float()
        ),
        dim=1,
    )
    reliable_union = pseudo_info["R_u"] | pseudo_info["R_m"]
    writer.add_scalar(
        "debug/ensemble_acc",
        safe_scalar(masked_label_accuracy(ensemble_pred, gt_label, reliable_union)),
        iter_num,
    )


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    model = create_model(num_classes=num_classes, train_args=train_args)

    db_train = ACDCDataSets(
        base_dir=train_args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        fold=train_args.fold,
        sup_type=train_args.sup_type,
        return_full_label=True,
    )
    db_val = ACDCDataSets(
        base_dir=train_args.root_path,
        fold=train_args.fold,
        split="val",
    )

    def worker_init_fn(worker_id):
        random.seed(train_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    model.train()
    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda().long()

            out = model(volume_batch)
            logits_u = out["logits_u"]
            logits_m = out["logits_m"]
            prob_u = out["prob_u"]
            prob_m = out["prob_m"]

            loss_pce_u = partial_ce_loss(logits_u, label_batch, ignore_index=num_classes)
            loss_pce_m = partial_ce_loss(logits_m, label_batch, ignore_index=num_classes)
            loss_pce = loss_pce_u + loss_pce_m

            tau_agree = linear_schedule(
                iter_num,
                train_args.warmup_iterations,
                max_iterations,
                train_args.agree_thresh_start,
                train_args.agree_thresh_end,
            )
            tau_disagree = linear_schedule(
                iter_num,
                train_args.warmup_iterations,
                max_iterations,
                train_args.disagree_thresh_start,
                train_args.disagree_thresh_end,
            )

            if iter_num >= train_args.warmup_iterations:
                pseudo_info = build_dual_branch_pseudo_label(
                    prob_u=prob_u,
                    prob_m=prob_m,
                    label=label_batch,
                    ignore_index=num_classes,
                    agree_thresh=tau_agree,
                    disagree_thresh=tau_disagree,
                    margin_thresh=train_args.margin_thresh,
                    pseudo_mask_mode=train_args.pseudo_mask_mode,
                )
                loss_u_to_m = masked_soft_ce_loss(logits_m, pseudo_info["soft_pseudo_u"], pseudo_info["R_u"])
                loss_m_to_u = masked_soft_ce_loss(logits_u, pseudo_info["soft_pseudo_m"], pseudo_info["R_m"])
                loss_cps = loss_u_to_m + loss_m_to_u
            else:
                pseudo_info = build_empty_pseudo_info(prob_u, prob_m, label_batch)
                loss_u_to_m = logits_u.new_tensor(0.0)
                loss_m_to_u = logits_u.new_tensor(0.0)
                loss_cps = logits_u.new_tensor(0.0)

            cps_weight = (
                get_current_consistency_weight(iter_num // len(trainloader), train_args)
                * train_args.pseudo_loss_weight
            )
            loss = loss_pce + cps_weight * train_args.lambda_cps * loss_cps

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            pseudo_stats = summarize_reliable_masks(pseudo_info)
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss_pce", loss_pce.item(), iter_num)
            writer.add_scalar("info/loss_pce_u", loss_pce_u.item(), iter_num)
            writer.add_scalar("info/loss_pce_m", loss_pce_m.item(), iter_num)
            writer.add_scalar("info/loss_cps", loss_cps.item(), iter_num)
            writer.add_scalar("info/loss_u_to_m", loss_u_to_m.item(), iter_num)
            writer.add_scalar("info/loss_m_to_u", loss_m_to_u.item(), iter_num)
            writer.add_scalar("info/cps_weight", cps_weight, iter_num)
            writer.add_scalar("pseudo/reliable_ratio_u", pseudo_stats["reliable_ratio_u"].item(), iter_num)
            writer.add_scalar("pseudo/reliable_ratio_m", pseudo_stats["reliable_ratio_m"].item(), iter_num)
            writer.add_scalar("pseudo/agreement_ratio_u", pseudo_stats["agreement_ratio_u"].item(), iter_num)
            writer.add_scalar("pseudo/agreement_ratio_m", pseudo_stats["agreement_ratio_m"].item(), iter_num)
            writer.add_scalar("pseudo/reliable_agree_ratio", safe_scalar(pseudo_stats.get("reliable_agree_ratio", 0.0)), iter_num)
            writer.add_scalar("pseudo/reliable_disagree_ratio", safe_scalar(pseudo_stats.get("reliable_disagree_ratio", 0.0)), iter_num)
            writer.add_scalar("pseudo/branch_reliable_ratio", safe_scalar(pseudo_stats.get("branch_reliable_ratio", 0.0)), iter_num)
            writer.add_scalar("pseudo/conf_u", safe_scalar(pseudo_stats["conf_u_mean"]), iter_num)
            writer.add_scalar("pseudo/conf_m", safe_scalar(pseudo_stats["conf_m_mean"]), iter_num)
            writer.add_scalar("pseudo/pseudo_conf_u", safe_scalar(pseudo_stats.get("pseudo_conf_u_mean", 0.0)), iter_num)
            writer.add_scalar("pseudo/pseudo_conf_m", safe_scalar(pseudo_stats.get("pseudo_conf_m_mean", 0.0)), iter_num)
            writer.add_scalar("pseudo/margin_u", safe_scalar(pseudo_stats.get("margin_u_mean", 0.0)), iter_num)
            writer.add_scalar("pseudo/margin_m", safe_scalar(pseudo_stats.get("margin_m_mean", 0.0)), iter_num)

            if iter_num % train_args.pseudo_metric_interval == 0:
                log_dense_debug(
                    writer,
                    pseudo_info,
                    sampled_batch,
                    iter_num,
                    train_args.dense_label_key,
                    num_classes,
                )

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss=%f, loss_pce=%f, loss_cps=%f, cps_weight=%f",
                    iter_num,
                    loss.item(),
                    loss_pce.item(),
                    loss_cps.item(),
                    cps_weight,
                )
                logging.info(
                    "thresholds: tau_agree=%.4f, tau_disagree=%.4f, margin=%.4f",
                    tau_agree,
                    tau_disagree,
                    train_args.margin_thresh,
                )
                logging.info(
                    "reliable_u=%f, reliable_m=%f, agree=%f, disagree=%f, conf_u=%f, conf_m=%f",
                    pseudo_stats["reliable_ratio_u"].item(),
                    pseudo_stats["reliable_ratio_m"].item(),
                    safe_scalar(pseudo_stats.get("reliable_agree_ratio", 0.0)),
                    safe_scalar(pseudo_stats.get("reliable_disagree_ratio", 0.0)),
                    safe_scalar(pseudo_stats["conf_u_mean"]),
                    safe_scalar(pseudo_stats["conf_m_mean"]),
                )

            if iter_num > 1 and iter_num % 400 == 0:
                performance_ens, mean_hd95_ens, performance_m, mean_hd95_m = validate(
                    model=model,
                    valloader=valloader,
                    db_val=db_val,
                    num_classes=num_classes,
                    writer=writer,
                    iter_num=iter_num,
                )

                if performance_ens > best_performance:
                    best_performance = performance_ens
                    save_ckpt = os.path.join(
                        snapshot_path,
                        "iter_{}_dice_{:.4f}.pth".format(iter_num, best_performance),
                    )
                    save_best = os.path.join(snapshot_path, "quanmambascrib_best_model.pth")
                    save_best_mamba = os.path.join(snapshot_path, "quanmambascrib_mamba_best_model.pth")
                    checkpoint = {
                        "iter_num": iter_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_performance": best_performance,
                        "args": vars(train_args),
                    }
                    torch.save(checkpoint, save_ckpt)
                    torch.save(checkpoint, save_best)
                    torch.save(model.state_dict(), os.path.join(snapshot_path, "quanmambascrib_best_model_state_dict.pth"))
                    torch.save(model.mamba_unet.state_dict(), save_best_mamba)
                    logging.info("save best model to %s", save_best)

                logging.info(
                    "iteration %d : val_mean_dice_ensemble=%f val_mean_hd95_ensemble=%f val_mean_dice_mamba=%f val_mean_hd95_mamba=%f",
                    iter_num,
                    performance_ens,
                    mean_hd95_ens,
                    performance_m,
                    mean_hd95_m,
                )

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, "iter_{}.pth".format(iter_num))
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

    snapshot_path = "../../checkpoints/{}_{}".format(args.data, args.exp)
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
