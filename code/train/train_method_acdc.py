import argparse
import logging
import math
import os
import random
import sys
from typing import Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.ema_optim import WeightEMA
from utils.pick_reliable_pixels import refine_high_confidence
from val import test_single_volume


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Weakly supervised ACDC training with two EMA teachers and "
            "entropy-guided winner-teacher distillation."
        )
    )

    # Dataset / experiment
    parser.add_argument("--root_path", type=str, default="../../data/ACDC")
    parser.add_argument("--exp", type=str, default="SDTNet_EntropyWinner")
    parser.add_argument("--data", type=str, default="ACDC")
    parser.add_argument("--fold", type=str, default="MAAGfold70")
    parser.add_argument("--sup_type", type=str, default="scribble")
    parser.add_argument("--snapshot_root", type=str, default="../../checkpoints")

    # Model / training
    parser.add_argument("--model", type=str, default="unet_hl")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--ignore_index", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--deterministic", type=int, default=1)

    # Existing pseudo-label branch
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_pseudo", type=float, default=0.5)

    # EMA teachers
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument(
        "--teacher_eval_mode",
        type=int,
        default=1,
        help=(
            "1: teacher runs in eval mode for stable entropy. "
            "0: teacher runs in train mode, matching the original script."
        ),
    )
    parser.add_argument(
        "--sync_teacher_buffers",
        type=int,
        default=1,
        help=(
            "Copy student buffers (for example BatchNorm running statistics) "
            "to the selected teacher after its EMA update."
        ),
    )

    # Entropy-guided consistency
    parser.add_argument("--lambda_uncertainty", type=float, default=0.5)
    parser.add_argument("--consistency_rampup", type=float, default=40.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--student_uncertainty_threshold", type=float, default=0.50)
    parser.add_argument("--teacher_uncertainty_threshold", type=float, default=0.35)
    parser.add_argument("--teacher_advantage_margin", type=float, default=0.05)
    parser.add_argument("--uncertainty_gate_temperature", type=float, default=0.10)
    parser.add_argument("--teacher_reliability_beta", type=float, default=2.0)
    parser.add_argument("--entropy_alignment_weight", type=float, default=0.10)
    parser.add_argument(
        "--hard_uncertainty_gate",
        type=int,
        default=0,
        help="1 uses a hard binary gate; 0 uses the recommended differentiable-shaped soft gate.",
    )
    parser.add_argument(
        "--uncertainty_on_unlabeled_only",
        type=int,
        default=1,
        help="Restrict entropy-guided distillation to pixels whose scribble label is ignore_index.",
    )

    # Logging / validation
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=400)
    parser.add_argument("--save_interval", type=int, default=3000)

    return parser


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def worker_init_fn_builder(seed: int):
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return worker_init_fn


def get_current_consistency_weight(current_epoch: float, args) -> float:
    """Sigmoid ramp-up; consistency_rampup is interpreted in epochs."""
    return float(ramps.sigmoid_rampup(current_epoch, args.consistency_rampup))


def extract_logits(model_output: torch.Tensor) -> torch.Tensor:
    """
    Support both networks returning logits directly and networks returning
    (logits, high_feature, low_feature), as in the original unet_hl code.
    """
    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise RuntimeError("The model returned an empty tuple/list.")
        return model_output[0]
    return model_output


@torch.no_grad()
def copy_model_buffers(source: nn.Module, target: nn.Module) -> None:
    """
    Keep non-parameter state such as BatchNorm running_mean/running_var aligned.
    WeightEMA implementations often update parameters only.
    """
    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())

    for name, source_buffer in source_buffers.items():
        target_buffer = target_buffers.get(name)
        if target_buffer is not None and target_buffer.shape == source_buffer.shape:
            target_buffer.copy_(source_buffer)


def configure_teacher_mode(teacher: nn.Module, use_eval_mode: bool) -> None:
    teacher.requires_grad_(False)
    if use_eval_mode:
        teacher.eval()
    else:
        teacher.train()


def safe_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def safe_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Avoid NaN when every target pixel is ignore_index."""
    if target.ne(ignore_index).any():
        return F.cross_entropy(logits, target, ignore_index=ignore_index)
    return logits.sum() * 0.0


def normalized_prediction_entropy(
    probabilities: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = probabilities.clamp_min(eps)
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    return entropy / math.log(probabilities.shape[1])


@torch.no_grad()
def teacher_match_uncertainty_stats(
    teacher_prob: torch.Tensor,
    scribble: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    teacher_pred = torch.argmax(teacher_prob, dim=1)
    teacher_uncertainty = normalized_prediction_entropy(teacher_prob)

    labeled_mask = scribble.ne(ignore_index)
    match_mask = labeled_mask & teacher_pred.eq(scribble)

    if match_mask.any():
        match_mean_uncertainty = teacher_uncertainty[match_mask].mean()
        match_mean_uncertainty_log = match_mean_uncertainty
    else:
        match_mean_uncertainty = teacher_uncertainty.new_tensor(float("inf"))
        match_mean_uncertainty_log = teacher_uncertainty.new_tensor(-1.0)

    if labeled_mask.any():
        labeled_mean_uncertainty = teacher_uncertainty[labeled_mask].mean()
    else:
        labeled_mean_uncertainty = teacher_uncertainty.new_tensor(float("inf"))

    return {
        "uncertainty_map": teacher_uncertainty.detach(),
        "match_mask": match_mask.detach(),
        "match_count": match_mask.sum().detach(),
        "labeled_count": labeled_mask.sum().detach(),
        "match_ratio": match_mask.float().mean().detach(),
        "match_mean_uncertainty": match_mean_uncertainty.detach(),
        "match_mean_uncertainty_log": match_mean_uncertainty_log.detach(),
        "labeled_mean_uncertainty": labeled_mean_uncertainty.detach(),
    }


# -----------------------------------------------------------------------------
# Entropy-guided winner-teacher loss
# -----------------------------------------------------------------------------
class EntropyGuidedWinnerTeacherLoss(nn.Module):
    """
    Pixel-wise teacher -> student distillation.

    A pixel receives a large weight when:
      * the student entropy is high;
      * the teacher entropy is low;
      * the teacher is more certain than the student.

    The primary loss is KL(teacher || student). A small Smooth-L1 term aligns
    the normalized entropy maps. The gate is detached so the student cannot
    manipulate its uncertainty simply to reduce its own loss weight.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 1.0,
        student_uncertainty_threshold: float = 0.50,
        teacher_uncertainty_threshold: float = 0.35,
        teacher_advantage_margin: float = 0.05,
        gate_temperature: float = 0.10,
        teacher_reliability_beta: float = 2.0,
        entropy_alignment_weight: float = 0.10,
        hard_gate: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")

        self.num_classes = num_classes
        self.temperature = temperature
        self.student_threshold = student_uncertainty_threshold
        self.teacher_threshold = teacher_uncertainty_threshold
        self.margin = teacher_advantage_margin
        self.gate_temperature = gate_temperature
        self.beta = teacher_reliability_beta
        self.entropy_alignment_weight = entropy_alignment_weight
        self.hard_gate = hard_gate
        self.eps = eps

    def normalized_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return normalized entropy map [B, H, W], approximately in [0, 1]."""
        probabilities = probabilities.clamp_min(self.eps)
        entropy = -(probabilities * probabilities.log()).sum(dim=1)
        return entropy / math.log(self.num_classes)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student_logits and teacher_logits must have identical shapes; "
                f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
            )

        teacher_logits = teacher_logits.detach()
        temperature = self.temperature

        # Distribution used by the distillation term.
        student_log_prob_t = F.log_softmax(student_logits / temperature, dim=1)
        teacher_prob_t = F.softmax(teacher_logits / temperature, dim=1)

        # Entropy should describe the actual predictive distributions, so T=1.
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_entropy = self.normalized_entropy(student_prob)
        teacher_entropy = self.normalized_entropy(teacher_prob)

        with torch.no_grad():
            student_entropy_gate = student_entropy.detach()
            teacher_entropy_gate = teacher_entropy.detach()

            hard_selected = (
                (student_entropy_gate >= self.student_threshold)
                & (teacher_entropy_gate <= self.teacher_threshold)
                & (
                    student_entropy_gate - teacher_entropy_gate
                    >= self.margin
                )
            )

            if self.hard_gate:
                weight_map = hard_selected.float()
                weight_map = weight_map * torch.exp(
                    -self.beta * teacher_entropy_gate
                )
            else:
                student_needs_help = torch.sigmoid(
                    (student_entropy_gate - self.student_threshold)
                    / self.gate_temperature
                )
                teacher_is_reliable = torch.sigmoid(
                    (self.teacher_threshold - teacher_entropy_gate)
                    / self.gate_temperature
                )
                teacher_is_better = torch.sigmoid(
                    (
                        student_entropy_gate
                        - teacher_entropy_gate
                        - self.margin
                    )
                    / self.gate_temperature
                )
                teacher_reliability = torch.exp(
                    -self.beta * teacher_entropy_gate
                )

                weight_map = (
                    student_needs_help
                    * teacher_is_reliable
                    * teacher_is_better
                    * teacher_reliability
                )

            valid_ratio = torch.ones((), device=student_logits.device)
            if pixel_mask is not None:
                if pixel_mask.ndim == 4 and pixel_mask.shape[1] == 1:
                    pixel_mask = pixel_mask[:, 0]
                if pixel_mask.shape != weight_map.shape:
                    raise ValueError(
                        f"pixel_mask shape {tuple(pixel_mask.shape)} does not match "
                        f"entropy map shape {tuple(weight_map.shape)}."
                    )
                valid_mask = pixel_mask.bool()
                valid_ratio = valid_mask.float().mean()
                weight_map = weight_map * valid_mask.float()
                hard_selected = hard_selected & valid_mask

        # Pixel-wise KL(teacher || student).
        pixel_kl = F.kl_div(
            student_log_prob_t,
            teacher_prob_t,
            reduction="none",
        ).sum(dim=1)
        pixel_kl = pixel_kl * (temperature ** 2)

        # Soft-gate weights must retain their absolute magnitude. Dividing by
        # weight_map.sum() would cancel a globally small confidence weight.
        # For a hard gate, normalize by the number of selected pixels; for a
        # soft gate, normalize by the number of valid candidate pixels.
        if self.hard_gate:
            normalizer = hard_selected.float().sum().clamp_min(1.0)
        elif pixel_mask is not None:
            normalizer = pixel_mask.bool().float().sum().clamp_min(1.0)
        else:
            normalizer = torch.tensor(
                weight_map.numel(),
                device=weight_map.device,
                dtype=weight_map.dtype,
            )

        loss_distillation = (weight_map * pixel_kl).sum() / normalizer

        pixel_entropy_alignment = F.smooth_l1_loss(
            student_entropy,
            teacher_entropy,
            reduction="none",
        )
        loss_entropy_alignment = (
            weight_map * pixel_entropy_alignment
        ).sum() / normalizer

        total_loss = (
            loss_distillation
            + self.entropy_alignment_weight * loss_entropy_alignment
        )

        stats = {
            "distillation": loss_distillation.detach(),
            "entropy_alignment": loss_entropy_alignment.detach(),
            "student_entropy": student_entropy.detach().mean(),
            "teacher_entropy": teacher_entropy.detach().mean(),
            "selected_ratio": hard_selected.float().mean().detach(),
            "valid_ratio": valid_ratio.detach(),
            "weight_mean": weight_map.mean().detach(),
            "weight_sum": weight_map.sum().detach(),
        }
        return total_loss, stats


# -----------------------------------------------------------------------------
# Validation / checkpointing
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    valloader: DataLoader,
    db_val,
    num_classes: int,
    writer: SummaryWriter,
    iter_num: int,
) -> Tuple[float, float]:
    model.eval()
    metric_items = []

    for sampled_batch in valloader:
        metric_i = test_single_volume(
            sampled_batch["image"],
            sampled_batch["label"],
            model,
            classes=num_classes,
        )
        metric_items.append(np.asarray(metric_i, dtype=np.float64))

    if not metric_items:
        raise RuntimeError("Validation loader is empty.")

    metric_list = np.mean(np.stack(metric_items, axis=0), axis=0)

    for class_i in range(num_classes - 1):
        writer.add_scalar(
            "info/val_{}_dice".format(class_i + 1),
            metric_list[class_i, 0],
            iter_num,
        )
        writer.add_scalar(
            "info/val_{}_hd95".format(class_i + 1),
            metric_list[class_i, 1],
            iter_num,
        )

    performance = float(np.mean(metric_list, axis=0)[0])
    mean_hd95 = float(np.mean(metric_list, axis=0)[1])

    writer.add_scalar("info/val_mean_dice", performance, iter_num)
    writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

    model.train()
    return performance, mean_hd95


def save_full_checkpoint(
    path: str,
    student: nn.Module,
    teacher1: nn.Module,
    teacher2: nn.Module,
    optimizer: optim.Optimizer,
    iter_num: int,
    best_performance: float,
    args,
) -> None:
    torch.save(
        {
            "student": student.state_dict(),
            "teacher1": teacher1.state_dict(),
            "teacher2": teacher2.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_performance": best_performance,
            "args": vars(args),
        },
        path,
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args, snapshot_path: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this training script but is not available.")

    device = torch.device("cuda")
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    def create_model(ema: bool = False) -> nn.Module:
        network = net_factory(
            net_type=args.model,
            in_chns=args.in_channels,
            class_num=args.num_classes,
        )
        if network is None:
            raise RuntimeError(
                f"net_factory returned None for model type '{args.model}'."
            )
        if ema:
            for parameter in network.parameters():
                parameter.detach_()
        return network

    # Models
    student = create_model(ema=False).to(device)
    teacher1 = create_model(ema=True).to(device)
    teacher2 = create_model(ema=True).to(device)

    student.train()
    configure_teacher_mode(teacher1, bool(args.teacher_eval_mode))
    configure_teacher_mode(teacher2, bool(args.teacher_eval_mode))

    # Dataset / data loader
    db_train = ACDCDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )
    db_val = ACDCDataSets(
        base_dir=args.root_path,
        fold=args.fold,
        split="val",
    )

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn_builder(args.seed),
        drop_last=False,
    )
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    if len(trainloader) == 0:
        raise RuntimeError("Training loader is empty.")

    # Optimizer / losses
    optimizer = optim.SGD(
        student.parameters(),
        lr=base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    tea1_optimizer = WeightEMA(student, teacher1, args.ema_decay)
    tea2_optimizer = WeightEMA(student, teacher2, args.ema_decay)

    ce_loss = CrossEntropyLoss(ignore_index=args.ignore_index)
    dice_loss = losses.pDLoss(
        num_classes,
        ignore_index=args.ignore_index,
    )
    uncertainty_loss_fn = EntropyGuidedWinnerTeacherLoss(
        num_classes=num_classes,
        temperature=args.distill_temperature,
        student_uncertainty_threshold=args.student_uncertainty_threshold,
        teacher_uncertainty_threshold=args.teacher_uncertainty_threshold,
        teacher_advantage_margin=args.teacher_advantage_margin,
        gate_temperature=args.uncertainty_gate_temperature,
        teacher_reliability_beta=args.teacher_reliability_beta,
        entropy_alignment_weight=args.entropy_alignment_weight,
        hard_gate=bool(args.hard_uncertainty_gate),
    ).to(device)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    teacher1_win_count = 0
    teacher2_win_count = 0

    iterator = tqdm(range(max_epoch), ncols=90)

    for epoch_num in iterator:
        for batch_idx, sampled in enumerate(trainloader):
            image = sampled["image"].to(device, non_blocking=True)
            scribble = sampled["label"].to(
                device,
                non_blocking=True,
            ).long()

            # Teacher predictions are targets only.
            with torch.no_grad():
                teacher1_logits = extract_logits(teacher1(image))
                teacher2_logits = extract_logits(teacher2(image))
                teacher1_prob = F.softmax(teacher1_logits, dim=1)
                teacher2_prob = F.softmax(teacher2_logits, dim=1)

            student_logits = extract_logits(student(image))
            student_prob = F.softmax(student_logits, dim=1)

            # Sparse supervised loss. The safe wrapper prevents NaN if a
            # sampled batch unexpectedly contains no labeled scribble pixel.
            loss_ce_student = safe_cross_entropy(
                student_logits,
                scribble,
                args.ignore_index,
            )

            # Winner teacher: lower mean uncertainty on correctly predicted
            # scribble pixels. If both teachers fail to match any scribble
            # pixel in the batch, fall back to the original CE criterion.
            with torch.no_grad():
                teacher1_match_stats = teacher_match_uncertainty_stats(
                    teacher1_prob,
                    scribble,
                    args.ignore_index,
                )
                teacher2_match_stats = teacher_match_uncertainty_stats(
                    teacher2_prob,
                    scribble,
                    args.ignore_index,
                )

                loss_ce_teacher1 = safe_cross_entropy(
                    teacher1_logits,
                    scribble,
                    args.ignore_index,
                )
                loss_ce_teacher2 = safe_cross_entropy(
                    teacher2_logits,
                    scribble,
                    args.ignore_index,
                )

                pseudo_label1 = refine_high_confidence(
                    teacher1_prob,
                    threshold=args.confidence_threshold,
                )
                pseudo_label2 = refine_high_confidence(
                    teacher2_prob,
                    threshold=args.confidence_threshold,
                )

                teacher1_has_match = bool(teacher1_match_stats["match_count"].item() > 0)
                teacher2_has_match = bool(teacher2_match_stats["match_count"].item() > 0)

                if teacher1_has_match and teacher2_has_match:
                    teacher1_wins = (
                        safe_scalar(teacher1_match_stats["match_mean_uncertainty"])
                        < safe_scalar(teacher2_match_stats["match_mean_uncertainty"])
                    )
                    winner_reason = "matched_uncertainty"
                elif teacher1_has_match and not teacher2_has_match:
                    teacher1_wins = True
                    winner_reason = "teacher1_has_matches"
                elif teacher2_has_match and not teacher1_has_match:
                    teacher1_wins = False
                    winner_reason = "teacher2_has_matches"
                else:
                    teacher1_wins = safe_scalar(loss_ce_teacher1) < safe_scalar(loss_ce_teacher2)
                    winner_reason = "ce_fallback"

            if teacher1_wins:
                winner_mode = 1
                winner_teacher_logits = teacher1_logits
                winner_pseudo_label = pseudo_label1
                winner_match_stats = teacher1_match_stats
                teacher1_win_count += 1
            else:
                winner_mode = 2
                winner_teacher_logits = teacher2_logits
                winner_pseudo_label = pseudo_label2
                winner_match_stats = teacher2_match_stats
                teacher2_win_count += 1

            # Keep the original hard pseudo-label branch. Guard against the
            # early-training case where every pseudo pixel is ignored.
            winner_pseudo_label = winner_pseudo_label.long()
            pseudo_valid = winner_pseudo_label.ne(args.ignore_index)
            if pseudo_valid.any():
                loss_pseudo_ce = ce_loss(student_logits, winner_pseudo_label)
                loss_pseudo_dice = dice_loss(
                    student_prob,
                    winner_pseudo_label.unsqueeze(1),
                )
            else:
                loss_pseudo_ce = student_logits.sum() * 0.0
                loss_pseudo_dice = student_logits.sum() * 0.0
            loss_pseudo = loss_pseudo_ce + loss_pseudo_dice

            # Apply uncertainty distillation mainly on unlabeled pixels.
            uncertainty_pixel_mask = None
            if args.uncertainty_on_unlabeled_only:
                uncertainty_pixel_mask = scribble.eq(args.ignore_index)

            loss_uncertainty, unc_stats = uncertainty_loss_fn(
                student_logits=student_logits,
                teacher_logits=winner_teacher_logits,
                pixel_mask=uncertainty_pixel_mask,
            )

            current_epoch = epoch_num + batch_idx / max(len(trainloader), 1)
            consistency_weight = get_current_consistency_weight(
                current_epoch,
                args,
            )

            total_loss = (
                loss_ce_student
                + args.lambda_pseudo * loss_pseudo
                + args.lambda_uncertainty
                * consistency_weight
                * loss_uncertainty
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # Update only the winner teacher, preserving the original design.
            if winner_mode == 1:
                tea1_optimizer.step()
                if args.sync_teacher_buffers:
                    copy_model_buffers(student, teacher1)
                configure_teacher_mode(teacher1, bool(args.teacher_eval_mode))
            else:
                tea2_optimizer.step()
                if args.sync_teacher_buffers:
                    copy_model_buffers(student, teacher2)
                configure_teacher_mode(teacher2, bool(args.teacher_eval_mode))

            # Polynomial LR decay.
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # TensorBoard
            writer.add_scalar("train/loss_total", total_loss.item(), iter_num)
            writer.add_scalar("train/loss_ce_student", loss_ce_student.item(), iter_num)
            writer.add_scalar("train/loss_pseudo", loss_pseudo.item(), iter_num)
            writer.add_scalar("train/loss_pseudo_ce", loss_pseudo_ce.item(), iter_num)
            writer.add_scalar("train/loss_pseudo_dice", loss_pseudo_dice.item(), iter_num)
            writer.add_scalar("train/loss_uncertainty", loss_uncertainty.item(), iter_num)
            writer.add_scalar(
                "train/loss_uncertainty_kl",
                safe_scalar(unc_stats["distillation"]),
                iter_num,
            )
            writer.add_scalar(
                "train/loss_entropy_alignment",
                safe_scalar(unc_stats["entropy_alignment"]),
                iter_num,
            )
            writer.add_scalar(
                "train/student_entropy",
                safe_scalar(unc_stats["student_entropy"]),
                iter_num,
            )
            writer.add_scalar(
                "train/winner_teacher_entropy",
                safe_scalar(unc_stats["teacher_entropy"]),
                iter_num,
            )
            writer.add_scalar(
                "train/uncertainty_selected_ratio",
                safe_scalar(unc_stats["selected_ratio"]),
                iter_num,
            )
            writer.add_scalar(
                "train/uncertainty_weight_mean",
                safe_scalar(unc_stats["weight_mean"]),
                iter_num,
            )
            writer.add_scalar(
                "train/consistency_weight",
                consistency_weight,
                iter_num,
            )
            writer.add_scalar("train/lr", lr_, iter_num)
            writer.add_scalar("train/winner_teacher", winner_mode, iter_num)
            writer.add_scalar(
                "train/teacher1_match_count",
                safe_scalar(teacher1_match_stats["match_count"].float()),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher2_match_count",
                safe_scalar(teacher2_match_stats["match_count"].float()),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher1_match_ratio",
                safe_scalar(teacher1_match_stats["match_ratio"]),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher2_match_ratio",
                safe_scalar(teacher2_match_stats["match_ratio"]),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher1_match_mean_uncertainty",
                safe_scalar(teacher1_match_stats["match_mean_uncertainty_log"]),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher2_match_mean_uncertainty",
                safe_scalar(teacher2_match_stats["match_mean_uncertainty_log"]),
                iter_num,
            )
            writer.add_scalar(
                "train/winner_match_mean_uncertainty",
                safe_scalar(winner_match_stats["match_mean_uncertainty_log"]),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher1_labeled_mean_uncertainty",
                safe_scalar(teacher1_match_stats["labeled_mean_uncertainty"]),
                iter_num,
            )
            writer.add_scalar(
                "train/teacher2_labeled_mean_uncertainty",
                safe_scalar(teacher2_match_stats["labeled_mean_uncertainty"]),
                iter_num,
            )

            iterator.set_description(
                "iter {:05d} loss {:.4f} unc {:.4f} T{} {}".format(
                    iter_num,
                    total_loss.item(),
                    loss_uncertainty.item(),
                    winner_mode,
                    winner_reason,
                )
            )

            if iter_num % args.log_interval == 0:
                logging.info(
                    "iteration %d | total %.6f | ce %.6f | pseudo %.6f | "
                    "unc %.6f | ramp %.4f | selected %.4f | weight %.6f | winner T%d | "
                    "reason %s | T1_match %d | T2_match %d | T1_u %.6f | T2_u %.6f",
                    iter_num,
                    total_loss.item(),
                    loss_ce_student.item(),
                    loss_pseudo.item(),
                    loss_uncertainty.item(),
                    consistency_weight,
                    safe_scalar(unc_stats["selected_ratio"]),
                    safe_scalar(unc_stats["weight_mean"]),
                    winner_mode,
                    winner_reason,
                    int(teacher1_match_stats["match_count"].item()),
                    int(teacher2_match_stats["match_count"].item()),
                    safe_scalar(teacher1_match_stats["match_mean_uncertainty_log"]),
                    safe_scalar(teacher2_match_stats["match_mean_uncertainty_log"]),
                )

            # Validation
            if iter_num > 1 and iter_num % args.val_interval == 0:
                performance, mean_hd95 = validate(
                    model=student,
                    valloader=valloader,
                    db_val=db_val,
                    num_classes=num_classes,
                    writer=writer,
                    iter_num=iter_num,
                )

                logging.info(
                    "iteration %d | mean_dice %.6f | mean_hd95 %.6f",
                    iter_num,
                    performance,
                    mean_hd95,
                )

                if performance > best_performance:
                    best_performance = performance
                    score_path = os.path.join(
                        snapshot_path,
                        "iter_{}_dice_{}.pth".format(
                            iter_num,
                            round(best_performance, 4),
                        ),
                    )
                    best_model_path = os.path.join(
                        snapshot_path,
                        "{}_best_model.pth".format(args.model),
                    )
                    best_checkpoint_path = os.path.join(
                        snapshot_path,
                        "best_training_checkpoint.pth",
                    )

                    torch.save(student.state_dict(), score_path)
                    torch.save(student.state_dict(), best_model_path)
                    save_full_checkpoint(
                        path=best_checkpoint_path,
                        student=student,
                        teacher1=teacher1,
                        teacher2=teacher2,
                        optimizer=optimizer,
                        iter_num=iter_num,
                        best_performance=best_performance,
                        args=args,
                    )

            # Periodic checkpoint
            if iter_num % args.save_interval == 0:
                model_path = os.path.join(
                    snapshot_path,
                    "iter_{}.pth".format(iter_num),
                )
                checkpoint_path = os.path.join(
                    snapshot_path,
                    "iter_{}_training_checkpoint.pth".format(iter_num),
                )
                torch.save(student.state_dict(), model_path)
                save_full_checkpoint(
                    path=checkpoint_path,
                    student=student,
                    teacher1=teacher1,
                    teacher2=teacher2,
                    optimizer=optimizer,
                    iter_num=iter_num,
                    best_performance=best_performance,
                    args=args,
                )
                logging.info("Saved model to %s", model_path)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.add_scalar("train/teacher1_win_count", teacher1_win_count, iter_num)
    writer.add_scalar("train/teacher2_win_count", teacher2_win_count, iter_num)
    writer.close()

    logging.info(
        "Training finished. Teacher-1 wins: %d; Teacher-2 wins: %d; best Dice: %.6f",
        teacher1_win_count,
        teacher2_win_count,
        best_performance,
    )
    return "Training Finished!"


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed, bool(args.deterministic))

    snapshot_path = os.path.join(
        args.snapshot_root,
        "{}_{}".format(args.data, args.exp),
    )
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Arguments: %s", args)

    train(args, snapshot_path)


if __name__ == "__main__":
    main()
