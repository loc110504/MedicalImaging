import argparse
import logging
import os
import random
import sys
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from val import test_single_volume_scribblevs


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="UF_DualScribble", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--model", type=str, default="unet", help="network name")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--max_iterations", type=int, default=60000, help="max training iterations")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="consistency rampup")
parser.add_argument("--evidence_activation", type=str, default="relu", choices=["relu", "softplus", "exp"], help="evidence activation")

parser.add_argument("--uncertainty_type", type=str, default="evidential", help="uncertainty type")
parser.add_argument("--tau_conf", type=float, default=0.35, help="confident uncertainty threshold")
parser.add_argument("--tau_uncertain", type=float, default=0.75, help="uncertain boundary threshold")
parser.add_argument("--pseudo_conf_thresh", type=float, default=0.60, help="pseudo confidence threshold")
parser.add_argument("--uncertainty_temp", type=float, default=0.5, help="uncertainty temperature")
parser.add_argument("--lambda_mutual_unc", type=float, default=0.0, help="teacher discrepancy uncertainty weight")

parser.add_argument("--lambda_pseudo_conf", type=float, default=1.0, help="student confident pseudo weight")
parser.add_argument("--lambda_pseudo_boundary", type=float, default=0.5, help="student boundary pseudo weight")
parser.add_argument("--lambda_boundary", type=float, default=0.2, help="boundary loss weight")
parser.add_argument("--lambda_cross_teacher", type=float, default=0.5, help="cross teacher weight")
parser.add_argument("--lambda_feedback", type=float, default=0.05, help="feedback weight")
parser.add_argument("--lambda_unc_cons", type=float, default=0.1, help="uncertainty consistency weight")

parser.add_argument("--feedback_start_iter", type=int, default=3000, help="feedback warmup iterations")
parser.add_argument("--feedback_interval", type=int, default=1, help="feedback interval")
parser.add_argument("--feedback_lr", type=float, default=0.1, help="virtual feedback step size")
parser.add_argument("--feedback_clip", type=float, default=1.0, help="feedback delta clip")
parser.add_argument("--normalize_trial_grad", type=int, default=1, help="normalize trial gradients")
parser.add_argument("--feedback_use_boundary_anchor", type=int, default=0, help="use boundary anchor in feedback")
parser.add_argument("--lambda_feedback_boundary_anchor", type=float, default=0.1, help="boundary anchor weight")
parser.add_argument("--feedback_bn_eval", type=int, default=0, help="set BN eval during feedback trial update")

parser.add_argument("--boundary_radius", type=int, default=3, help="boundary radius")
parser.add_argument("--boundary_mask_dilate", type=int, default=1, help="extra boundary dilation")
parser.add_argument("--use_boundary_region", type=int, default=1, help="use uncertainty boundary region")

parser.add_argument("--teacher_diversity_mode", type=str, default="aug_dropout", help="teacher diversity mode")
parser.add_argument("--teacher_ema_alpha", type=float, default=0.0, help="optional teacher EMA smoothing")
parser.add_argument("--save_debug_vis", type=int, default=1, help="save debug visualizations")
parser.add_argument("--debug_vis_interval", type=int, default=1000, help="debug visualization interval")
parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient clipping norm")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_current_consistency_weight(epoch, current_args):
    return ramps.sigmoid_rampup(epoch, current_args.consistency_rampup)


def resolve_path(path_value, anchor_dir):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(anchor_dir, path_value))


def unpack_model_output(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return unpack_model_output(self.model(x))


def zero_tensor(device):
    return torch.tensor(0.0, device=device)


def label_to_onehot(label, num_classes):
    return F.one_hot(label.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def minmax_norm(x, eps=1e-8):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def symmetric_kl(p, q, eps=1e-8):
    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    kl_pq = (p * (log_p - log_q)).sum(dim=1, keepdim=True)
    kl_qp = (q * (log_q - log_p)).sum(dim=1, keepdim=True)
    return 0.5 * (kl_pq + kl_qp)


def logits_to_evidence(logits, activation="relu"):
    if activation == "relu":
        return F.relu(logits)
    if activation == "softplus":
        return F.softplus(logits)
    if activation == "exp":
        return torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
    raise ValueError("Unsupported evidence activation: {}".format(activation))


def evidential_uncertainty_from_logits(logits, num_classes, activation="relu", eps=1e-8):
    evidence = logits_to_evidence(logits, activation=activation)
    alpha = evidence + 1.0
    return float(num_classes) / (torch.sum(alpha, dim=1, keepdim=True) + eps)


def maybe_dilate_mask(mask, dilate_iter):
    if dilate_iter <= 0:
        return mask
    kernel_size = 2 * dilate_iter + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=dilate_iter)


def boundary_from_label(label, num_classes, radius=3):
    onehot = label_to_onehot(label, num_classes)
    k = 2 * radius + 1
    dil = F.max_pool2d(onehot, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - onehot, kernel_size=k, stride=1, padding=radius)
    grad = (dil - ero).clamp(0.0, 1.0)
    return grad.max(dim=1, keepdim=True)[0]


def boundary_from_prob(prob, radius=3):
    k = 2 * radius + 1
    dil = F.max_pool2d(prob, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - prob, kernel_size=k, stride=1, padding=radius)
    grad = (dil - ero).clamp(0.0, 1.0)
    return grad.max(dim=1, keepdim=True)[0]


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


def masked_soft_ce_loss(logits, target_prob, mask, eps=1e-8):
    if mask.sum() < 1:
        return zero_tensor(logits.device)
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return masked_mean(ce_map, mask, eps)


def masked_hard_ce_loss(logits, target_label, mask, num_classes, eps=1e-8):
    if mask.sum() < 1:
        return zero_tensor(logits.device)
    target_prob = label_to_onehot(target_label, num_classes)
    return masked_soft_ce_loss(logits, target_prob, mask, eps)


def masked_soft_ce_loss_weighted(logits, target_prob, mask, weight=None, eps=1e-8):
    if weight is not None:
        mask = mask * weight
    if mask.sum() < 1:
        return zero_tensor(logits.device)
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return masked_mean(ce_map, mask, eps)


def boundary_bce_loss(logits, target_boundary, mask=None, radius=3, eps=1e-8):
    pred_boundary = boundary_from_prob(torch.softmax(logits, dim=1), radius=radius)
    loss_map = F.binary_cross_entropy(
        pred_boundary.clamp(1e-6, 1.0 - 1e-6),
        target_boundary.float().detach(),
        reduction="none",
    )
    if mask is None:
        return loss_map.mean()
    if mask.sum() < 1:
        return zero_tensor(logits.device)
    return masked_mean(loss_map, mask, eps)


def uncertainty_consistency_loss(student_unc, teacher_unc, loss_type="l1", mask=None):
    if mask is not None and mask.sum() < 1:
        return zero_tensor(student_unc.device)
    if loss_type == "mse":
        loss_map = (student_unc - teacher_unc).pow(2)
    else:
        loss_map = (student_unc - teacher_unc).abs()
    if mask is None:
        return loss_map.mean()
    return masked_mean(loss_map, mask)


def set_batchnorm_eval(module):
    bn_states = []
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            bn_states.append((submodule, submodule.training))
            submodule.eval()
    return bn_states


def restore_batchnorm_states(bn_states):
    for module, was_training in bn_states:
        module.train(was_training)


def compute_anchor_loss(student, volume_batch, label_batch, ce_loss, current_args, boundary_band=None):
    logits = unpack_model_output(student(volume_batch))
    loss = ce_loss(logits, label_batch.long())
    if current_args.feedback_use_boundary_anchor and boundary_band is not None:
        loss_bd = boundary_bce_loss(
            logits,
            target_boundary=boundary_band,
            mask=boundary_band,
            radius=current_args.boundary_radius,
        )
        loss = loss + current_args.lambda_feedback_boundary_anchor * loss_bd
    return loss


def compute_feedback_delta(
    student,
    volume_batch,
    label_batch,
    trial_loss_fn,
    ce_loss,
    current_args,
    boundary_band=None,
):
    student.train()
    bn_states = set_batchnorm_eval(student) if current_args.feedback_bn_eval else []
    try:
        with torch.no_grad():
            anchor_before = compute_anchor_loss(
                student, volume_batch, label_batch, ce_loss, current_args, boundary_band
            ).detach()

        backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
        logits = unpack_model_output(student(volume_batch))
        trial_loss = trial_loss_fn(logits)

        if (not torch.is_tensor(trial_loss)) or (not trial_loss.requires_grad):
            student.load_state_dict(backup)
            return zero_tensor(volume_batch.device)
        if not torch.isfinite(trial_loss):
            student.load_state_dict(backup)
            return zero_tensor(volume_batch.device)

        params = [p for p in student.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            trial_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_sq_sum = zero_tensor(volume_batch.device)
        has_grad = False
        for grad in grads:
            if grad is not None:
                grad_sq_sum = grad_sq_sum + grad.detach().pow(2).sum()
                has_grad = True

        if not has_grad:
            student.load_state_dict(backup)
            return zero_tensor(volume_batch.device)

        grad_norm = torch.sqrt(grad_sq_sum + 1e-8)
        if current_args.normalize_trial_grad:
            step_scale = current_args.feedback_lr / (grad_norm + 1e-8)
        else:
            step_scale = current_args.feedback_lr

        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-float(step_scale))

        with torch.no_grad():
            anchor_after = compute_anchor_loss(
                student, volume_batch, label_batch, ce_loss, current_args, boundary_band
            ).detach()
        student.load_state_dict(backup)

        delta = anchor_before - anchor_after
        if current_args.feedback_clip > 0:
            delta = delta.clamp(-current_args.feedback_clip, current_args.feedback_clip)
        return delta.detach()
    finally:
        if bn_states:
            restore_batchnorm_states(bn_states)


def save_debug_visualization(
    volume,
    label,
    pred_t1,
    pred_t2,
    pseudo,
    u_ens,
    conf_mask,
    boundary_mask,
    boundary_band,
    unreliable_mask,
    save_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    panels = [
        ("Input", volume, "gray", None),
        ("Scribble", np.ma.masked_where(label == 4, label), "tab10", None),
        ("Teacher1", pred_t1, "tab10", None),
        ("Teacher2", pred_t2, "tab10", None),
        ("Pseudo", pseudo, "tab10", None),
        ("Uncertainty", u_ens, "viridis", (0.0, 1.0)),
        ("Conf Mask", conf_mask, "gray", (0.0, 1.0)),
        ("Boundary Mask", boundary_mask, "gray", (0.0, 1.0)),
        ("Boundary Band", boundary_band, "gray", (0.0, 1.0)),
        ("Unreliable", unreliable_mask, "gray", (0.0, 1.0)),
    ]

    for axis, (title, image, cmap, value_range) in zip(axes.flatten(), panels):
        if value_range is None:
            axis.imshow(image, cmap=cmap)
        else:
            axis.imshow(image, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def maybe_apply_grad_clip(model, grad_clip):
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


def maybe_update_teacher_ema(model, shadow_state, alpha):
    if alpha <= 0:
        return shadow_state
    if shadow_state is None:
        shadow_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    current_state = model.state_dict()
    with torch.no_grad():
        for key in shadow_state:
            shadow_state[key].mul_(alpha).add_(current_state[key].detach(), alpha=1.0 - alpha)
        model.load_state_dict(shadow_state)
    return shadow_state


def validate_student(student, valloader, num_classes, patch_size):
    student_wrapper = LogitsOnlyWrapper(student)
    metric_list = 0.0
    for sampled_batch in valloader:
        metric_i = test_single_volume_scribblevs(
            sampled_batch["image"],
            sampled_batch["label"],
            student_wrapper,
            classes=num_classes,
            patch_size=patch_size,
        )
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(valloader.dataset)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    return metric_list, performance, mean_hd95


def create_model(num_classes=4):
    return net_factory(net_type=args.model, in_chns=1, class_num=num_classes)


def train(current_args, snapshot_path):
    if current_args.uncertainty_type != "evidential":
        raise ValueError("Only evidential uncertainty is supported in this implementation.")

    base_lr = current_args.base_lr
    num_classes = current_args.num_classes
    batch_size = current_args.batch_size
    max_iterations = current_args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(current_args.seed + worker_id)

    teacher1 = create_model(num_classes=num_classes)
    teacher2 = create_model(num_classes=num_classes)
    student = create_model(num_classes=num_classes)
    teacher1.train()
    teacher2.train()
    student.train()

    db_train = ACDCDataSets(
        base_dir=current_args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(current_args.patch_size)]),
        fold=current_args.fold,
        sup_type=current_args.sup_type,
    )
    db_val = ACDCDataSets(base_dir=current_args.root_path, split="val", fold=current_args.fold)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer_t1 = optim.SGD(teacher1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_t2 = optim.SGD(teacher2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_s = optim.SGD(student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(ignore_index=4)
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    debug_dir = os.path.join(snapshot_path, "debug_vis")
    if current_args.save_debug_vis:
        os.makedirs(debug_dir, exist_ok=True)

    logging.info("%d iterations per epoch", len(trainloader))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    last_performance = 0.0
    teacher1_ema_state = None
    teacher2_ema_state = None
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for sampled in trainloader:
            image = sampled["image"].cuda()
            label_batch = sampled["label"].cuda().long()
            scribble_mask = (label_batch != 4).float().unsqueeze(1)
            unlabeled_mask = (label_batch == 4).float().unsqueeze(1)

            logits_t1 = unpack_model_output(teacher1(image))
            logits_t2 = unpack_model_output(teacher2(image))
            prob_t1 = torch.softmax(logits_t1, dim=1)
            prob_t2 = torch.softmax(logits_t2, dim=1)

            u1 = evidential_uncertainty_from_logits(
                logits_t1, num_classes, activation=current_args.evidence_activation
            )
            u2 = evidential_uncertainty_from_logits(
                logits_t2, num_classes, activation=current_args.evidence_activation
            )
            u_mean = 0.5 * (u1 + u2)
            if current_args.lambda_mutual_unc > 0:
                u_kl = minmax_norm(symmetric_kl(prob_t1, prob_t2))
                u_ens = torch.clamp(u_mean + current_args.lambda_mutual_unc * u_kl, 0.0, 1.0)
            else:
                u_ens = u_mean

            w1 = torch.exp(-u1 / current_args.uncertainty_temp)
            w2 = torch.exp(-u2 / current_args.uncertainty_temp)
            w_sum = w1 + w2 + 1e-8
            w1 = w1 / w_sum
            w2 = w2 / w_sum

            pseudo_soft = (w1 * prob_t1 + w2 * prob_t2).detach()
            pseudo_hard = torch.argmax(pseudo_soft, dim=1)
            pseudo_conf = torch.max(pseudo_soft, dim=1, keepdim=True)[0]
            boundary_band = boundary_from_label(
                pseudo_hard, current_args.num_classes, current_args.boundary_radius
            ).detach()
            boundary_band = maybe_dilate_mask(boundary_band, current_args.boundary_mask_dilate)

            conf_mask = (
                (u_ens < current_args.tau_conf)
                & (pseudo_conf > current_args.pseudo_conf_thresh)
                & (unlabeled_mask > 0)
            ).float()

            if current_args.use_boundary_region:
                boundary_mask = (
                    (u_ens >= current_args.tau_conf)
                    & (u_ens <= current_args.tau_uncertain)
                    & (boundary_band > 0)
                    & (unlabeled_mask > 0)
                ).float()
            else:
                boundary_mask = torch.zeros_like(conf_mask)

            unreliable_mask = (
                (unlabeled_mask > 0)
                & (conf_mask <= 0)
                & (boundary_mask <= 0)
            ).float()

            consistency_weight = get_current_consistency_weight(iter_num // 300, current_args)
            use_feedback = (
                iter_num >= current_args.feedback_start_iter
                and iter_num % current_args.feedback_interval == 0
            )

            if use_feedback:
                delta_c = compute_feedback_delta(
                    student,
                    image,
                    label_batch,
                    lambda logits: masked_soft_ce_loss(logits, pseudo_soft, conf_mask),
                    ce_loss,
                    current_args,
                    boundary_band=boundary_band,
                )
                delta_b = compute_feedback_delta(
                    student,
                    image,
                    label_batch,
                    lambda logits: (
                        masked_soft_ce_loss(logits, pseudo_soft, boundary_mask)
                        + current_args.lambda_boundary
                        * boundary_bce_loss(
                            logits,
                            boundary_band,
                            boundary_mask,
                            current_args.boundary_radius,
                        )
                    ),
                    ce_loss,
                    current_args,
                    boundary_band=boundary_band,
                )
            else:
                delta_c = zero_tensor(image.device)
                delta_b = zero_tensor(image.device)

            reliable_t1 = ((u1 < current_args.tau_conf) & (unlabeled_mask > 0)).float()
            reliable_t2 = ((u2 < current_args.tau_conf) & (unlabeled_mask > 0)).float()

            loss_t1_scrib = ce_loss(logits_t1, label_batch)
            loss_t2_scrib = ce_loss(logits_t2, label_batch)
            loss_cross_t1 = masked_soft_ce_loss_weighted(
                logits_t1,
                prob_t2.detach(),
                reliable_t2,
                weight=torch.exp(-u2).detach(),
            )
            loss_cross_t2 = masked_soft_ce_loss_weighted(
                logits_t2,
                prob_t1.detach(),
                reliable_t1,
                weight=torch.exp(-u1).detach(),
            )
            loss_fb_t1 = (
                delta_c * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, conf_mask, w1.detach())
                + delta_b
                * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, boundary_mask, w1.detach())
            )
            loss_fb_t2 = (
                delta_c * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, conf_mask, w2.detach())
                + delta_b
                * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, boundary_mask, w2.detach())
            )
            loss_bd_t1 = boundary_bce_loss(
                logits_t1, boundary_band, boundary_mask, current_args.boundary_radius
            )
            loss_bd_t2 = boundary_bce_loss(
                logits_t2, boundary_band, boundary_mask, current_args.boundary_radius
            )

            loss_teacher1 = (
                loss_t1_scrib
                + consistency_weight * current_args.lambda_cross_teacher * loss_cross_t1
                + current_args.lambda_feedback * loss_fb_t1
                + consistency_weight * current_args.lambda_boundary * loss_bd_t1
            )
            loss_teacher2 = (
                loss_t2_scrib
                + consistency_weight * current_args.lambda_cross_teacher * loss_cross_t2
                + current_args.lambda_feedback * loss_fb_t2
                + consistency_weight * current_args.lambda_boundary * loss_bd_t2
            )

            optimizer_t1.zero_grad()
            optimizer_t2.zero_grad()
            (loss_teacher1 + loss_teacher2).backward()
            maybe_apply_grad_clip(teacher1, current_args.grad_clip)
            maybe_apply_grad_clip(teacher2, current_args.grad_clip)
            optimizer_t1.step()
            optimizer_t2.step()
            teacher1_ema_state = maybe_update_teacher_ema(
                teacher1, teacher1_ema_state, current_args.teacher_ema_alpha
            )
            teacher2_ema_state = maybe_update_teacher_ema(
                teacher2, teacher2_ema_state, current_args.teacher_ema_alpha
            )

            logits_s = unpack_model_output(student(image))
            u_s = evidential_uncertainty_from_logits(
                logits_s, num_classes, activation=current_args.evidence_activation
            )
            loss_s_scrib = ce_loss(logits_s, label_batch)
            loss_s_conf = masked_soft_ce_loss(logits_s, pseudo_soft, conf_mask)
            loss_s_boundary_soft = masked_soft_ce_loss(logits_s, pseudo_soft, boundary_mask)
            loss_s_boundary_bd = boundary_bce_loss(
                logits_s, boundary_band, boundary_mask, current_args.boundary_radius
            )
            loss_s_boundary = loss_s_boundary_soft + current_args.lambda_boundary * loss_s_boundary_bd
            loss_unc_s = uncertainty_consistency_loss(u_s, u_ens.detach(), loss_type="l1", mask=unlabeled_mask)

            loss_student = (
                loss_s_scrib
                + consistency_weight * current_args.lambda_pseudo_conf * loss_s_conf
                + consistency_weight * current_args.lambda_pseudo_boundary * loss_s_boundary
                + consistency_weight * current_args.lambda_unc_cons * loss_unc_s
            )

            optimizer_s.zero_grad()
            loss_student.backward()
            maybe_apply_grad_clip(student, current_args.grad_clip)
            optimizer_s.step()

            iter_num += 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for optimizer in [optimizer_t1, optimizer_t2, optimizer_s]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)

            writer.add_scalar("loss/student_total", loss_student.item(), iter_num)
            writer.add_scalar("loss/student_scrib", loss_s_scrib.item(), iter_num)
            writer.add_scalar("loss/student_conf", loss_s_conf.item(), iter_num)
            writer.add_scalar("loss/student_boundary", loss_s_boundary.item(), iter_num)
            writer.add_scalar("loss/student_unc", loss_unc_s.item(), iter_num)

            writer.add_scalar("loss/teacher1_total", loss_teacher1.item(), iter_num)
            writer.add_scalar("loss/teacher2_total", loss_teacher2.item(), iter_num)
            writer.add_scalar("loss/cross_t1", loss_cross_t1.item(), iter_num)
            writer.add_scalar("loss/cross_t2", loss_cross_t2.item(), iter_num)
            writer.add_scalar("loss/fb_t1", loss_fb_t1.item(), iter_num)
            writer.add_scalar("loss/fb_t2", loss_fb_t2.item(), iter_num)

            writer.add_scalar("feedback/delta_c", float(delta_c), iter_num)
            writer.add_scalar("feedback/delta_b", float(delta_b), iter_num)
            writer.add_scalar("feedback/delta_c_negative", float(delta_c < 0), iter_num)
            writer.add_scalar("feedback/delta_b_negative", float(delta_b < 0), iter_num)

            writer.add_scalar("unc/u1_mean", u1.mean().item(), iter_num)
            writer.add_scalar("unc/u2_mean", u2.mean().item(), iter_num)
            writer.add_scalar("unc/u_ens_mean", u_ens.mean().item(), iter_num)

            writer.add_scalar("mask/scribble_ratio", scribble_mask.mean().item(), iter_num)
            writer.add_scalar("mask/conf_ratio", conf_mask.mean().item(), iter_num)
            writer.add_scalar("mask/boundary_ratio", boundary_mask.mean().item(), iter_num)
            writer.add_scalar("mask/unreliable_ratio", unreliable_mask.mean().item(), iter_num)

            if current_args.save_debug_vis and iter_num % current_args.debug_vis_interval == 0:
                with torch.no_grad():
                    save_debug_visualization(
                        volume=image[0, 0].detach().cpu().numpy(),
                        label=label_batch[0].detach().cpu().numpy(),
                        pred_t1=torch.argmax(prob_t1[0], dim=0).detach().cpu().numpy(),
                        pred_t2=torch.argmax(prob_t2[0], dim=0).detach().cpu().numpy(),
                        pseudo=pseudo_hard[0].detach().cpu().numpy(),
                        u_ens=u_ens[0, 0].detach().cpu().numpy(),
                        conf_mask=conf_mask[0, 0].detach().cpu().numpy(),
                        boundary_mask=boundary_mask[0, 0].detach().cpu().numpy(),
                        boundary_band=boundary_band[0, 0].detach().cpu().numpy(),
                        unreliable_mask=unreliable_mask[0, 0].detach().cpu().numpy(),
                        save_path=os.path.join(debug_dir, "iter_{:06d}.png".format(iter_num)),
                    )

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss_s=%.4f loss_t1=%.4f loss_t2=%.4f dice=%.4f dc=%.4f db=%.4f conf=%.4f boundary=%.4f unc=%.4f",
                    iter_num,
                    loss_student.item(),
                    loss_teacher1.item(),
                    loss_teacher2.item(),
                    last_performance,
                    float(delta_c),
                    float(delta_b),
                    conf_mask.mean().item(),
                    boundary_mask.mean().item(),
                    u_ens.mean().item(),
                )

            if iter_num > 0 and iter_num % 200 == 0:
                student.eval()
                metric_list, performance, mean_hd95 = validate_student(
                    student, valloader, num_classes, current_args.patch_size
                )
                last_performance = performance
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/val_{}_dice".format(class_i + 1), metric_list[class_i, 0], iter_num
                    )
                    writer.add_scalar(
                        "info/val_{}_hd95".format(class_i + 1), metric_list[class_i, 1], iter_num
                    )
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, "iter_{}_dice_{:.4f}.pth".format(iter_num, best_performance)
                    )
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(current_args.model))
                    save_best_t1 = os.path.join(
                        snapshot_path, "{}_teacher1_best_model.pth".format(current_args.model)
                    )
                    save_best_t2 = os.path.join(
                        snapshot_path, "{}_teacher2_best_model.pth".format(current_args.model)
                    )
                    torch.save(student.state_dict(), save_mode_path)
                    torch.save(student.state_dict(), save_best)
                    torch.save(teacher1.state_dict(), save_best_t1)
                    torch.save(teacher2.state_dict(), save_best_t2)

                logging.info(
                    "validation %d : mean_dice=%.4f mean_hd95=%.4f best_dice=%.4f",
                    iter_num,
                    performance,
                    mean_hd95,
                    best_performance,
                )
                student.train()
                teacher1.train()
                teacher2.train()

            if iter_num % 3000 == 0:
                torch.save(
                    student.state_dict(),
                    os.path.join(snapshot_path, "iter_{}_student.pth".format(iter_num)),
                )
                torch.save(
                    teacher1.state_dict(),
                    os.path.join(snapshot_path, "iter_{}_teacher1.pth".format(iter_num)),
                )
                torch.save(
                    teacher2.state_dict(),
                    os.path.join(snapshot_path, "iter_{}_teacher2.pth".format(iter_num)),
                )

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.root_path = resolve_path(args.root_path, script_dir)

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

    snapshot_path = resolve_path("../../checkpoints/{}_{}".format(args.data, args.exp), script_dir)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("teacher_diversity_mode=%s", args.teacher_diversity_mode)
    train(args, snapshot_path)


if __name__ == "__main__":
    main()
