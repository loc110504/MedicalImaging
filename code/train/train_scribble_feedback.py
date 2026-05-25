import argparse
import logging
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.func import functional_call
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
parser.add_argument("--exp", type=str, default="ScribbleFeedback_TS", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--model", type=str, default="unet", help="network name")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=8, help="batch size per GPU")
parser.add_argument("--deterministic", type=int, default=1, help="use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="consistency rampup")

parser.add_argument(
    "--lambda_pseudo",
    type=float,
    default=1.0,
    help="weight for pseudo-label supervision on non-scribble pixels",
)
parser.add_argument(
    "--lambda_fb",
    type=float,
    default=0.1,
    help="weight for teacher feedback loss",
)
parser.add_argument(
    "--feedback_lr",
    type=float,
    default=0.01,
    help="virtual step size for student feedback estimation",
)
parser.add_argument(
    "--feedback_conf_thresh",
    type=float,
    default=0.0,
    help="confidence threshold for uncertain-unlabeled feedback mask",
)
parser.add_argument(
    "--feedback_mask_mode",
    type=str,
    default="unlabeled",
    choices=["unlabeled", "uncertain_unlabeled"],
    help="region where teacher feedback loss is applied",
)
parser.add_argument(
    "--pseudo_conf_thresh",
    type=float,
    default=0.7,
    help="confidence threshold for pseudo-label supervision",
)
parser.add_argument(
    "--feedback_warmup",
    type=int,
    default=1000,
    help="start feedback loss after this iteration",
)
parser.add_argument(
    "--pseudo_warmup",
    type=int,
    default=500,
    help="start pseudo-label supervision after this iteration",
)
parser.add_argument(
    "--delta_clip",
    type=float,
    default=1.0,
    help="clip feedback delta into [-delta_clip, delta_clip]",
)
parser.add_argument(
    "--normalize_delta",
    type=int,
    default=1,
    help="normalize feedback delta by current scribble loss",
)
parser.add_argument(
    "--teacher_scribble_loss",
    type=int,
    default=1,
    help="whether teacher is also trained by scribble partial CE",
)
parser.add_argument(
    "--feedback_interval",
    type=int,
    default=1,
    help="compute feedback every N iterations; 1 means every iteration",
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_current_consistency_weight(epoch, current_args):
    return ramps.sigmoid_rampup(epoch, current_args.consistency_rampup)


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return unpack_model_output(self.model(x))


def zero_tensor(device):
    return torch.tensor(0.0, device=device)


def create_model(model_name, num_classes=4):
    return net_factory(net_type=model_name, in_chns=1, class_num=num_classes)


def masked_hard_ce_loss(logits, target, mask, eps=1e-8):
    mask = mask.float()
    if mask.sum().item() < 1:
        return zero_tensor(logits.device)

    ce_map = F.cross_entropy(logits, target.long(), reduction="none")
    return (ce_map * mask.squeeze(1)).sum() / (mask.sum() + eps)


def masked_pseudo_nll_loss(teacher_logits, pseudo_hard, mask, eps=1e-8):
    mask = mask.float()
    if mask.sum().item() < 1:
        return zero_tensor(teacher_logits.device)

    log_prob = F.log_softmax(teacher_logits, dim=1)
    log_pseudo = torch.gather(log_prob, dim=1, index=pseudo_hard.long().unsqueeze(1))
    return -(log_pseudo * mask).sum() / (mask.sum() + eps)


def get_param_dict(model):
    return OrderedDict((name, param) for name, param in model.named_parameters())


def get_buffer_dict(model):
    # Clone buffers so the virtual forward does not mutate running stats in-place.
    return OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())


def compute_scribble_feedback_delta(
    student,
    volume_batch,
    label_batch,
    student_logits,
    pseudo_hard,
    feedback_mask,
    ce_loss,
    feedback_lr=0.01,
    normalize_delta=True,
    delta_clip=1.0,
):
    if feedback_mask.sum().item() < 1:
        return zero_tensor(student_logits.device)

    pseudo_loss = masked_hard_ce_loss(student_logits, pseudo_hard.detach(), feedback_mask)
    if not torch.isfinite(pseudo_loss):
        return zero_tensor(student_logits.device)

    params = get_param_dict(student)
    buffers = get_buffer_dict(student)
    grads = torch.autograd.grad(
        pseudo_loss,
        tuple(params.values()),
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    )

    has_grad = any(grad is not None for grad in grads)
    if not has_grad:
        return zero_tensor(student_logits.device)

    virtual_params = OrderedDict()
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            virtual_params[name] = param
        else:
            virtual_params[name] = param - feedback_lr * grad

    with torch.no_grad():
        virtual_logits = functional_call(student, {**virtual_params, **buffers}, (volume_batch,))
        virtual_logits = unpack_model_output(virtual_logits)
        loss_before = ce_loss(student_logits.detach(), label_batch.long())
        loss_after = ce_loss(virtual_logits, label_batch.long())
        delta = (loss_before - loss_after).detach()

        if normalize_delta:
            delta = delta / (loss_before.detach() + 1e-8)

        if delta_clip > 0:
            delta = torch.clamp(delta, -delta_clip, delta_clip)

    return delta


def validate_student(student, valloader, num_classes, patch_size):
    wrapper = LogitsOnlyWrapper(student)
    metric_list = 0.0
    for sampled_batch in valloader:
        metric_i = test_single_volume_scribblevs(
            sampled_batch["image"],
            sampled_batch["label"],
            wrapper,
            classes=num_classes,
            patch_size=patch_size,
        )
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(valloader.dataset)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    return metric_list, performance, mean_hd95


def train(current_args, snapshot_path):
    base_lr = current_args.base_lr
    num_classes = current_args.num_classes
    batch_size = current_args.batch_size
    max_iterations = current_args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(current_args.seed + worker_id)

    student = create_model(current_args.model, num_classes=num_classes)
    teacher = create_model(current_args.model, num_classes=num_classes)
    teacher.load_state_dict(student.state_dict())
    student.train()
    teacher.train()

    db_train = ACDCDataSets(
        base_dir=current_args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(current_args.patch_size)]),
        fold=current_args.fold,
        sup_type=current_args.sup_type,
    )
    db_val = ACDCDataSets(
        base_dir=current_args.root_path,
        fold=current_args.fold,
        split="val",
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer_student = optim.SGD(
        student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    optimizer_teacher = optim.SGD(
        teacher.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    ce_loss = CrossEntropyLoss(ignore_index=4)
    _pseudo_dice_loss = losses.pDLoss(num_classes, ignore_index=4)  # Kept for follow-up pseudo Dice ablations.

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    logging.info("%d iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda().long()

            teacher_logits = unpack_model_output(teacher(volume_batch))
            teacher_prob = torch.softmax(teacher_logits, dim=1)
            pseudo_conf, pseudo_hard = torch.max(teacher_prob.detach(), dim=1)
            pseudo_conf = pseudo_conf.unsqueeze(1).detach()
            pseudo_hard = pseudo_hard.detach()

            scribble_mask = (label_batch != 4).float().unsqueeze(1)
            unlabeled_mask = (label_batch == 4).float().unsqueeze(1)

            pseudo_mask = unlabeled_mask.clone()
            if current_args.pseudo_conf_thresh > 0:
                pseudo_mask = pseudo_mask * (pseudo_conf > current_args.pseudo_conf_thresh).float()

            feedback_mask = unlabeled_mask.clone()
            if current_args.feedback_mask_mode == "uncertain_unlabeled":
                feedback_mask = feedback_mask * (
                    pseudo_conf < current_args.feedback_conf_thresh
                ).float()

            student_logits = unpack_model_output(student(volume_batch))

            if current_args.feedback_interval > 0 and iter_num % current_args.feedback_interval == 0:
                delta = compute_scribble_feedback_delta(
                    student=student,
                    volume_batch=volume_batch,
                    label_batch=label_batch,
                    student_logits=student_logits,
                    pseudo_hard=pseudo_hard,
                    feedback_mask=feedback_mask,
                    ce_loss=ce_loss,
                    feedback_lr=current_args.feedback_lr,
                    normalize_delta=bool(current_args.normalize_delta),
                    delta_clip=current_args.delta_clip,
                )
            else:
                delta = zero_tensor(volume_batch.device)

            consistency_weight = get_current_consistency_weight(iter_num // 300, current_args)

            loss_scribble_student = ce_loss(student_logits, label_batch.long())
            loss_pseudo_student = masked_hard_ce_loss(student_logits, pseudo_hard, pseudo_mask)
            if iter_num < current_args.pseudo_warmup:
                loss_pseudo_student = zero_tensor(volume_batch.device)

            loss_student = (
                loss_scribble_student
                + current_args.lambda_pseudo * consistency_weight * loss_pseudo_student
            )

            loss_scribble_teacher = ce_loss(teacher_logits, label_batch.long())
            teacher_nll = masked_pseudo_nll_loss(teacher_logits, pseudo_hard, feedback_mask)
            loss_feedback = delta.detach() * teacher_nll
            if iter_num < current_args.feedback_warmup:
                loss_feedback = zero_tensor(volume_batch.device)

            if current_args.teacher_scribble_loss:
                loss_teacher = loss_scribble_teacher + current_args.lambda_fb * loss_feedback
            else:
                loss_teacher = current_args.lambda_fb * loss_feedback

            optimizer_student.zero_grad()
            loss_student.backward()
            optimizer_student.step()

            optimizer_teacher.zero_grad()
            loss_teacher.backward()
            optimizer_teacher.step()

            iter_num += 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for optimizer in [optimizer_student, optimizer_teacher]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("info/loss_student", loss_student.item(), iter_num)
            writer.add_scalar("info/loss_teacher", loss_teacher.item(), iter_num)
            writer.add_scalar(
                "info/loss_scribble_student", loss_scribble_student.item(), iter_num
            )
            writer.add_scalar(
                "info/loss_scribble_teacher", loss_scribble_teacher.item(), iter_num
            )
            writer.add_scalar("info/loss_pseudo_student", loss_pseudo_student.item(), iter_num)
            writer.add_scalar("info/loss_feedback", loss_feedback.item(), iter_num)
            writer.add_scalar("info/delta_feedback", delta.item(), iter_num)
            writer.add_scalar("info/teacher_nll", teacher_nll.item(), iter_num)
            writer.add_scalar("info/pseudo_conf_mean", pseudo_conf.mean().item(), iter_num)
            writer.add_scalar("info/pseudo_mask_ratio", pseudo_mask.mean().item(), iter_num)
            writer.add_scalar("info/feedback_mask_ratio", feedback_mask.mean().item(), iter_num)
            writer.add_scalar("info/scribble_mask_ratio", scribble_mask.mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss_student=%f, loss_teacher=%f, scribble_s=%f, "
                    "scribble_t=%f, pseudo=%f, fb=%f, delta=%f, pseudo_conf=%f, "
                    "pseudo_mask=%f, fb_mask=%f",
                    iter_num,
                    loss_student.item(),
                    loss_teacher.item(),
                    loss_scribble_student.item(),
                    loss_scribble_teacher.item(),
                    loss_pseudo_student.item(),
                    loss_feedback.item(),
                    delta.item(),
                    pseudo_conf.mean().item(),
                    pseudo_mask.mean().item(),
                    feedback_mask.mean().item(),
                )

            if iter_num > 0 and iter_num % 200 == 0:
                student.eval()
                metric_list, performance, mean_hd95 = validate_student(
                    student, valloader, num_classes, current_args.patch_size
                )
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
                    save_best_teacher = os.path.join(
                        snapshot_path, "{}_best_teacher.pth".format(current_args.model)
                    )
                    torch.save(student.state_dict(), save_mode_path)
                    torch.save(student.state_dict(), save_best)
                    torch.save(teacher.state_dict(), save_best_teacher)

                logging.info(
                    "validation %d : mean_dice=%f mean_hd95=%f best_dice=%f",
                    iter_num,
                    performance,
                    mean_hd95,
                    best_performance,
                )
                student.train()
                teacher.train()

            if iter_num % 3000 == 0:
                save_student = os.path.join(snapshot_path, "iter_{}_student.pth".format(iter_num))
                save_teacher = os.path.join(snapshot_path, "iter_{}_teacher.pth".format(iter_num))
                torch.save(student.state_dict(), save_student)
                torch.save(teacher.state_dict(), save_teacher)
                logging.info("save models to %s and %s", save_student, save_teacher)

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
    torch.cuda.manual_seed_all(args.seed)

    snapshot_path = "../../checkpoints/{}_{}".format(args.data, args.exp)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
