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
from utils.ema_optim import WeightEMA
from val import test_single_volume_scribblevs


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../../data/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="DTSF_ACDC", help="experiment name")
parser.add_argument("--data", type=str, default="ACDC", help="dataset name")
parser.add_argument("--fold", type=str, default="MAAGfold70", help="dataset fold")
parser.add_argument("--sup_type", type=str, default="scribble", help="supervision type")
parser.add_argument("--model", type=str, default="unet", help="network name")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum training iterations")
parser.add_argument("--batch_size", type=int, default=16, help="batch size per GPU")
parser.add_argument("--deterministic", type=int, default=1, help="use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=2022, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency_rampup", type=float, default=40.0, help="consistency rampup")
parser.add_argument("--lambda_pseudo", type=float, default=1.0, help="weight for student pseudo-label loss")
parser.add_argument("--lambda_fb", type=float, default=0.1, help="weight for dual-teacher feedback loss")
parser.add_argument("--lambda_cross", type=float, default=0.5, help="weight for cross-teacher pseudo supervision")
parser.add_argument(
    "--pseudo_agree_thresh",
    type=float,
    default=0.7,
    help="minimum min-confidence for agreement pseudo pixels",
)
parser.add_argument(
    "--pseudo_disagree_thresh",
    type=float,
    default=0.8,
    help="minimum max-confidence for disagreement pseudo pixels",
)
parser.add_argument(
    "--pseudo_margin_thresh",
    type=float,
    default=0.1,
    help="minimum confidence margin for disagreement pseudo pixels",
)
parser.add_argument("--feedback_warmup", type=int, default=1000, help="start feedback loss after this iteration")
parser.add_argument("--pseudo_warmup", type=int, default=500, help="start pseudo supervision after this iteration")
parser.add_argument("--cross_warmup", type=int, default=500, help="start cross-teacher loss after this iteration")
parser.add_argument(
    "--feedback_lr_factor",
    type=float,
    default=1.0,
    help="virtual student update step = current_lr * feedback_lr_factor",
)
parser.add_argument("--delta_clip", type=float, default=1.0, help="clip feedback delta into [-delta_clip, delta_clip]")
parser.add_argument("--normalize_delta", type=int, default=1, help="normalize delta by scribble loss before virtual update")
parser.add_argument("--step_normgrad", type=int, default=0, help="normalize virtual update gradient in BackupModel")
parser.add_argument("--teacher_scribble_loss", type=int, default=1, help="whether to train teachers with partial CE on scribbles")
parser.add_argument("--use_pseudo_dice", type=int, default=1, help="whether to add pDLoss on pseudo labels")
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


class BackupModel(object):
    def __init__(self, model, norm_grad=False):
        self.model = model
        self.backup = {}
        self.norm_grad = norm_grad

    def backup_param(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()

    def step(self, epsilon=1.0):
        for _, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                if self.norm_grad:
                    norm = torch.norm(grad)
                    if norm.item() > 0:
                        grad = grad / (norm + 1e-12)
                param.data.add_(grad, alpha=-epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].clone()
            if param.grad is not None:
                param.grad.data.zero_()


def create_model(num_classes=4, ema=False):
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net.cuda()


def masked_hard_ce_loss(logits, target, mask, eps=1e-8):
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    ce_map = F.cross_entropy(logits, target.long(), reduction="none")
    return (ce_map * mask).sum() / (mask.sum() + eps)


def masked_pseudo_nll_loss(logits, pseudo_hard, mask, eps=1e-8):
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    nll_map = F.cross_entropy(logits, pseudo_hard.detach().long(), reduction="none")
    return (nll_map * mask).sum() / (mask.sum() + eps)


def build_ignore_target(pseudo_hard, mask, ignore_index=4):
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    target = pseudo_hard.clone()
    target[mask < 0.5] = ignore_index
    return target


def build_dual_teacher_pseudo(
    logits_t1,
    logits_t2,
    label_batch,
    agree_thresh=0.7,
    disagree_thresh=0.8,
    margin_thresh=0.1,
    ignore_index=4,
):
    prob_t1 = torch.softmax(logits_t1, dim=1)
    prob_t2 = torch.softmax(logits_t2, dim=1)

    conf_t1, pl_t1 = torch.max(prob_t1, dim=1)
    conf_t2, pl_t2 = torch.max(prob_t2, dim=1)

    unlabeled = label_batch == ignore_index
    same = pl_t1 == pl_t2
    diff = ~same

    min_conf = torch.minimum(conf_t1, conf_t2)
    max_conf = torch.maximum(conf_t1, conf_t2)
    margin = torch.abs(conf_t1 - conf_t2)

    reliable_agree = same & (min_conf >= agree_thresh) & unlabeled
    reliable_disagree = diff & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & unlabeled

    choose_t1 = torch.where(same, torch.ones_like(same), conf_t1 > conf_t2)
    pseudo_hard = torch.where(choose_t1, pl_t1, pl_t2)
    reliable = reliable_agree | reliable_disagree

    t1_lowconf = reliable_agree & (conf_t1 < conf_t2)
    t2_lowconf = reliable_agree & (conf_t1 >= conf_t2)
    t1_highconf = reliable_disagree & (conf_t1 > conf_t2)
    t2_highconf = reliable_disagree & (conf_t1 <= conf_t2)

    return {
        "pseudo_hard": pseudo_hard.detach(),
        "reliable_mask": reliable.float().unsqueeze(1).detach(),
        "agreement_mask": reliable_agree.float().unsqueeze(1).detach(),
        "disagreement_mask": reliable_disagree.float().unsqueeze(1).detach(),
        "t1_lowconf_mask": t1_lowconf.float().unsqueeze(1).detach(),
        "t2_lowconf_mask": t2_lowconf.float().unsqueeze(1).detach(),
        "t1_highconf_mask": t1_highconf.float().unsqueeze(1).detach(),
        "t2_highconf_mask": t2_highconf.float().unsqueeze(1).detach(),
        "conf_t1": conf_t1.detach(),
        "conf_t2": conf_t2.detach(),
        "pl_t1": pl_t1.detach(),
        "pl_t2": pl_t2.detach(),
    }


def compute_feedback_delta(
    student,
    backup_student,
    volume_batch,
    label_batch,
    pseudo_hard,
    feedback_mask,
    ce_loss,
    feedback_step,
    normalize_delta=True,
    delta_clip=1.0,
):
    if feedback_mask.sum() < 1:
        return volume_batch.new_tensor(0.0)

    backup_student.restore()

    with torch.no_grad():
        logits_before = unpack_model_output(student(volume_batch))
        loss_before = ce_loss(logits_before, label_batch.long())

    logits_for_update = unpack_model_output(student(volume_batch))
    tmp_loss = masked_hard_ce_loss(logits_for_update, pseudo_hard.detach(), feedback_mask)

    student.zero_grad()
    tmp_loss.backward()
    backup_student.step(epsilon=feedback_step)

    with torch.no_grad():
        logits_after = unpack_model_output(student(volume_batch))
        loss_after = ce_loss(logits_after, label_batch.long())

    backup_student.restore()

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

    student = create_model(num_classes=num_classes)
    teacher_1 = create_model(num_classes=num_classes, ema=True)
    teacher_2 = create_model(num_classes=num_classes, ema=True)
    teacher_1.load_state_dict(student.state_dict())
    teacher_2.load_state_dict(student.state_dict())

    student.train()
    teacher_1.train()
    teacher_2.train()

    db_train = ACDCDataSets(
        base_dir=current_args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(current_args.patch_size)]),
        fold=current_args.fold,
        sup_type=current_args.sup_type,
    )
    db_val = ACDCDataSets(base_dir=current_args.root_path, fold=current_args.fold, split="val")

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer_student = optim.SGD(student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ema_teacher_1 = WeightEMA(student, teacher_1, 0.99)
    ema_teacher_2 = WeightEMA(student, teacher_2, 0.99)

    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    backup_student = BackupModel(student, norm_grad=bool(current_args.step_normgrad))

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

            with torch.no_grad():
                logits_t1 = unpack_model_output(teacher_1(volume_batch))
                logits_t2 = unpack_model_output(teacher_2(volume_batch))

            pseudo_info = build_dual_teacher_pseudo(
                logits_t1=logits_t1.detach(),
                logits_t2=logits_t2.detach(),
                label_batch=label_batch,
                agree_thresh=current_args.pseudo_agree_thresh,
                disagree_thresh=current_args.pseudo_disagree_thresh,
                margin_thresh=current_args.pseudo_margin_thresh,
                ignore_index=4,
            )

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            backup_student.backup_param()

            delta_agree = compute_feedback_delta(
                student=student,
                backup_student=backup_student,
                volume_batch=volume_batch,
                label_batch=label_batch,
                pseudo_hard=pseudo_info["pseudo_hard"],
                feedback_mask=pseudo_info["agreement_mask"],
                ce_loss=ce_loss,
                feedback_step=lr_ * current_args.feedback_lr_factor,
                normalize_delta=bool(current_args.normalize_delta),
                delta_clip=current_args.delta_clip,
            )
            delta_disagree = compute_feedback_delta(
                student=student,
                backup_student=backup_student,
                volume_batch=volume_batch,
                label_batch=label_batch,
                pseudo_hard=pseudo_info["pseudo_hard"],
                feedback_mask=pseudo_info["disagreement_mask"],
                ce_loss=ce_loss,
                feedback_step=lr_ * current_args.feedback_lr_factor,
                normalize_delta=bool(current_args.normalize_delta),
                delta_clip=current_args.delta_clip,
            )

            logits_s = unpack_model_output(student(volume_batch))
            prob_s = torch.softmax(logits_s, dim=1)

            loss_s_scribble = ce_loss(logits_s, label_batch.long())
            loss_s_pseudo_ce = masked_hard_ce_loss(
                logits_s, pseudo_info["pseudo_hard"], pseudo_info["reliable_mask"]
            )
            pseudo_target_ignore = build_ignore_target(
                pseudo_info["pseudo_hard"], pseudo_info["reliable_mask"], ignore_index=4
            )
            loss_s_pseudo_dice = dice_loss(prob_s, pseudo_target_ignore.unsqueeze(1))

            if current_args.use_pseudo_dice:
                loss_s_pseudo = 0.5 * (loss_s_pseudo_ce + loss_s_pseudo_dice)
            else:
                loss_s_pseudo = loss_s_pseudo_ce

            consistency_weight = get_current_consistency_weight(iter_num // 300, current_args)
            if iter_num < current_args.pseudo_warmup:
                loss_s_pseudo = logits_s.new_tensor(0.0)

            loss_student = loss_s_scribble + current_args.lambda_pseudo * consistency_weight * loss_s_pseudo

            optimizer_student.zero_grad()
            loss_student.backward()
            optimizer_student.step()
            ema_teacher_1.step()
            ema_teacher_2.step()

            iter_num += 1
            for param_group in optimizer_student.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("loss/student_total", loss_student.item(), iter_num)
            writer.add_scalar("loss/student_scribble", loss_s_scribble.item(), iter_num)
            writer.add_scalar("loss/student_pseudo", loss_s_pseudo.item(), iter_num)
            writer.add_scalar("feedback/delta_agree", delta_agree.item(), iter_num)
            writer.add_scalar("feedback/delta_disagree", delta_disagree.item(), iter_num)
            writer.add_scalar("mask/reliable_ratio", pseudo_info["reliable_mask"].mean().item(), iter_num)
            writer.add_scalar("mask/agreement_ratio", pseudo_info["agreement_mask"].mean().item(), iter_num)
            writer.add_scalar("mask/disagreement_ratio", pseudo_info["disagreement_mask"].mean().item(), iter_num)
            writer.add_scalar("conf/t1_mean", pseudo_info["conf_t1"].mean().item(), iter_num)
            writer.add_scalar("conf/t2_mean", pseudo_info["conf_t2"].mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    "iteration %d : loss_student=%f, scribble=%f, pseudo=%f, "
                    "delta_agree=%f, delta_disagree=%f, reliable=%f, agree=%f, disagree=%f, conf_t1=%f, conf_t2=%f",
                    iter_num,
                    loss_student.item(),
                    loss_s_scribble.item(),
                    loss_s_pseudo.item(),
                    delta_agree.item(),
                    delta_disagree.item(),
                    pseudo_info["reliable_mask"].mean().item(),
                    pseudo_info["agreement_mask"].mean().item(),
                    pseudo_info["disagreement_mask"].mean().item(),
                    pseudo_info["conf_t1"].mean().item(),
                    pseudo_info["conf_t2"].mean().item(),
                )
            if iter_num > 0 and iter_num % 200 == 0:
                student.eval()
                metric_list, performance, mean_hd95 = validate_student(
                    student, valloader, num_classes, current_args.patch_size
                )
                for class_i in range(num_classes - 1):
                    writer.add_scalar("info/val_{}_dice".format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar("info/val_{}_hd95".format(class_i + 1), metric_list[class_i, 1], iter_num)
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, "iter_{}_dice_{:.4f}.pth".format(iter_num, best_performance))
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(current_args.model))
                    save_best_teacher1 = os.path.join(snapshot_path, "{}_best_teacher1.pth".format(current_args.model))
                    save_best_teacher2 = os.path.join(snapshot_path, "{}_best_teacher2.pth".format(current_args.model))
                    torch.save(student.state_dict(), save_mode_path)
                    torch.save(student.state_dict(), save_best)
                    torch.save(teacher_1.state_dict(), save_best_teacher1)
                    torch.save(teacher_2.state_dict(), save_best_teacher2)

                logging.info(
                    "validation %d : mean_dice=%f mean_hd95=%f best_dice=%f",
                    iter_num,
                    performance,
                    mean_hd95,
                    best_performance,
                )
                student.train()
                teacher_1.train()
                teacher_2.train()

            if iter_num % 3000 == 0:
                save_student = os.path.join(snapshot_path, "iter_{}_student.pth".format(iter_num))
                save_teacher1 = os.path.join(snapshot_path, "iter_{}_teacher1.pth".format(iter_num))
                save_teacher2 = os.path.join(snapshot_path, "iter_{}_teacher2.pth".format(iter_num))
                torch.save(student.state_dict(), save_student)
                torch.save(teacher_1.state_dict(), save_teacher1)
                torch.save(teacher_2.state_dict(), save_teacher2)
                logging.info("save models to %s, %s, %s", save_student, save_teacher1, save_teacher2)

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
