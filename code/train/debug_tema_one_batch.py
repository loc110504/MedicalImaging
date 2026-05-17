import argparse
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses
from utils.tema_boundary import boundary_likelihood
from utils.tema_ema import copy_student_to_teacher, detach_model, update_ema_variables
from utils.tema_entropy import build_uncertain_mask, normalized_entropy, teacher_confidence_mask
from utils.tema_losses import weighted_feature_consistency_loss, weighted_soft_ce_loss
from utils.tema_region_switch import js_divergence_map, select_teacher_feature, temporal_teacher_arbitration


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/ACDC')
parser.add_argument('--fold', type=str, default='MAAGfold70')
parser.add_argument('--sup_type', type=str, default='scribble')
parser.add_argument('--model', type=str, default='unet_hl')
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2022)

parser.add_argument('--alpha_fast', type=float, default=0.99)
parser.add_argument('--alpha_slow', type=float, default=0.999)
parser.add_argument('--uncertain_mode', type=str, default='quantile', choices=['quantile', 'fixed'])
parser.add_argument('--uncertain_top_ratio', type=float, default=0.35)
parser.add_argument('--uncertain_threshold', type=float, default=0.5)
parser.add_argument('--teacher_conf_threshold', type=float, default=0.55)
parser.add_argument('--easy_threshold', type=float, default=0.2)
parser.add_argument('--boundary_lambda_image', type=float, default=1.0)
parser.add_argument('--gamma_boundary', type=float, default=0.5)
parser.add_argument('--gamma_disagree', type=float, default=0.5)
parser.add_argument('--gamma_stable', type=float, default=0.5)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    db_train = ACDCDataSets(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = create_model(ema=False).cuda().train()
    teacher_fast = create_model(ema=True).cuda().train()
    teacher_slow = create_model(ema=True).cuda().train()

    copy_student_to_teacher(model, teacher_fast)
    copy_student_to_teacher(model, teacher_slow)
    detach_model(teacher_fast)
    detach_model(teacher_slow)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(args.num_classes, ignore_index=4)

    sampled = next(iter(trainloader))
    image = sampled['image'].cuda()
    scrib = sampled['label'].cuda().long()

    assert image.dim() == 4 and image.size(1) == 1
    assert scrib.dim() == 3
    assert image.shape[-2:] == scrib.shape[-2:]
    assert scrib.max().item() <= 4 and scrib.min().item() >= 0

    with torch.no_grad():
        logits_fast, high_fast, low_fast = teacher_fast(image)
        logits_slow, high_slow, low_slow = teacher_slow(image)
        probs_fast = torch.softmax(logits_fast, dim=1)
        probs_slow = torch.softmax(logits_slow, dim=1)

    logits_stu, high_stu, low_stu = model(image)
    probs_stu = torch.softmax(logits_stu, dim=1)

    loss_scrib = ce_loss(logits_stu, scrib) + dice_loss(probs_stu, scrib.unsqueeze(1))

    entropy_stu = normalized_entropy(probs_stu)
    entropy_fast = normalized_entropy(probs_fast)
    entropy_slow = normalized_entropy(probs_slow)
    disagreement = js_divergence_map(probs_fast, probs_slow)

    uncertain_mask = build_uncertain_mask(
        entropy_stu,
        scrib,
        mode=args.uncertain_mode,
        top_ratio=args.uncertain_top_ratio,
        threshold=args.uncertain_threshold,
        ignore_index=4,
    )

    boundary = boundary_likelihood(image, probs_stu, lambda_image=args.boundary_lambda_image)

    switch = temporal_teacher_arbitration(
        probs_fast,
        probs_slow,
        entropy_fast,
        entropy_slow,
        disagreement,
        boundary,
        args.gamma_boundary,
        args.gamma_disagree,
        args.gamma_stable,
    )

    selected_probs = switch['selected_probs']
    selected_weight = switch['selected_weight']
    select_fast = switch['select_fast']

    conf_mask = teacher_confidence_mask(selected_probs, threshold=args.teacher_conf_threshold)
    pseudo_mask = uncertain_mask & conf_mask

    loss_pseudo = weighted_soft_ce_loss(logits_stu, selected_probs.detach(), pseudo_mask, selected_weight.detach())

    selected_low = select_teacher_feature(low_fast, low_slow, select_fast).detach()
    selected_high = select_teacher_feature(high_fast, high_slow, select_fast).detach()
    loss_low = weighted_feature_consistency_loss(low_stu, selected_low, pseudo_mask, selected_weight.detach())
    loss_high = weighted_feature_consistency_loss(high_stu, selected_high, pseudo_mask, selected_weight.detach())
    loss_hico = 0.5 * (loss_low + loss_high)

    easy_mask = (entropy_stu < args.easy_threshold) & (scrib.unsqueeze(1) == 4)
    loss_stable = weighted_soft_ce_loss(logits_stu, probs_slow.detach(), easy_mask, None)

    loss = loss_scrib + 0.5 * loss_pseudo + 0.5 * loss_hico + 0.1 * loss_stable

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_ema_variables(model, teacher_fast, alpha=args.alpha_fast, global_step=0)
    update_ema_variables(model, teacher_slow, alpha=args.alpha_slow, global_step=0)

    print('image shape:', tuple(image.shape))
    print('scrib shape:', tuple(scrib.shape))
    print('unique scribble labels:', sorted(scrib.unique().detach().cpu().tolist()))
    print('student logits shape:', tuple(logits_stu.shape))
    print('high/low shapes:', tuple(high_stu.shape), tuple(low_stu.shape))
    print('entropy_student mean/min/max:', float(entropy_stu.mean()), float(entropy_stu.min()), float(entropy_stu.max()))
    print('entropy_fast mean:', float(entropy_fast.mean()))
    print('entropy_slow mean:', float(entropy_slow.mean()))
    print('disagreement mean:', float(disagreement.mean()))
    print('boundary mean:', float(boundary.mean()))
    print('uncertain_ratio:', float(uncertain_mask.float().mean()))
    print('pseudo_ratio:', float(pseudo_mask.float().mean()))
    print('select_fast_ratio:', float(select_fast.float().mean()))
    print('easy_ratio:', float(easy_mask.float().mean()))
    print('loss_scrib:', float(loss_scrib.detach()))
    print('loss_pseudo:', float(loss_pseudo.detach()))
    print('loss_hico:', float(loss_hico.detach()))
    print('loss_stable:', float(loss_stable.detach()))


if __name__ == '__main__':
    main()
