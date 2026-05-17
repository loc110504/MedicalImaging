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
from utils.tema_boundary import boundary_likelihood
from utils.tema_ema import copy_student_to_teacher, detach_model, update_ema_variables
from utils.tema_entropy import build_uncertain_mask, normalized_entropy, teacher_confidence_mask
from utils.tema_losses import weighted_feature_consistency_loss, weighted_soft_ce_loss
from utils.tema_region_switch import js_divergence_map, select_teacher_feature, temporal_teacher_arbitration
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='TEMA_SDT', help='experiment_name')
parser.add_argument('--data', type=str, default='ACDC', help='experiment_name')
parser.add_argument('--fold', type=str, default='MAAGfold70', help='fold of dataset')
parser.add_argument('--sup_type', type=str, default='scribble', help='supervision')
parser.add_argument('--model', type=str, default='unet_hl', help='name of network')
parser.add_argument('--num_classes', type=int, default=4, help='output channel')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency_rampup', type=float, default=40)

parser.add_argument('--alpha_fast', type=float, default=0.99)
parser.add_argument('--alpha_slow', type=float, default=0.999)
parser.add_argument('--pseudo_warmup', type=int, default=3000)

parser.add_argument('--uncertain_mode', type=str, default='quantile', choices=['quantile', 'fixed'])
parser.add_argument('--uncertain_top_ratio', type=float, default=0.35)
parser.add_argument('--uncertain_threshold', type=float, default=0.5)
parser.add_argument('--teacher_conf_threshold', type=float, default=0.55)
parser.add_argument('--easy_threshold', type=float, default=0.2)

parser.add_argument('--use_boundary_prior', type=int, default=1)
parser.add_argument('--boundary_lambda_image', type=float, default=1.0)
parser.add_argument('--gamma_boundary', type=float, default=0.5)
parser.add_argument('--gamma_disagree', type=float, default=0.5)
parser.add_argument('--gamma_stable', type=float, default=0.5)

parser.add_argument('--lambda_pseudo', type=float, default=0.5)
parser.add_argument('--lambda_hico', type=float, default=0.5)
parser.add_argument('--lambda_stable', type=float, default=0.1)
parser.add_argument('--debug_shapes', action='store_true')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, args):
    return 1 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = create_model(ema=False)
    teacher_fast = create_model(ema=True)
    teacher_slow = create_model(ema=True)

    model.cuda()
    teacher_fast.cuda()
    teacher_slow.cuda()

    copy_student_to_teacher(model, teacher_fast)
    copy_student_to_teacher(model, teacher_slow)
    detach_model(teacher_fast)
    detach_model(teacher_slow)

    model.train()
    teacher_fast.train()
    teacher_slow.train()

    db_train = ACDCDataSets(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )
    db_val = ACDCDataSets(base_dir=args.root_path, fold=args.fold, split='val')

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled in enumerate(trainloader):
            image = sampled['image'].cuda()
            scrib = sampled['label'].cuda().long()

            if args.debug_shapes and iter_num < 3:
                assert image.dim() == 4 and image.size(1) == 1
                assert scrib.dim() == 3
                assert image.shape[-2:] == scrib.shape[-2:]
                assert scrib.max().item() <= 4
                assert scrib.min().item() >= 0

            with torch.no_grad():
                logits_fast, high_fast, low_fast = teacher_fast(image)
                probs_fast = torch.softmax(logits_fast, dim=1)

                logits_slow, high_slow, low_slow = teacher_slow(image)
                probs_slow = torch.softmax(logits_slow, dim=1)

            logits_stu, high_stu, low_stu = model(image)
            probs_stu = torch.softmax(logits_stu, dim=1)

            loss_ce_stu = ce_loss(logits_stu, scrib)
            loss_dice_stu = dice_loss(probs_stu, scrib.unsqueeze(1))
            loss_scrib = loss_ce_stu + loss_dice_stu

            entropy_stu = normalized_entropy(probs_stu)
            uncertain_mask = build_uncertain_mask(
                entropy=entropy_stu,
                scribble=scrib,
                mode=args.uncertain_mode,
                top_ratio=args.uncertain_top_ratio,
                threshold=args.uncertain_threshold,
                ignore_index=4,
            )

            entropy_fast = normalized_entropy(probs_fast)
            entropy_slow = normalized_entropy(probs_slow)
            disagreement = js_divergence_map(probs_fast, probs_slow)

            if args.use_boundary_prior:
                boundary = boundary_likelihood(image, probs_stu, lambda_image=args.boundary_lambda_image)
            else:
                boundary = torch.zeros_like(disagreement)

            switch = temporal_teacher_arbitration(
                probs_fast=probs_fast,
                probs_slow=probs_slow,
                entropy_fast=entropy_fast,
                entropy_slow=entropy_slow,
                disagreement=disagreement,
                boundary=boundary,
                gamma_boundary=args.gamma_boundary,
                gamma_disagree=args.gamma_disagree,
                gamma_stable=args.gamma_stable,
            )
            selected_probs = switch['selected_probs']
            selected_weight = switch['selected_weight']
            select_fast = switch['select_fast']

            conf_mask = teacher_confidence_mask(selected_probs, threshold=args.teacher_conf_threshold)
            pseudo_mask = uncertain_mask & conf_mask

            if iter_num >= args.pseudo_warmup:
                loss_pseudo = weighted_soft_ce_loss(
                    logits_stu,
                    selected_probs.detach(),
                    pseudo_mask,
                    selected_weight.detach(),
                )
            else:
                loss_pseudo = logits_stu.sum() * 0.0

            if iter_num >= args.pseudo_warmup:
                selected_low = select_teacher_feature(low_fast, low_slow, select_fast).detach()
                selected_high = select_teacher_feature(high_fast, high_slow, select_fast).detach()

                loss_low = weighted_feature_consistency_loss(
                    low_stu,
                    selected_low,
                    pseudo_mask,
                    selected_weight.detach(),
                )
                loss_high = weighted_feature_consistency_loss(
                    high_stu,
                    selected_high,
                    pseudo_mask,
                    selected_weight.detach(),
                )
                loss_hico = 0.5 * (loss_low + loss_high)
            else:
                loss_low = logits_stu.sum() * 0.0
                loss_high = logits_stu.sum() * 0.0
                loss_hico = logits_stu.sum() * 0.0

            easy_mask = (entropy_stu < args.easy_threshold) & (scrib.unsqueeze(1) == 4)
            if iter_num >= args.pseudo_warmup:
                loss_stable = weighted_soft_ce_loss(logits_stu, probs_slow.detach(), easy_mask, None)
            else:
                loss_stable = logits_stu.sum() * 0.0

            consistency_weight = get_current_consistency_weight(iter_num // 300, args)
            loss = (
                loss_scrib
                + consistency_weight * args.lambda_pseudo * loss_pseudo
                + consistency_weight * args.lambda_hico * loss_hico
                + consistency_weight * args.lambda_stable * loss_stable
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, teacher_fast, alpha=args.alpha_fast, global_step=iter_num)
            update_ema_variables(model, teacher_slow, alpha=args.alpha_slow, global_step=iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar('train/loss_total', loss.item(), iter_num)
            writer.add_scalar('train/loss_scrib', loss_scrib.item(), iter_num)
            writer.add_scalar('train/loss_pseudo', loss_pseudo.item(), iter_num)
            writer.add_scalar('train/loss_hico', loss_hico.item(), iter_num)
            writer.add_scalar('train/loss_stable', loss_stable.item(), iter_num)
            writer.add_scalar('train/loss_low', loss_low.item(), iter_num)
            writer.add_scalar('train/loss_high', loss_high.item(), iter_num)

            writer.add_scalar('train/entropy_stu_mean', entropy_stu.mean().item(), iter_num)
            writer.add_scalar('train/entropy_fast_mean', entropy_fast.mean().item(), iter_num)
            writer.add_scalar('train/entropy_slow_mean', entropy_slow.mean().item(), iter_num)
            writer.add_scalar('train/disagreement_mean', disagreement.mean().item(), iter_num)
            writer.add_scalar('train/boundary_mean', boundary.mean().item(), iter_num)
            writer.add_scalar('train/uncertain_ratio', uncertain_mask.float().mean().item(), iter_num)
            writer.add_scalar('train/pseudo_ratio', pseudo_mask.float().mean().item(), iter_num)
            writer.add_scalar('train/select_fast_ratio', select_fast.float().mean().item(), iter_num)
            writer.add_scalar('train/easy_ratio', easy_mask.float().mean().item(), iter_num)

            if iter_num % 400 == 0:
                logging.info('iteration %d : loss %.6f, loss_pseudo %.6f', iter_num, loss.item(), loss_pseudo.item())

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch['image'], sampled_batch['label'], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f', iter_num, performance, mean_hd95)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to {}'.format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return 'Training Finished!'


if __name__ == '__main__':
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

    snapshot_path = '../../checkpoints/{}_{}'.format(args.data, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
