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

from dataloader.mscmr import MSCMRDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import ramps
from utils.ema_optim import WeightEMA
from utils.evidential import (
    asymmetric_uncertainty_mse_loss,
    build_mt_confidence_pseudo_label,
    evidential_prediction,
    masked_soft_ce_from_prob,
    partial_ce_from_prob,
    unpack_model_output,
)
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/MSCMR', help='dataset root')
parser.add_argument('--exp', type=str, default='EMT_AUC', help='experiment name')
parser.add_argument('--data', type=str, default='MSCMR', help='dataset name')
parser.add_argument('--sup_type', type=str, default='scribble', help='supervision type')
parser.add_argument('--model', type=str, default='unet_hl', help='network name')
parser.add_argument('--num_classes', type=int, default=4, help='number of segmentation classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='network input patch size')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='pseudo-loss ramp-up')
parser.add_argument('--pseudo_agree_thresh', type=float, default=0.6, help='minimum confidence for both student and teacher when they agree')
parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.7, help='minimum confidence for the stronger prediction when student and teacher disagree')
parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1, help='minimum confidence margin between student and teacher when they disagree')
parser.add_argument('--pseudo_loss_weight', type=float, default=8.0, help='weight for reliable pseudo-label supervision')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled', choices=['unlabeled', 'all'], help='where to apply pseudo-label supervision')
parser.add_argument('--enable_uncertainty_loss', type=int, default=1, choices=[0, 1], help='Enable asymmetric evidential uncertainty consistency')
parser.add_argument('--uncertainty_loss_weight', type=float, default=0.5, help='Maximum uncertainty consistency weight')
parser.add_argument('--uncertainty_margin', type=float, default=0.0, help='Ignored Student-Teacher uncertainty gap')
parser.add_argument('--uncertainty_rampup', type=float, default=40.0, help='Epoch-length sigmoid ramp-up for uncertainty loss')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def get_current_uncertainty_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.uncertainty_rampup)


def create_model(ema=False, num_classes=4):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    model = create_model(ema=False, num_classes=num_classes)
    model_ema = create_model(ema=True, num_classes=num_classes)

    db_train = MSCMRDataSets(
        base_dir=train_args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        sup_type=train_args.sup_type,
    )
    db_val = MSCMRDataSets(base_dir=train_args.root_path, split='val')

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

    model.train()
    model_ema.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ema_optimizer = WeightEMA(model, model_ema, 0.99)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            with torch.no_grad():
                ema_output = unpack_model_output(model_ema(volume_batch))
                teacher_evi = evidential_prediction(
                    ema_output,
                    num_classes=num_classes,
                )
                teacher_prob = teacher_evi['prob']
                teacher_uncertainty = teacher_evi['uncertainty']

            outputs = unpack_model_output(model(volume_batch))
            student_evi = evidential_prediction(
                outputs,
                num_classes=num_classes,
            )
            student_prob = student_evi['prob']
            student_uncertainty = student_evi['uncertainty']

            loss_pce = partial_ce_from_prob(
                prob=student_prob,
                label=label_batch.long(),
                ignore_index=num_classes,
            )

            pseudo_info = build_mt_confidence_pseudo_label(
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                label=label_batch,
                agree_thresh=train_args.pseudo_agree_thresh,
                disagree_thresh=train_args.pseudo_disagree_thresh,
                margin_thresh=train_args.pseudo_margin_thresh,
                ignore_index=num_classes,
                pseudo_mask_mode=train_args.pseudo_mask_mode,
            )

            loss_pseudo = masked_soft_ce_from_prob(
                student_prob=student_prob,
                target_prob=pseudo_info['soft_pseudo_label'],
                mask=pseudo_info['reliable_mask'],
            )

            pseudo_weight = (
                get_current_consistency_weight(iter_num // len(trainloader), train_args)
                * train_args.pseudo_loss_weight
            )

            if train_args.enable_uncertainty_loss:
                loss_uncertainty = asymmetric_uncertainty_mse_loss(
                    student_uncertainty=student_uncertainty,
                    teacher_uncertainty=teacher_uncertainty,
                    reliable_mask=pseudo_info['reliable_mask'],
                    margin=train_args.uncertainty_margin,
                )
                uncertainty_weight = (
                    get_current_uncertainty_weight(iter_num // len(trainloader), train_args)
                    * train_args.uncertainty_loss_weight
                )
            else:
                loss_uncertainty = student_prob.new_zeros(())
                uncertainty_weight = 0.0

            loss = (
                loss_pce
                + pseudo_weight * loss_pseudo
                + uncertainty_weight * loss_uncertainty
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            with torch.no_grad():
                active_mask = (
                    (student_uncertainty > teacher_uncertainty + train_args.uncertainty_margin).float()
                    * pseudo_info['reliable_mask']
                )
                active_guidance_ratio = active_mask.sum() / (pseudo_info['reliable_mask'].sum() + 1e-8)

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_pce', loss_pce.item(), iter_num)
            writer.add_scalar('info/loss_pseudo', loss_pseudo.item(), iter_num)
            writer.add_scalar('info/loss_uncertainty', loss_uncertainty.item(), iter_num)
            writer.add_scalar('info/pseudo_weight', pseudo_weight, iter_num)
            writer.add_scalar('info/uncertainty_weight', uncertainty_weight, iter_num)
            writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)
            writer.add_scalar('uncertainty/student_mean', student_uncertainty.detach().mean().item(), iter_num)
            writer.add_scalar('uncertainty/teacher_mean', teacher_uncertainty.detach().mean().item(), iter_num)
            writer.add_scalar('uncertainty/active_guidance_ratio', active_guidance_ratio.item(), iter_num)
            writer.add_scalar('evidence/student_strength_mean', student_evi['strength'].detach().mean().item(), iter_num)
            writer.add_scalar('evidence/teacher_strength_mean', teacher_evi['strength'].detach().mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss=%f, loss_pce=%f, loss_pseudo=%f, loss_uncertainty=%f, '
                    'pseudo_weight=%f, uncertainty_weight=%f, student_u=%f, teacher_u=%f, '
                    'active_guidance=%f, reliable=%f, agree=%f, disagree=%f, pseudo_conf=%f',
                    iter_num,
                    loss.item(),
                    loss_pce.item(),
                    loss_pseudo.item(),
                    loss_uncertainty.item(),
                    pseudo_weight,
                    uncertainty_weight,
                    student_uncertainty.detach().mean().item(),
                    teacher_uncertainty.detach().mean().item(),
                    active_guidance_ratio.item(),
                    pseudo_info['reliable_ratio'].item(),
                    pseudo_info['agreement_ratio'].item(),
                    pseudo_info['disagreement_ratio'].item(),
                    pseudo_info['pseudo_conf'].mean().item(),
                )

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                metric_list = 0.0
                for sampled_val in valloader:
                    metric_i = test_single_volume(
                        sampled_val['image'],
                        sampled_val['label'],
                        model,
                        classes=num_classes,
                    )
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
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(train_args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f', iter_num, performance, mean_hd95)
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to %s', save_mode_path)

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
