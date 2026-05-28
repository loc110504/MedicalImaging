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
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import ramps
from utils.ema_optim import WeightEMA
from val import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/ACDC', help='dataset root')
parser.add_argument('--exp', type=str, default='MT_Confidence', help='experiment name')
parser.add_argument('--data', type=str, default='ACDC', help='dataset name')
parser.add_argument('--fold', type=str, default='MAAGfold70', help='dataset fold')
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
parser.add_argument('--pseudo_agree_thresh', type=float, default=0.6,
                    help='minimum confidence for both student and teacher when they agree')
parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.7,
                    help='minimum confidence for the stronger prediction when student and teacher disagree')
parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='minimum confidence margin between student and teacher when they disagree')
parser.add_argument('--pseudo_loss_weight', type=float, default=8.0,
                    help='weight for reliable pseudo-label supervision')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'all'],
                    help='where to apply pseudo-label supervision')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)


def build_mt_confidence_pseudo_label(
    student_prob,
    teacher_prob,
    label,
    agree_thresh=0.7,
    disagree_thresh=0.8,
    margin_thresh=0.1,
    ignore_index=4,
    pseudo_mask_mode='unlabeled',
    eps=1e-8,
):
    student_prob = student_prob.detach()
    teacher_prob = teacher_prob.detach()

    conf_s, pred_s = torch.max(student_prob, dim=1)
    conf_t, pred_t = torch.max(teacher_prob, dim=1)

    if pseudo_mask_mode == 'unlabeled':
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == 'all':
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError('Unsupported pseudo_mask_mode: {}'.format(pseudo_mask_mode))

    same_pred = pred_s == pred_t
    diff_pred = ~same_pred

    min_conf = torch.minimum(conf_s, conf_t)
    max_conf = torch.maximum(conf_s, conf_t)
    margin = torch.abs(conf_s - conf_t)

    reliable_agree = same_pred & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = diff_pred & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & candidate_mask

    mean_pseudo = 0.5 * (student_prob + teacher_prob)
    choose_student = (conf_s > conf_t).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_student, student_prob, teacher_prob)

    soft_pseudo_label = torch.where(
        reliable_disagree.unsqueeze(1),
        high_conf_pseudo,
        mean_pseudo,
    )
    soft_pseudo_label = soft_pseudo_label / (soft_pseudo_label.sum(dim=1, keepdim=True) + eps)

    reliable_mask = (reliable_agree | reliable_disagree).float().unsqueeze(1)
    pseudo_conf = torch.maximum(conf_s, conf_t).unsqueeze(1)

    return {
        'soft_pseudo_label': soft_pseudo_label.detach(),
        'reliable_mask': reliable_mask.detach(),
        'reliable_agree': reliable_agree,
        'reliable_disagree': reliable_disagree,
        'agreement_ratio': reliable_agree.float().mean(),
        'disagreement_ratio': reliable_disagree.float().mean(),
        'reliable_ratio': reliable_mask.mean(),
        'pseudo_conf': pseudo_conf.detach(),
    }


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

    db_train = ACDCDataSets(
        base_dir=train_args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        fold=train_args.fold,
        sup_type=train_args.sup_type,
    )
    db_val = ACDCDataSets(base_dir=train_args.root_path, fold=train_args.fold, split='val')

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
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)

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
                teacher_prob = torch.softmax(ema_output, dim=1)

            outputs = unpack_model_output(model(volume_batch))
            student_prob = torch.softmax(outputs, dim=1)

            loss_pce = ce_loss(outputs, label_batch.long())

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

            loss_pseudo = masked_soft_ce_loss(
                logits=outputs,
                target_prob=pseudo_info['soft_pseudo_label'],
                mask=pseudo_info['reliable_mask'],
            )

            pseudo_weight = (
                get_current_consistency_weight(iter_num // len(trainloader), train_args)
                * train_args.pseudo_loss_weight
            )
            loss = loss_pce + pseudo_weight * loss_pseudo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_pce', loss_pce.item(), iter_num)
            writer.add_scalar('info/loss_pseudo', loss_pseudo.item(), iter_num)
            writer.add_scalar('info/pseudo_weight', pseudo_weight, iter_num)
            writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss=%f, loss_pce=%f, loss_pseudo=%f, pseudo_weight=%f, '
                    'reliable=%f, agree=%f, disagree=%f, pseudo_conf=%f',
                    iter_num,
                    loss.item(),
                    loss_pce.item(),
                    loss_pseudo.item(),
                    pseudo_weight,
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
