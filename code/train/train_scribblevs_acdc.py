import argparse
import logging
import os
import random
import shutil
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from val import test_single_volume_scribblevs

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ScribbleVS', help='experiment_name')
parser.add_argument('--data', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--tau', type=float,
                    default=0.5, help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='MAAGfold70', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency_rampup', type=float, default=40.0,
                    help='ramp-up for consistency regularization')
parser.add_argument('--lambda_unc_cons', type=float, default=0.5,
                    help='weight for evidential uncertainty map consistency loss')
parser.add_argument('--uncertainty_consistency', type=str, default='l1',
                    choices=['l1', 'mse', 'smooth_l1'],
                    help='loss type for uncertainty map consistency')
parser.add_argument('--evidence_activation', type=str, default='relu',
                    choices=['relu', 'softplus', 'exp'],
                    help='activation used to convert logits into non-negative evidence')
parser.add_argument('--detach_teacher_uncertainty', type=int, default=1,
                    help='detach teacher uncertainty map before consistency loss')
parser.add_argument('--uncertainty_mask_mode', type=str, default='all',
                    choices=['all', 'unlabeled', 'labeled'],
                    help='where to apply uncertainty consistency for scribble training')
parser.add_argument('--uncertainty_temp', type=float, default=0.5,
                    help='temperature for uncertainty-weighted pseudo-label fusion')
parser.add_argument('--pseudo_conf_thresh', type=float, default=0.0,
                    help='optional confidence threshold for fused pseudo-label; 0 disables filtering')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['all', 'unlabeled'],
                    help='where to apply pseudo-label supervision')
parser.add_argument('--detach_student_fusion', type=int, default=1,
                    help='detach student branch when building fused pseudo-label target')
parser.add_argument('--fusion_use_teacher_only_warmup', type=int, default=0,
                    help='if >0, use teacher-only pseudo label before this iteration')
parser.add_argument('--uncertainty_target', type=str, default='teacher',
                    choices=['teacher', 'fused'],
                    help='target uncertainty map for uncertainty consistency')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def logits_to_evidence(logits, activation='relu'):
    if activation == 'relu':
        return F.relu(logits)
    if activation == 'softplus':
        return F.softplus(logits)
    if activation == 'exp':
        return torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
    raise ValueError('Unsupported evidence activation: {}'.format(activation))


def evidential_uncertainty_from_logits(logits, num_classes, activation='relu', eps=1e-8):
    evidence = logits_to_evidence(logits, activation=activation)
    alpha = evidence + 1.0
    uncertainty = float(num_classes) / (torch.sum(alpha, dim=1, keepdim=True) + eps)
    return uncertainty


def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)


def uncertainty_weighted_fusion(
    student_prob,
    teacher_prob,
    student_unc,
    teacher_unc,
    temperature=0.5,
    detach_student=True,
    detach_teacher=True,
    eps=1e-8,
):
    if detach_student:
        student_prob = student_prob.detach()
        student_unc = student_unc.detach()

    if detach_teacher:
        teacher_prob = teacher_prob.detach()
        teacher_unc = teacher_unc.detach()

    student_unc = student_unc.clamp(0.0, 1.0)
    teacher_unc = teacher_unc.clamp(0.0, 1.0)
    temperature = max(float(temperature), eps)

    ws = torch.exp(-student_unc / temperature)
    wt = torch.exp(-teacher_unc / temperature)

    w_sum = ws + wt + eps
    ws = ws / w_sum
    wt = wt / w_sum

    fused_prob = ws * student_prob + wt * teacher_prob
    fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + eps)

    return fused_prob.detach(), ws.detach(), wt.detach()


def uncertainty_consistency_loss(student_unc, teacher_unc, loss_type='l1', mask=None):
    if mask is None:
        if loss_type == 'l1':
            return F.l1_loss(student_unc, teacher_unc)
        if loss_type == 'mse':
            return F.mse_loss(student_unc, teacher_unc)
        if loss_type == 'smooth_l1':
            return F.smooth_l1_loss(student_unc, teacher_unc)
        raise ValueError('Unsupported uncertainty consistency loss: {}'.format(loss_type))

    if loss_type == 'l1':
        diff = torch.abs(student_unc - teacher_unc)
    elif loss_type == 'mse':
        diff = torch.square(student_unc - teacher_unc)
    elif loss_type == 'smooth_l1':
        diff = F.smooth_l1_loss(student_unc, teacher_unc, reduction='none')
    else:
        raise ValueError('Unsupported uncertainty consistency loss: {}'.format(loss_type))
    return (diff * mask).sum() / (mask.sum() + 1e-8)


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def create_model(ema=False,num_classes=4):
    # Network definition
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.01
        # self.wd = 0.02 * args.base_lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):

        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:

                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    model = create_model(ema=False, num_classes=num_classes)
    model_ema = create_model(ema=True, num_classes=num_classes)

    db_train = ACDCDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = ACDCDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ema_optimizer = WeightEMA(model, model_ema, alpha=0.99)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    print(len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                ema_output = unpack_model_output(model_ema(volume_batch))
                outputs_soft_ema = torch.softmax(ema_output, dim=1)
            outputs = unpack_model_output(model(volume_batch))
            outputs_soft1 = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())

            iter_num = iter_num + 1

            if iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss=%f, loss_ce=%f, loss_pse_sup=%f, loss_uncertainty=%f, '
                    'pseudo_conf=%f, mask_ratio=%f, ws=%f, wt=%f' %
                    (
                        iter_num,
                        loss.item(),
                        loss_ce.item(),
                        loss_pse_sup.item(),
                        loss_uncertainty.item(),
                        pseudo_conf.mean().item(),
                        pseudo_mask.mean().item(),
                        weight_student.mean().item(),
                        weight_teacher.mean().item(),
                    ))

            if iter_num > 1 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_scribblevs(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

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
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
