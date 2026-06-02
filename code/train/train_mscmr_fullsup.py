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
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.mscmr import MSCMRDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses
from val import calculate_metric_percase


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/MSCMR', help='dataset root')
parser.add_argument('--exp', type=str, default='FullSup', help='experiment name')
parser.add_argument('--data', type=str, default='MSCMR', help='dataset name')
parser.add_argument('--sup_type', type=str, default='label', help='supervision type, use dense label for full supervision')
parser.add_argument('--model', type=str, default='unet', help='network name')
parser.add_argument('--num_classes', type=int, default=4, help='number of segmentation classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per gpu')
parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
parser.add_argument('--deterministic', type=int, default=1, help='use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256], help='network input patch size')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--eval_interval', type=int, default=400, help='validation interval in iterations')
parser.add_argument('--save_interval', type=int, default=3000, help='checkpoint interval in iterations')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def extract_main_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


@torch.no_grad()
def validate(valloader, model, num_classes, patch_size):
    model.eval()
    metric_list = 0.0

    for sampled_batch in valloader:
        image = sampled_batch['image'].squeeze(0).cpu().numpy()
        label = sampled_batch['label'].squeeze(0).cpu().numpy()

        if len(image.shape) == 3:
            prediction = np.zeros_like(label)
            for slice_idx in range(image.shape[0]):
                image_slice = image[slice_idx]
                x, y = image_slice.shape
                image_slice = zoom(
                    image_slice,
                    (patch_size[0] / x, patch_size[1] / y),
                    order=0,
                )
                input_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda()
                logits = extract_main_logits(model(input_tensor))
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
                prediction[slice_idx] = zoom(
                    pred,
                    (x / patch_size[0], y / patch_size[1]),
                    order=0,
                )
        else:
            input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = extract_main_logits(model(input_tensor))
            prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

        metric_i = []
        for class_i in range(1, num_classes):
            metric_i.append(calculate_metric_percase(prediction == class_i, label == class_i))
        metric_list += np.array(metric_i)

    return metric_list / len(valloader.dataset)


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    model = net_factory(net_type=train_args.model, in_chns=1, class_num=num_classes).cuda()

    db_train = MSCMRDataSets(
        base_dir=train_args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        sup_type=train_args.sup_type,
    )
    db_val = MSCMRDataSets(base_dir=train_args.root_path, split='val')

    def worker_init_fn(worker_id):
        random.seed(train_args.seed + worker_id)
        np.random.seed(train_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()

    for _ in iterator:
        for sampled_batch in trainloader:
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda().long()

            logits = extract_main_logits(model(image_batch))
            outputs_soft = torch.softmax(logits, dim=1)

            loss_ce = ce_loss(logits, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

            if iter_num % train_args.eval_interval == 0:
                metric_list = validate(valloader, model, num_classes, train_args.patch_size)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
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

                logging.info(
                    'iteration %d : loss=%f, mean_dice=%f, mean_hd95=%f',
                    iter_num,
                    loss.item(),
                    performance,
                    mean_hd95,
                )
                model.train()

            if iter_num % train_args.save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to %s', save_mode_path)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    final_model_path = os.path.join(snapshot_path, '{}_final_model.pth'.format(train_args.model))
    torch.save(model.state_dict(), final_model_path)
    logging.info('save final model to %s', final_model_path)
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
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
