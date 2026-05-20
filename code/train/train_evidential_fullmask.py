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

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from val import test_single_volume
from utils.evidential_losses import EvidentialSegmentationLoss

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../../data/ACDC')
parser.add_argument('--exp', type=str, default='EvidentialFullMask')
parser.add_argument('--data', type=str, default='ACDC')
parser.add_argument('--fold', type=str, default='MAAGfold70')
parser.add_argument('--sup_type', type=str, default='label')
parser.add_argument('--model', type=str, default='unet_hl')
parser.add_argument('--num_classes', type=int, default=4)

parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--lambda_kl', type=float, default=0.2)
parser.add_argument('--lambda_dice', type=float, default=1.0)
parser.add_argument('--alpha0', type=float, default=1.0)
parser.add_argument('--kl_annealing_epochs', type=int, default=50)
parser.add_argument(
    '--evidence_activation',
    type=str,
    default='relu',
    choices=['relu', 'softplus', 'exp'],
)
parser.add_argument('--include_background_dice', action='store_true')

parser.add_argument('--ignore_index', type=int, default=-1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.ignore_index < 0:
    args.ignore_index = None


def unpack_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class LogitsOnlyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)

        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        return logits, None, None


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(
        net_type=args.model,
        in_chns=1,
        class_num=args.num_classes,
    )

    model.cuda()
    model.train()

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    criterion = EvidentialSegmentationLoss(
        num_classes=args.num_classes,
        lambda_kl=args.lambda_kl,
        lambda_dice=args.lambda_dice,
        ignore_index=args.ignore_index,
        activation=args.evidence_activation,
        alpha0=args.alpha0,
        kl_annealing_epochs=args.kl_annealing_epochs,
        include_background=args.include_background_dice,
    )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled in enumerate(trainloader):
            image = sampled['image'].cuda()
            label = sampled['label'].cuda().long()

            output = model(image)
            logits = unpack_logits(output)

            loss, loss_dict, _ = criterion(
                logits=logits,
                target=label,
                epoch=epoch_num,
                max_epoch=max_epoch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar("train/loss_total", loss_dict["loss_total"].item(), iter_num)
            writer.add_scalar("train/loss_ece", loss_dict["loss_ece"].item(), iter_num)
            writer.add_scalar("train/loss_ceu", loss_dict["loss_ceu"].item(), iter_num)
            writer.add_scalar("train/loss_kl", loss_dict["loss_kl"].item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dict["loss_dice"].item(), iter_num)
            writer.add_scalar("train/uncertainty_mean", loss_dict["uncertainty_mean"].item(), iter_num)
            writer.add_scalar("train/evidence_mean", loss_dict["evidence_mean"].item(), iter_num)
            writer.add_scalar("train/lr", lr_, iter_num)

            if iter_num % 400 == 0:
                logging.info(
                    "iteration %d : loss=%f, ce=%f, ceu=%f, kl=%f, dice=%f, uncertainty=%f"
                    % (
                        iter_num,
                        loss_dict["loss_total"].item(),
                        loss_dict["loss_ece"].item(),
                        loss_dict["loss_ceu"].item(),
                        loss_dict["loss_kl"].item(),
                        loss_dict["loss_dice"].item(),
                        loss_dict["uncertainty_mean"].item(),
                    )
                )

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                eval_model = LogitsOnlyModel(model)
                metric_list = 0.0

                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        eval_model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        'info/val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance

                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(
                            iter_num,
                            round(best_performance, 4),
                        ),
                    )

                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(args.model),
                    )

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f'
                    % (iter_num, performance, mean_hd95)
                )

                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path,
                    'iter_' + str(iter_num) + '.pth',
                )
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

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
