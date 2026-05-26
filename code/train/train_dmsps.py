"""ACDC: total 1356 samples; 30 samples for vadilation;
57 iterations per epoch; max epoch: 527.
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

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
from scipy.ndimage import zoom

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from val import calculate_metric_percase

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../../data/ACDC', help='Data root path')
    parser.add_argument('--data_name', type=str,
                        default='ACDC', help='Data name')  
    parser.add_argument('--model', type=str,
                        default='unet_cct', help='model_name, select: unet_cct, \
                            NestedUNet2d_2dual, swinunet_2dual')
    parser.add_argument('--exp', type=str,
                        default='DMSPS_Stage1', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='MAAGfold70', help='cross validation fold')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=60000, help='maximum epoch number to train')
    parser.add_argument('--ES_interval', type=int,
                        default=10000, help='maximum iteration iternal for early-stopping')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input. Specially, [224, 224] for swinunet')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    args = parser.parse_args()
    return args


def build_soft_pseudo_label_with_agreement(
    outputs_soft1,
    outputs_soft2,
    label,
    agree_thresh=0.6,
    disagree_thresh=0.7,
    margin_thresh=0.1,
    ignore_index=4,
):
    conf1, pred1 = torch.max(outputs_soft1, dim=1)
    conf2, pred2 = torch.max(outputs_soft2, dim=1)

    unlabeled = label == ignore_index
    same = pred1 == pred2
    diff = ~same

    min_conf = torch.minimum(conf1, conf2)
    max_conf = torch.maximum(conf1, conf2)
    margin = torch.abs(conf1 - conf2)

    reliable_agree = same & (min_conf >= agree_thresh) & unlabeled
    reliable_disagree = diff & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & unlabeled

    mean_pseudo = 0.5 * (outputs_soft1.detach() + outputs_soft2.detach())
    choose_model1 = (conf1 > conf2).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_model1, outputs_soft1.detach(), outputs_soft2.detach())

    disagreement_mask = reliable_disagree.unsqueeze(1)
    soft_pseudo_label = torch.where(disagreement_mask, high_conf_pseudo, mean_pseudo)

    return soft_pseudo_label, reliable_agree.float().mean(), reliable_disagree.float().mean()


def masked_soft_ce_loss(logits, target_prob, mask, eps=1e-8):
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return (ce_map * mask).sum() / (mask.sum() + eps)


def _extract_main_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


@torch.no_grad()
def test_all_case_2D(valloader, model, args):
    metric_list = 0.0
    for sampled_batch in valloader:
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        image_np = image.squeeze(0).cpu().numpy()
        label_np = label.squeeze(0).cpu().numpy()

        if len(image_np.shape) == 3:
            prediction = np.zeros_like(label_np)
            for ind in range(image_np.shape[0]):
                image_slice = image_np[ind]
                x, y = image_slice.shape
                image_slice = zoom(
                    image_slice,
                    (args.patch_size[0] / x, args.patch_size[1] / y),
                    order=0,
                )
                input_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda()
                logits = _extract_main_logits(model(input_tensor))
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
                prediction[ind] = zoom(
                    pred,
                    (x / args.patch_size[0], y / args.patch_size[1]),
                    order=0,
                )
        else:
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = _extract_main_logits(model(input_tensor))
            prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

        metric_i = []
        for class_idx in range(1, args.num_classes):
            metric_i.append(calculate_metric_percase(prediction == class_idx, label_np == class_idx))
        metric_list += np.array(metric_i)

    return metric_list / len(valloader.dataset)


def train(args, snapshot_path):

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    ES_interval = args.ES_interval

    # Create model
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model_parameter = sum(p.numel() for p in model.parameters())
    logging.info("model_parameter:{}M".format(round(model_parameter / (1024*1024),2)))

    # create Dataset
    db_train = ACDCDataSets( base_dir=args.root_path, split="train", transform=transforms.Compose(
                            [RandomGenerator(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    db_val = ACDCDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    random.seed(args.seed)
    np.random.seed(args.seed)   
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Data loader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0

    # Training
    model.train()
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for iter, sampled_batch in enumerate(trainloader):

            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()

            outputs, outputs_aux1 = model(img)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            
            # pCE
            loss_ce1 = ce_loss(outputs, label[:].cuda().long())
            loss_ce2 = ce_loss(outputs_aux1, label[:].cuda().long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            # Fuse pseudo-labels by agreement/disagreement regions.
            soft_pseudo_label, agreement_ratio, disagreement_ratio = build_soft_pseudo_label_with_agreement(
                outputs_soft1=outputs_soft1,
                outputs_soft2=outputs_soft2,
                label=label,
                ignore_index=num_classes,
            )
            reliable_mask = (label == num_classes).float().unsqueeze(1)
            loss_pse_sup_soft = 0.5 * (
                masked_soft_ce_loss(outputs, soft_pseudo_label, reliable_mask) +
                masked_soft_ce_loss(outputs_aux1, soft_pseudo_label, reliable_mask)
            )

            # total loss
            loss = loss_ce + 8.0 * loss_pse_sup_soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_pse_sup_soft',loss_pse_sup_soft,iter_num)
            agreement_ratio_value = agreement_ratio.item()
            disagreement_ratio_value = disagreement_ratio.item()
            writer.add_scalar('info/agreement_ratio', agreement_ratio_value, iter_num)
            writer.add_scalar('info/disagreement_ratio', disagreement_ratio_value, iter_num)
            
            # Validation
            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup_soft: %f, agree: %f, disagree: %f' 
                    %(
                        iter_num,
                        loss.item(),
                        loss_ce.item(),
                        loss_pse_sup_soft.item(),
                        agreement_ratio_value,
                        disagreement_ratio_value,
                    ))
                
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
             
                if metric_list[:, 0].mean() > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = metric_list[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', metric_list[:, 0].mean(), iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, metric_list[:, 0].mean()))

                model.train()


            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num - fresh_iter_num >= ES_interval:
                logging.info("early stooping since there is no model updating over 1w \
                    iteration, iter:{} ".format(iter_num))
                break

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations or (iter_num - fresh_iter_num >= ES_interval):
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = parse_args()
    snapshot_path = "../../checkpoints/{}_{}".format(args.data_name, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(snapshot_path+"/train_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s')) 
    logger.addHandler(console_handler)
    logger.info(str(args))
    start_time = time.time()
    train(args, snapshot_path)
    time_s = time.time()-start_time
    logging.info("time cost: {} s, i.e, {} h".format(time_s,time_s/3600))
