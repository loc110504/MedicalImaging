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
from scipy.ndimage import zoom
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import ramps
from utils.ema_optim import WeightEMA
from val import calculate_metric_percase


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/ACDC', help='dataset root')
parser.add_argument('--exp', type=str, default='RPG_MT', help='experiment name')
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
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='ramp-up length in epochs')
parser.add_argument('--eval_ema', type=int, default=0, help='evaluate EMA teacher instead of student')

parser.add_argument('--pseudo_agree_thresh', type=float, default=0.7,
                    help='confidence threshold for agreement pixels')
parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.8,
                    help='confidence threshold for stronger prediction in disagreement pixels')
parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='minimum confidence margin in disagreement pixels')
parser.add_argument('--pseudo_loss_weight', type=float, default=8.0,
                    help='weight for graph-diffused pseudo-label loss')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'all'],
                    help='candidate region for pseudo-label selection')

parser.add_argument('--use_graph_diffusion', type=int, default=1,
                    help='enable graph-harmonic pseudo-label diffusion')
parser.add_argument('--graph_diffusion_iters', type=int, default=3,
                    help='number of local graph diffusion iterations')
parser.add_argument('--graph_alpha', type=float, default=0.7,
                    help='diffusion strength')
parser.add_argument('--graph_kernel_size', type=int, default=7,
                    help='local window size for graph diffusion')
parser.add_argument('--graph_sigma_feat', type=float, default=0.5,
                    help='feature similarity bandwidth')
parser.add_argument('--graph_sigma_spatial', type=float, default=4.0,
                    help='spatial similarity bandwidth')
parser.add_argument('--graph_sigma_intensity', type=float, default=0.2,
                    help='intensity similarity bandwidth')
parser.add_argument('--graph_use_intensity', type=int, default=1,
                    help='use image intensity in graph affinity')
parser.add_argument('--graph_expand_weight', type=float, default=0.5,
                    help='weight for non-seed diffused pixels')
parser.add_argument('--graph_entropy_thresh', type=float, default=0.6,
                    help='threshold for low-entropy diffused pseudo-labels')
parser.add_argument('--graph_supervise_mode', type=str, default='seed_plus_diffused',
                    choices=['seed_only', 'seed_plus_diffused'],
                    help='where to apply graph-diffused pseudo-label loss')

parser.add_argument('--use_proto', type=int, default=1, help='enable prototype loss')
parser.add_argument('--proto_loss_weight', type=float, default=0.1, help='weight for prototype loss')
parser.add_argument('--proto_temperature', type=float, default=0.2,
                    help='temperature for prototype contrastive loss')
parser.add_argument('--proto_margin', type=float, default=0.3, help='margin for prototype loss')
parser.add_argument('--proto_momentum', type=float, default=0.99, help='EMA momentum for prototype memory')
parser.add_argument('--proto_loss_type', type=str, default='contrastive',
                    choices=['contrastive', 'margin', 'both'],
                    help='prototype loss formulation')
parser.add_argument('--proto_min_pixels', type=int, default=5,
                    help='minimum pixels required to update a class prototype in a batch')
parser.add_argument('--proto_warmup_iters', type=int, default=1000,
                    help='start prototype loss after this number of iterations')

parser.add_argument('--feature_level', type=str, default='decoder',
                    choices=['encoder', 'bottleneck', 'decoder'],
                    help='which feature map to use for graph and prototype learning')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, train_args):
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)


def build_confidence_agree_disagree_pseudo(
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

    same = pred_s == pred_t
    diff = ~same

    min_conf = torch.minimum(conf_s, conf_t)
    max_conf = torch.maximum(conf_s, conf_t)
    margin = torch.abs(conf_s - conf_t)

    reliable_agree = same & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = diff & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & candidate_mask

    mean_pseudo = 0.5 * (student_prob + teacher_prob)
    choose_student = (conf_s > conf_t).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_student, student_prob, teacher_prob)

    soft_pseudo_label = torch.where(reliable_disagree.unsqueeze(1), high_conf_pseudo, mean_pseudo)
    soft_pseudo_label = soft_pseudo_label / (soft_pseudo_label.sum(dim=1, keepdim=True) + eps)

    reliable_mask = (reliable_agree | reliable_disagree).float().unsqueeze(1)
    pseudo_conf = torch.maximum(conf_s, conf_t).unsqueeze(1)

    return {
        'soft_pseudo_label': soft_pseudo_label.detach(),
        'reliable_mask': reliable_mask.detach(),
        'reliable_agree': reliable_agree.detach(),
        'reliable_disagree': reliable_disagree.detach(),
        'agreement_ratio': reliable_agree.float().mean(),
        'disagreement_ratio': reliable_disagree.float().mean(),
        'reliable_ratio': reliable_mask.mean(),
        'pseudo_conf': pseudo_conf.detach(),
    }


def build_local_spatial_distance(kernel_size, device):
    radius = kernel_size // 2
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1, device=device),
            torch.arange(-radius, radius + 1, device=device),
            indexing='ij',
        ),
        dim=-1,
    ).float()
    return (coords[..., 0] ** 2 + coords[..., 1] ** 2).reshape(-1)


def graph_harmonic_diffusion(
    seed_pseudo,
    seed_mask,
    feature,
    image=None,
    num_classes=4,
    num_iters=3,
    kernel_size=7,
    alpha=0.7,
    sigma_feat=0.5,
    sigma_spatial=4.0,
    sigma_intensity=0.2,
    use_intensity=True,
    expand_weight=0.5,
    entropy_thresh=0.6,
    supervise_mode='seed_plus_diffused',
    eps=1e-8,
):
    batch_size, num_channels, height, width = seed_pseudo.shape
    feat_dim = feature.shape[1]

    z = F.normalize(feature, p=2, dim=1)
    z_patches = F.unfold(z, kernel_size=kernel_size, padding=kernel_size // 2)
    z_patches = z_patches.view(batch_size, feat_dim, kernel_size * kernel_size, height * width)
    z_center = z.view(batch_size, feat_dim, 1, height * width)

    feat_dist = ((z_patches - z_center) ** 2).sum(dim=1)
    spatial_dist = build_local_spatial_distance(kernel_size, feature.device).view(1, kernel_size * kernel_size, 1)

    affinity_log = -feat_dist / (sigma_feat ** 2 + eps)
    affinity_log = affinity_log - spatial_dist / (sigma_spatial ** 2 + eps)

    if use_intensity and image is not None:
        image_low = F.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
        image_patches = F.unfold(image_low, kernel_size=kernel_size, padding=kernel_size // 2)
        image_patches = image_patches.view(batch_size, 1, kernel_size * kernel_size, height * width)
        image_center = image_low.view(batch_size, 1, 1, height * width)
        intensity_dist = ((image_patches - image_center) ** 2).sum(dim=1)
        affinity_log = affinity_log - intensity_dist / (sigma_intensity ** 2 + eps)

    affinity = torch.softmax(affinity_log, dim=1)
    diffusion = seed_pseudo.clone()

    for _ in range(num_iters):
        diff_patches = F.unfold(diffusion, kernel_size=kernel_size, padding=kernel_size // 2)
        diff_patches = diff_patches.view(batch_size, num_channels, kernel_size * kernel_size, height * width)

        neighbor = (affinity.unsqueeze(1) * diff_patches).sum(dim=2)
        neighbor = neighbor.view(batch_size, num_channels, height, width)

        diffusion = alpha * neighbor + (1.0 - alpha) * diffusion
        diffusion = seed_mask * seed_pseudo + (1.0 - seed_mask) * diffusion
        diffusion = diffusion / (diffusion.sum(dim=1, keepdim=True) + eps)

    entropy = -(diffusion * torch.log(diffusion + eps)).sum(dim=1, keepdim=True)
    diffusion_conf = 1.0 - entropy / np.log(num_classes)

    if supervise_mode == 'seed_only':
        weight = seed_mask
    elif supervise_mode == 'seed_plus_diffused':
        diffused_mask = (diffusion_conf >= entropy_thresh).float() * (1.0 - seed_mask)
        weight = seed_mask + expand_weight * diffusion_conf * diffused_mask
    else:
        raise ValueError('Unsupported supervise_mode: {}'.format(supervise_mode))

    return diffusion.detach(), weight.detach(), diffusion_conf.detach()


def build_scribble_reliable_proto_targets(label, seed_pseudo, seed_mask, feature_size, ignore_index=4):
    height, width = feature_size

    label_low = F.interpolate(label.unsqueeze(1).float(), size=(height, width), mode='nearest').squeeze(1).long()
    seed_mask_low = F.interpolate(seed_mask.float(), size=(height, width), mode='nearest')

    seed_pseudo_low = F.interpolate(seed_pseudo, size=(height, width), mode='bilinear', align_corners=False)
    seed_pseudo_low = seed_pseudo_low / (seed_pseudo_low.sum(dim=1, keepdim=True) + 1e-8)
    pseudo_cls = torch.argmax(seed_pseudo_low, dim=1)

    labeled_mask = label_low != ignore_index
    pseudo_mask = seed_mask_low.squeeze(1) > 0.5

    targets = torch.full_like(label_low, fill_value=ignore_index)
    targets[labeled_mask] = label_low[labeled_mask]

    use_pseudo = (~labeled_mask) & pseudo_mask
    targets[use_pseudo] = pseudo_cls[use_pseudo]

    valid_mask = (targets != ignore_index).float().unsqueeze(1)
    return targets, valid_mask


class ClassPrototypeMemory:
    def __init__(self, num_classes, feat_dim, momentum=0.99, min_pixels=5, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.min_pixels = min_pixels
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, features, targets, mask):
        _, feat_dim, _, _ = features.shape
        feats = F.normalize(features, p=2, dim=1).permute(0, 2, 3, 1).reshape(-1, feat_dim)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.squeeze(1).reshape(-1).bool()

        for cls_idx in range(self.num_classes):
            class_mask = mask_flat & (targets_flat == cls_idx)
            if class_mask.sum() < self.min_pixels:
                continue

            proto = feats[class_mask].mean(dim=0)
            proto = F.normalize(proto, p=2, dim=0)

            if not self.initialized[cls_idx]:
                self.prototypes[cls_idx] = proto
                self.initialized[cls_idx] = True
            else:
                self.prototypes[cls_idx] = (
                    self.momentum * self.prototypes[cls_idx]
                    + (1.0 - self.momentum) * proto
                )
                self.prototypes[cls_idx] = F.normalize(self.prototypes[cls_idx], p=2, dim=0)

    def compute_loss(self, features, targets, mask, temperature=0.2, margin=0.3, loss_type='contrastive'):
        _, feat_dim, _, _ = features.shape
        if self.initialized.sum() < self.num_classes:
            return features.new_tensor(0.0)

        feats = F.normalize(features, p=2, dim=1).permute(0, 2, 3, 1).reshape(-1, feat_dim)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.squeeze(1).reshape(-1).bool()

        valid = mask_flat & (targets_flat >= 0) & (targets_flat < self.num_classes)
        if valid.sum() < 1:
            return features.new_tensor(0.0)

        z_valid = feats[valid]
        y_valid = targets_flat[valid].long()
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        sim = torch.matmul(z_valid, prototypes.t())

        losses = []
        if loss_type in ['contrastive', 'both']:
            logits_proto = sim / temperature
            losses.append(F.cross_entropy(logits_proto, y_valid))

        if loss_type in ['margin', 'both']:
            dist = 1.0 - sim
            pos_dist = dist.gather(1, y_valid.view(-1, 1)).squeeze(1)
            neg_mask = torch.ones_like(dist, dtype=torch.bool)
            neg_mask.scatter_(1, y_valid.view(-1, 1), False)
            neg_dist = dist.masked_fill(~neg_mask, 1e6).min(dim=1)[0]
            losses.append(F.relu(margin + pos_dist - neg_dist).mean())

        if not losses:
            raise ValueError('Unsupported proto loss type: {}'.format(loss_type))
        return sum(losses)


def decode_from_feature_list(model, features):
    if hasattr(model, 'decoder_s'):
        return model.decoder_s(features)

    decoder = getattr(model, 'decoder', None)
    if decoder is None:
        raise ValueError('Model does not expose decoder features needed for RPG-MT')

    x0, x1, x2, x3, x4 = features
    x = decoder.up1(x4, x3)
    high_feature = x
    x = decoder.up2(x, x2)
    x = decoder.up3(x, x1)
    low_feature = decoder.up4(x, x0)
    if hasattr(decoder, 'refine'):
        low_feature = decoder.refine(low_feature)
    logits = decoder.out_conv(low_feature)
    return logits, high_feature, low_feature


def forward_with_feature(model, x, feature_level='decoder'):
    if not hasattr(model, 'encoder'):
        raise ValueError('Model must expose encoder for RPG-MT feature extraction')

    features = model.encoder(x)
    logits, high_feature, low_feature = decode_from_feature_list(model, features)

    if feature_level == 'encoder':
        feature = features[0]
    elif feature_level == 'bottleneck':
        feature = features[-1]
    elif feature_level == 'decoder':
        feature = low_feature
    else:
        raise ValueError('Unsupported feature_level: {}'.format(feature_level))

    return logits, F.normalize(feature, p=2, dim=1), {
        'encoder': features[0],
        'bottleneck': features[-1],
        'decoder_high': high_feature,
        'decoder_low': low_feature,
    }


def model_predict_logits(model, x):
    logits, _, _ = forward_with_feature(model, x, feature_level=args.feature_level)
    return logits


@torch.no_grad()
def test_single_volume_generic(image, label, net, classes, patch_size=(256, 256)):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            image_slice = image[ind, :, :]
            x_size, y_size = image_slice.shape[0], image_slice.shape[1]
            image_slice = zoom(image_slice, (patch_size[0] / x_size, patch_size[1] / y_size), order=0)
            input_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            logits = model_predict_logits(net, input_tensor)
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = zoom(out, (x_size / patch_size[0], y_size / patch_size[1]), order=0)
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        logits = model_predict_logits(net, input_tensor)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

    metric_list = []
    for cls_idx in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == cls_idx, label == cls_idx))
    return metric_list


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

    prototype_memory = None
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            with torch.no_grad():
                teacher_logits, _, _ = forward_with_feature(
                    model_ema, volume_batch, feature_level=train_args.feature_level
                )
                teacher_prob = torch.softmax(teacher_logits, dim=1)

            student_logits, student_feature, _ = forward_with_feature(
                model, volume_batch, feature_level=train_args.feature_level
            )
            student_prob = torch.softmax(student_logits, dim=1)
            loss_pce = ce_loss(student_logits, label_batch.long())

            pseudo_info = build_confidence_agree_disagree_pseudo(
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                label=label_batch,
                agree_thresh=train_args.pseudo_agree_thresh,
                disagree_thresh=train_args.pseudo_disagree_thresh,
                margin_thresh=train_args.pseudo_margin_thresh,
                ignore_index=num_classes,
                pseudo_mask_mode=train_args.pseudo_mask_mode,
            )

            q_seed = pseudo_info['soft_pseudo_label']
            m_seed = pseudo_info['reliable_mask']

            if train_args.use_graph_diffusion:
                feature_height, feature_width = student_feature.shape[-2:]
                q_low = F.interpolate(q_seed, size=(feature_height, feature_width), mode='bilinear', align_corners=False)
                q_low = q_low / (q_low.sum(dim=1, keepdim=True) + 1e-8)
                m_low = F.interpolate(m_seed, size=(feature_height, feature_width), mode='nearest')

                u_low, w_low, diffusion_conf_low = graph_harmonic_diffusion(
                    seed_pseudo=q_low,
                    seed_mask=m_low,
                    feature=student_feature.detach(),
                    image=volume_batch,
                    num_classes=num_classes,
                    num_iters=train_args.graph_diffusion_iters,
                    kernel_size=train_args.graph_kernel_size,
                    alpha=train_args.graph_alpha,
                    sigma_feat=train_args.graph_sigma_feat,
                    sigma_spatial=train_args.graph_sigma_spatial,
                    sigma_intensity=train_args.graph_sigma_intensity,
                    use_intensity=bool(train_args.graph_use_intensity),
                    expand_weight=train_args.graph_expand_weight,
                    entropy_thresh=train_args.graph_entropy_thresh,
                    supervise_mode=train_args.graph_supervise_mode,
                )

                u_high = F.interpolate(u_low, size=student_logits.shape[-2:], mode='bilinear', align_corners=False)
                u_high = u_high / (u_high.sum(dim=1, keepdim=True) + 1e-8)
                w_high = F.interpolate(w_low, size=student_logits.shape[-2:], mode='nearest')
            else:
                u_high = q_seed
                w_high = m_seed
                diffusion_conf_low = None

            loss_ghrd = masked_soft_ce_loss(
                logits=student_logits,
                target_prob=u_high.detach(),
                mask=w_high.detach(),
            )

            if prototype_memory is None:
                prototype_memory = ClassPrototypeMemory(
                    num_classes=num_classes,
                    feat_dim=student_feature.shape[1],
                    momentum=train_args.proto_momentum,
                    min_pixels=train_args.proto_min_pixels,
                    device=student_feature.device,
                )

            if train_args.use_proto and iter_num >= train_args.proto_warmup_iters:
                proto_targets, proto_mask = build_scribble_reliable_proto_targets(
                    label=label_batch,
                    seed_pseudo=q_seed,
                    seed_mask=m_seed,
                    feature_size=student_feature.shape[-2:],
                    ignore_index=num_classes,
                )

                prototype_memory.update(
                    features=student_feature.detach(),
                    targets=proto_targets,
                    mask=proto_mask,
                )

                loss_proto = prototype_memory.compute_loss(
                    features=student_feature,
                    targets=proto_targets,
                    mask=proto_mask,
                    temperature=train_args.proto_temperature,
                    margin=train_args.proto_margin,
                    loss_type=train_args.proto_loss_type,
                )
            else:
                loss_proto = student_logits.new_tensor(0.0)

            ramp = get_current_consistency_weight(epoch_num, train_args)
            lambda_pseudo = ramp * train_args.pseudo_loss_weight
            if train_args.use_proto and iter_num >= train_args.proto_warmup_iters:
                lambda_proto = ramp * train_args.proto_loss_weight
            else:
                lambda_proto = 0.0

            loss = loss_pce + lambda_pseudo * loss_ghrd + lambda_proto * loss_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/loss_pce', loss_pce.item(), iter_num)
            writer.add_scalar('loss/loss_ghrd', loss_ghrd.item(), iter_num)
            writer.add_scalar('loss/loss_proto', loss_proto.item(), iter_num)
            writer.add_scalar('weight/lambda_pseudo', lambda_pseudo, iter_num)
            writer.add_scalar('weight/lambda_proto', lambda_proto, iter_num)
            writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)
            writer.add_scalar('graph/w_high_ratio', (w_high > 0).float().mean().item(), iter_num)
            writer.add_scalar('graph/w_high_mean', w_high.mean().item(), iter_num)

            if diffusion_conf_low is not None:
                writer.add_scalar('graph/diffusion_conf_mean', diffusion_conf_low.mean().item(), iter_num)

            if prototype_memory is not None:
                writer.add_scalar('proto/initialized_classes',
                                  prototype_memory.initialized.float().sum().item(), iter_num)
                writer.add_scalar('proto/prototype_norm_mean',
                                  prototype_memory.prototypes.norm(dim=1).mean().item(), iter_num)

            if iter_num % 200 == 0:
                logging.info(
                    'iter %d | loss %.4f | pce %.4f | ghrd %.4f | proto %.4f | '
                    'lambda_p %.4f | lambda_proto %.4f | rel %.4f | agree %.4f | disagree %.4f | '
                    'graph_w %.4f',
                    iter_num,
                    loss.item(),
                    loss_pce.item(),
                    loss_ghrd.item(),
                    loss_proto.item(),
                    lambda_pseudo,
                    lambda_proto,
                    pseudo_info['reliable_ratio'].item(),
                    pseudo_info['agreement_ratio'].item(),
                    pseudo_info['disagreement_ratio'].item(),
                    w_high.mean().item(),
                )

            if iter_num > 1 and iter_num % 400 == 0:
                eval_model = model_ema if train_args.eval_ema else model
                eval_model.eval()
                metric_list = 0.0
                for sampled_val in valloader:
                    metric_i = test_single_volume_generic(
                        sampled_val['image'],
                        sampled_val['label'],
                        eval_model,
                        classes=num_classes,
                        patch_size=tuple(train_args.patch_size),
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                for class_idx in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_idx + 1), metric_list[class_idx, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_idx + 1), metric_list[class_idx, 1], iter_num)

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
                model_ema.train()

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
