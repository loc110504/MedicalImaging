import argparse
import csv
import json
import logging
import math
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

# =========================
# Basic training arguments
# =========================
parser.add_argument('--root_path', type=str, default='../../data/ACDC', help='dataset root')
parser.add_argument('--exp', type=str, default='MT_Confidence_RAC', help='experiment name')
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

# =========================
# Mean Teacher / pseudo loss
# =========================
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='pseudo-loss ramp-up epoch length')
parser.add_argument('--pseudo_loss_weight', type=float, default=8.0, help='weight for reliable pseudo-label supervision')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'all'],
                    help='where to apply pseudo-label supervision')

# Fixed-threshold fallback. These are used only when --use_threshold_curriculum 0.
parser.add_argument('--pseudo_agree_thresh', type=float, default=0.6,
                    help='fixed confidence threshold for agreement pixels')
parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.7,
                    help='fixed confidence threshold for stronger prediction in disagreement pixels')
parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='fixed confidence margin threshold in disagreement pixels')

# =========================
# Reliability-Anchored Confidence Curriculum
# =========================
parser.add_argument('--use_threshold_curriculum', type=int, default=1,
                    help='enable conservative-to-permissive confidence threshold curriculum')
parser.add_argument('--threshold_schedule', type=str, default='cosine',
                    choices=['cosine', 'linear'],
                    help='threshold decay schedule')
# parser.add_argument('--threshold_warmup_iters', type=int, default=1500,
#                     help='iterations before decreasing confidence thresholds')
# parser.add_argument('--threshold_decay_iters', type=int, default=20000,
#                     help='iterations used to decrease confidence thresholds')

parser.add_argument('--agree_thresh_start', type=float, default=0.80,
                    help='initial high threshold for agreement pixels')
parser.add_argument('--agree_thresh_end', type=float, default=0.50,
                    help='final lower threshold for agreement pixels')

parser.add_argument('--disagree_thresh_start', type=float, default=0.90,
                    help='initial high threshold for disagreement pixels')
parser.add_argument('--disagree_thresh_end', type=float, default=0.60,
                    help='final lower threshold for disagreement pixels')

parser.add_argument('--threshold_warmup_iters', type=int, default=1500,
                    help='iterations before decreasing confidence thresholds')
parser.add_argument('--threshold_decay_iters', type=int, default=25000,
                    help='iterations used to decrease confidence thresholds')

parser.add_argument('--margin_thresh_start', type=float, default=0.15,
                    help='initial confidence margin threshold')
parser.add_argument('--margin_thresh_end', type=float, default=0.10,
                    help='final confidence margin threshold')

parser.add_argument('--disagree_decay_power', type=float, default=1.5,
                    help='larger value makes disagreement threshold decay slower than agreement threshold')
parser.add_argument('--min_disagree_gap', type=float, default=0.10,
                    help='minimum gap: disagree threshold should be at least agree threshold + this value')
parser.add_argument('--oracle_metric_logging', type=int, default=0,
                    help='log pseudo-label selection quality against full training masks without using them in loss')
parser.add_argument('--threshold_sweep_logging', type=int, default=0,
                    help='evaluate multiple threshold configs during one training run and export rankings')
parser.add_argument('--threshold_sweep_interval', type=int, default=50,
                    help='evaluate threshold sweep every N iterations')
parser.add_argument('--threshold_sweep_report_interval', type=int, default=1000,
                    help='refresh sweep summary files every N iterations')
parser.add_argument('--threshold_sweep_start_iter', type=int, default=500,
                    help='start sweep analysis from this iteration')
parser.add_argument('--threshold_sweep_late_start_iter', type=int, default=-1,
                    help='late-stage boundary for sweep ranking; -1 means 60 percent of max_iterations')
parser.add_argument('--threshold_sweep_topk', type=int, default=5,
                    help='number of top sweep candidates to print and export')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, train_args):
    """Sigmoid ramp-up for pseudo-label loss weight."""
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def unpack_model_output(output):
    """Support models that return either logits or tuple/list where first item is logits."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    """
    Soft cross entropy with an optional spatial mask.

    Args:
        logits:      [B, C, H, W]
        target_prob: [B, C, H, W], detached soft pseudo-label distribution
        mask:        [B, 1, H, W], detached reliable mask
    """
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)


def _schedule_progress(iter_num, warmup_iters, decay_iters, schedule='cosine'):
    """
    Returns smooth progress in [0, 1].
    progress = 0 during warmup, then increases to 1.
    """
    if iter_num <= warmup_iters:
        raw = 0.0
    else:
        raw = (iter_num - warmup_iters) / float(max(decay_iters, 1))
        raw = min(max(raw, 0.0), 1.0)

    if schedule == 'linear':
        return raw

    if schedule == 'cosine':
        return 0.5 * (1.0 - math.cos(math.pi * raw))

    raise ValueError('Unsupported threshold_schedule: {}'.format(schedule))


def _get_config_value(config_source, key):
    if isinstance(config_source, dict):
        return config_source[key]
    return getattr(config_source, key)


def get_threshold_curriculum_from_config(iter_num, config_source):
    """
    Reliability-Anchored Confidence Curriculum.

    Early stage:
        high thresholds -> few but very reliable pseudo-labels.

    Late stage:
        lower thresholds -> more pseudo-label coverage.

    Disagreement threshold decays slower and remains stricter than agreement threshold.
    """
    if not _get_config_value(config_source, 'use_threshold_curriculum'):
        return (
            _get_config_value(config_source, 'pseudo_agree_thresh'),
            _get_config_value(config_source, 'pseudo_disagree_thresh'),
            _get_config_value(config_source, 'pseudo_margin_thresh'),
        )

    smooth = _schedule_progress(
        iter_num=iter_num,
        warmup_iters=_get_config_value(config_source, 'threshold_warmup_iters'),
        decay_iters=_get_config_value(config_source, 'threshold_decay_iters'),
        schedule=_get_config_value(config_source, 'threshold_schedule'),
    )

    agree_thresh = (
        _get_config_value(config_source, 'agree_thresh_start')
        - smooth * (
            _get_config_value(config_source, 'agree_thresh_start')
            - _get_config_value(config_source, 'agree_thresh_end')
        )
    )

    # Disagreement pseudo-labels are riskier, so relax them more slowly.
    smooth_disagree = smooth ** _get_config_value(config_source, 'disagree_decay_power')
    disagree_thresh = (
        _get_config_value(config_source, 'disagree_thresh_start')
        - smooth_disagree * (
            _get_config_value(config_source, 'disagree_thresh_start')
            - _get_config_value(config_source, 'disagree_thresh_end')
        )
    )

    margin_thresh = (
        _get_config_value(config_source, 'margin_thresh_start')
        - smooth * (
            _get_config_value(config_source, 'margin_thresh_start')
            - _get_config_value(config_source, 'margin_thresh_end')
        )
    )

    # Safety constraint: disagreement must remain stricter than agreement.
    disagree_thresh = max(
        disagree_thresh,
        agree_thresh + _get_config_value(config_source, 'min_disagree_gap'),
    )

    # Clamp into valid confidence range.
    agree_thresh = float(min(max(agree_thresh, 0.0), 1.0))
    disagree_thresh = float(min(max(disagree_thresh, 0.0), 1.0))
    margin_thresh = float(min(max(margin_thresh, 0.0), 1.0))

    return agree_thresh, disagree_thresh, margin_thresh


def get_threshold_curriculum(iter_num, train_args):
    return get_threshold_curriculum_from_config(iter_num, train_args)


def export_threshold_config_from_args(train_args, name='active_cli'):
    return {
        'name': name,
        'use_threshold_curriculum': int(train_args.use_threshold_curriculum),
        'threshold_schedule': train_args.threshold_schedule,
        'pseudo_agree_thresh': float(train_args.pseudo_agree_thresh),
        'pseudo_disagree_thresh': float(train_args.pseudo_disagree_thresh),
        'pseudo_margin_thresh': float(train_args.pseudo_margin_thresh),
        'agree_thresh_start': float(train_args.agree_thresh_start),
        'agree_thresh_end': float(train_args.agree_thresh_end),
        'disagree_thresh_start': float(train_args.disagree_thresh_start),
        'disagree_thresh_end': float(train_args.disagree_thresh_end),
        'threshold_warmup_iters': int(train_args.threshold_warmup_iters),
        'threshold_decay_iters': int(train_args.threshold_decay_iters),
        'margin_thresh_start': float(train_args.margin_thresh_start),
        'margin_thresh_end': float(train_args.margin_thresh_end),
        'disagree_decay_power': float(train_args.disagree_decay_power),
        'min_disagree_gap': float(train_args.min_disagree_gap),
    }


def _clamp_float(value, lower=0.0, upper=1.0):
    return float(min(max(value, lower), upper))


def build_threshold_sweep_candidates(train_args):
    base = export_threshold_config_from_args(train_args, name='active_cli')
    candidates = [base]

    def add_variant(
        name,
        agree_shift=(0.0, 0.0),
        disagree_shift=(0.0, 0.0),
        margin_shift=(0.0, 0.0),
        warmup_scale=1.0,
        decay_scale=1.0,
        power_shift=0.0,
        gap_shift=0.0,
    ):
        cfg = dict(base)
        cfg['name'] = name
        cfg['agree_thresh_start'] = _clamp_float(cfg['agree_thresh_start'] + agree_shift[0])
        cfg['agree_thresh_end'] = _clamp_float(cfg['agree_thresh_end'] + agree_shift[1])
        cfg['disagree_thresh_start'] = _clamp_float(cfg['disagree_thresh_start'] + disagree_shift[0])
        cfg['disagree_thresh_end'] = _clamp_float(cfg['disagree_thresh_end'] + disagree_shift[1])
        cfg['margin_thresh_start'] = _clamp_float(cfg['margin_thresh_start'] + margin_shift[0])
        cfg['margin_thresh_end'] = _clamp_float(cfg['margin_thresh_end'] + margin_shift[1])
        cfg['threshold_warmup_iters'] = max(0, int(round(cfg['threshold_warmup_iters'] * warmup_scale)))
        cfg['threshold_decay_iters'] = max(1, int(round(cfg['threshold_decay_iters'] * decay_scale)))
        cfg['disagree_decay_power'] = max(0.1, float(cfg['disagree_decay_power'] + power_shift))
        cfg['min_disagree_gap'] = _clamp_float(cfg['min_disagree_gap'] + gap_shift, lower=0.0, upper=0.5)
        candidates.append(cfg)

    add_variant(
        'strict_slow',
        agree_shift=(0.05, 0.05),
        disagree_shift=(0.05, 0.08),
        margin_shift=(0.03, 0.02),
        warmup_scale=1.5,
        decay_scale=1.25,
        power_shift=0.4,
        gap_shift=0.03,
    )
    add_variant(
        'strict_mid',
        agree_shift=(0.05, 0.05),
        disagree_shift=(0.05, 0.08),
        margin_shift=(0.03, 0.02),
        power_shift=0.4,
        gap_shift=0.03,
    )
    add_variant(
        'strict_fast',
        agree_shift=(0.05, 0.05),
        disagree_shift=(0.05, 0.08),
        margin_shift=(0.03, 0.02),
        warmup_scale=0.5,
        decay_scale=0.75,
        power_shift=0.4,
        gap_shift=0.03,
    )
    add_variant('balanced_slow', warmup_scale=1.5, decay_scale=1.25)
    add_variant('balanced_mid')
    add_variant('balanced_fast', warmup_scale=0.5, decay_scale=0.75)
    add_variant(
        'permissive_slow',
        agree_shift=(-0.05, -0.05),
        disagree_shift=(-0.05, -0.05),
        margin_shift=(-0.03, -0.02),
        warmup_scale=1.25,
        decay_scale=1.0,
        power_shift=-0.2,
        gap_shift=-0.02,
    )
    add_variant(
        'permissive_mid',
        agree_shift=(-0.05, -0.05),
        disagree_shift=(-0.05, -0.05),
        margin_shift=(-0.03, -0.02),
        power_shift=-0.2,
        gap_shift=-0.02,
    )
    add_variant(
        'permissive_fast',
        agree_shift=(-0.05, -0.05),
        disagree_shift=(-0.05, -0.05),
        margin_shift=(-0.03, -0.02),
        warmup_scale=0.5,
        decay_scale=0.75,
        power_shift=-0.2,
        gap_shift=-0.02,
    )
    add_variant(
        'strict_disagree',
        disagree_shift=(0.05, 0.05),
        margin_shift=(0.04, 0.03),
        power_shift=0.6,
        gap_shift=0.04,
    )
    add_variant(
        'relaxed_disagree',
        disagree_shift=(-0.04, -0.04),
        margin_shift=(-0.03, -0.02),
        power_shift=-0.4,
        gap_shift=-0.03,
    )
    add_variant('wide_margin', margin_shift=(0.05, 0.03))
    add_variant('narrow_margin', margin_shift=(-0.05, -0.03))

    deduped = []
    seen = set()
    for cfg in candidates:
        key = (
            cfg['use_threshold_curriculum'],
            cfg['threshold_schedule'],
            cfg['pseudo_agree_thresh'],
            cfg['pseudo_disagree_thresh'],
            cfg['pseudo_margin_thresh'],
            cfg['agree_thresh_start'],
            cfg['agree_thresh_end'],
            cfg['disagree_thresh_start'],
            cfg['disagree_thresh_end'],
            cfg['threshold_warmup_iters'],
            cfg['threshold_decay_iters'],
            cfg['margin_thresh_start'],
            cfg['margin_thresh_end'],
            cfg['disagree_decay_power'],
            cfg['min_disagree_gap'],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


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
    """
    Build reliable soft pseudo-labels through Mean Teacher agreement/disagreement logic.

    Agreement reliable seed:
        student and teacher predict the same class
        AND both have confidence >= agree_thresh.

    Disagreement reliable seed:
        student and teacher predict different classes
        AND the stronger prediction has confidence >= disagree_thresh
        AND the confidence gap >= margin_thresh.
    """
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
    reliable_disagree = (
        diff_pred
        & (max_conf >= disagree_thresh)
        & (margin >= margin_thresh)
        & candidate_mask
    )

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
        'reliable_agree': reliable_agree.detach(),
        'reliable_disagree': reliable_disagree.detach(),
        'agreement_ratio': reliable_agree.float().mean().detach(),
        'disagreement_ratio': reliable_disagree.float().mean().detach(),
        'reliable_ratio': reliable_mask.mean().detach(),
        'pseudo_conf': pseudo_conf.detach(),
    }


def compute_oracle_selection_metrics(pseudo_info, scribble_label, gt_label, ignore_index=4, eps=1e-8):
    """
    Measure pseudo-label selection quality against full masks.

    This is for analysis only and must not be used inside the optimization target.
    """
    candidate_mask = scribble_label == ignore_index
    reliable_agree = pseudo_info['reliable_agree'] & candidate_mask
    reliable_disagree = pseudo_info['reliable_disagree'] & candidate_mask
    reliable_mask = (pseudo_info['reliable_mask'].squeeze(1) > 0.5) & candidate_mask

    pseudo_hard = torch.argmax(pseudo_info['soft_pseudo_label'], dim=1)
    correct_mask = pseudo_hard == gt_label

    candidate_count = candidate_mask.float().sum()
    selected_count = reliable_mask.float().sum()
    agree_count = reliable_agree.float().sum()
    disagree_count = reliable_disagree.float().sum()
    agree_correct = (correct_mask & reliable_agree).float().sum()
    disagree_correct = (correct_mask & reliable_disagree).float().sum()

    def masked_accuracy(mask):
        if mask.float().sum() < 1:
            return gt_label.new_tensor(0.0, dtype=torch.float32)
        return correct_mask[mask].float().mean()

    selected_correct = (correct_mask & reliable_mask).float().sum()

    return {
        'candidate_count': candidate_count,
        'selected_count': selected_count,
        'selected_correct': selected_correct,
        'agree_count': agree_count,
        'agree_correct': agree_correct,
        'disagree_count': disagree_count,
        'disagree_correct': disagree_correct,
        'coverage': selected_count / (candidate_count + eps),
        'selected_accuracy': masked_accuracy(reliable_mask),
        'selected_correct_ratio': selected_correct / (candidate_count + eps),
        'agree_coverage': agree_count / (candidate_count + eps),
        'agree_accuracy': masked_accuracy(reliable_agree),
        'disagree_coverage': disagree_count / (candidate_count + eps),
        'disagree_accuracy': masked_accuracy(reliable_disagree),
    }


def format_threshold_cli_args(config):
    return (
        '--agree_thresh_start {agree_thresh_start:.2f} '
        '--agree_thresh_end {agree_thresh_end:.2f} '
        '--disagree_thresh_start {disagree_thresh_start:.2f} '
        '--disagree_thresh_end {disagree_thresh_end:.2f} '
        '--threshold_warmup_iters {threshold_warmup_iters:d} '
        '--threshold_decay_iters {threshold_decay_iters:d} '
        '--margin_thresh_start {margin_thresh_start:.2f} '
        '--margin_thresh_end {margin_thresh_end:.2f} '
        '--disagree_decay_power {disagree_decay_power:.2f} '
        '--min_disagree_gap {min_disagree_gap:.2f}'
    ).format(**config)


class ThresholdSweepTracker:
    def __init__(self, train_args, snapshot_path):
        self.train_args = train_args
        self.snapshot_path = snapshot_path
        self.candidates = build_threshold_sweep_candidates(train_args)
        self.topk = max(1, int(train_args.threshold_sweep_topk))
        self.eval_interval = max(1, int(train_args.threshold_sweep_interval))
        self.report_interval = max(1, int(train_args.threshold_sweep_report_interval))
        self.start_iter = max(0, int(train_args.threshold_sweep_start_iter))
        if train_args.threshold_sweep_late_start_iter >= 0:
            self.late_start_iter = int(train_args.threshold_sweep_late_start_iter)
        else:
            self.late_start_iter = int(0.6 * train_args.max_iterations)
        self.best_active_val = None
        self.summary_json_path = os.path.join(snapshot_path, 'threshold_sweep_summary.json')
        self.summary_csv_path = os.path.join(snapshot_path, 'threshold_sweep_summary.csv')
        self.best_txt_path = os.path.join(snapshot_path, 'threshold_sweep_best.txt')
        self.candidates_json_path = os.path.join(snapshot_path, 'threshold_sweep_candidates.json')
        with open(self.candidates_json_path, 'w') as f:
            json.dump(self.candidates, f, indent=2)
        self.stats = {}
        for cfg in self.candidates:
            self.stats[cfg['name']] = {
                'config': cfg,
                'global': self._new_counter(),
                'late': self._new_counter(),
            }

    @staticmethod
    def _new_counter():
        return {
            'updates': 0.0,
            'candidate_count': 0.0,
            'selected_count': 0.0,
            'selected_correct': 0.0,
            'agree_count': 0.0,
            'agree_correct': 0.0,
            'disagree_count': 0.0,
            'disagree_correct': 0.0,
            'sum_agree_thresh': 0.0,
            'sum_disagree_thresh': 0.0,
            'sum_margin_thresh': 0.0,
        }

    @staticmethod
    def _safe_div(num, den):
        if den <= 0:
            return 0.0
        return float(num) / float(den)

    def _accumulate_phase(self, counter, oracle_metrics, agree_thresh, disagree_thresh, margin_thresh):
        counter['updates'] += 1.0
        counter['candidate_count'] += float(oracle_metrics['candidate_count'].item())
        counter['selected_count'] += float(oracle_metrics['selected_count'].item())
        counter['selected_correct'] += float(oracle_metrics['selected_correct'].item())
        counter['agree_count'] += float(oracle_metrics['agree_count'].item())
        counter['agree_correct'] += float(oracle_metrics['agree_correct'].item())
        counter['disagree_count'] += float(oracle_metrics['disagree_count'].item())
        counter['disagree_correct'] += float(oracle_metrics['disagree_correct'].item())
        counter['sum_agree_thresh'] += float(agree_thresh)
        counter['sum_disagree_thresh'] += float(disagree_thresh)
        counter['sum_margin_thresh'] += float(margin_thresh)

    def _phase_summary(self, counter):
        updates = counter['updates']
        return {
            'updates': int(updates),
            'coverage': self._safe_div(counter['selected_count'], counter['candidate_count']),
            'selected_accuracy': self._safe_div(counter['selected_correct'], counter['selected_count']),
            'selected_correct_ratio': self._safe_div(counter['selected_correct'], counter['candidate_count']),
            'agree_coverage': self._safe_div(counter['agree_count'], counter['candidate_count']),
            'agree_accuracy': self._safe_div(counter['agree_correct'], counter['agree_count']),
            'disagree_coverage': self._safe_div(counter['disagree_count'], counter['candidate_count']),
            'disagree_accuracy': self._safe_div(counter['disagree_correct'], counter['disagree_count']),
            'avg_agree_thresh': self._safe_div(counter['sum_agree_thresh'], updates),
            'avg_disagree_thresh': self._safe_div(counter['sum_disagree_thresh'], updates),
            'avg_margin_thresh': self._safe_div(counter['sum_margin_thresh'], updates),
            'candidate_pixels': int(counter['candidate_count']),
            'selected_pixels': int(counter['selected_count']),
        }

    def _score_row(self, row):
        global_ratio = row['global_selected_correct_ratio']
        late_ratio = row['late_selected_correct_ratio']
        late_coverage = row['late_coverage']
        if row['late_updates'] < 1:
            late_ratio = global_ratio
            late_coverage = row['global_coverage']
        score = 0.4 * global_ratio + 0.6 * late_ratio
        if late_coverage < 0.05:
            score *= max(late_coverage / 0.05, 0.2)
        return score

    def update(self, iter_num, student_prob, teacher_prob, scribble_label, gt_label, ignore_index, pseudo_mask_mode):
        if iter_num < self.start_iter:
            return
        if iter_num % self.eval_interval != 0:
            return

        for cfg in self.candidates:
            agree_thresh, disagree_thresh, margin_thresh = get_threshold_curriculum_from_config(
                iter_num=iter_num,
                config_source=cfg,
            )
            pseudo_info = build_mt_confidence_pseudo_label(
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                label=scribble_label,
                agree_thresh=agree_thresh,
                disagree_thresh=disagree_thresh,
                margin_thresh=margin_thresh,
                ignore_index=ignore_index,
                pseudo_mask_mode=pseudo_mask_mode,
            )
            oracle_metrics = compute_oracle_selection_metrics(
                pseudo_info=pseudo_info,
                scribble_label=scribble_label,
                gt_label=gt_label,
                ignore_index=ignore_index,
            )
            stat = self.stats[cfg['name']]
            self._accumulate_phase(stat['global'], oracle_metrics, agree_thresh, disagree_thresh, margin_thresh)
            if iter_num >= self.late_start_iter:
                self._accumulate_phase(stat['late'], oracle_metrics, agree_thresh, disagree_thresh, margin_thresh)

    def set_best_active_val(self, best_val):
        self.best_active_val = float(best_val)

    def build_rows(self):
        rows = []
        for cfg in self.candidates:
            stat = self.stats[cfg['name']]
            global_summary = self._phase_summary(stat['global'])
            late_summary = self._phase_summary(stat['late'])
            row = {
                'name': cfg['name'],
                'score': 0.0,
                'best_active_val_dice': self.best_active_val,
                'cli_args': format_threshold_cli_args(cfg),
                'global_updates': global_summary['updates'],
                'global_coverage': global_summary['coverage'],
                'global_selected_accuracy': global_summary['selected_accuracy'],
                'global_selected_correct_ratio': global_summary['selected_correct_ratio'],
                'global_agree_coverage': global_summary['agree_coverage'],
                'global_agree_accuracy': global_summary['agree_accuracy'],
                'global_disagree_coverage': global_summary['disagree_coverage'],
                'global_disagree_accuracy': global_summary['disagree_accuracy'],
                'global_avg_agree_thresh': global_summary['avg_agree_thresh'],
                'global_avg_disagree_thresh': global_summary['avg_disagree_thresh'],
                'global_avg_margin_thresh': global_summary['avg_margin_thresh'],
                'late_updates': late_summary['updates'],
                'late_coverage': late_summary['coverage'],
                'late_selected_accuracy': late_summary['selected_accuracy'],
                'late_selected_correct_ratio': late_summary['selected_correct_ratio'],
                'late_agree_coverage': late_summary['agree_coverage'],
                'late_agree_accuracy': late_summary['agree_accuracy'],
                'late_disagree_coverage': late_summary['disagree_coverage'],
                'late_disagree_accuracy': late_summary['disagree_accuracy'],
                'late_avg_agree_thresh': late_summary['avg_agree_thresh'],
                'late_avg_disagree_thresh': late_summary['avg_disagree_thresh'],
                'late_avg_margin_thresh': late_summary['avg_margin_thresh'],
                'agree_thresh_start': cfg['agree_thresh_start'],
                'agree_thresh_end': cfg['agree_thresh_end'],
                'disagree_thresh_start': cfg['disagree_thresh_start'],
                'disagree_thresh_end': cfg['disagree_thresh_end'],
                'threshold_warmup_iters': cfg['threshold_warmup_iters'],
                'threshold_decay_iters': cfg['threshold_decay_iters'],
                'margin_thresh_start': cfg['margin_thresh_start'],
                'margin_thresh_end': cfg['margin_thresh_end'],
                'disagree_decay_power': cfg['disagree_decay_power'],
                'min_disagree_gap': cfg['min_disagree_gap'],
            }
            row['score'] = self._score_row(row)
            rows.append(row)
        rows.sort(
            key=lambda x: (
                x['score'],
                x['late_selected_correct_ratio'],
                x['global_selected_correct_ratio'],
                x['late_selected_accuracy'],
            ),
            reverse=True,
        )
        return rows

    def write_summary(self, current_iter):
        rows = self.build_rows()
        payload = {
            'iter': int(current_iter),
            'late_start_iter': int(self.late_start_iter),
            'best_active_val_dice': self.best_active_val,
            'topk': rows[:self.topk],
            'all_candidates': rows,
        }
        with open(self.summary_json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(self.summary_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with open(self.best_txt_path, 'w') as f:
                f.write('iteration: {}\n'.format(int(current_iter)))
                f.write('late_start_iter: {}\n'.format(int(self.late_start_iter)))
                if self.best_active_val is not None:
                    f.write('best_active_val_dice: {:.6f}\n'.format(float(self.best_active_val)))
                f.write('score = 0.4 * global_selected_correct_ratio + 0.6 * late_selected_correct_ratio\n')
                f.write('late coverage under 0.05 receives a penalty\n')
                f.write('\nTop {} threshold candidates:\n'.format(self.topk))
                for rank, row in enumerate(rows[:self.topk], start=1):
                    f.write(
                        '{}. {} score={:.6f} global_hit={:.6f} late_hit={:.6f} '
                        'global_acc={:.6f} late_acc={:.6f} global_cov={:.6f} late_cov={:.6f}\n'.format(
                            rank,
                            row['name'],
                            row['score'],
                            row['global_selected_correct_ratio'],
                            row['late_selected_correct_ratio'],
                            row['global_selected_accuracy'],
                            row['late_selected_accuracy'],
                            row['global_coverage'],
                            row['late_coverage'],
                        )
                    )
                    f.write('   {}\n'.format(row['cli_args']))
        return rows

def create_model(ema=False, num_classes=4):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def validate(model, valloader, db_val, num_classes, writer, iter_num):
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

    # Usually background is excluded, so num_classes - 1 foreground classes are logged.
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

    model.train()
    return performance, mean_hd95


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    model = create_model(ema=False, num_classes=num_classes)
    model_ema = create_model(ema=True, num_classes=num_classes)
    need_full_label = bool(train_args.oracle_metric_logging) or bool(train_args.threshold_sweep_logging)

    db_train = ACDCDataSets(
        base_dir=train_args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        fold=train_args.fold,
        sup_type=train_args.sup_type,
        return_full_label=need_full_label,
    )
    db_val = ACDCDataSets(
        base_dir=train_args.root_path,
        fold=train_args.fold,
        split='val',
    )

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

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    model.train()
    model_ema.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    ema_optimizer = WeightEMA(model, model_ema, 0.99)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    threshold_sweep_tracker = None
    if train_args.threshold_sweep_logging:
        threshold_sweep_tracker = ThresholdSweepTracker(train_args, snapshot_path)
        logging.info(
            'threshold sweep enabled with %d candidates, eval_interval=%d, report_interval=%d',
            len(threshold_sweep_tracker.candidates),
            threshold_sweep_tracker.eval_interval,
            threshold_sweep_tracker.report_interval,
        )

    for epoch_num in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            gt_label_batch = None
            if need_full_label and 'gt_label' in sampled_batch:
                gt_label_batch = sampled_batch['gt_label'].cuda()

            # -------------------------
            # 1. EMA Teacher forward
            # -------------------------
            with torch.no_grad():
                ema_output = unpack_model_output(model_ema(volume_batch))
                teacher_prob = torch.softmax(ema_output, dim=1)

            # -------------------------
            # 2. Student forward
            # -------------------------
            outputs = unpack_model_output(model(volume_batch))
            student_prob = torch.softmax(outputs, dim=1)

            # -------------------------
            # 3. Partial CE on scribble labels
            # -------------------------
            loss_pce = ce_loss(outputs, label_batch.long())

            # -------------------------
            # 4. Dynamic confidence thresholds
            # -------------------------
            cur_agree_thresh, cur_disagree_thresh, cur_margin_thresh = get_threshold_curriculum(
                iter_num=iter_num,
                train_args=train_args,
            )

            # -------------------------
            # 5. Reliable pseudo-label selection
            # -------------------------
            pseudo_info = build_mt_confidence_pseudo_label(
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                label=label_batch,
                agree_thresh=cur_agree_thresh,
                disagree_thresh=cur_disagree_thresh,
                margin_thresh=cur_margin_thresh,
                ignore_index=num_classes,
                pseudo_mask_mode=train_args.pseudo_mask_mode,
            )

            # -------------------------
            # 6. Pseudo-label loss
            # -------------------------
            loss_pseudo = masked_soft_ce_loss(
                logits=outputs,
                target_prob=pseudo_info['soft_pseudo_label'],
                mask=pseudo_info['reliable_mask'],
            )

            # -------------------------
            # 7. Pseudo loss ramp-up
            # -------------------------
            pseudo_weight = (
                get_current_consistency_weight(iter_num // len(trainloader), train_args)
                * train_args.pseudo_loss_weight
            )

            loss = loss_pce + pseudo_weight * loss_pseudo

            # -------------------------
            # 8. Student optimization
            # -------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------------------------
            # 9. EMA teacher update
            # -------------------------
            ema_optimizer.step()

            # -------------------------
            # 10. Poly LR decay
            # -------------------------
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # -------------------------
            # 11. TensorBoard logging
            # -------------------------
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_pce', loss_pce.item(), iter_num)
            writer.add_scalar('info/loss_pseudo', loss_pseudo.item(), iter_num)
            writer.add_scalar('info/pseudo_weight', pseudo_weight, iter_num)

            writer.add_scalar('threshold/agree', cur_agree_thresh, iter_num)
            writer.add_scalar('threshold/disagree', cur_disagree_thresh, iter_num)
            writer.add_scalar('threshold/margin', cur_margin_thresh, iter_num)

            writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)

            oracle_metrics = None
            if gt_label_batch is not None:
                oracle_metrics = compute_oracle_selection_metrics(
                    pseudo_info=pseudo_info,
                    scribble_label=label_batch,
                    gt_label=gt_label_batch.long(),
                    ignore_index=num_classes,
                )
                writer.add_scalar('oracle/coverage', oracle_metrics['coverage'].item(), iter_num)
                writer.add_scalar(
                    'oracle/selected_accuracy',
                    oracle_metrics['selected_accuracy'].item(),
                    iter_num,
                )
                writer.add_scalar(
                    'oracle/selected_correct_ratio',
                    oracle_metrics['selected_correct_ratio'].item(),
                    iter_num,
                )
                writer.add_scalar('oracle/agree_coverage', oracle_metrics['agree_coverage'].item(), iter_num)
                writer.add_scalar('oracle/agree_accuracy', oracle_metrics['agree_accuracy'].item(), iter_num)
                writer.add_scalar(
                    'oracle/disagree_coverage',
                    oracle_metrics['disagree_coverage'].item(),
                    iter_num,
                )
                writer.add_scalar(
                    'oracle/disagree_accuracy',
                    oracle_metrics['disagree_accuracy'].item(),
                    iter_num,
                )
                if threshold_sweep_tracker is not None:
                    threshold_sweep_tracker.update(
                        iter_num=iter_num,
                        student_prob=student_prob,
                        teacher_prob=teacher_prob,
                        scribble_label=label_batch,
                        gt_label=gt_label_batch.long(),
                        ignore_index=num_classes,
                        pseudo_mask_mode=train_args.pseudo_mask_mode,
                    )
                    if iter_num % threshold_sweep_tracker.report_interval == 0:
                        sweep_rows = threshold_sweep_tracker.write_summary(iter_num)
                        if sweep_rows:
                            top_row = sweep_rows[0]
                            logging.info(
                                'threshold sweep @ %d : best=%s score=%.6f global_hit=%.6f '
                                'late_hit=%.6f global_acc=%.6f late_acc=%.6f',
                                iter_num,
                                top_row['name'],
                                top_row['score'],
                                top_row['global_selected_correct_ratio'],
                                top_row['late_selected_correct_ratio'],
                                top_row['global_selected_accuracy'],
                                top_row['late_selected_accuracy'],
                            )

            # -------------------------
            # 12. Console logging
            # -------------------------
            if iter_num % 200 == 0:
                log_msg = (
                    'iteration %d : loss=%f, loss_pce=%f, loss_pseudo=%f, pseudo_weight=%f, '
                    'agree_th=%.4f, disagree_th=%.4f, margin_th=%.4f, '
                    'reliable=%f, agree=%f, disagree=%f, pseudo_conf=%f'
                )
                log_values = [
                    iter_num,
                    loss.item(),
                    loss_pce.item(),
                    loss_pseudo.item(),
                    pseudo_weight,
                    cur_agree_thresh,
                    cur_disagree_thresh,
                    cur_margin_thresh,
                    pseudo_info['reliable_ratio'].item(),
                    pseudo_info['agreement_ratio'].item(),
                    pseudo_info['disagreement_ratio'].item(),
                    pseudo_info['pseudo_conf'].mean().item(),
                ]
                if oracle_metrics is not None:
                    log_msg += (
                        ', oracle_cov=%f, oracle_acc=%f, oracle_hit=%f, '
                        'oracle_agree_acc=%f, oracle_disagree_acc=%f'
                    )
                    log_values.extend([
                        oracle_metrics['coverage'].item(),
                        oracle_metrics['selected_accuracy'].item(),
                        oracle_metrics['selected_correct_ratio'].item(),
                        oracle_metrics['agree_accuracy'].item(),
                        oracle_metrics['disagree_accuracy'].item(),
                    ])
                logging.info(log_msg, *log_values)

            # -------------------------
            # 13. Validation and best checkpoint
            # -------------------------
            if iter_num > 1 and iter_num % 400 == 0:
                performance, mean_hd95 = validate(
                    model=model,
                    valloader=valloader,
                    db_val=db_val,
                    num_classes=num_classes,
                    writer=writer,
                    iter_num=iter_num,
                )

                if performance > best_performance:
                    best_performance = performance
                    if threshold_sweep_tracker is not None:
                        threshold_sweep_tracker.set_best_active_val(best_performance)
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(train_args.model),
                    )
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info('save best model to %s', save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f',
                    iter_num,
                    performance,
                    mean_hd95,
                )

            # -------------------------
            # 14. Regular checkpoint
            # -------------------------
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to %s', save_mode_path)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    if threshold_sweep_tracker is not None:
        threshold_sweep_tracker.set_best_active_val(best_performance)
        threshold_sweep_tracker.write_summary(iter_num)
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

    result = train(args, snapshot_path)
    logging.info(result)
