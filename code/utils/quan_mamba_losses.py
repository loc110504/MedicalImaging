import torch
import torch.nn.functional as F


def partial_ce_loss(logits, label, ignore_index):
    return F.cross_entropy(logits, label.long(), ignore_index=ignore_index)


def partial_prob_ce(prob, label, ignore_index, eps=1e-8):
    valid = label != ignore_index
    if valid.sum() == 0:
        return prob.new_tensor(0.0)
    prob = torch.clamp(prob, eps, 1.0)
    selected = prob.permute(0, 2, 3, 1)[valid]
    target = label[valid].long()
    return F.nll_loss(torch.log(selected), target)


def masked_hard_ce(logits, pseudo, mask, eps=1e-8):
    ce = F.cross_entropy(logits, pseudo.long(), reduction="none")
    mask = mask.float()
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    return (ce * mask).sum() / (mask.sum() + eps)


def linear_threshold(iter_num, warmup, max_iterations, start, end):
    if iter_num < warmup:
        return float(start)
    progress = min(1.0, (iter_num - warmup) / float(max(1, max_iterations - warmup)))
    return float(start + progress * (end - start))


def get_current_thresholds(iter_num, warmup, max_iterations, tau_u_start, tau_u_end, tau_m_start, tau_m_end, tau_q_start, tau_q_end):
    return (
        linear_threshold(iter_num, warmup, max_iterations, tau_u_start, tau_u_end),
        linear_threshold(iter_num, warmup, max_iterations, tau_m_start, tau_m_end),
        linear_threshold(iter_num, warmup, max_iterations, tau_q_start, tau_q_end),
    )
