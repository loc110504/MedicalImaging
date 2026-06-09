# 04 - Losses and Pseudo-Label Specification

Files:

```text
code/utils/quan_mamba_losses.py
code/utils/quan_mamba_pseudo.py
```

## Label Convention
For ACDC with 4 classes:

```text
classes:      0, 1, 2, 3
unknown:      4
ignore_index: num_classes
```

`label_batch` shape: `[B, H, W]`.

## Partial Cross Entropy `L_pCE`
Use logits, not probabilities.

```python
def partial_ce_loss(logits, label, ignore_index):
    return F.cross_entropy(logits, label.long(), ignore_index=ignore_index)
```

For QuanMambaScrib:

```python
loss_pce_u = partial_ce_loss(logits_u, label, ignore_index)
loss_pce_m = partial_ce_loss(logits_m, label, ignore_index)
loss_pce = loss_pce_u + loss_pce_m
```

## Quantum Prototype Supervision `L_q`
`Q_u` and `Q_m` are probabilities. Use negative log likelihood on scribble pixels.

```python
def partial_prob_ce(prob, label, ignore_index, eps=1e-8):
    valid = label != ignore_index
    if valid.sum() == 0:
        return prob.new_tensor(0.0)
    prob = torch.clamp(prob, eps, 1.0)
    selected = prob.permute(0,2,3,1)[valid]       # [N_valid, K]
    target = label[valid].long()                  # [N_valid]
    return F.nll_loss(torch.log(selected), target)
```

Then:

```python
loss_q_u = partial_prob_ce(Q_u, label, ignore_index)
loss_q_m = partial_prob_ce(Q_m, label, ignore_index)
loss_q = loss_q_u + loss_q_m
```

## Quantum-Guided Cross Pseudo Supervision `L_qcps`

### Function API

```python
def build_quantum_guided_reliable_masks(
    prob_u,
    prob_m,
    Q_u,
    Q_m,
    label,
    ignore_index,
    tau_u,
    tau_m,
    tau_q,
):
    ...
```

Return:

```python
{
    "pseudo_u": pseudo_u,            # [B,H,W]
    "pseudo_m": pseudo_m,            # [B,H,W]
    "R_u": R_u,                      # [B,H,W] float
    "R_m": R_m,                      # [B,H,W] float
    "conf_u": conf_u,                # [B,H,W]
    "conf_m": conf_m,                # [B,H,W]
    "qconf_u": qconf_u,              # [B,H,W]
    "qconf_m": qconf_m,              # [B,H,W]
    "agreement_u": agreement_u,      # bool [B,H,W]
    "agreement_m": agreement_m,      # bool [B,H,W]
}
```

### Exact Logic

```python
conf_u, pseudo_u = torch.max(prob_u.detach(), dim=1)
conf_m, pseudo_m = torch.max(prob_m.detach(), dim=1)
qconf_u, qlabel_u = torch.max(Q_u.detach(), dim=1)
qconf_m, qlabel_m = torch.max(Q_m.detach(), dim=1)

unknown = label == ignore_index
agreement_u = pseudo_u == qlabel_u
agreement_m = pseudo_m == qlabel_m

R_u = agreement_u & (conf_u > tau_u) & (qconf_u > tau_q) & unknown
R_m = agreement_m & (conf_m > tau_m) & (qconf_m > tau_q) & unknown
```

`R_u` means U-Net pseudo-label is reliable and will supervise Mamba-UNet.
`R_m` means Mamba pseudo-label is reliable and will supervise U-Net.

### Loss

```python
def masked_hard_ce(logits, pseudo, mask, eps=1e-8):
    ce = F.cross_entropy(logits, pseudo.long(), reduction="none")
    mask = mask.float()
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    return (ce * mask).sum() / (mask.sum() + eps)
```

Then:

```python
loss_u_to_m = masked_hard_ce(logits_m, pseudo_u, R_u)
loss_m_to_u = masked_hard_ce(logits_u, pseudo_m, R_m)
loss_qcps = loss_u_to_m + loss_m_to_u
```

## Threshold Schedule
Add args:

```python
--warmup_iterations 5000
--tau_u_start 0.95
--tau_u_end 0.75
--tau_m_start 0.95
--tau_m_end 0.75
--tau_q_start 0.90
--tau_q_end 0.70
--threshold_schedule linear
```

Function:

```python
def linear_threshold(iter_num, warmup, max_iterations, start, end):
    if iter_num < warmup:
        return start
    progress = min(1.0, (iter_num - warmup) / max(1, max_iterations - warmup))
    return start + progress * (end - start)
```

Default strict-to-relaxed schedule:

```text
before warmup: no Q-CPS
at warmup:    strict thresholds
late train:   relaxed thresholds
```

## Optional Disabled Loss Hooks
The finalized PDF objective does not include these losses. Implement only if easy, and keep disabled by default:

```python
--lambda_con 0.0
--lambda_aff 0.0
```

If implemented:

- `L_con`: consistency between `prob_u` and `prob_m` on `R_u & R_m`.
- `L_aff`: KL from `Q` to branch probability.

Do not enable them in default training commands unless the paper section is updated.

## Logging Statistics
For pseudo masks:

```python
reliable_ratio_u = R_u.float().mean()
reliable_ratio_m = R_m.float().mean()
agreement_ratio_u = agreement_u.float().mean()
agreement_ratio_m = agreement_m.float().mean()
conf_u_mean = conf_u[R_u].mean() if R_u.any() else nan
conf_m_mean = conf_m[R_m].mean() if R_m.any() else nan
qconf_u_mean = qconf_u[R_u].mean() if R_u.any() else nan
qconf_m_mean = qconf_m[R_m].mean() if R_m.any() else nan
```

## Dense-Label Debug Metrics
Reuse existing helper logic if `sampled_batch['gt_label']` is available. Compute pseudo-label quality separately for:

1. U-Net reliable pseudo-labels `pseudo_u` on `R_u`.
2. Mamba reliable pseudo-labels `pseudo_m` on `R_m`.
3. Ensemble hard prediction.

Report:

```text
pseudo_u_mean_dice
pseudo_m_mean_dice
pseudo_u_selected_ratio
pseudo_m_selected_ratio
pseudo_u_boundary_loss
pseudo_m_boundary_loss
```
