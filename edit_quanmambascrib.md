# SPEC: Replace Current Q-CPS with U-Net/Mamba Agreement-Disagreement Pseudo-Label + Quantum Verification

## 1. Target Change

Modify the current `train.py` pipeline for `QuanMambaScrib` so that pseudo-labels are generated jointly from the two branches:

```text
U-Net prediction P^u + Mamba-UNet prediction P^m
        ↓
Agreement / Disagreement pseudo-label construction
        ↓
Quantum verification using Q^u and Q^m
        ↓
Reliable U-view pseudo-label supervises Mamba
Reliable M-view pseudo-label supervises U-Net
```

Important: **Do not add Mean Teacher. Do not create EMA teacher models.**

The model should remain a single `QuanMambaScrib` model with two branches:

```python
logits_u, prob_u, Q_u
logits_m, prob_m, Q_m
```

The new pseudo-label construction must use only:

```python
prob_u
prob_m
Q_u
Q_m
label_batch
```

---

## 2. Current Code Behavior to Replace

The current code uses:

```python
pseudo_info = build_quantum_guided_reliable_masks(
    prob_u=prob_u,
    prob_m=prob_m,
    Q_u=q_u,
    Q_m=q_m,
    label=label_batch,
    ignore_index=num_classes,
    tau_u=tau_u,
    tau_m=tau_m,
    tau_q=tau_q,
)
loss_u_to_m = masked_hard_ce(logits_m, pseudo_info["pseudo_u"], pseudo_info["R_u"])
loss_m_to_u = masked_hard_ce(logits_u, pseudo_info["pseudo_m"], pseudo_info["R_m"])
loss_qcps = loss_u_to_m + loss_m_to_u
```

Replace this with a new builder:

```python
pseudo_info = build_dual_branch_quantum_pseudo_label(
    prob_u=prob_u,
    prob_m=prob_m,
    Q_u=q_u,
    Q_m=q_m,
    label=label_batch,
    ignore_index=num_classes,
    agree_thresh=tau_agree,
    disagree_thresh=tau_disagree,
    margin_thresh=train_args.margin_thresh,
    tau_q=tau_q,
    pseudo_mask_mode=train_args.pseudo_mask_mode,
)
```

Then use soft pseudo-label loss:

```python
loss_u_to_m = masked_soft_ce_loss(
    logits=logits_m,
    soft_target=pseudo_info["soft_pseudo_u"],
    mask=pseudo_info["R_u"],
)

loss_m_to_u = masked_soft_ce_loss(
    logits=logits_u,
    soft_target=pseudo_info["soft_pseudo_m"],
    mask=pseudo_info["R_m"],
)

loss_qcps = loss_u_to_m + loss_m_to_u
```

---

## 3. Core Design

The new module should build pseudo-labels from **U-Net vs Mamba-UNet predictions**, not from student-teacher.

Given:

```python
prob_u: [B, K, H, W]
prob_m: [B, K, H, W]
Q_u:    [B, K, H, W]
Q_m:    [B, K, H, W]
label:  [B, H, W]
```

Compute branch predictions:

```python
conf_u, pred_u = torch.max(prob_u, dim=1)
conf_m, pred_m = torch.max(prob_m, dim=1)
```

Then separate pixels into:

```text
Agreement pixels:
    pred_u == pred_m

Disagreement pixels:
    pred_u != pred_m
```

Pseudo-label generation rules:

```text
Agreement:
    If U-Net and Mamba predict the same class
    AND both are confident enough,
    then use mean soft pseudo-label:
        soft_pseudo = 0.5 * (prob_u + prob_m)

Disagreement:
    If U-Net and Mamba predict different classes
    BUT one branch is much more confident,
    then use the more confident branch's probability distribution.
```

Then verify with quantum predictions:

```text
For U-view pseudo-label:
    soft_pseudo_u must agree with Q_u
    AND Q_u confidence must be high enough.

For M-view pseudo-label:
    soft_pseudo_m must agree with Q_m
    AND Q_m confidence must be high enough.
```

Finally:

```text
soft_pseudo_u + R_u supervise Mamba branch
soft_pseudo_m + R_m supervise U-Net branch
```

---

## 4. New CLI Arguments

Add these arguments to `argparse`:

```python
parser.add_argument("--agree_thresh_start", type=float, default=0.80)
parser.add_argument("--agree_thresh_end", type=float, default=0.70)

parser.add_argument("--disagree_thresh_start", type=float, default=0.90)
parser.add_argument("--disagree_thresh_end", type=float, default=0.80)

parser.add_argument("--margin_thresh", type=float, default=0.10)

parser.add_argument(
    "--pseudo_mask_mode",
    type=str,
    default="unlabeled",
    choices=["unlabeled", "all"],
)
```

Keep existing quantum threshold arguments:

```python
--tau_q_start
--tau_q_end
```

The old `tau_u_start`, `tau_u_end`, `tau_m_start`, `tau_m_end` can remain for backward compatibility, but the new pseudo-label builder should preferably use:

```text
agree_thresh
disagree_thresh
margin_thresh
tau_q
```

Recommended initial values:

```text
agree_thresh:    0.80 → 0.70
disagree_thresh: 0.90 → 0.80
margin_thresh:   0.10 fixed
tau_q:           0.55 → 0.40
```

---

## 5. Threshold Scheduling

Add a helper function:

```python
def linear_schedule(iter_num, warmup, max_iterations, start, end):
    if iter_num < warmup:
        return start
    progress = float(iter_num - warmup) / float(max_iterations - warmup + 1e-8)
    progress = min(max(progress, 0.0), 1.0)
    return start + progress * (end - start)
```

Inside training loop:

```python
tau_agree = linear_schedule(
    iter_num,
    train_args.warmup_iterations,
    max_iterations,
    train_args.agree_thresh_start,
    train_args.agree_thresh_end,
)

tau_disagree = linear_schedule(
    iter_num,
    train_args.warmup_iterations,
    max_iterations,
    train_args.disagree_thresh_start,
    train_args.disagree_thresh_end,
)

_, _, tau_q = get_current_thresholds(
    iter_num=iter_num,
    warmup=train_args.warmup_iterations,
    max_iterations=max_iterations,
    tau_u_start=train_args.tau_u_start,
    tau_u_end=train_args.tau_u_end,
    tau_m_start=train_args.tau_m_start,
    tau_m_end=train_args.tau_m_end,
    tau_q_start=train_args.tau_q_start,
    tau_q_end=train_args.tau_q_end,
)
```

Only `tau_q` is needed from the old function.

---

## 6. New Function: `masked_soft_ce_loss`

Create this function either in `utils/quan_mamba_losses.py` or inside `train.py`.

```python
def masked_soft_ce_loss(logits, soft_target, mask, eps=1e-8):
    """
    Args:
        logits:      [B, K, H, W]
        soft_target: [B, K, H, W]
        mask:        [B, H, W] or [B, 1, H, W]

    Returns:
        scalar loss
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask = mask.float()

    log_prob = torch.log_softmax(logits, dim=1)
    loss_map = -torch.sum(soft_target.detach() * log_prob, dim=1, keepdim=True)
    loss = torch.sum(loss_map * mask) / (torch.sum(mask) + eps)
    return loss
```

This replaces hard-label CPS loss for the new pseudo-label module.

---

## 7. New Function: `build_dual_branch_quantum_pseudo_label`

Create this function in `utils/quan_mamba_pseudo.py`.

```python
def build_dual_branch_quantum_pseudo_label(
    prob_u,
    prob_m,
    Q_u,
    Q_m,
    label,
    ignore_index,
    agree_thresh=0.70,
    disagree_thresh=0.80,
    margin_thresh=0.10,
    tau_q=0.50,
    pseudo_mask_mode="unlabeled",
    eps=1e-8,
):
    """
    Build reliable soft pseudo-labels using U-Net/Mamba agreement-disagreement
    and quantum prototype verification.

    Agreement reliable seed:
        U-Net and Mamba predict the same class
        AND both have confidence >= agree_thresh.

    Disagreement reliable seed:
        U-Net and Mamba predict different classes
        AND the stronger branch has confidence >= disagree_thresh
        AND confidence margin >= margin_thresh.

    Quantum verification:
        The selected soft pseudo-label must agree with branch-specific quantum
        prototype prediction Q.
    """
```

### 7.1 Detach all pseudo-label sources

```python
prob_u = prob_u.detach()
prob_m = prob_m.detach()
Q_u = Q_u.detach()
Q_m = Q_m.detach()
```

If label shape is `[B, 1, H, W]`:

```python
if label.dim() == 4:
    label = label.squeeze(1)
```

### 7.2 Candidate mask

```python
if pseudo_mask_mode == "unlabeled":
    candidate_mask = label == ignore_index
elif pseudo_mask_mode == "all":
    candidate_mask = torch.ones_like(label, dtype=torch.bool)
else:
    raise ValueError("Unsupported pseudo_mask_mode: {}".format(pseudo_mask_mode))
```

For scribble-supervised training, default should be:

```python
pseudo_mask_mode = "unlabeled"
```

### 7.3 Compute U-Net/Mamba predictions

```python
conf_u, pred_u = torch.max(prob_u, dim=1)
conf_m, pred_m = torch.max(prob_m, dim=1)

same_pred = pred_u == pred_m
diff_pred = pred_u != pred_m

min_conf = torch.minimum(conf_u, conf_m)
max_conf = torch.maximum(conf_u, conf_m)
margin = torch.abs(conf_u - conf_m)
```

### 7.4 Agreement reliability

```python
reliable_agree = (
    same_pred
    & (min_conf >= agree_thresh)
    & candidate_mask
)
```

### 7.5 Disagreement reliability

```python
reliable_disagree = (
    diff_pred
    & (max_conf >= disagree_thresh)
    & (margin >= margin_thresh)
    & candidate_mask
)
```

### 7.6 Build shared soft pseudo-label

For agreement pixels, use averaged probability:

```python
mean_pseudo = 0.5 * (prob_u + prob_m)
```

For disagreement pixels, use the more confident branch:

```python
choose_u = (conf_u > conf_m).unsqueeze(1)
high_conf_pseudo = torch.where(choose_u, prob_u, prob_m)
```

Construct soft pseudo-label:

```python
soft_pseudo = torch.where(
    reliable_disagree.unsqueeze(1),
    high_conf_pseudo,
    mean_pseudo,
)

soft_pseudo = soft_pseudo / (soft_pseudo.sum(dim=1, keepdim=True) + eps)
```

Important:

* `soft_pseudo` is the semantic pseudo-label distribution selected from U-Net/Mamba agreement-disagreement logic.
* It is not yet quantum-filtered.
* It is shared as the base pseudo-label for both cross-supervision directions.

### 7.7 Quantum verification for U-view and M-view

Compute pseudo prediction:

```python
pseudo_conf, pseudo_pred = torch.max(soft_pseudo, dim=1)
```

Compute quantum predictions:

```python
qconf_u, qpred_u = torch.max(Q_u, dim=1)
qconf_m, qpred_m = torch.max(Q_m, dim=1)
```

Define mean-branch reliability:

```python
branch_reliable = reliable_agree | reliable_disagree
```

Quantum verification for U-view:

```python
quantum_reliable_u = (
    (pseudo_pred == qpred_u)
    & (qconf_u >= tau_q)
    & candidate_mask
)
```

Quantum verification for M-view:

```python
quantum_reliable_m = (
    (pseudo_pred == qpred_m)
    & (qconf_m >= tau_q)
    & candidate_mask
)
```

Final masks:

```python
R_u = branch_reliable & quantum_reliable_u
R_m = branch_reliable & quantum_reliable_m
```

### 7.8 Direction-specific pseudo-labels

Use the same soft pseudo-label distribution for both directions, but different quantum masks:

```python
soft_pseudo_u = soft_pseudo
soft_pseudo_m = soft_pseudo
```

Then in training:

```text
soft_pseudo_u + R_u supervises Mamba branch
soft_pseudo_m + R_m supervises U-Net branch
```

Explanation:

```text
R_u means the pseudo-label is reliable under U-branch quantum verification.
R_m means the pseudo-label is reliable under Mamba-branch quantum verification.
```

### 7.9 Return dictionary

Return:

```python
return {
    "soft_pseudo_u": soft_pseudo_u.detach(),
    "soft_pseudo_m": soft_pseudo_m.detach(),

    "pseudo_u": torch.argmax(soft_pseudo_u, dim=1).detach(),
    "pseudo_m": torch.argmax(soft_pseudo_m, dim=1).detach(),

    "R_u": R_u.detach(),
    "R_m": R_m.detach(),

    "reliable_agree": reliable_agree.detach(),
    "reliable_disagree": reliable_disagree.detach(),
    "branch_reliable": branch_reliable.detach(),

    "quantum_reliable_u": quantum_reliable_u.detach(),
    "quantum_reliable_m": quantum_reliable_m.detach(),

    "conf_u": conf_u.detach(),
    "conf_m": conf_m.detach(),
    "pseudo_conf": pseudo_conf.detach(),

    "qconf_u": qconf_u.detach(),
    "qconf_m": qconf_m.detach(),

    "agreement_u": reliable_agree.detach(),
    "agreement_m": reliable_agree.detach(),

    "disagreement": reliable_disagree.detach(),
}
```

Keep keys like `"pseudo_u"`, `"pseudo_m"`, `"R_u"`, `"R_m"`, `"conf_u"`, `"conf_m"`, `"qconf_u"`, `"qconf_m"`, `"agreement_u"`, `"agreement_m"` because the current logging functions already expect them.

---

## 8. Update `summarize_reliable_masks`

Ensure `summarize_reliable_masks(pseudo_info)` supports new keys.

It should return at least:

```python
{
    "reliable_ratio_u": pseudo_info["R_u"].float().mean(),
    "reliable_ratio_m": pseudo_info["R_m"].float().mean(),

    "agreement_ratio_u": pseudo_info["reliable_agree"].float().mean(),
    "agreement_ratio_m": pseudo_info["reliable_agree"].float().mean(),

    "disagreement_ratio": pseudo_info["reliable_disagree"].float().mean(),

    "branch_reliable_ratio": pseudo_info["branch_reliable"].float().mean(),

    "quantum_reliable_ratio_u": pseudo_info["quantum_reliable_u"].float().mean(),
    "quantum_reliable_ratio_m": pseudo_info["quantum_reliable_m"].float().mean(),

    "conf_u_mean": pseudo_info["conf_u"].mean(),
    "conf_m_mean": pseudo_info["conf_m"].mean(),

    "qconf_u_mean": pseudo_info["qconf_u"].mean(),
    "qconf_m_mean": pseudo_info["qconf_m"].mean(),
}
```

If keeping backward compatibility, use `.get(...)` with safe fallbacks.

---

## 9. Training Loop Replacement

In `train.py`, replace this block:

```python
if iter_num >= train_args.warmup_iterations:
    pseudo_info = build_quantum_guided_reliable_masks(...)
    loss_u_to_m = masked_hard_ce(logits_m, pseudo_info["pseudo_u"], pseudo_info["R_u"])
    loss_m_to_u = masked_hard_ce(logits_u, pseudo_info["pseudo_m"], pseudo_info["R_m"])
    loss_qcps = loss_u_to_m + loss_m_to_u
```

with:

```python
if iter_num >= train_args.warmup_iterations:
    tau_agree = linear_schedule(
        iter_num,
        train_args.warmup_iterations,
        max_iterations,
        train_args.agree_thresh_start,
        train_args.agree_thresh_end,
    )

    tau_disagree = linear_schedule(
        iter_num,
        train_args.warmup_iterations,
        max_iterations,
        train_args.disagree_thresh_start,
        train_args.disagree_thresh_end,
    )

    _, _, tau_q = get_current_thresholds(
        iter_num=iter_num,
        warmup=train_args.warmup_iterations,
        max_iterations=max_iterations,
        tau_u_start=train_args.tau_u_start,
        tau_u_end=train_args.tau_u_end,
        tau_m_start=train_args.tau_m_start,
        tau_m_end=train_args.tau_m_end,
        tau_q_start=train_args.tau_q_start,
        tau_q_end=train_args.tau_q_end,
    )

    pseudo_info = build_dual_branch_quantum_pseudo_label(
        prob_u=prob_u,
        prob_m=prob_m,
        Q_u=q_u,
        Q_m=q_m,
        label=label_batch,
        ignore_index=num_classes,
        agree_thresh=tau_agree,
        disagree_thresh=tau_disagree,
        margin_thresh=train_args.margin_thresh,
        tau_q=tau_q,
        pseudo_mask_mode=train_args.pseudo_mask_mode,
    )

    loss_u_to_m = masked_soft_ce_loss(
        logits=logits_m,
        soft_target=pseudo_info["soft_pseudo_u"],
        mask=pseudo_info["R_u"],
    )

    loss_m_to_u = masked_soft_ce_loss(
        logits=logits_u,
        soft_target=pseudo_info["soft_pseudo_m"],
        mask=pseudo_info["R_m"],
    )

    loss_qcps = loss_u_to_m + loss_m_to_u
```

Warm-up fallback can remain similar, but add soft pseudo fields:

```python
else:
    pseudo_u_hard = torch.argmax(prob_u.detach(), dim=1)
    pseudo_m_hard = torch.argmax(prob_m.detach(), dim=1)
    zero_mask = torch.zeros_like(label_batch, dtype=torch.bool)

    pseudo_info = {
        "soft_pseudo_u": prob_u.detach(),
        "soft_pseudo_m": prob_m.detach(),

        "pseudo_u": pseudo_u_hard,
        "pseudo_m": pseudo_m_hard,

        "R_u": zero_mask,
        "R_m": zero_mask,

        "reliable_agree": zero_mask,
        "reliable_disagree": zero_mask,
        "branch_reliable": zero_mask,

        "quantum_reliable_u": zero_mask,
        "quantum_reliable_m": zero_mask,

        "conf_u": torch.max(prob_u.detach(), dim=1)[0],
        "conf_m": torch.max(prob_m.detach(), dim=1)[0],
        "pseudo_conf": torch.max(0.5 * (prob_u.detach() + prob_m.detach()), dim=1)[0],

        "qconf_u": torch.max(q_u.detach(), dim=1)[0],
        "qconf_m": torch.max(q_m.detach(), dim=1)[0],

        "agreement_u": zero_mask,
        "agreement_m": zero_mask,
        "disagreement": zero_mask,
    }

    loss_u_to_m = logits_u.new_tensor(0.0)
    loss_m_to_u = logits_u.new_tensor(0.0)
    loss_qcps = logits_u.new_tensor(0.0)
```

---

## 10. Import Changes

In `train.py`, replace or extend imports.

Current imports:

```python
from utils.quan_mamba_losses import (
    get_current_thresholds,
    masked_hard_ce,
    partial_ce_loss,
    partial_prob_ce,
)
from utils.quan_mamba_pseudo import (
    build_quantum_guided_reliable_masks,
    masked_label_accuracy,
    summarize_reliable_masks,
)
```

New imports should be:

```python
from utils.quan_mamba_losses import (
    get_current_thresholds,
    masked_hard_ce,
    partial_ce_loss,
    partial_prob_ce,
    masked_soft_ce_loss,
)
from utils.quan_mamba_pseudo import (
    build_dual_branch_quantum_pseudo_label,
    masked_label_accuracy,
    summarize_reliable_masks,
)
```

`masked_hard_ce` can remain imported for backward compatibility, but the new Q-CPS must use `masked_soft_ce_loss`.

---

## 11. Logging Updates

Keep current logs:

```python
writer.add_scalar("pseudo/reliable_ratio_u", ...)
writer.add_scalar("pseudo/reliable_ratio_m", ...)
writer.add_scalar("pseudo/agreement_ratio_u", ...)
writer.add_scalar("pseudo/agreement_ratio_m", ...)
writer.add_scalar("pseudo/conf_u", ...)
writer.add_scalar("pseudo/conf_m", ...)
writer.add_scalar("quantum/q_conf_u", ...)
writer.add_scalar("quantum/q_conf_m", ...)
```

Add new logs:

```python
writer.add_scalar("pseudo/disagreement_ratio", pseudo_stats["disagreement_ratio"].item(), iter_num)
writer.add_scalar("pseudo/branch_reliable_ratio", pseudo_stats["branch_reliable_ratio"].item(), iter_num)
writer.add_scalar("quantum/quantum_reliable_ratio_u", pseudo_stats["quantum_reliable_ratio_u"].item(), iter_num)
writer.add_scalar("quantum/quantum_reliable_ratio_m", pseudo_stats["quantum_reliable_ratio_m"].item(), iter_num)
writer.add_scalar("threshold/tau_agree", tau_agree, iter_num)
writer.add_scalar("threshold/tau_disagree", tau_disagree, iter_num)
writer.add_scalar("threshold/tau_q", tau_q, iter_num)
```

Update console logging:

```python
logging.info(
    "thresholds: tau_agree=%.4f, tau_disagree=%.4f, tau_q=%.4f",
    tau_agree,
    tau_disagree,
    tau_q,
)

logging.info(
    "reliable_u=%f, reliable_m=%f, agree=%f, disagree=%f, qrel_u=%f, qrel_m=%f, qconf_u=%f, qconf_m=%f",
    pseudo_stats["reliable_ratio_u"].item(),
    pseudo_stats["reliable_ratio_m"].item(),
    pseudo_stats["agreement_ratio_u"].item(),
    pseudo_stats["disagreement_ratio"].item(),
    pseudo_stats["quantum_reliable_ratio_u"].item(),
    pseudo_stats["quantum_reliable_ratio_m"].item(),
    safe_scalar(pseudo_stats["qconf_u_mean"]),
    safe_scalar(pseudo_stats["qconf_m_mean"]),
)
```

---

## 12. Dense Debug Compatibility

Current debug function uses:

```python
masked_label_accuracy(pseudo_info["pseudo_u"], gt_label, pseudo_info["R_u"])
masked_label_accuracy(pseudo_info["pseudo_m"], gt_label, pseudo_info["R_m"])
```

This can remain unchanged because the new function still returns hard versions:

```python
"pseudo_u": torch.argmax(soft_pseudo_u, dim=1)
"pseudo_m": torch.argmax(soft_pseudo_m, dim=1)
```

---

## 13. Loss Objective

Keep the same total loss:

```python
loss = loss_pce + train_args.lambda_q * loss_q + cps_weight * train_args.lambda_cps * loss_qcps
```

Where:

```text
loss_pce:
    partial scribble CE for U-Net and Mamba branches.

loss_q:
    quantum prototype supervision on scribble pixels.

loss_qcps:
    new dual-branch agreement/disagreement + quantum-verified soft cross pseudo supervision.
```

---

## 14. Conceptual Method Name

Use this name in comments or method description:

```text
Quantum-Verified Dual-Branch Cross Pseudo Supervision
```

Suggested abbreviation:

```text
QV-DB-CPS
```

More explicit name:

```text
Agreement-Disagreement Quantum-Verified Cross Pseudo Supervision
```

Suggested abbreviation:

```text
AD-QCPS
```

Recommended final name:

```text
AD-QCPS: Agreement-Disagreement Quantum-Verified Cross Pseudo Supervision
```

---

## 15. Expected Behavior

Early training:

```text
reliable_ratio_u ≈ 0
reliable_ratio_m ≈ 0
loss_qcps ≈ 0
```

After warm-up:

```text
agreement_ratio should increase
disagreement_ratio should be smaller than agreement_ratio
branch_reliable_ratio should become non-zero
quantum_reliable_ratio_u/m should become non-zero
reliable_ratio_u/m should become non-zero
loss_qcps should start contributing
```

If `reliable_ratio_u` and `reliable_ratio_m` stay zero for too long:

```text
Lower tau_q
Lower agree_thresh
Lower disagree_thresh
Lower margin_thresh
Check whether Q_u/Q_m are meaningful
Check whether label unlabeled pixels really equal num_classes
```

Recommended debugging values:

```text
agree_thresh_start=0.75
agree_thresh_end=0.65
disagree_thresh_start=0.85
disagree_thresh_end=0.75
margin_thresh=0.05
tau_q_start=0.50
tau_q_end=0.35
```

---

## 16. Important Notes

1. Do not add teacher models.
2. Do not add EMA update.
3. Do not create separate student/teacher architecture.
4. Use only the existing U-Net and Mamba-UNet branches inside `QuanMambaScrib`.
5. Replace hard CPS with soft pseudo-label CPS.
6. Keep partial CE and quantum prototype CE unchanged.
7. Keep validation and checkpoint logic unchanged.
8. Preserve current dictionary keys used by logging/debugging.
9. Ensure all pseudo-label tensors are detached.
10. Ensure `masked_soft_ce_loss` handles zero reliable pixels without NaN.

---

## 17. Minimal Implementation Checklist

The agent must implement:

```text
[ ] Add parser args: agree_thresh_start, agree_thresh_end, disagree_thresh_start, disagree_thresh_end, margin_thresh, pseudo_mask_mode.
[ ] Add linear_schedule function.
[ ] Add masked_soft_ce_loss in quan_mamba_losses.py.
[ ] Add build_dual_branch_quantum_pseudo_label in quan_mamba_pseudo.py.
[ ] Update summarize_reliable_masks for new statistics.
[ ] Replace build_quantum_guided_reliable_masks call in train.py.
[ ] Replace masked_hard_ce CPS with masked_soft_ce_loss CPS.
[ ] Keep log_dense_debug compatible by returning pseudo_u/pseudo_m hard labels.
[ ] Add new TensorBoard logs for disagreement and quantum reliable ratios.
[ ] Verify training does not produce NaN when masks are empty.
```
