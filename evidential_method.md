# SPEC — Evidential Mean Teacher with Asymmetric Uncertainty Consistency
## Scribble-Supervised Medical Image Segmentation

**Method name:** Evidential Mean Teacher with Asymmetric Uncertainty Consistency  
**Short name:** EMT-AUC  
**Target baseline:** Current Mean Teacher U-Net training script for ACDC scribble supervision  
**Implementation goal:** Keep the current pseudo-label mechanism unchanged, convert Student and EMA Teacher outputs to evidential predictions, generate uncertainty maps, and add one one-sided MSE loss in which lower Teacher uncertainty guides higher Student uncertainty.

---

## 1. Required behavior

The final training pipeline must do exactly this:

1. Student U-Net and EMA Teacher U-Net process the same image batch.
2. Each raw U-Net output is converted to:
   - non-negative evidence;
   - Dirichlet concentration parameters;
   - class probabilities;
   - a one-channel uncertainty map.
3. Existing confidence-based pseudo-label construction is kept unchanged.
4. Existing partial scribble supervision is kept semantically unchanged.
5. Add a custom uncertainty loss:
   - active only on the existing reliable pseudo-label mask;
   - active only where Student uncertainty is higher than Teacher uncertainty;
   - Teacher is detached;
   - only Student receives gradient.
6. Total loss:
   \[
   L = L_{PCE} + \lambda_p(t)L_{pseudo} + \lambda_u(t)L_{unc}
   \]

Do not add any other modules in version 1.

---

## 2. Non-goals

Do not implement:

- a second uncertainty head;
- MC Dropout;
- model ensembles;
- boundary-aware losses;
- uncertainty calibration modules;
- OOD training;
- Dempster–Shafer fusion;
- extra KL regularization;
- uncertainty-based pseudo-label filtering;
- bidirectional Student/Teacher backpropagation;
- changes to current agreement/disagreement pseudo-label logic.

This must remain a minimal ablation-friendly extension.

---

## 3. Baseline invariants

The following current behaviors must remain unchanged:

- Student is optimized by SGD.
- Teacher is updated only by EMA.
- Teacher forward is under `torch.no_grad()`.
- Current confidence thresholds and curriculum remain unchanged.
- Agreement pseudo-labels use the mean Student/Teacher distribution.
- Disagreement pseudo-labels use the stronger branch.
- `reliable_mask` remains the mask used by pseudo-label supervision.
- Current pseudo-loss ramp-up remains unchanged.
- Poly learning-rate decay remains unchanged.
- Validation cadence, Dice/HD95 evaluation, and checkpoint cadence remain unchanged.

The only semantic replacement is:

```python
softmax probability
```

becomes:

```python
Dirichlet expected probability
```

---

## 4. Evidential formulation

For raw U-Net output:

\[
z \in \mathbb{R}^{B	imes C	imes H	imes W}
\]

convert to evidence:

\[
e = \operatorname{Softplus}(z)
\]

Dirichlet parameters:

\[
lpha = e + 1
\]

Dirichlet strength:

\[
S = \sum_{c=1}^{C}lpha_c
\]

Expected class probability:

\[
p_c = rac{lpha_c}{S}
\]

Uncertainty:

\[
u = rac{C}{S}
\]

Required tensor shapes:

```text
raw_output   [B, C, H, W]
evidence     [B, C, H, W]
alpha        [B, C, H, W]
strength     [B, 1, H, W]
probability  [B, C, H, W]
uncertainty  [B, 1, H, W]
```

Required properties:

```text
evidence >= 0
alpha >= 1
sum(probability over classes) == 1
0 < uncertainty <= 1
```

Use `Softplus`. Do not use ReLU or exponential in version 1.

---

## 5. Architecture decision

Do not add a new head.

The existing U-Net final output remains a tensor with `num_classes` channels. Reinterpret that tensor as raw evidence logits.

Therefore:

- no change to U-Net parameter count;
- no change to decoder;
- no change to `net_factory`;
- existing checkpoints remain shape-compatible.

If the network returns a tuple/list, continue using the first tensor.

Required safer helper:

```python
def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Model output must be a tensor or tuple/list with tensor first")

    if output.ndim != 4:
        raise ValueError(
            f"Expected [B, C, H, W], received {tuple(output.shape)}"
        )

    return output
```

---

## 6. New utility module

Recommended file:

```text
utils/evidential.py
```

It must contain the four functions below.

### 6.1 `evidential_prediction`

```python
from typing import Dict
import torch
import torch.nn.functional as F


def evidential_prediction(
    raw_output: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if raw_output.ndim != 4:
        raise ValueError(
            f"Expected [B, C, H, W], received {tuple(raw_output.shape)}"
        )

    if raw_output.shape[1] != int(num_classes):
        raise ValueError(
            f"Output channels={raw_output.shape[1]} "
            f"but num_classes={num_classes}"
        )

    evidence = F.softplus(raw_output)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    prob = alpha / strength
    uncertainty = float(num_classes) / strength

    return {
        "evidence": evidence,
        "alpha": alpha,
        "strength": strength,
        "prob": prob,
        "uncertainty": uncertainty,
    }
```

Do not apply softmax anywhere inside this function.

---

### 6.2 `partial_ce_from_prob`

The current `CrossEntropyLoss` cannot receive evidential probabilities because it internally applies `log_softmax`.

Replace the supervised scribble loss with:

```python
def partial_ce_from_prob(
    prob: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prob.ndim != 4:
        raise ValueError("prob must be [B, C, H, W]")

    if label.ndim != 3:
        raise ValueError("label must be [B, H, W]")

    valid_mask = label != int(ignore_index)

    if valid_mask.sum().item() == 0:
        return prob.new_zeros(())

    num_classes = prob.shape[1]
    safe_label = label.clamp(0, num_classes - 1).long()

    gt_prob = prob.gather(
        dim=1,
        index=safe_label.unsqueeze(1),
    ).squeeze(1)

    loss_map = -torch.log(gt_prob.clamp_min(eps))
    return loss_map[valid_mask].mean()
```

Required semantics:

- only scribble pixels contribute;
- unlabeled pixels with `label == num_classes` contribute zero;
- output is a scalar tensor on the same device.

---

### 6.3 `masked_soft_ce_from_prob`

Replace the current pseudo-label CE that applies `log_softmax` to logits.

```python
def masked_soft_ce_from_prob(
    student_prob: torch.Tensor,
    target_prob: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_prob.shape != target_prob.shape:
        raise ValueError(
            "student_prob and target_prob must have identical shapes"
        )

    target_prob = target_prob.detach()
    log_prob = torch.log(student_prob.clamp_min(eps))

    ce_map = -(target_prob * log_prob).sum(
        dim=1,
        keepdim=True,
    )

    if mask is None:
        return ce_map.mean()

    mask = mask.detach().float()

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B, 1, H, W]")

    if mask.sum().item() < 1:
        return student_prob.new_zeros(())

    return (ce_map * mask).sum() / (mask.sum() + eps)
```

Do not call `F.log_softmax(student_prob)`.

---

### 6.4 `asymmetric_uncertainty_mse_loss`

This is the proposed custom loss.

Mathematical definition:

\[
L_{unc}
=
rac{
\sum_i M_i
\left[
\max(0,u_i^s-u_i^t-m)

ight]^2
}{
\sum_iM_i+\epsilon
}
\]

where:

- \(M_i\) is `pseudo_info["reliable_mask"]`;
- \(m\) is an optional margin;
- Teacher uncertainty is detached.

Required function:

```python
def asymmetric_uncertainty_mse_loss(
    student_uncertainty: torch.Tensor,
    teacher_uncertainty: torch.Tensor,
    reliable_mask: torch.Tensor | None,
    margin: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_uncertainty.shape != teacher_uncertainty.shape:
        raise ValueError(
            "Student and Teacher uncertainty shapes must match"
        )

    if student_uncertainty.ndim != 4:
        raise ValueError(
            "Uncertainty maps must be [B, 1, H, W]"
        )

    if student_uncertainty.shape[1] != 1:
        raise ValueError(
            "Uncertainty maps must have one channel"
        )

    if margin < 0:
        raise ValueError("margin must be non-negative")

    teacher_target = teacher_uncertainty.detach()

    positive_gap = F.relu(
        student_uncertainty
        - teacher_target
        - float(margin)
    )

    loss_map = positive_gap.pow(2)

    if reliable_mask is None:
        return loss_map.mean()

    mask = reliable_mask.detach().float()

    if mask.shape != student_uncertainty.shape:
        raise ValueError(
            "reliable_mask must match [B, 1, H, W]"
        )

    if mask.sum().item() < 1:
        return student_uncertainty.new_zeros(())

    return (loss_map * mask).sum() / (mask.sum() + eps)
```

Required behavior per pixel:

```text
teacher_u < student_u - margin:
    positive loss
    Student uncertainty is pulled downward

student_u <= teacher_u + margin:
    zero loss
    Student is not pulled upward

Teacher:
    never receives gradient
```

Do not use symmetric MSE as the default.

---

## 7. Pseudo-label generation

Keep `build_mt_confidence_pseudo_label` logically unchanged.

Only replace its inputs.

Old:

```python
student_prob = torch.softmax(outputs, dim=1)
teacher_prob = torch.softmax(ema_output, dim=1)
```

New:

```python
student_prob = student_evi["prob"]
teacher_prob = teacher_evi["prob"]
```

Do not modify:

- confidence extraction;
- predicted-class extraction;
- agreement condition;
- disagreement condition;
- confidence thresholds;
- confidence margin;
- mean pseudo-label;
- stronger-branch selection;
- normalization;
- reliable mask;
- detach behavior.

Do not add uncertainty thresholds to this function in version 1.

---

## 8. New CLI arguments

Add:

```python
parser.add_argument(
    "--enable_uncertainty_loss",
    type=int,
    default=1,
    choices=[0, 1],
    help="Enable asymmetric evidential uncertainty consistency",
)

parser.add_argument(
    "--uncertainty_loss_weight",
    type=float,
    default=0.5,
    help="Maximum uncertainty consistency weight",
)

parser.add_argument(
    "--uncertainty_margin",
    type=float,
    default=0.0,
    help="Ignored Student-Teacher uncertainty gap",
)

parser.add_argument(
    "--uncertainty_rampup",
    type=float,
    default=40.0,
    help="Epoch-length sigmoid ramp-up for uncertainty loss",
)
```

Add:

```python
def get_current_uncertainty_weight(epoch, train_args):
    return ramps.sigmoid_rampup(
        epoch,
        train_args.uncertainty_rampup,
    )
```

Required weight:

```python
uncertainty_weight = (
    get_current_uncertainty_weight(
        iter_num // len(trainloader),
        train_args,
    )
    * train_args.uncertainty_loss_weight
)
```

Do not reuse `pseudo_weight` as the uncertainty weight.

---

## 9. Exact training-loop modification

### Step 1 — EMA Teacher forward

```python
with torch.no_grad():
    ema_output = unpack_model_output(
        model_ema(volume_batch)
    )

    teacher_evi = evidential_prediction(
        ema_output,
        num_classes=num_classes,
    )

    teacher_prob = teacher_evi["prob"]
    teacher_uncertainty = teacher_evi["uncertainty"]
```

Teacher tensors must have `requires_grad=False`.

### Step 2 — Student forward

```python
outputs = unpack_model_output(
    model(volume_batch)
)

student_evi = evidential_prediction(
    outputs,
    num_classes=num_classes,
)

student_prob = student_evi["prob"]
student_uncertainty = student_evi["uncertainty"]
```

Student uncertainty must retain gradient.

### Step 3 — Scribble loss

Replace:

```python
loss_pce = ce_loss(outputs, label_batch.long())
```

with:

```python
loss_pce = partial_ce_from_prob(
    prob=student_prob,
    label=label_batch.long(),
    ignore_index=num_classes,
)
```

The old `CrossEntropyLoss` object is no longer required for the evidential path.

### Step 4 — Threshold curriculum

No changes.

### Step 5 — Pseudo-label construction

```python
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
```

No internal logic changes.

### Step 6 — Pseudo-label loss

Replace the current logit-based helper with:

```python
loss_pseudo = masked_soft_ce_from_prob(
    student_prob=student_prob,
    target_prob=pseudo_info["soft_pseudo_label"],
    mask=pseudo_info["reliable_mask"],
)
```

### Step 7 — Pseudo weight

No changes.

### Step 8 — Uncertainty loss

```python
if train_args.enable_uncertainty_loss:
    loss_uncertainty = asymmetric_uncertainty_mse_loss(
        student_uncertainty=student_uncertainty,
        teacher_uncertainty=teacher_uncertainty,
        reliable_mask=pseudo_info["reliable_mask"],
        margin=train_args.uncertainty_margin,
    )

    uncertainty_weight = (
        get_current_uncertainty_weight(
            iter_num // len(trainloader),
            train_args,
        )
        * train_args.uncertainty_loss_weight
    )
else:
    loss_uncertainty = student_prob.new_zeros(())
    uncertainty_weight = 0.0
```

### Step 9 — Total loss

```python
loss = (
    loss_pce
    + pseudo_weight * loss_pseudo
    + uncertainty_weight * loss_uncertainty
)
```

Do not add another loss in version 1.

### Step 10 — Optimization and EMA

Keep:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
ema_optimizer.step()
```

Teacher update must remain after Student optimization.

---

## 10. Complete reference pseudocode

```python
for sampled_batch in trainloader:
    volume_batch = sampled_batch["image"].cuda()
    label_batch = sampled_batch["label"].cuda()

    with torch.no_grad():
        ema_raw = unpack_model_output(
            model_ema(volume_batch)
        )
        teacher_evi = evidential_prediction(
            ema_raw,
            num_classes,
        )
        teacher_prob = teacher_evi["prob"]
        teacher_u = teacher_evi["uncertainty"]

    student_raw = unpack_model_output(
        model(volume_batch)
    )
    student_evi = evidential_prediction(
        student_raw,
        num_classes,
    )
    student_prob = student_evi["prob"]
    student_u = student_evi["uncertainty"]

    loss_pce = partial_ce_from_prob(
        student_prob,
        label_batch.long(),
        ignore_index=num_classes,
    )

    agree_th, disagree_th, margin_th = (
        get_threshold_curriculum(
            iter_num,
            train_args,
        )
    )

    pseudo_info = build_mt_confidence_pseudo_label(
        student_prob=student_prob,
        teacher_prob=teacher_prob,
        label=label_batch,
        agree_thresh=agree_th,
        disagree_thresh=disagree_th,
        margin_thresh=margin_th,
        ignore_index=num_classes,
        pseudo_mask_mode=train_args.pseudo_mask_mode,
    )

    loss_pseudo = masked_soft_ce_from_prob(
        student_prob=student_prob,
        target_prob=pseudo_info["soft_pseudo_label"],
        mask=pseudo_info["reliable_mask"],
    )

    pseudo_weight = (
        get_current_consistency_weight(
            iter_num // len(trainloader),
            train_args,
        )
        * train_args.pseudo_loss_weight
    )

    if train_args.enable_uncertainty_loss:
        loss_unc = asymmetric_uncertainty_mse_loss(
            student_uncertainty=student_u,
            teacher_uncertainty=teacher_u,
            reliable_mask=pseudo_info["reliable_mask"],
            margin=train_args.uncertainty_margin,
        )

        unc_weight = (
            get_current_uncertainty_weight(
                iter_num // len(trainloader),
                train_args,
            )
            * train_args.uncertainty_loss_weight
        )
    else:
        loss_unc = student_prob.new_zeros(())
        unc_weight = 0.0

    loss = (
        loss_pce
        + pseudo_weight * loss_pseudo
        + unc_weight * loss_unc
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema_optimizer.step()
```

---

## 11. Validation requirements

The coding agent must inspect the validation path, especially:

```text
val.py
test_single_volume
```

If validation currently does:

```python
prob = torch.softmax(output, dim=1)
```

it must use evidential probability for this method:

```python
raw_output = unpack_model_output(model(image))
evi = evidential_prediction(
    raw_output,
    num_classes=classes,
)
prob = evi["prob"]
pred = torch.argmax(prob, dim=1)
```

Never apply softmax after evidential conversion.

Training and validation must use the same probability definition.

Preferred reusable helper:

```python
def segmentation_prob_from_output(
    raw_output,
    num_classes,
    evidential=True,
):
    if evidential:
        return evidential_prediction(
            raw_output,
            num_classes,
        )["prob"]

    return torch.softmax(raw_output, dim=1)
```

---

## 12. Logging requirements

Add TensorBoard scalars:

```python
writer.add_scalar(
    "info/loss_uncertainty",
    loss_uncertainty.item(),
    iter_num,
)

writer.add_scalar(
    "info/uncertainty_weight",
    uncertainty_weight,
    iter_num,
)

writer.add_scalar(
    "uncertainty/student_mean",
    student_uncertainty.detach().mean().item(),
    iter_num,
)

writer.add_scalar(
    "uncertainty/teacher_mean",
    teacher_uncertainty.detach().mean().item(),
    iter_num,
)

writer.add_scalar(
    "evidence/student_strength_mean",
    student_evi["strength"].detach().mean().item(),
    iter_num,
)

writer.add_scalar(
    "evidence/teacher_strength_mean",
    teacher_evi["strength"].detach().mean().item(),
    iter_num,
)
```

Add active-guidance ratio:

```python
with torch.no_grad():
    active_mask = (
        student_uncertainty
        > teacher_uncertainty
        + train_args.uncertainty_margin
    ).float() * pseudo_info["reliable_mask"]

    active_guidance_ratio = (
        active_mask.sum()
        / (
            pseudo_info["reliable_mask"].sum()
            + 1e-8
        )
    )
```

Log:

```python
writer.add_scalar(
    "uncertainty/active_guidance_ratio",
    active_guidance_ratio.item(),
    iter_num,
)
```

Add to console logging every 200 iterations:

- `loss_uncertainty`;
- `uncertainty_weight`;
- Student mean uncertainty;
- Teacher mean uncertainty;
- active guidance ratio.

---

## 13. Numerical safety

Required:

- `Softplus` for evidence;
- `alpha = evidence + 1`;
- clamp strength with `eps`;
- clamp probability before `log`;
- return tensor zero for empty masks;
- detach all masks;
- detach Teacher uncertainty inside the loss;
- reject negative uncertainty margin;
- no NumPy conversion in the loss path;
- no `.data`;
- no in-place modification of gradient-requiring tensors.

Do not manually clamp uncertainty to `[0, 1]`.

---

## 14. Checkpoint behavior

The architecture is unchanged, so state-dict shapes remain compatible.

However, a model trained with softmax semantics is not automatically a calibrated evidential model.

Requirements:

- existing checkpoint loading may be used for initialization;
- emit a warning when loading a softmax-trained checkpoint for evidential training;
- save best model in the same format expected by current evaluation;
- optionally save metadata separately.

Recommended metadata:

```python
{
    "method": "EMT-AUC",
    "iteration": iter_num,
    "num_classes": num_classes,
    "uncertainty_loss_weight":
        train_args.uncertainty_loss_weight,
    "uncertainty_margin":
        train_args.uncertainty_margin,
}
```

---

## 15. Unit tests

Create:

```text
tests/test_evidential_mean_teacher.py
```

Required tests:

### Test 1 — probability normalization

```python
raw = torch.randn(2, 4, 16, 16)
result = evidential_prediction(raw, 4)

assert result["prob"].shape == (2, 4, 16, 16)
assert result["uncertainty"].shape == (2, 1, 16, 16)
assert torch.allclose(
    result["prob"].sum(dim=1),
    torch.ones_like(result["prob"][:, 0]),
    atol=1e-5,
)
```

### Test 2 — uncertainty range

```python
assert result["uncertainty"].min() > 0
assert result["uncertainty"].max() <= 1 + 1e-6
```

### Test 3 — zero-evidence limit

```python
raw = torch.full((1, 4, 2, 2), -100.0)
```

Expected:

```text
probability ≈ 0.25
uncertainty ≈ 1.0
```

### Test 4 — more evidence means less uncertainty

Compare zero logits with large positive logits.

Expected:

```python
u_high_evidence.mean() < u_low_evidence.mean()
```

### Test 5 — one-sided loss

```python
student_u = torch.tensor(
    [[[[0.8, 0.2]]]],
    requires_grad=True,
)
teacher_u = torch.tensor(
    [[[[0.3, 0.7]]]]
)
mask = torch.ones_like(student_u)
```

Expected with margin 0:

```text
first pixel loss = 0.25
second pixel loss = 0
mean = 0.125
```

### Test 6 — Teacher receives no gradient

Set Teacher tensor to `requires_grad=True`, call backward, and assert:

```python
student_u.grad is not None
teacher_u.grad is None
```

### Test 7 — reliable mask

All active loss pixels excluded by the mask must yield zero.

### Test 8 — empty mask

All-zero reliable mask must return a scalar zero without error.

### Test 9 — margin

For Student 0.55, Teacher 0.50:

```text
margin 0.10 -> loss 0
margin 0.00 -> positive loss
```

### Test 10 — partial CE ignores unlabeled pixels

Changing probabilities at ignored pixels must not change `partial_ce_from_prob`.

### Test 11 — integration backward

Run:

- evidential prediction;
- partial CE;
- pseudo-label construction;
- pseudo loss;
- uncertainty loss;
- total backward.

Assert:

- no NaN;
- no Inf;
- Student output receives gradient;
- Teacher output receives no gradient.

---

## 16. Acceptance criteria

Implementation is complete only when:

- training runs at least 500 iterations without NaN/Inf;
- Student and Teacher probabilities sum to one;
- uncertainty maps are `[B, 1, H, W]`;
- uncertainty loss is non-negative;
- uncertainty loss is zero when Student is never more uncertain than Teacher on reliable pixels;
- Teacher has no gradient;
- EMA update remains after Student optimizer step;
- pseudo-label generation remains logically unchanged;
- validation uses evidential probability;
- Dice and HD95 evaluation still runs;
- checkpoint saving still runs;
- required TensorBoard statistics are present;
- all unit tests pass.

---

## 17. Initial experiment configuration

Use first:

```text
--enable_uncertainty_loss 1
--uncertainty_loss_weight 0.5
--uncertainty_margin 0.0
--uncertainty_rampup 40.0
```

Keep all existing baseline hyperparameters unchanged.

Recommended ablation:

```text
uncertainty_loss_weight: 0.1, 0.5, 1.0
uncertainty_margin:      0.0, 0.02, 0.05
```

Required experiment variants:

1. Original softmax Mean Teacher baseline.
2. Evidential output with uncertainty loss disabled.
3. Full EMT-AUC.
4. Optional symmetric uncertainty MSE ablation, never default.

---

## 18. Common mistakes to avoid

Wrong:

```python
prob = torch.softmax(alpha / strength, dim=1)
```

Correct:

```python
prob = alpha / strength
```

Wrong:

```python
CrossEntropyLoss()(student_prob, label)
```

Correct:

```python
partial_ce_from_prob(student_prob, label, ignore_index)
```

Wrong:

```python
F.log_softmax(student_prob, dim=1)
```

Correct:

```python
torch.log(student_prob.clamp_min(eps))
```

Wrong default uncertainty loss:

```python
(student_u - teacher_u.detach()).pow(2)
```

Correct:

```python
F.relu(
    student_u
    - teacher_u.detach()
    - margin
).pow(2)
```

Wrong:

```python
loss_uncertainty(..., mask=None)
```

for the proposed full method.

Correct:

```python
reliable_mask=pseudo_info["reliable_mask"]
```

Wrong:

- uncertainty threshold inside pseudo-label builder;
- direct optimization of Teacher;
- validation with softmax;
- Python float `0.0` for empty masks.

---

## 19. File-level change plan

### Training script

Modify:

- imports;
- parser arguments;
- Student probability conversion;
- Teacher probability conversion;
- supervised scribble loss;
- pseudo-label loss;
- uncertainty loss;
- total loss;
- TensorBoard logging;
- console logging.

### `utils/evidential.py`

Add:

- `evidential_prediction`;
- `partial_ce_from_prob`;
- `masked_soft_ce_from_prob`;
- `asymmetric_uncertainty_mse_loss`.

### Validation path

Modify probability conversion to use Dirichlet expected probability.

### Tests

Add `tests/test_evidential_mean_teacher.py`.

Do not modify U-Net architecture in version 1.

---

## 20. Final implementation contract

Implement exactly:

```text
Student raw output
    -> Softplus evidence
    -> alpha = evidence + 1
    -> probability = alpha / sum(alpha)
    -> uncertainty = C / sum(alpha)

EMA Teacher raw output
    -> same evidential conversion

Student probability + Teacher probability
    -> current pseudo-label generation unchanged
    -> current reliable mask

Student uncertainty + detached Teacher uncertainty
    -> ReLU(Student - Teacher - margin)^2
    -> multiply by current reliable mask
    -> normalize by reliable-mask size

Total loss
    = partial scribble CE
    + ramped pseudo-label loss
    + ramped asymmetric uncertainty loss
```

Do not add components outside this contract without explicit approval.
