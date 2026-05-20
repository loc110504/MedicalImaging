# Spec: Replace High/Low Feature Consistency with Evidential Uncertainty Map Consistency in `train_method_acdc.py`

## 1. Goal

Modify the existing ACDC scribble-training script `train_method_acdc.py` so that it keeps the current Dual-Teacher + scribble supervision pipeline, but removes the current high-level / low-level feature consistency terms:

```python
loss_low = ...
loss_high = ...
(loss_low + loss_high) * 0.5
```

and replaces them with an **uncertainty map consistency loss** computed from the **student logits** and the **selected teacher logits**.

The uncertainty map must be extracted using the same evidential principle used in `train_evidential_fullmask.py`:

```python
evidence = activation(logits)
alpha = evidence + 1.0
uncertainty = num_classes / alpha.sum(dim=1)
```

This spec only covers training-loss modification. Do **not** add prompt encoder, point-click generation, or EUGIS Stage-II interactive refinement here.

---

## 2. Current code context

### 2.1 Existing scribble training script

The current `train_method_acdc.py` has:

- one student model: `model`
- two EMA teachers: `teacher1`, `teacher2`
- scribble labels loaded from `sampled['label']`
- supervised CE/Dice with `ignore_index=4`
- pseudo labels refined from high-confidence teacher predictions
- high/low feature consistency between selected teacher and student features

Current relevant flow:

```python
with torch.no_grad():
    teacher1_output, high1, low1 = teacher1(image)
    outputs_soft_teacher1 = torch.softmax(teacher1_output, dim=1)

    teacher2_output, high2, low2 = teacher2(image)
    outputs_soft_teacher2 = torch.softmax(teacher2_output, dim=1)

student_output, high, low = model(image)
outputs_soft_student = torch.softmax(student_output, dim=1)

loss_ce_stu = ce_loss(student_output, scrib[:])
loss_ce_tea1 = ce_loss(teacher1_output, scrib[:])
loss_ce_tea2 = ce_loss(teacher2_output, scrib[:])

pseudo_label1 = refine_high_confidence(outputs_soft_teacher1, threshold=args.confidence_threshold)
pseudo_label2 = refine_high_confidence(outputs_soft_teacher2, threshold=args.confidence_threshold)
```

Teacher selection currently depends on the teacher with lower scribble CE:

```python
if loss_ce_tea1 < loss_ce_tea2:
    selected_teacher = teacher1
else:
    selected_teacher = teacher2
```

Keep this teacher-selection logic unchanged.

### 2.2 Existing evidential full-mask script

The `train_evidential_fullmask.py` script trains a single model with `EvidentialSegmentationLoss`, which internally uses evidential uncertainty from logits. The same evidential uncertainty extraction rule must be reused for this scribble script.

---

## 3. Required behavior after modification

After modification, each iteration should optimize:

```python
loss = loss_ce_stu + 0.5 * loss_pseudo + lambda_unc_cons * consistency_weight * loss_uncertainty
```

where:

- `loss_ce_stu`: scribble-supervised CE using sparse scribble annotation.
- `loss_pseudo`: pseudo-label CE + pseudo-label Dice, same as current code.
- `loss_uncertainty`: uncertainty map consistency between selected teacher and student.
- `consistency_weight`: existing ramp-up from `get_current_consistency_weight`.
- `lambda_unc_cons`: new argument.

Remove `loss_low` and `loss_high` entirely from the total loss.

---

## 4. New command-line arguments

Add the following arguments to `argparse`:

```python
parser.add_argument(
    '--lambda_unc_cons',
    type=float,
    default=0.5,
    help='weight for evidential uncertainty map consistency loss'
)

parser.add_argument(
    '--uncertainty_consistency',
    type=str,
    default='l1',
    choices=['l1', 'mse', 'smooth_l1'],
    help='loss type for uncertainty map consistency'
)

parser.add_argument(
    '--evidence_activation',
    type=str,
    default='relu',
    choices=['relu', 'softplus', 'exp'],
    help='activation used to convert logits into non-negative evidence'
)

parser.add_argument(
    '--detach_teacher_uncertainty',
    action='store_true',
    default=True,
    help='detach teacher uncertainty map before consistency loss'
)
```

Implementation note:

- If `argparse` does not support `action='store_true', default=True` in the intended way for this codebase, replace it with:

```python
parser.add_argument('--detach_teacher_uncertainty', type=int, default=1)
```

and use:

```python
if args.detach_teacher_uncertainty:
    teacher_unc = teacher_unc.detach()
```

Recommended default: detach teacher uncertainty.

---

## 5. Add helper functions

Add these helper functions near `get_current_consistency_weight`.

### 5.1 Evidence activation helper

```python
def logits_to_evidence(logits, activation='relu'):
    if activation == 'relu':
        return F.relu(logits)
    if activation == 'softplus':
        return F.softplus(logits)
    if activation == 'exp':
        return torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
    raise ValueError('Unsupported evidence activation: {}'.format(activation))
```

Rationale:

- `relu` matches the default evidential-training configuration.
- `softplus` is smoother and may be useful for ablation.
- `exp` should be clamped to avoid numerical overflow.

### 5.2 Evidential uncertainty map helper

```python
def evidential_uncertainty_from_logits(logits, num_classes, activation='relu', eps=1e-8):
    evidence = logits_to_evidence(logits, activation=activation)
    alpha = evidence + 1.0
    uncertainty = float(num_classes) / (torch.sum(alpha, dim=1, keepdim=True) + eps)
    return uncertainty
```

Expected output shape:

```python
[B, 1, H, W]
```

Do **not** apply min-max normalization before consistency loss. The raw evidential uncertainty already lies approximately in `[0, 1]` because:

```python
uncertainty = K / S
S = sum(alpha)
alpha >= 1
```

When evidence is zero for all classes, `S = K`, so uncertainty is `1.0`.

### 5.3 Uncertainty consistency loss helper

```python
def uncertainty_consistency_loss(student_unc, teacher_unc, loss_type='l1'):
    if loss_type == 'l1':
        return F.l1_loss(student_unc, teacher_unc)
    if loss_type == 'mse':
        return F.mse_loss(student_unc, teacher_unc)
    if loss_type == 'smooth_l1':
        return F.smooth_l1_loss(student_unc, teacher_unc)
    raise ValueError('Unsupported uncertainty consistency loss: {}'.format(loss_type))
```

---

## 6. Required changes inside the training loop

### 6.1 Keep current forward pass structure

Keep this:

```python
with torch.no_grad():
    teacher1_output, high1, low1 = teacher1(image)
    outputs_soft_teacher1 = torch.softmax(teacher1_output, dim=1)

    teacher2_output, high2, low2 = teacher2(image)
    outputs_soft_teacher2 = torch.softmax(teacher2_output, dim=1)

student_output, high, low = model(image)
outputs_soft_student = torch.softmax(student_output, dim=1)
```

However, `high1`, `low1`, `high2`, `low2`, `high`, and `low` will no longer be used for the loss. It is acceptable to keep the unpacking if the model returns `(logits, high, low)`.

If the agent wants cleaner code, it may rename unused variables:

```python
teacher1_output, _, _ = teacher1(image)
teacher2_output, _, _ = teacher2(image)
student_output, _, _ = model(image)
```

Only do this if it does not break compatibility with other `net_factory` models.

---

### 6.2 Keep pseudo-label logic unchanged

Keep:

```python
pseudo_label1 = refine_high_confidence(outputs_soft_teacher1, threshold=args.confidence_threshold)
pseudo_label2 = refine_high_confidence(outputs_soft_teacher2, threshold=args.confidence_threshold)
```

Keep the current pseudo-label losses:

```python
loss_pseudo = ce_loss(student_output, pseudo_label.long()) + dice_loss(outputs_soft_student, pseudo_label.unsqueeze(1))
```

---

### 6.3 Replace feature consistency with uncertainty consistency

Current branch for teacher1:

```python
if loss_ce_tea1 < loss_ce_tea2:
    mode = 1
    loss_pseudo = ce_loss(student_output, pseudo_label1[:].long()) + dice_loss(outputs_soft_student, pseudo_label1.unsqueeze(1))
    loss_low = (F.l1_loss(low1, low) + (1 - F.cosine_similarity(low1.flatten(1), low.flatten(1)).mean())) / 2
    loss_high = (F.l1_loss(high1, high) + (1 - F.cosine_similarity(high1.flatten(1), high.flatten(1)).mean())) / 2
```

Replace with:

```python
if loss_ce_tea1 < loss_ce_tea2:
    mode = 1
    selected_teacher_output = teacher1_output
    selected_pseudo_label = pseudo_label1

    loss_pseudo = (
        ce_loss(student_output, selected_pseudo_label.long())
        + dice_loss(outputs_soft_student, selected_pseudo_label.unsqueeze(1))
    )

else:
    mode = 0
    selected_teacher_output = teacher2_output
    selected_pseudo_label = pseudo_label2

    loss_pseudo = (
        ce_loss(student_output, selected_pseudo_label.long())
        + dice_loss(outputs_soft_student, selected_pseudo_label.unsqueeze(1))
    )
```

Then compute uncertainty consistency after the branch:

```python
student_unc = evidential_uncertainty_from_logits(
    student_output,
    num_classes=args.num_classes,
    activation=args.evidence_activation,
)

teacher_unc = evidential_uncertainty_from_logits(
    selected_teacher_output,
    num_classes=args.num_classes,
    activation=args.evidence_activation,
)

if args.detach_teacher_uncertainty:
    teacher_unc = teacher_unc.detach()

loss_uncertainty = uncertainty_consistency_loss(
    student_unc,
    teacher_unc,
    loss_type=args.uncertainty_consistency,
)
```

Then update total loss:

```python
consistency_weight = get_current_consistency_weight(iter_num // 300, args)

loss = (
    loss_ce_stu
    + 0.5 * loss_pseudo
    + args.lambda_unc_cons * consistency_weight * loss_uncertainty
)
```

Delete or stop using:

```python
loss_low
loss_high
```

---

## 7. Important scribble-specific masking option

The default uncertainty consistency should be computed on the full image:

```python
loss_uncertainty = loss(student_unc, teacher_unc)
```

However, add an optional argument if possible:

```python
parser.add_argument(
    '--uncertainty_mask_mode',
    type=str,
    default='all',
    choices=['all', 'unlabeled', 'labeled'],
    help='where to apply uncertainty consistency for scribble training'
)
```

Behavior:

```python
if args.uncertainty_mask_mode == 'all':
    mask = None

elif args.uncertainty_mask_mode == 'unlabeled':
    mask = (scrib == 4).float().unsqueeze(1)

elif args.uncertainty_mask_mode == 'labeled':
    mask = (scrib != 4).float().unsqueeze(1)
```

If `mask is not None`, compute masked consistency:

```python
diff = torch.abs(student_unc - teacher_unc)  # for l1
loss_uncertainty = (diff * mask).sum() / (mask.sum() + 1e-8)
```

For `mse`, use squared diff. For `smooth_l1`, use elementwise smooth L1 with reduction `'none'`.

Recommended default:

```python
--uncertainty_mask_mode all
```

Ablation candidates:

- `all`: strongest regularization over whole image.
- `unlabeled`: focuses on regions without scribble supervision.
- `labeled`: debug only; less useful because labeled scribble pixels already have CE supervision.

---

## 8. TensorBoard logging changes

Remove or stop logging:

```python
writer.add_scalar("train/loss_low", loss_low.item(), iter_num)
writer.add_scalar("train/loss_high", loss_high.item(), iter_num)
```

Add:

```python
writer.add_scalar("train/loss_uncertainty", loss_uncertainty.item(), iter_num)
writer.add_scalar("train/uncertainty_student_mean", student_unc.mean().item(), iter_num)
writer.add_scalar("train/uncertainty_teacher_mean", teacher_unc.mean().item(), iter_num)
writer.add_scalar("train/consistency_weight", consistency_weight, iter_num)
writer.add_scalar("train/lr", lr_, iter_num)
```

Keep existing logs:

```python
writer.add_scalar("train/loss_total", loss.item(), iter_num)
writer.add_scalar("train/loss_pseudo_label", loss_pseudo.item(), iter_num)
writer.add_scalar("train/loss_ce", loss_ce_stu.item(), iter_num)
```

Update console logging from:

```python
'iteration %d : loss : %f, loss_pseudo_label: %f'
```

to:

```python
'iteration %d : loss=%f, ce=%f, pseudo=%f, unc_cons=%f, cw=%f'
```

---

## 9. Validation compatibility

Current validation likely expects the model to return either logits or a tuple. If `test_single_volume` can already handle the current `unet_hl` output, no change is required.

If validation breaks because the model returns `(logits, high, low)`, add a wrapper equivalent to `LogitsOnlyModel` from `train_evidential_fullmask.py`:

```python
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
```

Then validate with:

```python
eval_model = LogitsOnlyModel(model)
metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], eval_model, classes=num_classes)
```

Only apply this if needed, because the current scribble file may already validate correctly.

---

## 10. Suggested final script name

Do not overwrite the original file first. Create a new file:

```text
train_method_acdc_uncertainty_consistency.py
```

or:

```text
train_method_acdc_scribble_uncertainty.py
```

Recommended:

```text
train_method_acdc_scribble_uncertainty.py
```

---

## 11. Expected training command

Example:

```bash
python train_method_acdc_scribble_uncertainty.py \
  --root_path ../../data/ACDC \
  --sup_type scribble \
  --exp DualTeacher_UncertaintyConsistency \
  --model unet_hl \
  --num_classes 4 \
  --batch_size 8 \
  --base_lr 0.01 \
  --max_iterations 30000 \
  --lambda_unc_cons 0.5 \
  --uncertainty_consistency l1 \
  --evidence_activation relu \
  --uncertainty_mask_mode all
```

For an unlabeled-region-only ablation:

```bash
python train_method_acdc_scribble_uncertainty.py \
  --root_path ../../data/ACDC \
  --sup_type scribble \
  --exp DualTeacher_UncertaintyConsistency_Unlabeled \
  --model unet_hl \
  --num_classes 4 \
  --batch_size 8 \
  --base_lr 0.01 \
  --max_iterations 30000 \
  --lambda_unc_cons 0.5 \
  --uncertainty_consistency l1 \
  --evidence_activation relu \
  --uncertainty_mask_mode unlabeled
```

---

## 12. Acceptance criteria

The implementation is acceptable only if all conditions below are satisfied.

### Functional criteria

- [ ] Script runs with `--sup_type scribble`.
- [ ] Student and both teachers still load from `net_factory`.
- [ ] Teacher selection by lower scribble CE is preserved.
- [ ] Pseudo-label refinement via `refine_high_confidence` is preserved.
- [ ] `loss_low` and `loss_high` are removed from the final loss.
- [ ] New `loss_uncertainty` is computed from evidential uncertainty maps.
- [ ] Teacher uncertainty is detached by default.
- [ ] TensorBoard logs include `train/loss_uncertainty`.
- [ ] Checkpoint saving and validation still work.
- [ ] Best model is still saved as `{args.model}_best_model.pth`.

### Numerical criteria

- [ ] `student_unc.shape == [B, 1, H, W]`.
- [ ] `teacher_unc.shape == [B, 1, H, W]`.
- [ ] No NaN/Inf in `student_unc`, `teacher_unc`, or `loss_uncertainty`.
- [ ] `student_unc.min() >= 0` and `student_unc.max()` is reasonably near `<= 1`.
- [ ] Initial `loss_uncertainty` is finite and non-negative.

### Logging criteria

At every 400 iterations, log:

```text
iteration {iter_num} : loss={...}, ce={...}, pseudo={...}, unc_cons={...}, cw={...}
```

---

## 13. Debug checklist

If training crashes:

### Case 1: model output is not 3 values

Use a safe unpack helper:

```python
def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        logits = output[0]
        high = output[1] if len(output) > 1 else None
        low = output[2] if len(output) > 2 else None
    else:
        logits, high, low = output, None, None
    return logits, high, low
```

Then:

```python
teacher1_output, _, _ = unpack_model_output(teacher1(image))
teacher2_output, _, _ = unpack_model_output(teacher2(image))
student_output, _, _ = unpack_model_output(model(image))
```

### Case 2: `refine_high_confidence` returns invalid labels

Check that ignored pixels are `4`, matching:

```python
ce_loss = CrossEntropyLoss(ignore_index=4)
dice_loss = losses.pDLoss(num_classes, ignore_index=4)
```

### Case 3: uncertainty becomes always 1

This means evidence is near zero everywhere. Try:

```bash
--evidence_activation softplus
```

or lower the uncertainty consistency weight:

```bash
--lambda_unc_cons 0.1
```

### Case 4: uncertainty consistency dominates training

Log these values:

```python
print(loss_ce_stu.item(), loss_pseudo.item(), loss_uncertainty.item(), consistency_weight)
```

Then reduce:

```bash
--lambda_unc_cons 0.1
```

---

## 14. Important implementation note

Do **not** import or directly reuse `EvidentialSegmentationLoss` in this scribble script unless explicitly needed. The goal is not to replace the whole scribble training objective with full-mask evidential supervised learning. The goal is only to reuse the evidential uncertainty extraction rule from logits:

```python
evidence = activation(logits)
alpha = evidence + 1.0
uncertainty = num_classes / alpha.sum(dim=1)
```

The scribble method should still remain a Dual-Teacher scribble method with:

- sparse scribble CE,
- pseudo-label supervision,
- EMA teacher update,
- evidential uncertainty map consistency.

---

## 15. Summary of exact replacement

Before:

```python
loss = loss_ce_stu + loss_pseudo * 0.5 + (loss_low + loss_high) * 0.5
```

After:

```python
student_unc = evidential_uncertainty_from_logits(
    student_output,
    num_classes=args.num_classes,
    activation=args.evidence_activation,
)

teacher_unc = evidential_uncertainty_from_logits(
    selected_teacher_output,
    num_classes=args.num_classes,
    activation=args.evidence_activation,
)

teacher_unc = teacher_unc.detach()

loss_uncertainty = uncertainty_consistency_loss(
    student_unc,
    teacher_unc,
    loss_type=args.uncertainty_consistency,
)

consistency_weight = get_current_consistency_weight(iter_num // 300, args)

loss = (
    loss_ce_stu
    + 0.5 * loss_pseudo
    + args.lambda_unc_cons * consistency_weight * loss_uncertainty
)
```
