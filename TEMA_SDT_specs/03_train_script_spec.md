# 03 - Training Script Spec: `train_acdc_tema_sdt.py`

Create a new script based on `train_acdc.py`.

## Parser changes

Keep original args. Change patch size parser:

```python
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
```

Add:

```python
parser.add_argument('--alpha_fast', type=float, default=0.99)
parser.add_argument('--alpha_slow', type=float, default=0.999)
parser.add_argument('--pseudo_warmup', type=int, default=3000)

parser.add_argument('--uncertain_mode', type=str, default='quantile', choices=['quantile', 'fixed'])
parser.add_argument('--uncertain_top_ratio', type=float, default=0.35)
parser.add_argument('--uncertain_threshold', type=float, default=0.5)
parser.add_argument('--teacher_conf_threshold', type=float, default=0.55)
parser.add_argument('--easy_threshold', type=float, default=0.2)

parser.add_argument('--use_boundary_prior', type=int, default=1)
parser.add_argument('--boundary_lambda_image', type=float, default=1.0)
parser.add_argument('--gamma_boundary', type=float, default=0.5)
parser.add_argument('--gamma_disagree', type=float, default=0.5)
parser.add_argument('--gamma_stable', type=float, default=0.5)

parser.add_argument('--lambda_pseudo', type=float, default=0.5)
parser.add_argument('--lambda_hico', type=float, default=0.5)
parser.add_argument('--lambda_stable', type=float, default=0.1)
parser.add_argument('--debug_shapes', action='store_true')
```

## Imports

Remove:

```python
from utils.pick_reliable_pixels import refine_high_confidence
from utils.ema_optim import WeightEMA
```

Add:

```python
from utils.tema_entropy import normalized_entropy, build_uncertain_mask, teacher_confidence_mask
from utils.tema_boundary import boundary_likelihood
from utils.tema_region_switch import js_divergence_map, temporal_teacher_arbitration, select_teacher_feature
from utils.tema_losses import weighted_soft_ce_loss, weighted_feature_consistency_loss
from utils.tema_ema import detach_model, copy_student_to_teacher, update_ema_variables
```

## Model creation

```python
model = create_model(ema=False)
teacher_fast = create_model(ema=True)
teacher_slow = create_model(ema=True)

copy_student_to_teacher(model, teacher_fast)
copy_student_to_teacher(model, teacher_slow)
detach_model(teacher_fast)
detach_model(teacher_slow)

model.train()
teacher_fast.train()
teacher_slow.train()
```

Do not define `WeightEMA` objects.

## Optimizer

```python
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
```

## Forward pass

```python
with torch.no_grad():
    logits_fast, high_fast, low_fast = teacher_fast(image)
    probs_fast = torch.softmax(logits_fast, dim=1)

    logits_slow, high_slow, low_slow = teacher_slow(image)
    probs_slow = torch.softmax(logits_slow, dim=1)

logits_stu, high_stu, low_stu = model(image)
probs_stu = torch.softmax(logits_stu, dim=1)
```

## Loss construction

Scribble:

```python
loss_ce_stu = ce_loss(logits_stu, scrib)
loss_dice_stu = dice_loss(probs_stu, scrib.unsqueeze(1))
loss_scrib = loss_ce_stu + loss_dice_stu
```

Entropy and uncertain mask:

```python
entropy_stu = normalized_entropy(probs_stu)
uncertain_mask = build_uncertain_mask(
    entropy=entropy_stu,
    scribble=scrib,
    mode=args.uncertain_mode,
    top_ratio=args.uncertain_top_ratio,
    threshold=args.uncertain_threshold,
    ignore_index=4,
)
```

Teacher signals:

```python
entropy_fast = normalized_entropy(probs_fast)
entropy_slow = normalized_entropy(probs_slow)
disagreement = js_divergence_map(probs_fast, probs_slow)

if args.use_boundary_prior:
    boundary = boundary_likelihood(image, probs_stu, lambda_image=args.boundary_lambda_image)
else:
    boundary = torch.zeros_like(disagreement)
```

Arbitration:

```python
switch = temporal_teacher_arbitration(
    probs_fast=probs_fast,
    probs_slow=probs_slow,
    entropy_fast=entropy_fast,
    entropy_slow=entropy_slow,
    disagreement=disagreement,
    boundary=boundary,
    gamma_boundary=args.gamma_boundary,
    gamma_disagree=args.gamma_disagree,
    gamma_stable=args.gamma_stable,
)
selected_probs = switch['selected_probs']
selected_weight = switch['selected_weight']
select_fast = switch['select_fast']
```

Pseudo mask:

```python
conf_mask = teacher_confidence_mask(selected_probs, threshold=args.teacher_conf_threshold)
pseudo_mask = uncertain_mask & conf_mask
```

Pseudo loss:

```python
if iter_num >= args.pseudo_warmup:
    loss_pseudo = weighted_soft_ce_loss(
        logits_stu,
        selected_probs.detach(),
        pseudo_mask,
        selected_weight.detach(),
    )
else:
    loss_pseudo = logits_stu.sum() * 0.0
```

HiCo:

```python
if iter_num >= args.pseudo_warmup:
    selected_low = select_teacher_feature(low_fast, low_slow, select_fast).detach()
    selected_high = select_teacher_feature(high_fast, high_slow, select_fast).detach()

    loss_low = weighted_feature_consistency_loss(low_stu, selected_low, pseudo_mask, selected_weight.detach())
    loss_high = weighted_feature_consistency_loss(high_stu, selected_high, pseudo_mask, selected_weight.detach())
    loss_hico = 0.5 * (loss_low + loss_high)
else:
    loss_low = logits_stu.sum() * 0.0
    loss_high = logits_stu.sum() * 0.0
    loss_hico = logits_stu.sum() * 0.0
```

Stable slow loss:

```python
easy_mask = (entropy_stu < args.easy_threshold) & (scrib.unsqueeze(1) == 4)
if iter_num >= args.pseudo_warmup:
    loss_stable = weighted_soft_ce_loss(logits_stu, probs_slow.detach(), easy_mask, None)
else:
    loss_stable = logits_stu.sum() * 0.0
```

Total:

```python
consistency_weight = get_current_consistency_weight(iter_num // 300, args)
loss = (
    loss_scrib
    + consistency_weight * args.lambda_pseudo * loss_pseudo
    + consistency_weight * args.lambda_hico * loss_hico
    + consistency_weight * args.lambda_stable * loss_stable
)
```

Update:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()

update_ema_variables(model, teacher_fast, alpha=args.alpha_fast, global_step=iter_num)
update_ema_variables(model, teacher_slow, alpha=args.alpha_slow, global_step=iter_num)
```

## Logging

Add TensorBoard scalars:

```text
train/loss_total
train/loss_scrib
train/loss_pseudo
train/loss_hico
train/loss_stable
train/loss_low
train/loss_high
train/entropy_stu_mean
train/entropy_fast_mean
train/entropy_slow_mean
train/disagreement_mean
train/boundary_mean
train/uncertain_ratio
train/pseudo_ratio
train/select_fast_ratio
train/easy_ratio
```

## Validation/checkpoint

Keep current validation logic using `model` only. Save only student `model.state_dict()`.
