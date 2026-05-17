# 06 - Training Script Spec: `train_acdc_lgdt.py`

## Goal

Create a new training script based on the current `train_acdc.py`.

Do not overwrite old training script. Create:

```text
train_acdc_lgdt.py
```

## Major differences from current `train_acdc.py`

Remove:
```python
teacher1 = create_model(ema=True)
teacher2 = create_model(ema=True)
tea1_optimizer = WeightEMA(...)
tea2_optimizer = WeightEMA(...)
teacher1_output, high1, low1 = teacher1(image)
teacher2_output, high2, low2 = teacher2(image)
if loss_ce_tea1 < loss_ce_tea2: ...
```

Add:
```python
model = net_factory(net_type="unet_lgdt", ...)
out = model(image, return_all=True)
```

## Parser additions

Keep old args and add:

```python
parser.add_argument('--uncertain_mode', type=str, default='quantile',
                    choices=['quantile', 'fixed'])
parser.add_argument('--uncertain_top_ratio', type=float, default=0.35)
parser.add_argument('--uncertain_threshold', type=float, default=0.5)
parser.add_argument('--teacher_conf_threshold', type=float, default=0.55)
parser.add_argument('--boundary_gamma', type=float, default=0.5)
parser.add_argument('--boundary_lambda_image', type=float, default=1.0)

parser.add_argument('--lambda_pseudo', type=float, default=0.5)
parser.add_argument('--lambda_hico', type=float, default=0.5)
parser.add_argument('--lambda_aux', type=float, default=0.4)
parser.add_argument('--lambda_consensus', type=float, default=0.1)
parser.add_argument('--lambda_div', type=float, default=0.0)

parser.add_argument('--pseudo_warmup', type=int, default=3000)
parser.add_argument('--debug_shapes', action='store_true')
```

Change patch size parser if needed:

```python
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
```

The current parser uses `type=list`, which is fragile. Prefer `nargs=2`.

## Imports

Add:

```python
from utils.entropy_utils import normalized_entropy, build_uncertain_mask, teacher_confidence_mask
from utils.boundary_utils import boundary_likelihood
from utils.region_switch import region_wise_teacher_selection, select_teacher_feature
from utils.weighted_losses import (
    weighted_soft_ce_loss,
    weighted_feature_consistency_loss,
    masked_symmetric_kl_loss,
)
```

## Model creation

Use only one model:

```python
model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
model.cuda()
model.train()
```

Optimizer:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001
)
```

## Forward pass

```python
out = model(image, return_all=True)

logits_s = out["logits_s"]
logits_l = out["logits_l"]
logits_g = out["logits_g"]

probs_s = torch.softmax(logits_s, dim=1)
probs_l = torch.softmax(logits_l, dim=1)
probs_g = torch.softmax(logits_g, dim=1)
```

## Supervised scribble loss

```python
loss_ce_s = ce_loss(logits_s, scrib)
loss_dice_s = dice_loss(probs_s, scrib.unsqueeze(1))
loss_scrib = loss_ce_s + loss_dice_s
```

If `pDLoss` cannot handle `ignore_index=4` for scribble, use only CE for scribble initially.

## Auxiliary teacher scribble loss

```python
loss_aux = 0.5 * (
    ce_loss(logits_l, scrib) + ce_loss(logits_g, scrib)
)
```

Start with CE only for stability.

## Entropy and masks

```python
entropy_s = normalized_entropy(probs_s)  # [B,1,H,W]

uncertain_mask = build_uncertain_mask(
    entropy=entropy_s,
    scribble=scrib,
    mode=args.uncertain_mode,
    top_ratio=args.uncertain_top_ratio,
    threshold=args.uncertain_threshold,
    ignore_index=4,
)
```

Boundary:

```python
boundary = boundary_likelihood(
    image=image,
    probs_s=probs_s,
    lambda_image=args.boundary_lambda_image
)
```

Region switching:

```python
switch = region_wise_teacher_selection(
    probs_l=probs_l,
    probs_g=probs_g,
    boundary=boundary,
    gamma=args.boundary_gamma,
)
selected_probs = switch["selected_probs"]
selected_weight = switch["selected_weight"]
select_local = switch["select_local"]
```

Confidence mask:

```python
conf_mask = teacher_confidence_mask(
    selected_probs,
    threshold=args.teacher_conf_threshold
)
pseudo_mask = uncertain_mask & conf_mask
```

Detach teacher targets:

```python
selected_probs_detached = selected_probs.detach()
selected_weight_detached = selected_weight.detach()
```

## Pseudo-label loss

Only after warmup:

```python
if iter_num >= args.pseudo_warmup:
    loss_pseudo = weighted_soft_ce_loss(
        logits=logits_s,
        soft_targets=selected_probs_detached,
        mask=pseudo_mask,
        weight=selected_weight_detached,
    )
else:
    loss_pseudo = logits_s.sum() * 0.0
```

## HiCo feature consistency

Select teacher features:

```python
feat_low_t = select_teacher_feature(
    out["low_l"],
    out["low_g"],
    select_local=select_local,
).detach()

feat_high_t = select_teacher_feature(
    out["high_l"],
    out["high_g"],
    select_local=select_local,
).detach()
```

Compute feature consistency after warmup:

```python
if iter_num >= args.pseudo_warmup:
    loss_low = weighted_feature_consistency_loss(
        feat_s=out["low_s"],
        feat_t=feat_low_t,
        mask=pseudo_mask,
        weight=selected_weight_detached,
    )
    loss_high = weighted_feature_consistency_loss(
        feat_s=out["high_s"],
        feat_t=feat_high_t,
        mask=pseudo_mask,
        weight=selected_weight_detached,
    )
    loss_hico = 0.5 * (loss_low + loss_high)
else:
    loss_low = logits_s.sum() * 0.0
    loss_high = logits_s.sum() * 0.0
    loss_hico = logits_s.sum() * 0.0
```

## Consensus loss on easy pixels

Easy mask:

```python
easy_mask = (entropy_s < 0.2) & (scrib.unsqueeze(1) == 4)
```

Loss:

```python
loss_consensus = masked_symmetric_kl_loss(
    probs_a=probs_l,
    probs_b=probs_g,
    mask=easy_mask,
)
```

Set `lambda_consensus=0.1`.

## Diversity loss

For v1, keep zero:

```python
loss_div = logits_s.sum() * 0.0
```

Do not implement until baseline works. If requested later, implement JS-margin loss.

## Total loss

Use ramp-up for pseudo and hico:

```python
consistency_weight = get_current_consistency_weight(iter_num // 300, args)
pseudo_weight = args.lambda_pseudo * consistency_weight
hico_weight = args.lambda_hico * consistency_weight
cons_weight = args.lambda_consensus * consistency_weight

loss = (
    loss_scrib
    + args.lambda_aux * loss_aux
    + pseudo_weight * loss_pseudo
    + hico_weight * loss_hico
    + cons_weight * loss_consensus
    + args.lambda_div * loss_div
)
```

## Backprop

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

No EMA teacher update.

## Logging

Add scalars:
- `train/loss_total`
- `train/loss_scrib`
- `train/loss_aux`
- `train/loss_pseudo`
- `train/loss_hico`
- `train/loss_low`
- `train/loss_high`
- `train/loss_consensus`
- `train/entropy_s_mean`
- `train/uncertain_ratio`
- `train/pseudo_ratio`
- `train/select_local_ratio`
- `train/boundary_mean`
