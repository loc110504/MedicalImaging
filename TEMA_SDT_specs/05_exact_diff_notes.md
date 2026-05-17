# 05 - Exact Diff Notes from Current `train_acdc.py`

## Remove PRP and WeightEMA imports

Remove:

```python
from utils.pick_reliable_pixels import refine_high_confidence
from utils.ema_optim import WeightEMA
```

Add TEMA utilities.

## Replace teacher naming

Current:

```python
teacher1 = create_model(ema=True)
teacher2 = create_model(ema=True)
```

New:

```python
teacher_fast = create_model(ema=True)
teacher_slow = create_model(ema=True)
copy_student_to_teacher(model, teacher_fast)
copy_student_to_teacher(model, teacher_slow)
detach_model(teacher_fast)
detach_model(teacher_slow)
```

## Remove teacher optimizers

Remove:

```python
tea1_optimizer = WeightEMA(model, teacher1, 0.99)
tea2_optimizer = WeightEMA(model, teacher2, 0.99)
```

## Replace batch-level DTS

Remove:

```python
loss_ce_tea1 = ce_loss(teacher1_output, scrib[:])
loss_ce_tea2 = ce_loss(teacher2_output, scrib[:])

if loss_ce_tea1 < loss_ce_tea2:
    mode = 1
    ...
else:
    mode = 0
    ...
```

New: compute entropy/disagreement/boundary and do pixel-wise arbitration.

## Replace hard PRP pseudo-labels

Remove:

```python
pseudo_label1 = refine_high_confidence(...)
pseudo_label2 = refine_high_confidence(...)
loss_pseudo = ce_loss(student_output, pseudo_label.long()) + dice_loss(...)
```

New:

```python
loss_pseudo = weighted_soft_ce_loss(student_output, selected_probs.detach(), pseudo_mask, selected_weight.detach())
```

## Replace HiCo target

Current HiCo uses either teacher1 or teacher2 for the whole batch.

New HiCo uses:

```python
selected_low = select_teacher_feature(low_fast, low_slow, select_fast).detach()
selected_high = select_teacher_feature(high_fast, high_slow, select_fast).detach()
```

## Replace teacher update

Remove:

```python
if mode == 1:
    tea1_optimizer.step()
else:
    tea2_optimizer.step()
```

New:

```python
update_ema_variables(model, teacher_fast, args.alpha_fast, iter_num)
update_ema_variables(model, teacher_slow, args.alpha_slow, iter_num)
```

## Keep unchanged

- Dataloader.
- Validation loop.
- LR schedule.
- Snapshot save logic.
- TensorBoard writer setup.
