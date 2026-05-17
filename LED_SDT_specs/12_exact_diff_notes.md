# 12 - Exact Diff Notes from Current SDT-Net Code

This file maps current code blocks to new implementation blocks.

## Current model block

Current:

```python
model = create_model(ema=False) # student
teacher1 = create_model(ema=True) # teacher1
teacher2 = create_model(ema=True) # teacher2
model.cuda()
teacher1.cuda()
teacher2.cuda()
model.train()
teacher1.train()
teacher2.train()
```

Replace with:

```python
model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
model.cuda()
model.train()
```

## Current optimizer block

Current:

```python
optimizer = optim.SGD(model.parameters(), lr=base_lr, 
                      momentum=0.9, weight_decay=0.0001)
tea1_optimizer = WeightEMA(model, teacher1, 0.99)
tea2_optimizer = WeightEMA(model, teacher2, 0.99)
```

Replace with:

```python
optimizer = optim.SGD(model.parameters(), lr=base_lr,
                      momentum=0.9, weight_decay=0.0001)
```

Remove `WeightEMA` import.

## Current forward block

Current:

```python
with torch.no_grad():
    teacher1_output, high1, low1 = teacher1(image)
    outputs_soft_teacher1 = torch.softmax(teacher1_output, dim=1)
    teacher2_output, high2, low2 = teacher2(image)
    outputs_soft_teacher2 = torch.softmax(teacher2_output, dim=1)

student_output, high, low = model(image)
outputs_soft_student = torch.softmax(student_output, dim=1)
```

Replace with:

```python
out = model(image, return_all=True)

student_output = out["logits_s"]
local_output = out["logits_l"]
global_output = out["logits_g"]

outputs_soft_student = torch.softmax(student_output, dim=1)
outputs_soft_local = torch.softmax(local_output, dim=1)
outputs_soft_global = torch.softmax(global_output, dim=1)
```

## Current teacher selection block

Current:

```python
loss_ce_tea1 = ce_loss(teacher1_output, scrib[:])
loss_ce_tea2 = ce_loss(teacher2_output, scrib[:])

pseudo_label1 = refine_high_confidence(outputs_soft_teacher1, threshold=args.confidence_threshold)
pseudo_label2 = refine_high_confidence(outputs_soft_teacher2, threshold=args.confidence_threshold)

if loss_ce_tea1 < loss_ce_tea2:
    ...
else:
    ...
```

Replace with region-wise switching:

```python
entropy_s = normalized_entropy(outputs_soft_student)
uncertain_mask = build_uncertain_mask(entropy_s, scrib, ...)
boundary = boundary_likelihood(image, outputs_soft_student, ...)
switch = region_wise_teacher_selection(outputs_soft_local, outputs_soft_global, boundary, ...)
selected_probs = switch["selected_probs"]
selected_weight = switch["selected_weight"]
select_local = switch["select_local"]

conf_mask = teacher_confidence_mask(selected_probs, args.teacher_conf_threshold)
pseudo_mask = uncertain_mask & conf_mask

loss_pseudo = weighted_soft_ce_loss(
    student_output,
    selected_probs.detach(),
    pseudo_mask,
    selected_weight.detach()
)
```

## Current HiCo block

Current:

```python
loss_low = (F.l1_loss(low1, low) + (1 - F.cosine_similarity(low1.flatten(1), low.flatten(1)).mean())) / 2
loss_high = (F.l1_loss(high1, high) + (1 - F.cosine_similarity(high1.flatten(1), high.flatten(1)).mean())) / 2
```

Replace with:

```python
feat_low_t = select_teacher_feature(out["low_l"], out["low_g"], select_local).detach()
feat_high_t = select_teacher_feature(out["high_l"], out["high_g"], select_local).detach()

loss_low = weighted_feature_consistency_loss(out["low_s"], feat_low_t, pseudo_mask, selected_weight.detach())
loss_high = weighted_feature_consistency_loss(out["high_s"], feat_high_t, pseudo_mask, selected_weight.detach())
loss_hico = 0.5 * (loss_low + loss_high)
```

## Current loss block

Current:

```python
loss = loss_ce_stu + loss_pseudo * 0.5 + (loss_low + loss_high) * 0.5
```

Replace with:

```python
loss = (
    loss_scrib
    + args.lambda_aux * loss_aux
    + args.lambda_pseudo * consistency_weight * loss_pseudo
    + args.lambda_hico * consistency_weight * loss_hico
    + args.lambda_consensus * consistency_weight * loss_consensus
)
```

## Current EMA step

Current:

```python
if mode == 1:
    tea1_optimizer.step()
else:
    tea2_optimizer.step()
```

Remove completely.

## Current alpha schedule

Current:

```python
if iter_num > 0 and iter_num % 500 == 0:
    if alpha > 0.01:
        alpha = alpha - 0.01
    else:
        alpha = 0.01
```

Remove if not used anywhere.
