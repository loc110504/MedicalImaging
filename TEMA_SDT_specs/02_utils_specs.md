# 02 - Utility Module Specs

Create these modules:

```text
utils/tema_entropy.py
utils/tema_boundary.py
utils/tema_region_switch.py
utils/tema_losses.py
utils/tema_ema.py
```

## `utils/tema_entropy.py`

### `normalized_entropy`

```python
def normalized_entropy(probs, eps=1e-8, keepdim=True):
    """
    probs: Tensor[B,C,H,W]
    returns: Tensor[B,1,H,W] in [0,1]
    """
```

Implementation:

```python
ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
ent = ent / math.log(probs.shape[1])
ent = ent.clamp(0.0, 1.0)
```

### `build_uncertain_mask`

```python
def build_uncertain_mask(entropy, scribble, mode='quantile', top_ratio=0.35,
                         threshold=0.5, ignore_index=4):
    """
    entropy: Tensor[B,1,H,W]
    scribble: Tensor[B,H,W]
    returns BoolTensor[B,1,H,W]
    """
```

Rules:

- Only select from `(scribble == 4)`.
- `fixed`: entropy >= threshold.
- `quantile`: per-image top `top_ratio` among unlabeled pixels.

### `teacher_confidence_mask`

```python
def teacher_confidence_mask(probs, threshold=0.55):
    return probs.max(dim=1, keepdim=True)[0] >= threshold
```

## `utils/tema_boundary.py`

### `sobel_magnitude`

GPU Sobel magnitude using `torch.nn.functional.conv2d`.

Input: `[B,1,H,W]` or `[B,C,H,W]`. If C > 1, average channels first.

Output: `[B,1,H,W]`, normalized per image.

### `boundary_likelihood`

```python
def boundary_likelihood(image, probs_s=None, lambda_image=1.0, eps=1e-8):
    b_img = sobel_magnitude(image)
    if probs_s is None or lambda_image >= 1.0:
        return b_img
    conf = probs_s.max(dim=1, keepdim=True)[0]
    b_prob = sobel_magnitude(conf)
    return (lambda_image*b_img + (1-lambda_image)*b_prob).clamp(0,1)
```

## `utils/tema_region_switch.py`

### `js_divergence_map`

```python
def js_divergence_map(probs_a, probs_b, eps=1e-8, normalize=True):
    """returns Tensor[B,1,H,W]"""
```

Formula:

```math
m = 0.5 * (p_a + p_b)
JS = 0.5 KL(p_a || m) + 0.5 KL(p_b || m)
```

If `normalize=True`, divide by `log(2)` and clamp to `[0,1]`.

### `temporal_teacher_arbitration`

```python
def temporal_teacher_arbitration(
    probs_fast, probs_slow,
    entropy_fast, entropy_slow,
    disagreement,
    boundary=None,
    gamma_boundary=0.5,
    gamma_disagree=0.5,
    gamma_stable=0.5,
    eps=1e-8,
):
    """returns dict with selected_probs, selected_weight, select_fast, R_fast, R_slow"""
```

Reliability:

```python
R_fast = (1 - entropy_fast) * (1 + gamma_boundary * boundary) * (1 + gamma_disagree * disagreement)
R_slow = (1 - entropy_slow) * (1 + gamma_boundary * (1 - boundary)) * (1 + gamma_stable * (1 - disagreement))
```

Selection:

```python
select_fast = R_fast > R_slow
selected_probs = torch.where(select_fast, probs_fast, probs_slow)
selected_weight = torch.where(select_fast, R_fast, R_slow)
selected_weight = selected_weight / (selected_weight.detach().mean() + eps)
selected_weight = selected_weight.clamp(0.0, 3.0)
```

### `select_teacher_feature`

```python
def select_teacher_feature(feat_fast, feat_slow, select_fast):
    mask = F.interpolate(select_fast.float(), size=feat_fast.shape[-2:], mode='nearest') > 0.5
    return torch.where(mask, feat_fast, feat_slow)
```

## `utils/tema_losses.py`

### `weighted_soft_ce_loss`

```python
def weighted_soft_ce_loss(logits, soft_targets, mask, weight=None, eps=1e-8):
    mask_f = mask.float()
    if mask_f.sum() < 1:
        return logits.sum() * 0.0
    log_probs = F.log_softmax(logits, dim=1)
    loss_map = -(soft_targets * log_probs).sum(dim=1, keepdim=True)
    if weight is not None:
        loss_map = loss_map * weight.float()
    return (loss_map * mask_f).sum() / (mask_f.sum() + eps)
```

### `weighted_feature_consistency_loss`

Weighted L1 + cosine feature consistency. Resize mask/weight to feature spatial size. If mask empty, return zero graph loss.

## `utils/tema_ema.py`

### `copy_student_to_teacher`

```python
@torch.no_grad()
def copy_student_to_teacher(student, teacher):
    teacher.load_state_dict(student.state_dict())
```

### `detach_model`

```python
def detach_model(model):
    for p in model.parameters():
        p.detach_()
        p.requires_grad_(False)
```

### `update_ema_variables`

```python
@torch.no_grad()
def update_ema_variables(student, teacher, alpha, global_step=None):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0-alpha)
    for ema_buf, buf in zip(teacher.buffers(), student.buffers()):
        ema_buf.copy_(buf)
```
