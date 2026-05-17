# 05 - Weighted Loss Specs

Create:

```text
utils/weighted_losses.py
```

## Function 1: weighted soft cross entropy

```python
def weighted_soft_ce_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        logits: Tensor[B, C, H, W]
        soft_targets: Tensor[B, C, H, W], detached teacher soft probabilities.
        mask: BoolTensor or FloatTensor[B, 1, H, W], valid pseudo-label pixels.
        weight: optional Tensor[B, 1, H, W], reliability weights.
    Returns:
        scalar loss.
    """
```

Implementation:

```python
log_probs = F.log_softmax(logits, dim=1)
per_pixel = -(soft_targets * log_probs).sum(dim=1, keepdim=True)
mask = mask.float()
if weight is not None:
    per_pixel = per_pixel * weight.float()
loss = (per_pixel * mask).sum() / (mask.sum() + eps)
return loss
```

If `mask.sum()==0`, return `logits.sum() * 0.0` to preserve graph.

## Function 2: weighted hard CE

Optional helper:

```python
def masked_hard_ce_loss(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor = None,
    ignore_index: int = 255,
    eps: float = 1e-8,
) -> torch.Tensor:
    ...
```

Use only if hard pseudo-label loss is needed. Default method uses soft CE.

## Function 3: weighted feature L1 + cosine

```python
def weighted_feature_consistency_loss(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        feat_s: Tensor[B, C, h, w]
        feat_t: Tensor[B, C, h, w], detached selected teacher feature.
        mask: BoolTensor/FloatTensor[B, 1, H, W] or [B,1,h,w]
        weight: optional Tensor[B, 1, H, W] or [B,1,h,w]
    Returns:
        scalar loss.
    """
```

Implementation details:
- resize mask and weight to feature size by nearest interpolation;
- L1:
  ```python
  l1_map = torch.abs(feat_s - feat_t).mean(dim=1, keepdim=True)
  l1 = (l1_map * mask * weight).sum() / (mask.sum() + eps)
  ```
- Cosine:
  Compute channel-wise cosine at each spatial location:
  ```python
  cos_map = F.cosine_similarity(feat_s, feat_t, dim=1).unsqueeze(1)
  cos_loss_map = 1.0 - cos_map
  cos = (cos_loss_map * mask * weight).sum() / (mask.sum() + eps)
  ```
- return `(l1 + cos) / 2`.

## Function 4: symmetric KL for teacher consensus

```python
def masked_symmetric_kl_loss(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        probs_a/probs_b: Tensor[B, C, H, W]
        mask: Tensor[B, 1, H, W]
    """
```

Implementation:
```python
log_a = torch.log(probs_a.clamp_min(eps))
log_b = torch.log(probs_b.clamp_min(eps))
kl_ab = (probs_a.detach() * (log_a.detach() - log_b)).sum(dim=1, keepdim=True)
kl_ba = (probs_b.detach() * (log_b.detach() - log_a)).sum(dim=1, keepdim=True)
loss_map = 0.5 * (kl_ab + kl_ba)
loss = (loss_map * mask.float()).sum() / (mask.float().sum() + eps)
```
