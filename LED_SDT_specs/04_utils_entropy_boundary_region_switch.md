# 04 - Utility Specs: Entropy, Boundary, Region Switching

Create the following files:

```text
utils/entropy_utils.py
utils/boundary_utils.py
utils/region_switch.py
```

---

# A. `utils/entropy_utils.py`

## Function 1: normalized entropy

```python
def normalized_entropy(
    probs: torch.Tensor,
    eps: float = 1e-8,
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Args:
        probs: Tensor[B, C, H, W], softmax probabilities.
        eps: numerical epsilon.
        keepdim: if True return Tensor[B, 1, H, W], else Tensor[B, H, W].

    Returns:
        entropy normalized to [0, 1].
    """
```

Implementation:

```python
import math
ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
ent = ent / math.log(probs.shape[1])
```

Clamp final result to `[0, 1]`.

## Function 2: entropy uncertain mask

```python
def build_uncertain_mask(
    entropy: torch.Tensor,
    scribble: torch.Tensor,
    mode: str = "quantile",
    top_ratio: float = 0.35,
    threshold: float = 0.5,
    ignore_index: int = 4,
) -> torch.Tensor:
    """
    Args:
        entropy: Tensor[B, 1, H, W] normalized entropy.
        scribble: Tensor[B, H, W].
        mode: "quantile" or "fixed".
        top_ratio: top entropy ratio among unlabeled pixels for quantile mode.
        threshold: fixed threshold for fixed mode.
        ignore_index: unlabeled pixel value.

    Returns:
        uncertain_mask: BoolTensor[B, 1, H, W].
    """
```

Rules:
- Only select uncertain pixels among unlabeled pixels:
  ```python
  unlabeled = (scribble == ignore_index).unsqueeze(1)
  ```
- If mode is fixed:
  ```python
  mask = (entropy >= threshold) & unlabeled
  ```
- If mode is quantile:
  compute threshold per image using only unlabeled pixels.
- If an image has no unlabeled pixels, return all false for that image.

## Function 3: teacher confidence mask

```python
def teacher_confidence_mask(
    selected_probs: torch.Tensor,
    threshold: float = 0.55,
) -> torch.Tensor:
    """
    Args:
        selected_probs: Tensor[B, C, H, W]
    Returns:
        BoolTensor[B, 1, H, W]
    """
```

---

# B. `utils/boundary_utils.py`

## Function 1: sobel magnitude

```python
def sobel_magnitude(
    x: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        x: Tensor[B, 1, H, W] or Tensor[B, C, H, W].
           For C > 1, convert to mean channel first.
    Returns:
        Tensor[B, 1, H, W], Sobel gradient magnitude.
    """
```

Implementation details:
- Use torch conv2d, not cv2, to stay on GPU.
- Define Sobel kernels:

```python
kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
```

- Use padding=1.
- Normalize per image.

## Function 2: boundary likelihood

```python
def boundary_likelihood(
    image: torch.Tensor,
    probs_s: torch.Tensor = None,
    lambda_image: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        image: Tensor[B, 1, H, W].
        probs_s: optional Tensor[B, C, H, W].
        lambda_image: weight for image-gradient boundary.
    Returns:
        Tensor[B, 1, H, W] in [0,1].
    """
```

Default v1:
- if `probs_s is None`: return Sobel magnitude of image.
- if `probs_s is not None`:
  - compute `conf_s = probs_s.max(dim=1, keepdim=True)[0]`
  - `b_prob = sobel_magnitude(conf_s)`
  - return `lambda_image * b_image + (1-lambda_image) * b_prob`
  - clamp to `[0,1]`.

Recommended default:
```python
lambda_image = 1.0
```

---

# C. `utils/region_switch.py`

## Function 1: select teacher by entropy and boundary prior

```python
def region_wise_teacher_selection(
    probs_l: torch.Tensor,
    probs_g: torch.Tensor,
    boundary: torch.Tensor,
    gamma: float = 0.5,
    eps: float = 1e-8,
) -> dict:
    """
    Args:
        probs_l: Tensor[B, C, H, W], local teacher probabilities.
        probs_g: Tensor[B, C, H, W], global teacher probabilities.
        boundary: Tensor[B, 1, H, W] in [0,1].
        gamma: prior strength.

    Returns:
        {
            "selected_probs": Tensor[B, C, H, W],
            "selected_hard": Tensor[B, H, W],
            "selected_weight": Tensor[B, 1, H, W],
            "select_local": BoolTensor[B, 1, H, W],
            "R_l": Tensor[B, 1, H, W],
            "R_g": Tensor[B, 1, H, W],
            "entropy_l": Tensor[B, 1, H, W],
            "entropy_g": Tensor[B, 1, H, W],
        }
    """
```

Reliability:

```python
H_l = normalized_entropy(probs_l)
H_g = normalized_entropy(probs_g)

R_l = (1.0 - H_l) * (1.0 + gamma * boundary)
R_g = (1.0 - H_g) * (1.0 + gamma * (1.0 - boundary))

select_local = R_l > R_g
selected_probs = torch.where(select_local, probs_l, probs_g)
selected_weight = torch.where(select_local, R_l, R_g)

selected_weight = selected_weight / (selected_weight.detach().mean() + eps)
selected_weight = selected_weight.clamp(0.0, 3.0)
selected_hard = selected_probs.argmax(dim=1)
```

## Function 2: select feature maps according to local/global mask

```python
def select_teacher_feature(
    feat_l: torch.Tensor,
    feat_g: torch.Tensor,
    select_local: torch.Tensor,
    target_size: tuple = None,
) -> torch.Tensor:
    """
    Args:
        feat_l: Tensor[B, C, h, w]
        feat_g: Tensor[B, C, h, w]
        select_local: BoolTensor[B, 1, H, W]
        target_size: optional spatial size; if None use feat_l spatial size.

    Returns:
        selected_feat: Tensor[B, C, h, w]
    """
```
