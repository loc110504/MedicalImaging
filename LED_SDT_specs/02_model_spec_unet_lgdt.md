# 02 - Model Spec: `networks/unet_lgdt.py`

## Goal

Create a new efficient model with:
- one shared encoder;
- one student decoder;
- one local teacher decoder;
- one global teacher decoder.

The model must avoid two full teacher networks.

## File to create

```text
networks/unet_lgdt.py
```

## Main class

```python
class UNet_LGDT(nn.Module):
    def __init__(
        self,
        in_chns: int = 1,
        class_num: int = 4,
        base_chns: int = 16,
        return_all_default: bool = False,
        use_aspp: bool = True,
    ):
        ...
```

## Forward API

The forward function must support two modes.

### Training mode

```python
out = model(image, return_all=True)
```

Return a dictionary:

```python
{
    "logits_s": Tensor[B, C, H, W],
    "logits_l": Tensor[B, C, H, W],
    "logits_g": Tensor[B, C, H, W],

    "low_s": Tensor[B, Ch_low, H_low, W_low],
    "low_l": Tensor[B, Ch_low, H_low, W_low],
    "low_g": Tensor[B, Ch_low, H_low, W_low],

    "high_s": Tensor[B, Ch_high, H_high, W_high],
    "high_l": Tensor[B, Ch_high, H_high, W_high],
    "high_g": Tensor[B, Ch_high, H_high, W_high],
}
```

### Compatibility mode

```python
logits_s, high_s, low_s = model(image)
```

Return exactly this tuple by default, because the old training/validation code expects:

```python
student_output, high, low = model(image)
```

This keeps compatibility with `test_single_volume` if it expects the same tuple behavior.

## Architecture details

### Shared encoder

Reuse the existing encoder logic from `networks/unet.py` if possible.

Expected encoder features:

```python
f1: Tensor[B, base, H, W]
f2: Tensor[B, base*2, H/2, W/2]
f3: Tensor[B, base*4, H/4, W/4]
f4: Tensor[B, base*8, H/8, W/8]
f5: Tensor[B, base*16, H/16, W/16]  # optional bottleneck
```

If the existing UNet uses four levels only, adapt accordingly. The critical requirement is that all decoders receive the same shared features.

### Student decoder

Standard U-Net decoder.

Requirements:
- output `logits_s` shape `[B, C, H, W]`;
- output `low_s` and `high_s` for HiCo;
- feature channels must match local/global features.

Recommended:
- `low_*`: high-resolution decoder feature near final layer, e.g. `[B, base, H, W]` or `[B, base, H/2, W/2]`;
- `high_*`: bottleneck or high-level decoder feature, e.g. `[B, base*8, H/8, W/8]`.

### Local teacher decoder

Bias: local detail / boundary.

Design:
- use high-resolution skip features strongly;
- use 3x3 conv refinement;
- no large context module;
- no heavy attention.

Pseudo-architecture:

```python
x = up(f5, f4)
high_l = x
x = up(x, f3)
x = up(x, f2)
x = up(x, f1)
low_l = local_refine(x)
logits_l = Conv1x1(low_l, class_num)
```

`local_refine`:

```python
Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
```

### Global teacher decoder

Bias: global shape / context.

Design:
- use bottleneck context module before decoding;
- use ASPP or large-kernel depthwise conv;
- rely more on deep features.

Pseudo-architecture:

```python
f5_context = ASPP(f5)  # or LargeKernelContext
x = up(f5_context, f4)
high_g = x
x = up(x, f3)
x = up(x, f2)
x = up(x, f1)
low_g = global_refine(x)
logits_g = Conv1x1(low_g, class_num)
```

ASPP spec:
- dilation rates: `[1, 6, 12, 18]`
- each branch: Conv3x3 or Conv1x1 -> BN -> ReLU
- concatenate branches -> Conv1x1 projection

If ASPP causes memory issue, use:

```python
DepthwiseConv7x7 -> PointwiseConv1x1 -> BN -> ReLU
```

## Feature shape contract for HiCo

The following must be true:

```python
out["low_s"].shape == out["low_l"].shape == out["low_g"].shape
out["high_s"].shape == out["high_l"].shape == out["high_g"].shape
```

If global/local channels differ, add 1x1 adapters.

## Initialization

Use Kaiming normal for Conv2d and constant BN initialization.

## Unit test

```python
from networks.unet_lgdt import UNet_LGDT
import torch

model = UNet_LGDT(in_chns=1, class_num=4).cuda()
x = torch.randn(2, 1, 256, 256).cuda()

out = model(x, return_all=True)
assert out["logits_s"].shape == (2, 4, 256, 256)
assert out["logits_l"].shape == (2, 4, 256, 256)
assert out["logits_g"].shape == (2, 4, 256, 256)
assert out["low_s"].shape == out["low_l"].shape == out["low_g"].shape
assert out["high_s"].shape == out["high_l"].shape == out["high_g"].shape

logits, high, low = model(x)
assert logits.shape == (2, 4, 256, 256)
assert high.shape == out["high_s"].shape
assert low.shape == out["low_s"].shape
```

## Do not

- Do not instantiate teacher1/teacher2 full models.
- Do not use EMA in v1.
- Do not use evidential uncertainty.
- Do not modify dataset labels.
