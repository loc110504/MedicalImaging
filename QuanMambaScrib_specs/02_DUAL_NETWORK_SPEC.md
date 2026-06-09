# 02 - Dual Network Specification: U-Net + Mamba-UNet

## Purpose
Implement two independent segmentation branches with different inductive biases:

1. U-Net: local convolutional view.
2. Mamba-UNet: long-range state-space view.

The two branches must not share an encoder. Co-training relies on their prediction diversity.

## U-Net Branch Requirements
Use the existing U-Net implementation if possible through `net_factory(net_type="unet_hl")`, but it must expose an intermediate feature map.

### Preferred Option: Modify Existing U-Net
Edit the U-Net class forward to support:

```python
def forward(self, x, return_features=False):
    ...
    if return_features:
        return logits, feature
    return logits
```

Feature should come from one of:
- final decoder feature before classifier, shape `[B, C, H, W]`; or
- bottleneck/low-resolution decoder feature, shape `[B, C, h, w]`.

For QPIM speed, low-resolution feature is preferred, e.g. `[B, 128, 32, 32]` for input `[B, 1, 256, 256]`.

### Alternative Option: Forward Hook
If changing U-Net source is risky, implement `FeatureHookWrapper`:

```python
class FeatureHookWrapper(nn.Module):
    def __init__(self, base_model, feature_layer_name):
        ...
    def forward(self, x):
        logits = self.base_model(x)
        return logits, self.cached_feature
```

Avoid this if layer names are unstable.

## Mamba-UNet Branch Requirements
Create:

```python
class MambaUNet2D(nn.Module):
    def __init__(
        self,
        in_chns: int,
        class_num: int,
        img_size: int = 256,
        base_channels: int = 32,
        return_feature_stage: str = "decoder_low",
        **kwargs,
    ):
        ...

    def forward(self, x, return_features=True):
        ...
        if return_features:
            return logits, feature
        return logits
```

Output:

```text
logits:  [B, K, H, W]
feature: [B, C_m, h, w]
```

## Recommended Source Repos for Mamba-UNet
Use these as references when implementing/adapting the Mamba branch:

1. **VM-UNet** official repo
   - Useful for Visual State Space blocks and U-shaped medical segmentation design.
   - Reference class names: VMUNet, VSSBlock, VSSLayer, SS2D.

2. **Mamba-UNet** repo
   - Useful for a pure visual Mamba U-Net architecture and medical segmentation training conventions.
   - Also links related Semi-Mamba-UNet and Weak-Mamba-UNet work.

3. **Weak-Mamba-UNet** repo
   - Useful because it is directly scribble-supervised and uses CNN/ViT/Mamba collaborative learning.
   - Do not copy the three-network setting; adapt only Mamba-UNet implementation details and training conventions.

4. **Semi-Mamba-UNet** paper/repo patterns
   - Useful for pixel-level cross-supervised U-Net + Mamba-UNet logic.

## Vendor Strategy
If using public code, copy only the required model files under:

```text
code/networks/vmunet/
```

Then create a clean adapter:

```python
# code/networks/mamba_unet_2d.py
from networks.vmunet.vmunet import VMUNet

class MambaUNet2D(nn.Module):
    ...
```

Do not let external code hard-code:
- number of classes;
- image size;
- dataset path;
- CUDA device;
- checkpoint path;
- training loop.

## Feature Extraction from Mamba-UNet
The Mamba branch must return a feature map for QPIM. Choose one:

1. **Decoder-low feature**: lower memory, better for QPIM speed.
2. **Final decoder feature**: better spatial detail, more expensive.

Recommended default:

```text
return_feature_stage = "decoder_low"
feature stride = 4 or 8
```

For input 256x256:

```text
preferred QPIM feature sizes:
16x16, 32x32, or 64x64 max
avoid full 256x256 for quantum kernels
```

## Wrapper Model
Create:

```python
class QuanMambaScrib(nn.Module):
    def __init__(
        self,
        in_chns,
        class_num,
        unet_type="unet_hl",
        mamba_variant="vmunet",
        qpim_backend="torch_angle_fidelity",
        qpim_dim=8,
        qpim_tau=0.5,
        ignore_index=None,
    ):
        ...
```

Forward:

```python
def forward(self, x, scribble_label=None, update_memory=True, return_q=True):
    logits_u, feat_u = self.unet(x, return_features=True)
    logits_m, feat_m = self.mamba_unet(x, return_features=True)
    prob_u = torch.softmax(logits_u, dim=1)
    prob_m = torch.softmax(logits_m, dim=1)

    output = {...}

    if return_q and scribble_label is not None:
        q_out = self.qpim(feat_u, feat_m, scribble_label, target_size=x.shape[-2:], update_memory=update_memory)
        output.update(q_out)

    return output
```

## Shape Acceptance Test
Create a local test script or function:

```python
x = torch.randn(2, 1, 256, 256).cuda()
y = torch.randint(0, 5, (2, 256, 256)).cuda()  # 5 means ignore if K=4
model = QuanMambaScrib(in_chns=1, class_num=4).cuda()
out = model(x, scribble_label=y)
assert out["logits_u"].shape == (2, 4, 256, 256)
assert out["logits_m"].shape == (2, 4, 256, 256)
assert out["Q_u"].shape == (2, 4, 256, 256)
assert out["Q_m"].shape == (2, 4, 256, 256)
```
