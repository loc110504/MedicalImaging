# 06 - Mamba-UNet Adaptation Specification

## Objective
Implement a 2D Mamba-UNet branch that can be trained in the existing ACDC/MSCMR 2D slice pipeline and can return both logits and intermediate features.

## Recommended Reference Repositories
Use the following codebases as implementation references:

1. `JCruan519/VM-UNet`
   - Official Vision Mamba UNet for medical image segmentation.
   - Key concepts: Visual State Space block, asymmetrical U-shaped encoder-decoder.

2. `ziyangwang007/Mamba-UNet`
   - Mamba-UNet for medical image segmentation.
   - Also references Semi-Mamba-UNet and Weak-Mamba-UNet.

3. `ziyangwang007/Weak-Mamba-UNet`
   - Directly relevant because it is scribble-based weakly supervised medical segmentation.
   - Its training idea uses collaborative learning, but QuanMambaScrib uses only two networks and QPIM verifier.

4. `openmedlab/Swin-UMamba`
   - Useful if ImageNet-pretrained Mamba-style backbones are desired later.
   - Not required for first implementation.

## Dependency Strategy
Try to use `mamba-ssm` only if compatible with the user's CUDA/PyTorch environment. If it is hard to install, use the VM-UNet implementation that vendors its required selective scan code or provide a fallback simplified VSS block.

Add to `requirements.txt` as optional:

```text
mamba-ssm
causal-conv1d
pennylane
pennylane-lightning
```

If install fails, document fallback:

```text
--mamba_variant simple_vss
```

## Architecture Requirements
Mamba branch must preserve U-Net-like dense prediction behavior:

```text
Input [B,1,256,256]
Encoder stages: 256->128->64->32->16 or similar
Mamba/VSS blocks at selected scales
Decoder stages with skip connections
Output logits [B,K,256,256]
Return feature [B,C,h,w]
```

## Minimal Adapter Contract

```python
class MambaUNet2D(nn.Module):
    def __init__(
        self,
        in_chns=1,
        class_num=4,
        img_size=256,
        base_channels=32,
        depths=(2,2,2,2),
        dims=(32,64,128,256),
        return_feature_stage="decoder_low",
    ):
        ...

    def forward(self, x, return_features=True):
        logits = ...
        feature = ...
        if return_features:
            return logits, feature
        return logits
```

## Feature Stage
Recommended:

```python
return_feature_stage = "decoder_low"
```

For `256x256` input, feature should be `32x32` or `64x64`, not full resolution. QPIM will upsample outputs back to full size.

## Implementation Choices

### Option A - Vendor VM-UNet
Copy required model files from VM-UNet into:

```text
code/networks/vmunet/
```

Then implement adapter in `mamba_unet_2d.py`.

Required edits:
- remove command-line/path code from vendored model files;
- expose `num_classes` argument;
- expose `in_channels` argument;
- return intermediate feature;
- keep all tensors on the input device;
- no hard-coded `.cuda()` in model code.

### Option B - Implement Lightweight Mamba-UNet
If vendoring is too complex, implement a lightweight VSS-inspired block:

```python
class SimpleMambaBlock(nn.Module):
    def __init__(self, dim):
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        # x [B,C,H,W]
        # flatten to [B,H*W,C], apply sequence mixing approximation, reshape
```

This is acceptable only as a temporary fallback. Final experiments should use a real Mamba/VMamba block if possible.

## Common Pitfalls
1. Mamba/VMamba code often assumes channel-last tensors `[B,H,W,C]`; existing U-Net code uses `[B,C,H,W]`. Convert carefully.
2. Some repos hard-code image size. Remove hard-coded assumptions.
3. Selective scan kernels may fail on CPU. Provide fallback or meaningful error.
4. Avoid using pretrained weights initially. Get the pipeline running from scratch.
5. Ensure `model.train()` and `model.eval()` work without changing QPIM memory incorrectly.

## Acceptance Tests
1. Forward pass with random input returns correct shape.
2. Backward pass works:

```python
x = torch.randn(2,1,256,256).cuda()
y = torch.randint(0,4,(2,256,256)).cuda()
logits, feat = mamba_unet(x, return_features=True)
loss = F.cross_entropy(logits, y)
loss.backward()
```

3. Feature map spatial size is no larger than 64x64 by default.
4. No hard-coded dataset or checkpoint path exists in model files.
