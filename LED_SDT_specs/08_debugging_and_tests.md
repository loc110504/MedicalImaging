# 08 - Debugging and Tests

## 1. Shape test

Run before training:

```bash
python -m tests.test_lgdt_shapes
```

If no tests folder exists, create a temporary script:

```python
import torch
from networks.net_factory import net_factory

model = net_factory("unet_lgdt", in_chns=1, class_num=4)
x = torch.randn(2, 1, 256, 256).cuda()

out = model(x, return_all=True)
for k, v in out.items():
    print(k, v.shape)

assert out["logits_s"].shape == (2, 4, 256, 256)
assert out["logits_l"].shape == (2, 4, 256, 256)
assert out["logits_g"].shape == (2, 4, 256, 256)
assert out["low_s"].shape == out["low_l"].shape == out["low_g"].shape
assert out["high_s"].shape == out["high_l"].shape == out["high_g"].shape

y = model(x)
assert isinstance(y, tuple)
assert y[0].shape == (2, 4, 256, 256)
```

## 2. Utility test

```python
from utils.entropy_utils import normalized_entropy, build_uncertain_mask
from utils.boundary_utils import boundary_likelihood
from utils.region_switch import region_wise_teacher_selection

B, C, H, W = 2, 4, 256, 256
logits = torch.randn(B, C, H, W).cuda()
probs = torch.softmax(logits, dim=1)
scrib = torch.full((B, H, W), 4, dtype=torch.long).cuda()
image = torch.randn(B, 1, H, W).cuda()

ent = normalized_entropy(probs)
assert ent.shape == (B,1,H,W)
assert ent.min() >= 0 and ent.max() <= 1

umask = build_uncertain_mask(ent, scrib, mode="quantile", top_ratio=0.35)
assert umask.dtype == torch.bool

bd = boundary_likelihood(image)
assert bd.shape == (B,1,H,W)
assert bd.min() >= 0 and bd.max() <= 1

out = region_wise_teacher_selection(probs, probs, bd)
assert out["selected_probs"].shape == (B,C,H,W)
```

## 3. One-batch overfit test

Run training for 20 iterations on one batch:
- temporarily set dataset to use first batch repeatedly or break after first batch;
- check loss decreases;
- check no NaN.

Expected:
- `loss_scrib` should be finite immediately;
- `loss_pseudo` should be 0 before warmup;
- `uncertain_ratio` should be close to `uncertain_top_ratio * unlabeled_ratio`;
- `pseudo_ratio` should be lower than `uncertain_ratio`.

## 4. 200-iteration smoke test

Command:

```bash
python train_acdc_lgdt.py \
  --root_path ../../data/ACDC \
  --exp LED_SDT_smoke \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_lgdt \
  --num_classes 4 \
  --max_iterations 200 \
  --batch_size 2 \
  --pseudo_warmup 50 \
  --debug_shapes \
  --gpu 0
```

Success criteria:
- no crash;
- TensorBoard logs created;
- checkpoint directory created;
- validation either skipped or runs if iteration hits validation interval;
- `pseudo_ratio` is not always zero after warmup.

## 5. Common bugs

### Bug: `pDLoss` crashes with label value 4

Fix:
- only use CE for `loss_scrib`;
- keep `pDLoss` for pseudo labels only;
- or verify `losses.pDLoss(num_classes, ignore_index=4)` handles ignore.

### Bug: feature channels mismatch

Fix:
- add 1x1 adapters in `UNet_LGDT`;
- ensure all low features have same channels;
- ensure all high features have same channels.

### Bug: no pseudo pixels selected

Possible causes:
- `teacher_conf_threshold` too high;
- `uncertain_top_ratio` too low;
- entropy threshold mode too strict.

Fix:
- lower `teacher_conf_threshold` from `0.55` to `0.45`;
- increase `uncertain_top_ratio` to `0.5`;
- reduce `pseudo_warmup`.

### Bug: local teacher selected everywhere

Possible causes:
- boundary prior too strong;
- image gradient noisy.

Fix:
- lower `boundary_gamma` from `0.5` to `0.2`;
- set `boundary_lambda_image=0.5` and include student probability gradient;
- log `boundary_mean`.

### Bug: global teacher selected everywhere

Possible causes:
- boundary map too weak or all zeros.

Fix:
- check Sobel normalization;
- visualize boundary maps;
- increase `boundary_gamma`.

### Bug: validation cannot handle tuple output

Patch `val.py`:

```python
outputs = net(input)
if isinstance(outputs, dict):
    outputs = outputs["logits_s"]
elif isinstance(outputs, (tuple, list)):
    outputs = outputs[0]
```

## 6. Visualization debug

Every 1000 iterations, optionally save:
- image;
- scribble;
- entropy map;
- boundary map;
- select local mask;
- pseudo mask;
- student prediction;
- local teacher prediction;
- global teacher prediction.

Create utility later:

```text
utils/visualize_lgdt.py
```

Not required for v1.
