# LED-SDT Implementation Specs

**Method name:** LED-SDT / LGDT-Net  
**Full name:** Lightweight Entropy-guided Dual-Teacher Head Network for Scribble-supervised Medical Image Segmentation.

This spec package is written for a coding agent that will modify the current SDT-Net-style ACDC codebase.

The current codebase has:
- `dataloader/acdc.py`
- `networks/net_factory.py`
- `train_acdc.py`
- `networks/unet.py` with `UNet` and `UNet_HL`
- `utils/losses.py`
- `utils/pick_reliable_pixels.py`
- `utils/ema_optim.py`
- `val.py`

The current SDT-Net implementation uses:
- one student model;
- two full EMA teachers;
- batch-level teacher selection based on partial CE over scribble pixels;
- Pick Reliable Pixels based on confidence threshold;
- Hierarchical Consistency with `high` and `low` features returned by `UNet_HL`.

The proposed implementation removes the two full teacher networks and replaces them with one efficient shared model:

```text
Input image x
  |
Shared Encoder E
  |
  +-- Student Decoder D_s  -> logits_s, high_s, low_s
  +-- Local Teacher Head D_l -> logits_l, high_l, low_l
  +-- Global Teacher Head D_g -> logits_g, high_g, low_g
```

Only the student decoder is used at inference time. Teacher heads are used during training only.

## Core changes

1. Add a new network:
   - `networks/unet_lgdt.py`
   - class: `UNet_LGDT`

2. Update factory:
   - `networks/net_factory.py`
   - add `net_type == "unet_lgdt"`

3. Add utility modules:
   - `utils/entropy_utils.py`
   - `utils/boundary_utils.py`
   - `utils/region_switch.py`
   - `utils/weighted_losses.py`

4. Add new training script:
   - `train_acdc_lgdt.py`

5. Keep dataloader unchanged:
   - ACDC image shape after transform: `[B, 1, 256, 256]`
   - scribble label shape: `[B, 256, 256]`
   - classes: `0,1,2,3`
   - unlabeled/ignore pixels: `4`
   - `num_classes = 4`
   - `ignore_index = 4`

## Default training command

```bash
python train_acdc_lgdt.py \
  --root_path ../../data/ACDC \
  --exp LED_SDT \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_lgdt \
  --num_classes 4 \
  --max_iterations 30000 \
  --batch_size 8 \
  --base_lr 0.01 \
  --patch_size 256 256 \
  --gpu 0 \
  --uncertain_mode quantile \
  --uncertain_top_ratio 0.35 \
  --teacher_conf_threshold 0.55 \
  --lambda_pseudo 0.5 \
  --lambda_hico 0.5 \
  --lambda_aux 0.4 \
  --lambda_consensus 0.1 \
  --lambda_div 0.0
```

## Recommended implementation order

1. Implement utility functions and unit tests first.
2. Implement `UNet_LGDT`.
3. Update `net_factory.py`.
4. Implement `train_acdc_lgdt.py`.
5. Run shape/debug test for 1 batch.
6. Run 200 iterations smoke training.
7. Run full training.
8. Run ablations.
