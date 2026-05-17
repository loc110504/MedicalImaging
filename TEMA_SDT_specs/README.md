# TEMA-SDT Implementation Specs

**Method:** TEMA-SDT — Two-timescale EMA Dynamic Teacher Switching for Scribble-supervised Medical Image Segmentation.

This package is designed for the current SDT-Net-style codebase you provided:

```text
dataloader/acdc.py
networks/net_factory.py
train_acdc.py
networks/unet.py     # contains UNet and UNet_HL
utils/losses.py
utils/pick_reliable_pixels.py
utils/ema_optim.py
utils/ramps.py
val.py
```

TEMA-SDT keeps the current **student + two EMA teachers** structure, but changes the teacher roles and switching logic:

```text
Student S_theta
Fast EMA Teacher T_fast: alpha_fast = 0.99
Slow EMA Teacher T_slow: alpha_slow = 0.999

Student entropy map -> uncertain unlabeled region
Fast/slow teacher entropy + fast-slow disagreement + optional boundary prior
-> region-wise teacher arbitration
-> selected soft teacher target
-> student pseudo-label loss + region-wise HiCo
```

## Critical constraints

- Do not change `dataloader/acdc.py`.
- Do not change label convention: `0,1,2,3` are classes, `4` is unlabeled/ignore.
- Do not use evidential/Dirichlet uncertainty.
- Do not train teachers by gradient.
- Do not update only the selected teacher. Update **both** teachers every iteration with different EMA decay.
- Do not overwrite the original `train_acdc.py`; create `train_acdc_tema_sdt.py`.
- Inference/validation uses the student model only.

## Files to create

```text
utils/tema_entropy.py
utils/tema_boundary.py
utils/tema_region_switch.py
utils/tema_losses.py
utils/tema_ema.py
train_acdc_tema_sdt.py
debug_tema_one_batch.py
```

## Default command

```bash
python train_acdc_tema_sdt.py \
  --root_path ../../data/ACDC \
  --exp TEMA_SDT \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 30000 \
  --batch_size 8 \
  --base_lr 0.01 \
  --patch_size 256 256 \
  --gpu 0 \
  --alpha_fast 0.99 \
  --alpha_slow 0.999 \
  --pseudo_warmup 3000 \
  --uncertain_mode quantile \
  --uncertain_top_ratio 0.35 \
  --teacher_conf_threshold 0.55 \
  --lambda_pseudo 0.5 \
  --lambda_hico 0.5 \
  --lambda_stable 0.1
```
