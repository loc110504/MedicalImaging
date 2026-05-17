# 14 - Expected Final Code Deliverables

After the coding agent finishes, the repo should contain:

```text
dataloader/
  acdc.py                         # unchanged

networks/
  unet.py                         # unchanged unless reusable blocks are exposed
  unet_lgdt.py                    # new
  net_factory.py                  # modified

utils/
  entropy_utils.py                # new
  boundary_utils.py               # new
  region_switch.py                # new
  weighted_losses.py              # new
  losses.py                       # unchanged
  ramps.py                        # unchanged
  pick_reliable_pixels.py         # unchanged, not used in v1
  ema_optim.py                    # unchanged, not used in v1

train_acdc.py                     # unchanged
train_acdc_lgdt.py                # new

debug_lgdt_one_batch.py           # new, optional but recommended
```

## Minimal success run

```bash
python debug_lgdt_one_batch.py --root_path ../../data/ACDC --gpu 0
```

Then:

```bash
python train_acdc_lgdt.py \
  --root_path ../../data/ACDC \
  --exp LED_SDT_smoke \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_lgdt \
  --num_classes 4 \
  --max_iterations 200 \
  --batch_size 2 \
  --pseudo_warmup 50 \
  --gpu 0
```

Then full:

```bash
python train_acdc_lgdt.py \
  --root_path ../../data/ACDC \
  --exp LED_SDT \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_lgdt \
  --num_classes 4 \
  --max_iterations 30000 \
  --batch_size 8 \
  --gpu 0
```

## What the agent must not do

- Do not use evidential uncertainty.
- Do not instantiate two full teacher networks.
- Do not alter ACDC label conventions.
- Do not treat unlabeled label `4` as background.
- Do not compute pseudo-label loss on scribble pixels in v1.
- Do not remove the original SDT-Net script.
- Do not break old `unet` and `unet_hl` factory options.
