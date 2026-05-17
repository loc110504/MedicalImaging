# 10 - Agent Task Breakdown

This is the exact task list for the coding agent.

## Task 1: Implement utilities

Create:
- `utils/entropy_utils.py`
- `utils/boundary_utils.py`
- `utils/region_switch.py`
- `utils/weighted_losses.py`

Acceptance criteria:
- all functions import successfully;
- synthetic unit tests pass;
- no CPU/GPU device mismatch;
- no NaN for random inputs.

## Task 2: Implement `UNet_LGDT`

Create:
- `networks/unet_lgdt.py`

Acceptance criteria:
- `model(x, return_all=True)` returns dict with all required keys;
- `model(x)` returns `(logits_s, high_s, low_s)`;
- output logits are full resolution;
- low/high feature shapes match across student/local/global;
- one forward pass works on `[2,1,256,256]`.

## Task 3: Update factory

Modify:
- `networks/net_factory.py`

Acceptance criteria:
- `net_factory("unet_lgdt", 1, 4)` returns CUDA model;
- invalid net type raises `ValueError`.

## Task 4: Implement new trainer

Create:
- `train_acdc_lgdt.py`

Base it on current `train_acdc.py`, but:
- use one model only;
- remove EMA teacher models;
- use region-wise teacher switching;
- use entropy uncertainty map;
- use pseudo warmup;
- preserve validation and checkpoint logic.

Acceptance criteria:
- smoke run for 200 iterations passes;
- TensorBoard logs include required loss/mask ratios;
- checkpoints save correctly.

## Task 5: Patch validation only if needed

If validation fails due to output type, patch `val.py` so it handles:
- dict output;
- tuple/list output;
- raw tensor output.

Acceptance criteria:
- old `unet_hl` still works;
- new `unet_lgdt` works.

## Task 6: Add debug script

Create:
- `debug_lgdt_one_batch.py`

It should:
- instantiate dataset and model;
- run one forward pass;
- compute all losses;
- print shapes, unique labels, masks ratios;
- exit without backprop.

Acceptance criteria:
- no exception;
- all printed shapes correct.

## Task 7: Full training

Run:

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
  --gpu 0
```

Acceptance criteria:
- best model checkpoint saved;
- validation Dice/HD95 logged;
- no NaN;
- pseudo ratio non-zero after warmup.

## Task 8: Ablations

Run A1-A7 from `09_ablation_plan.md`.

Acceptance criteria:
- each experiment has separate checkpoint dir;
- export final mean Dice/HD95 into a CSV.
