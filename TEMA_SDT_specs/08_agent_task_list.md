# 08 - Agent Task List

## Task 1: Implement utilities

Create:

```text
utils/tema_entropy.py
utils/tema_boundary.py
utils/tema_region_switch.py
utils/tema_losses.py
utils/tema_ema.py
```

Acceptance:

- imports work;
- synthetic tensor test passes;
- no CPU/GPU mismatch;
- empty masks return zero graph loss;
- entropy/disagreement/boundary maps are `[B,1,H,W]`.

## Task 2: Implement training script

Create:

```text
train_acdc_tema_sdt.py
```

Acceptance:

- based on current `train_acdc.py`;
- creates student, fast teacher, slow teacher;
- copies student weights into teachers;
- does not use `WeightEMA`;
- does not use `refine_high_confidence`;
- updates both teachers every iteration;
- logs required scalars;
- validates student only.

## Task 3: Implement debug script

Create:

```text
debug_tema_one_batch.py
```

Acceptance:

- runs one batch;
- prints shapes and ratios;
- computes all losses;
- runs one backward step;
- updates both EMA teachers.

## Task 4: Run smoke training

```bash
python train_acdc_tema_sdt.py \
  --root_path ../../data/ACDC \
  --exp TEMA_SDT_smoke \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 200 \
  --batch_size 2 \
  --pseudo_warmup 50 \
  --gpu 0 \
  --debug_shapes
```

Acceptance:

- no crash;
- TensorBoard logs saved;
- pseudo ratio non-zero after warmup or explained by confidence threshold.

## Task 5: Full training

Run 30k iterations and save best checkpoint.

## Task 6: Ablations

Run ablations from `07_hyperparameters_and_ablations.md` and summarize Dice/HD95.
