# 06 - Debugging and Tests

## Debug script

Create:

```text
debug_tema_one_batch.py
```

It should:

1. Load one ACDC training batch.
2. Instantiate student, fast teacher, slow teacher.
3. Copy student weights to teachers.
4. Forward all three networks.
5. Compute entropy, uncertain mask, disagreement, boundary, switch mask, pseudo mask.
6. Compute losses.
7. Run one optimizer step.
8. Update both EMA teachers.
9. Print all shapes and ratios.

## Required printout

```text
image shape
scrib shape
unique scribble labels
student logits shape
high/low shapes
entropy_student mean/min/max
entropy_fast mean
entropy_slow mean
disagreement mean
boundary mean
uncertain_ratio
pseudo_ratio
select_fast_ratio
easy_ratio
loss_scrib
loss_pseudo
loss_hico
loss_stable
```

## Smoke command

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

## Common bugs

### `pseudo_ratio` stays zero

Try:

```bash
--teacher_conf_threshold 0.45 --uncertain_top_ratio 0.5
```

### `select_fast_ratio` always 1

Try:

```bash
--gamma_boundary 0.2 --gamma_disagree 0.2 --gamma_stable 1.0
```

### `select_fast_ratio` always 0

Try:

```bash
--gamma_boundary 0.8 --gamma_disagree 0.8
```

### `pDLoss` crashes with ignore index

Temporarily set:

```python
loss_scrib = loss_ce_stu
```

### Validation fails on tuple output

Patch `val.py` to use first output if model returns tuple.
