# 01 - Data Contract for ACDC

The current `dataloader/acdc.py` should remain unchanged for the first implementation.

## Dataset object

```python
db_train = ACDCDataSets(
    base_dir=args.root_path,
    split="train",
    transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    fold=args.fold,
    sup_type=args.sup_type
)
```

## Expected training sample

Each training batch from `trainloader` contains:

```python
sampled = {
    "image": Tensor[B, 1, H, W], dtype=torch.float32,
    "label": Tensor[B, H, W], dtype=torch.uint8 or torch.long,
    "idx": int or Tensor[B]
}
```

After `.cuda()` and `.long()`:

```python
image: Tensor[B, 1, 256, 256], float32
scrib: Tensor[B, 256, 256], long
```

## Label semantics

For ACDC scribble supervision:

```text
0 = background scribble
1 = class 1
2 = class 2
3 = class 3
4 = unlabeled / ignore
```

The training code must use:

```python
ignore_index = 4
num_classes = 4
ce_loss = CrossEntropyLoss(ignore_index=4)
dice_loss = losses.pDLoss(num_classes, ignore_index=4)
```

## Important rules

1. Do not change the dataloader for v1.
2. Do not convert unlabeled `4` to background.
3. All pseudo-label loss must only apply to unlabeled pixels:
   ```python
   unlabeled_mask = (scrib == 4)
   ```
4. Partial CE over scribbles is already handled by `ignore_index=4`.
5. Validation uses dense labels from `ACDC_training_volumes`, not scribbles.
6. The new model must remain compatible with `test_single_volume`.

## Transform behavior

`RandomGenerator` applies:
- random rotation/flip;
- possible random rotate with `cval=4` if label contains `4`;
- zoom to output size;
- image -> `[1, H, W]`;
- label -> `[H, W]`.

This means rotated background outside scribble may be `4`, not `0`, when training with scribbles.

## Debug assertions

Add these assertions in the training script for the first 2 iterations:

```python
assert image.dim() == 4 and image.size(1) == 1
assert scrib.dim() == 3
assert image.shape[-2:] == scrib.shape[-2:]
assert scrib.max().item() <= 4
assert scrib.min().item() >= 0
```

Log unique labels occasionally:

```python
if iter_num < 3:
    logging.info(f"scribble unique labels: {torch.unique(scrib).detach().cpu().tolist()}")
```
