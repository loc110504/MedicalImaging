# 01 - Codebase Contract

## Dataset

Use the current loader unchanged:

```python
db_train = ACDCDataSets(
    base_dir=args.root_path,
    split="train",
    transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    fold=args.fold,
    sup_type=args.sup_type,
)
```

For `sup_type='scribble'`, the loader reads:

```python
label = h5f['scribble'][:]
```

Training batch:

```python
image: Tensor[B, 1, 256, 256], float32
scrib: Tensor[B, 256, 256], long
```

## Label semantics

```text
0,1,2,3 = valid segmentation classes
4 = ignore / unlabeled
```

Must use:

```python
ce_loss = CrossEntropyLoss(ignore_index=4)
dice_loss = losses.pDLoss(num_classes, ignore_index=4)
```

## Model

Use current `UNet_HL` through:

```bash
--model unet_hl
```

Expected output:

```python
logits, high, low = model(image)
```

The fast and slow teachers use the exact same architecture.

## Validation

Validation should use only the student:

```python
metric_i = test_single_volume(sampled_batch['image'], sampled_batch['label'], model, classes=num_classes)
```

If `val.py` cannot handle tuple output, patch it minimally:

```python
outputs = net(input)
if isinstance(outputs, dict):
    outputs = outputs['logits_s']
elif isinstance(outputs, (tuple, list)):
    outputs = outputs[0]
```

## Debug assertions

Add for first iterations:

```python
assert image.dim() == 4 and image.size(1) == 1
assert scrib.dim() == 3
assert image.shape[-2:] == scrib.shape[-2:]
assert scrib.max().item() <= 4
assert scrib.min().item() >= 0
```
