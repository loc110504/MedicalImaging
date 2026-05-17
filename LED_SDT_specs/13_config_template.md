# 13 - Suggested Hyperparameter Defaults

Use these defaults for ACDC.

```yaml
model: unet_lgdt
num_classes: 4
ignore_index: 4

training:
  max_iterations: 30000
  batch_size: 8
  base_lr: 0.01
  optimizer: SGD
  momentum: 0.9
  weight_decay: 0.0001
  lr_schedule: poly
  power: 0.9
  pseudo_warmup: 3000
  consistency_rampup: 40

data:
  root_path: ../../data/ACDC
  fold: MAAGfold70
  sup_type: scribble
  patch_size: [256, 256]

uncertainty:
  type: entropy
  uncertain_mode: quantile
  uncertain_top_ratio: 0.35
  uncertain_threshold: 0.5

teacher_switch:
  teacher_conf_threshold: 0.55
  boundary_gamma: 0.5
  boundary_lambda_image: 1.0
  switch_mode: region

loss_weights:
  lambda_aux: 0.4
  lambda_pseudo: 0.5
  lambda_hico: 0.5
  lambda_consensus: 0.1
  lambda_div: 0.0

logging:
  val_interval: 400
  save_interval: 3000
```

## If training is unstable

Use safer settings:

```yaml
pseudo_warmup: 5000
teacher_conf_threshold: 0.65
uncertain_top_ratio: 0.25
lambda_pseudo: 0.3
lambda_hico: 0.3
```

## If pseudo supervision is too sparse

Use more permissive settings:

```yaml
teacher_conf_threshold: 0.45
uncertain_top_ratio: 0.5
pseudo_warmup: 1500
```

## If local teacher dominates

```yaml
boundary_gamma: 0.2
boundary_lambda_image: 0.5
```

## If global teacher dominates

```yaml
boundary_gamma: 0.8
boundary_lambda_image: 1.0
```
