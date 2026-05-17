# 07 - Hyperparameters and Ablations

## Default hyperparameters

```yaml
alpha_fast: 0.99
alpha_slow: 0.999
pseudo_warmup: 3000
uncertain_mode: quantile
uncertain_top_ratio: 0.35
teacher_conf_threshold: 0.55
easy_threshold: 0.2
use_boundary_prior: 1
boundary_lambda_image: 1.0
gamma_boundary: 0.5
gamma_disagree: 0.5
gamma_stable: 0.5
lambda_pseudo: 0.5
lambda_hico: 0.5
lambda_stable: 0.1
```

## Safer setting

Use if unstable:

```bash
--pseudo_warmup 5000 \
--teacher_conf_threshold 0.65 \
--uncertain_top_ratio 0.25 \
--lambda_pseudo 0.3 \
--lambda_hico 0.3
```

## More pseudo-label coverage

Use if pseudo ratio too low:

```bash
--pseudo_warmup 1500 \
--teacher_conf_threshold 0.45 \
--uncertain_top_ratio 0.5
```

## Ablations

### A0 Original SDT-Net

Run original `train_acdc.py`.

### A1 Single slow teacher

Always select `probs_slow`, `low_slow`, `high_slow`.

### A2 Single fast teacher

Always select `probs_fast`, `low_fast`, `high_fast`.

### A3 Same decay two teachers

```bash
--alpha_fast 0.999 --alpha_slow 0.999
```

### A4 No boundary prior

```bash
--use_boundary_prior 0
```

### A5 No temporal disagreement

```bash
--gamma_disagree 0 --gamma_stable 0
```

### A6 No HiCo

```bash
--lambda_hico 0
```

### A7 No stable slow loss

```bash
--lambda_stable 0
```

### A8 All unlabeled confident pixels instead of entropy uncertainty

Add flag later:

```bash
--use_uncertain_mask 0
```

Then use:

```python
pseudo_mask = (scrib.unsqueeze(1) == 4) & conf_mask
```

## Metrics to report

- Dice
- HD95
- pseudo ratio
- select fast ratio
- training time per iteration
- inference time with student only
