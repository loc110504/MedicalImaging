# 09 - Ablation Plan

The implementation should support these ablations via CLI flags or minimal code changes.

## A0: Current baseline

Original SDT-Net code:
- student + two full EMA teachers;
- global/batch teacher switching based on scribble CE;
- PRP;
- HiCo.

Purpose: compare with current implementation.

## A1: Shared encoder + student only

Disable teacher heads in loss:

```bash
--lambda_aux 0 \
--lambda_pseudo 0 \
--lambda_hico 0 \
--lambda_consensus 0
```

This gives pCE + pDice baseline using the student decoder.

## A2: Shared encoder + two teacher heads, no region switching

Use averaged teacher pseudo target:

\[
\hat{p} = 0.5p_l + 0.5p_g
\]

Purpose: show region-wise teacher switching is better than averaging.

Implementation flag:

```bash
--switch_mode average
```

## A3: Entropy-only switching without boundary prior

Set:

```bash
--boundary_gamma 0
```

Then selection depends only on teacher entropy:

\[
R_t = 1-H(p_t)
\]

Purpose: show boundary-aware local/global prior improves boundary metrics.

## A4: Full method

Default:

```bash
--uncertain_mode quantile \
--uncertain_top_ratio 0.35 \
--teacher_conf_threshold 0.55 \
--boundary_gamma 0.5
```

## A5: All reliable pixels instead of uncertain pixels

Use pseudo-labels for all unlabeled pixels passing teacher confidence:

```bash
--use_uncertain_mask 0
```

Purpose: show focusing on student uncertain regions is better than supervising everywhere.

## A6: No HiCo

```bash
--lambda_hico 0
```

Purpose: verify hierarchical feature alignment contributes beyond pseudo-labeling.

## A7: No consensus

```bash
--lambda_consensus 0
```

Purpose: check whether teacher heads drift without easy-region consistency.

## A8: Different uncertain ratios

Run:
- `top_ratio = 0.2`
- `top_ratio = 0.35`
- `top_ratio = 0.5`

Expected:
- too low: not enough pseudo supervision;
- too high: noisy pseudo supervision;
- default 0.35 likely stable.

## A9: Different teacher confidence thresholds

Run:
- `0.45`
- `0.55`
- `0.65`

Expected:
- lower threshold increases coverage but more noise;
- higher threshold increases precision but lower coverage.

## Metrics

Report:
- Dice
- HD95
- optionally ASD
- boundary F1 if implemented

Also log efficiency:
- number of parameters;
- training iteration time;
- inference FPS or ms per slice.

Important claim:
- Full method should train much cheaper than two full teachers and infer with only student decoder.
