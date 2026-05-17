# 00 - Method Overview

## Goal

Train dense medical segmentation from sparse scribble labels on ACDC.

Current ACDC scribble labels:

```text
0 = background scribble
1 = class 1
2 = class 2
3 = class 3
4 = unlabeled / ignore
```

## Main idea

Use two EMA teachers from the same student:

- `T_fast`: lower EMA decay, tracks current student quickly.
- `T_slow`: higher EMA decay, provides stable long-term target.

Student entropy identifies uncertain unlabeled regions. Then a region-wise arbitration module selects the teacher per pixel/region.

## Student entropy

For student probability `p_s = softmax(z_s)`:

```math
U_s(i) = -1/log(C) * sum_c p_s^c(i) log(p_s^c(i) + eps)
```

For ACDC, `C=4`.

## Uncertain region

Use only unlabeled pixels:

```python
unlabeled = (scrib == 4).unsqueeze(1)
```

Default quantile mode:

```math
Omega_u = top-ratio entropy pixels within unlabeled region
```

Default:

```text
uncertain_top_ratio = 0.35
```

## Teacher reliability

Teacher probabilities:

```python
p_fast = softmax(z_fast)
p_slow = softmax(z_slow)
```

Teacher entropies:

```math
H_fast = entropy(p_fast)
H_slow = entropy(p_slow)
```

Fast-slow disagreement:

```math
D_FS = JS(p_fast, p_slow)
```

Optional boundary likelihood:

```math
B = normalize(Sobel(image))
```

Reliability:

```math
R_fast = (1-H_fast) * (1 + gamma_boundary * B) * (1 + gamma_disagree * D_FS)
R_slow = (1-H_slow) * (1 + gamma_boundary * (1-B)) * (1 + gamma_stable * (1-D_FS))
```

Interpretation:

- Fast teacher is favored near boundaries and high temporal disagreement.
- Slow teacher is favored in interior/stable regions.

## Region-wise arbitration

```python
select_fast = R_fast > R_slow
selected_probs = where(select_fast, p_fast, p_slow)
selected_weight = where(select_fast, R_fast, R_slow)
```

Pseudo mask:

```python
conf_mask = selected_probs.max(dim=1, keepdim=True)[0] >= teacher_conf_threshold
pseudo_mask = uncertain_mask & conf_mask
```

## Losses

```math
L = L_scrib + lambda_pseudo L_pseudo + lambda_hico L_hico + lambda_stable L_stable
```

### Scribble loss

```math
L_scrib = CE(z_s, scrib) + pDLoss(p_s, scrib)
```

Use `ignore_index=4`.

### Soft pseudo-label loss

```math
L_pseudo = - sum_i M_i w_i sum_c selected_probs^c_i log p_s^c_i / (sum_i M_i + eps)
```

### Region-wise HiCo

Use the selected teacher's low/high features:

```python
selected_low = where(resize(select_fast), low_fast, low_slow)
selected_high = where(resize(select_fast), high_fast, high_slow)
```

Feature loss = average of L1 and cosine distance, weighted by `pseudo_mask`.

### Stable slow-teacher loss

For easy unlabeled pixels:

```python
easy_mask = (entropy_student < easy_threshold) & (scrib.unsqueeze(1) == 4)
```

Use slow teacher soft target on this easy region.

## EMA updates

After every optimizer step:

```math
theta_fast = alpha_fast theta_fast + (1-alpha_fast) theta_student
theta_slow = alpha_slow theta_slow + (1-alpha_slow) theta_student
```

Default:

```text
alpha_fast = 0.99
alpha_slow = 0.999
```
