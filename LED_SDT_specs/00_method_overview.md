# 00 - Method Overview

## Problem

We train a segmentation model using scribble supervision only. In the ACDC loader, `scribble` labels use:
- `0`: background scribble pixel
- `1`: class 1
- `2`: class 2
- `3`: class 3
- `4`: unlabeled / ignored pixel

The model must learn dense segmentation despite sparse scribbles.

## Motivation

The current SDT-Net code uses two full EMA teacher networks. This is expensive because each iteration requires:
- student forward,
- teacher1 forward,
- teacher2 forward.

The proposed model keeps the teacher-student idea but makes it efficient:
- one shared encoder;
- one student decoder;
- two lightweight teacher heads.

This gives teacher diversity without training three full networks.

## Main idea

The student first predicts a probability map and entropy map. High-entropy pixels represent uncertain regions. These regions are usually:
- boundary pixels,
- ambiguous unlabeled pixels,
- noisy shape/interior regions.

At each uncertain pixel, the method asks:

> Should the student trust the local teacher or the global teacher?

The local teacher is biased toward boundary/detail. The global teacher is biased toward shape/context.

## Architecture

```text
x
|
Shared Encoder E
|
|-- Student Decoder D_s
|     output: logits_s, high_s, low_s
|
|-- Local Teacher Decoder D_l
|     output: logits_l, high_l, low_l
|     bias: high-resolution skip fusion, boundary/detail
|
|-- Global Teacher Decoder D_g
      output: logits_g, high_g, low_g
      bias: ASPP or large-kernel bottleneck context, global shape
```

## Entropy map

For student probability \(p_s\):

\[
U_s(i) = -\frac{1}{\log C}\sum_{c=1}^{C} p_s^c(i)\log(p_s^c(i)+\epsilon)
\]

where:
- \(C=4\) for ACDC;
- \(U_s(i)\in[0,1]\).

## Uncertain region

Default: top entropy pixels among unlabeled pixels.

\[
\Omega_u = \{i \mid U_s(i) \ge Q_{1-r}(U_s)\}
\]

where:
- \(r=\texttt{uncertain_top_ratio}\), default `0.35`;
- quantile is computed per image.

Only unlabeled pixels should be used for pseudo-labeling:

\[
\Omega_{unlab} = \{i \mid scrib(i)=4\}
\]

Final candidate region:

\[
\Omega_c = \Omega_u \cap \Omega_{unlab}
\]

## Boundary likelihood

Boundary likelihood is computed from image gradient and optionally student probability gradient.

Simple default:

\[
B(i)=\text{normalize}(\text{SobelMagnitude}(x_i))
\]

Optional extended version:

\[
B(i)=\lambda_b B_x(i)+(1-\lambda_b)B_p(i)
\]

where:
- \(B_x\): image Sobel magnitude;
- \(B_p\): Sobel magnitude over max student probability map.

## Teacher reliability

Local teacher reliability:

\[
R_l(i)=\left(1-H(p_l(i))\right)\left(1+\gamma B(i)\right)
\]

Global teacher reliability:

\[
R_g(i)=\left(1-H(p_g(i))\right)\left(1+\gamma(1-B(i))\right)
\]

where:
- \(H(p)\) is normalized entropy;
- \(B(i)\) is boundary likelihood;
- \(\gamma\) controls prior strength, default `0.5`.

## Teacher switching

For each pixel:

\[
t^*(i)=
\begin{cases}
l, & R_l(i) > R_g(i)\\
g, & otherwise
\end{cases}
\]

The selected soft pseudo-target is:

\[
\hat{p}(i)=
\begin{cases}
p_l(i), & t^*(i)=l\\
p_g(i), & t^*(i)=g
\end{cases}
\]

The target must be detached from gradient.

## Reliable pseudo-label mask

Pseudo-label is used only if selected teacher is confident enough:

\[
M_{pl}(i)=
\Omega_c(i)\land \max_c\hat{p}^c(i) \ge \tau_{conf}
\]

Default `teacher_conf_threshold = 0.55`.

## Loss

Total loss:

\[
L = L_{scrib} + \lambda_{aux}L_{aux} + \lambda_{pl}L_{pseudo}
+ \lambda_{hico}L_{hico} + \lambda_{cons}L_{consensus}
+ \lambda_{div}L_{div}
\]

Default:
- `lambda_pseudo = 0.5`
- `lambda_hico = 0.5`
- `lambda_aux = 0.4`
- `lambda_consensus = 0.1`
- `lambda_div = 0.0` initially disabled.

## Loss definitions

### Scribble loss

Use CE with `ignore_index=4` and existing partial Dice loss:

\[
L_{scrib} = CE(z_s, y_{scrib}) + Dice(p_s, y_{scrib})
\]

### Auxiliary teacher scribble loss

Teacher heads must learn from scribble labels:

\[
L_{aux} = \frac{1}{2}\left[
CE(z_l,y_{scrib}) + CE(z_g,y_{scrib})
\right]
\]

### Soft pseudo-label loss

Use weighted soft CE:

\[
L_{pseudo} =
-\frac{\sum_i w_i M_{pl}(i)\sum_c \hat{p}^c(i)\log p_s^c(i)}
{\sum_i M_{pl}(i)+\epsilon}
\]

where:
- \(w_i = \max(R_l(i), R_g(i))\), normalized or clamped;
- target \(\hat{p}\) is detached.

### Hierarchical consistency

Select teacher features region-wise:
- local feature where local teacher wins,
- global feature where global teacher wins.

For each feature level \(f\in\{low,high\}\):

\[
L_f = \frac{1}{2}\left[
L1(F_s^f, stopgrad(F_{t^*}^f))
+
1-\cos(F_s^f, stopgrad(F_{t^*}^f))
\right]
\]

Use resized `M_pl` as weight.

\[
L_{hico} = \frac{1}{2}(L_{low}+L_{high})
\]

### Teacher consensus loss

Encourage teacher heads to be consistent on easy pixels:

\[
\Omega_e = \{i \mid U_s(i)<\tau_e, scrib(i)=4\}
\]

\[
L_{consensus} = KL(stopgrad(p_l) || p_g) + KL(stopgrad(p_g) || p_l)
\]

Apply only to easy pixels. This prevents local/global heads from drifting too far.

### Diversity loss

Optional, disabled by default. Encourage mild diversity only on uncertain pixels:

\[
L_{div} = -\min(JS(p_l,p_g), m)
\]

Do not enable until baseline works. If enabled, use small weight `0.01`.
