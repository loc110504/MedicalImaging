# RPG-MT SPEC: Reliable Pseudo-label Graph-guided Mean Teacher for Scribble-supervised Medical Image Segmentation

## 0. Purpose

This spec describes how to implement the proposed method:

**RPG-MT: Reliable Pseudo-label Graph-guided Mean Teacher for Scribble-supervised Medical Image Segmentation**

The method extends the current Mean Teacher scribble-supervised baseline:

```text
Student + EMA Teacher
LpCE on scribble pixels
+
confidence-based agreement/disagreement pseudo-label loss
```

into:

```text
Student + EMA Teacher
        ↓
Confidence Agreement–Disagreement Reliable Seed Selection
        ↓
Graph-Harmonic Reliable Pseudo-label Diffusion
        ↓
Scribble-Reliable Prototype Margin Learning
        ↓
L_total = L_pCE + λ_pseudo L_GHRD + λ_proto L_proto
```

The implementation must be compatible with the current ACDC scribble training code style:

```python
model = create_model(ema=False, num_classes=num_classes)
model_ema = create_model(ema=True, num_classes=num_classes)

ce_loss = CrossEntropyLoss(ignore_index=num_classes)
ema_optimizer = WeightEMA(model, model_ema, alpha=0.99)
```

---

# 1. Literature-grounded motivation

Recent semi-supervised and scribble-supervised medical segmentation papers motivate three directions that RPG-MT combines:

1. **Reliable pseudo-label region selection**  
   Mean Teacher pseudo-labels are useful but noisy, so recent methods partition pseudo-labels into reliable/unreliable or consistent/inconsistent regions rather than using all pixels equally.

2. **Scribble-specific feature discrimination**  
   Scribble labels are sparse. Output-level pseudo-supervision is not enough; the model also needs class-discriminative foreground/background or anatomy-aware feature learning.

3. **Graph/manifold label propagation**  
   Graph-based semi-supervised learning assumes that nearby samples on the data manifold should have similar labels. RPG-MT adapts this classical idea to dense medical segmentation by treating reliable pseudo-label pixels as graph anchors.

The core novelty of RPG-MT:

```text
Reliable pseudo-label pixels are not used merely as sparse output targets.
They are treated as graph anchors for pseudo-label propagation and as prototype anchors for representation learning.
```

---

# 2. Notation

For one training mini-batch:

```text
x ∈ R^{B×1×H×W}
    input image batch

y ∈ {0, ..., C-1, ignore_index}^{B×H×W}
    scribble label batch

C
    number of segmentation classes

ignore_index = C
    unlabeled / ignored scribble pixels

Ω_l = {i | y_i ≠ ignore_index}
    scribble-labeled pixel set

Ω_u = {i | y_i = ignore_index}
    unlabeled pixel set
```

Student network:

```text
f_θ(x) → (s, z)

s ∈ R^{B×C×H×W}
    student logits

p_s = softmax(s)
    student probability map

z ∈ R^{B×D×h×w}
    student feature map for graph/prototype learning
```

EMA teacher network:

```text
f_{θ'}(x) → (t, z_t)

t ∈ R^{B×C×H×W}
    EMA teacher logits

p_t = softmax(t)
    teacher probability map
```

EMA update:

```text
θ' ← α θ' + (1 - α) θ
```

where `α = 0.99` or `0.999`.

---

# 3. Final objective

The full method optimizes:

```text
L_total = L_pCE + λ_pseudo(t) L_GHRD + λ_proto(t) L_proto
```

where:

```text
L_pCE
    partial cross-entropy on scribble pixels.

L_GHRD
    graph-harmonic reliable pseudo-label diffusion loss.

L_proto
    scribble-reliable prototype contrastive / margin loss.

λ_pseudo(t)
    ramped pseudo-label weight.

λ_proto(t)
    ramped prototype weight.
```

Recommended ramp-up:

```text
λ_pseudo(t) = λ_pseudo × sigmoid_rampup(t, T_ramp)
λ_proto(t)  = λ_proto  × sigmoid_rampup(t, T_ramp)
```

Recommended defaults:

```text
λ_pseudo = 8.0
λ_proto = 0.1 or 0.2
T_ramp = 40 epochs
```

Minimal version:

```text
L_total = L_pCE + λ_pseudo(t) L_GHRD
```

Full Q1-target version:

```text
L_total = L_pCE + λ_pseudo(t) L_GHRD + λ_proto(t) L_proto
```

---

# 4. Required new arguments

Add these arguments to `argparse`.

```python
# Reliable pseudo-label fusion
parser.add_argument('--pseudo_agree_thresh', type=float, default=0.7,
                    help='confidence threshold for agreement pixels')

parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.8,
                    help='confidence threshold for stronger prediction in disagreement pixels')

parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='minimum confidence margin in disagreement pixels')

parser.add_argument('--pseudo_loss_weight', type=float, default=8.0,
                    help='weight for graph-diffused pseudo-label loss')

parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'all'],
                    help='candidate region for pseudo-label selection')

# Graph diffusion
parser.add_argument('--use_graph_diffusion', type=int, default=1,
                    help='enable graph-harmonic pseudo-label diffusion')

parser.add_argument('--graph_diffusion_iters', type=int, default=3,
                    help='number of local graph diffusion iterations')

parser.add_argument('--graph_alpha', type=float, default=0.7,
                    help='diffusion strength')

parser.add_argument('--graph_kernel_size', type=int, default=7,
                    help='local window size for graph diffusion')

parser.add_argument('--graph_sigma_feat', type=float, default=0.5,
                    help='feature similarity bandwidth')

parser.add_argument('--graph_sigma_spatial', type=float, default=4.0,
                    help='spatial similarity bandwidth')

parser.add_argument('--graph_sigma_intensity', type=float, default=0.2,
                    help='intensity similarity bandwidth')

parser.add_argument('--graph_use_intensity', type=int, default=1,
                    help='use image intensity in graph affinity')

parser.add_argument('--graph_expand_weight', type=float, default=0.5,
                    help='weight for non-seed diffused pixels')

parser.add_argument('--graph_entropy_thresh', type=float, default=0.6,
                    help='optional threshold for low-entropy diffused pseudo-labels')

parser.add_argument('--graph_supervise_mode', type=str, default='seed_plus_diffused',
                    choices=['seed_only', 'seed_plus_diffused'],
                    help='where to apply graph-diffused pseudo-label loss')

# Prototype module
parser.add_argument('--use_proto', type=int, default=1,
                    help='enable scribble-reliable prototype margin loss')

parser.add_argument('--proto_loss_weight', type=float, default=0.1,
                    help='weight for prototype loss')

parser.add_argument('--proto_temperature', type=float, default=0.2,
                    help='temperature for prototype contrastive loss')

parser.add_argument('--proto_margin', type=float, default=0.3,
                    help='margin for prototype margin loss')

parser.add_argument('--proto_momentum', type=float, default=0.99,
                    help='EMA momentum for class prototype memory')

parser.add_argument('--proto_loss_type', type=str, default='contrastive',
                    choices=['contrastive', 'margin', 'both'],
                    help='prototype loss formulation')

parser.add_argument('--proto_min_pixels', type=int, default=5,
                    help='minimum pixels required to update a class prototype in a batch')

parser.add_argument('--proto_warmup_iters', type=int, default=1000,
                    help='start prototype loss after this number of iterations')

# Feature extraction
parser.add_argument('--feature_level', type=str, default='decoder',
                    choices=['encoder', 'bottleneck', 'decoder'],
                    help='which feature map to use for graph and prototype learning')
```

---

# 5. Network output requirement

The current model may only return logits. RPG-MT needs a feature map for graph diffusion and prototype learning.

Modify the network forward function or wrapper to support:

```python
logits, feature = model(x, return_feature=True)
```

If modifying the network is difficult, implement a wrapper or forward hook.

Recommended feature source:

```text
For graph diffusion:
    Use decoder feature before the final 1×1 segmentation head.
    Resolution should be close to output, e.g. H/4×W/4 or H/2×W/2.

For prototype loss:
    Use the same decoder feature, then upsample or downsample labels/masks to feature resolution.
```

Expected shapes:

```text
logits:  [B, C, H, W]
feature: [B, D, h, w]
```

Feature normalization:

```python
feature = F.normalize(feature, p=2, dim=1)
```

---

# 6. Module 1: Confidence Agreement–Disagreement Reliable Seed Selection

## 6.1 Mathematical definition

For each pixel `i`:

```text
c_s(i) = max_c p_s(i,c)
ŷ_s(i) = argmax_c p_s(i,c)

c_t(i) = max_c p_t(i,c)
ŷ_t(i) = argmax_c p_t(i,c)
```

Candidate mask:

```text
a_i = 1[y_i = ignore_index]
```

Agreement reliable seed:

```text
m_agree(i) =
    1[ŷ_s(i) = ŷ_t(i)]
    · 1[min(c_s(i), c_t(i)) ≥ τ_agree]
    · a_i
```

Disagreement reliable seed:

```text
m_disagree(i) =
    1[ŷ_s(i) ≠ ŷ_t(i)]
    · 1[max(c_s(i), c_t(i)) ≥ τ_disagree]
    · 1[|c_s(i) - c_t(i)| ≥ τ_margin]
    · a_i
```

Final seed mask:

```text
m_seed(i) = m_agree(i) ∨ m_disagree(i)
```

Pseudo-label target for agreement:

```text
q_i = 0.5 · (p_s(i) + p_t(i))
```

Pseudo-label target for disagreement:

```text
q_i =
    p_s(i), if c_s(i) > c_t(i)
    p_t(i), otherwise
```

General formula:

```text
q_i = m_disagree(i) · p_high(i)
    + (1 - m_disagree(i)) · 0.5(p_s(i) + p_t(i))
```

where:

```text
p_high(i) = p_s(i) if c_s(i) > c_t(i), else p_t(i)
```

Important:

```text
q must be detached.
m_seed must be detached.
Teacher must not receive gradient.
```

## 6.2 Implementation function

Add:

```python
def build_confidence_agree_disagree_pseudo(
    student_prob,
    teacher_prob,
    label,
    agree_thresh=0.7,
    disagree_thresh=0.8,
    margin_thresh=0.1,
    ignore_index=4,
    pseudo_mask_mode='unlabeled',
    eps=1e-8,
):
    student_prob = student_prob.detach()
    teacher_prob = teacher_prob.detach()

    conf_s, pred_s = torch.max(student_prob, dim=1)
    conf_t, pred_t = torch.max(teacher_prob, dim=1)

    if pseudo_mask_mode == 'unlabeled':
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == 'all':
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError('Unsupported pseudo_mask_mode: {}'.format(pseudo_mask_mode))

    same = pred_s == pred_t
    diff = ~same

    min_conf = torch.minimum(conf_s, conf_t)
    max_conf = torch.maximum(conf_s, conf_t)
    margin = torch.abs(conf_s - conf_t)

    reliable_agree = same & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = (
        diff
        & (max_conf >= disagree_thresh)
        & (margin >= margin_thresh)
        & candidate_mask
    )

    mean_pseudo = 0.5 * (student_prob + teacher_prob)

    choose_student = (conf_s > conf_t).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_student, student_prob, teacher_prob)

    soft_pseudo_label = torch.where(
        reliable_disagree.unsqueeze(1),
        high_conf_pseudo,
        mean_pseudo,
    )
    soft_pseudo_label = soft_pseudo_label / (
        soft_pseudo_label.sum(dim=1, keepdim=True) + eps
    )

    reliable_mask = (reliable_agree | reliable_disagree).float().unsqueeze(1)
    pseudo_conf = torch.maximum(conf_s, conf_t).unsqueeze(1)

    return {
        'soft_pseudo_label': soft_pseudo_label.detach(),   # [B, C, H, W]
        'reliable_mask': reliable_mask.detach(),           # [B, 1, H, W]
        'reliable_agree': reliable_agree.detach(),         # [B, H, W]
        'reliable_disagree': reliable_disagree.detach(),   # [B, H, W]
        'agreement_ratio': reliable_agree.float().mean(),
        'disagreement_ratio': reliable_disagree.float().mean(),
        'reliable_ratio': reliable_mask.mean(),
        'pseudo_conf': pseudo_conf.detach(),
    }
```

---

# 7. Module 2: Graph-Harmonic Reliable Pseudo-label Diffusion

## 7.1 Motivation

After Module 1, reliable pseudo-labels are sparse:

```text
Ω_seed = {i | m_seed(i) = 1}
```

These pixels are reliable but do not cover all useful unlabeled regions.

Graph diffusion treats them as graph anchors and propagates pseudo-labels to nearby pixels that are similar in:

```text
feature space
spatial position
image intensity
```

This converts sparse reliable pseudo-labels into richer dense soft supervision while avoiding naive propagation across boundaries.

## 7.2 Graph construction

Work at feature resolution `[h, w]`.

Downsample:

```python
Q_low = interpolate(Q, size=(h, w), mode='bilinear')
M_low = interpolate(M_seed, size=(h, w), mode='nearest')
X_low = interpolate(x, size=(h, w), mode='bilinear')
```

Normalize pseudo-label:

```python
Q_low = Q_low / (Q_low.sum(dim=1, keepdim=True) + eps)
```

Feature:

```python
Z = F.normalize(feature, p=2, dim=1)  # [B, D, h, w]
```

For each node `i`, define local neighbors:

```text
N(i) = k×k local window around i
```

Affinity:

```text
A_ij = exp(
    - ||z_i - z_j||² / σ_f²
    - ||r_i - r_j||² / σ_x²
    - 1[use_intensity] · |I_i - I_j|² / σ_I²
)
```

Normalize over local neighbors:

```text
P_ij = A_ij / Σ_{j∈N(i)} A_ij
```

where `P` is the local transition matrix.

## 7.3 Graph-harmonic objective

The desired diffused pseudo-label `U` minimizes:

```text
U* = argmin_U
        Σ_i Σ_{j∈N(i)} A_ij ||U_i - U_j||²
        + μ Σ_i M_i ||U_i - Q_i||²
```

Interpretation:

```text
Graph smoothness:
    similar pixels should have similar pseudo-labels.

Seed fidelity:
    reliable seed pixels should stay close to the confidence-fused pseudo-label.
```

Equivalent Laplacian form:

```text
U* = argmin_U Tr(U^T L U) + μ ||M^(1/2)(U - Q)||_F²
```

where:

```text
L = D - A
```

Closed-form condition:

```text
(L + μM)U = μMQ
```

Do not solve this system explicitly in the code. Use iterative local diffusion.

## 7.4 Practical iterative diffusion

Initialize:

```text
U^(0) = Q_low
```

Iterate for `K` steps:

```text
Ū_i^(k) = Σ_{j∈N(i)} P_ij U_j^(k)

U_i^(k+1) =
    Q_i,                                  if M_i = 1
    α Ū_i^(k) + (1 - α) U_i^(k),          otherwise
```

This keeps reliable seeds fixed while propagating their soft labels to neighbors.

After each iteration:

```text
U_i ← U_i / Σ_c U_i,c
```

## 7.5 Diffusion reliability weight

Compute entropy:

```text
H(U_i) = -Σ_c U_i,c log(U_i,c + ε)
```

Normalized confidence:

```text
r_i = 1 - H(U_i) / log(C)
```

Final training weight:

Option A: seed only

```text
W_i = M_i
```

Option B: seed plus low-entropy diffused region

```text
W_i = M_i + γ · (1 - M_i) · r_i · 1[r_i ≥ τ_diff]
```

Recommended default:

```text
graph_supervise_mode = seed_plus_diffused
γ = graph_expand_weight = 0.5
τ_diff = graph_entropy_thresh = 0.6
```

This means:

```text
Seed pixels have weight 1.
Diffused non-seed pixels are used only if diffusion confidence is high.
```

## 7.6 Loss

Upsample `U` and `W` back to output resolution:

```python
U_high = F.interpolate(U_low, size=(H, W), mode='bilinear', align_corners=False)
U_high = U_high / (U_high.sum(dim=1, keepdim=True) + eps)

W_high = F.interpolate(W_low, size=(H, W), mode='nearest')
```

Graph-harmonic pseudo-label loss:

```text
L_GHRD =
    - 1 / (Σ_i W_i + ε)
      Σ_i W_i Σ_c U_i,c log p_s(i,c)
```

Implementation:

```python
loss_ghrd = masked_soft_ce_loss(
    logits=student_logits,
    target_prob=U_high.detach(),
    mask=W_high.detach(),
)
```

## 7.7 Efficient local window implementation with unfold

Implement a local graph diffusion function using `torch.nn.functional.unfold`.

Expected function signature:

```python
def graph_harmonic_diffusion(
    seed_pseudo,
    seed_mask,
    feature,
    image=None,
    num_classes=4,
    num_iters=3,
    kernel_size=7,
    alpha=0.7,
    sigma_feat=0.5,
    sigma_spatial=4.0,
    sigma_intensity=0.2,
    use_intensity=True,
    expand_weight=0.5,
    entropy_thresh=0.6,
    supervise_mode='seed_plus_diffused',
    eps=1e-8,
):
    # returns:
    # U_low: [B, C, h, w]
    # W_low: [B, 1, h, w]
    # diffusion_conf: [B, 1, h, w]
```

Pseudo-code:

```python
def graph_harmonic_diffusion(...):
    B, C, h, w = seed_pseudo.shape
    _, D, _, _ = feature.shape

    Z = F.normalize(feature, p=2, dim=1)

    # Extract local patches of features: [B, D*K*K, h*w]
    Z_patches = F.unfold(Z, kernel_size=kernel_size, padding=kernel_size // 2)
    Z_patches = Z_patches.view(B, D, kernel_size * kernel_size, h * w)

    Z_center = Z.view(B, D, 1, h * w)

    feat_dist = ((Z_patches - Z_center) ** 2).sum(dim=1)  # [B, K*K, h*w]

    # spatial distance prior
    spatial_dist = build_local_spatial_distance(kernel_size, device=feature.device)
    spatial_dist = spatial_dist.view(1, kernel_size * kernel_size, 1)

    affinity_log = - feat_dist / (sigma_feat ** 2 + eps)
    affinity_log = affinity_log - spatial_dist / (sigma_spatial ** 2 + eps)

    if use_intensity and image is not None:
        I = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        I_patches = F.unfold(I, kernel_size=kernel_size, padding=kernel_size // 2)
        I_patches = I_patches.view(B, 1, kernel_size * kernel_size, h * w)
        I_center = I.view(B, 1, 1, h * w)
        intensity_dist = ((I_patches - I_center) ** 2).sum(dim=1)
        affinity_log = affinity_log - intensity_dist / (sigma_intensity ** 2 + eps)

    A = torch.softmax(affinity_log, dim=1)  # [B, K*K, h*w]

    U = seed_pseudo.clone()

    for _ in range(num_iters):
        U_patches = F.unfold(U, kernel_size=kernel_size, padding=kernel_size // 2)
        U_patches = U_patches.view(B, C, kernel_size * kernel_size, h * w)

        U_neighbor = (A.unsqueeze(1) * U_patches).sum(dim=2)
        U_neighbor = U_neighbor.view(B, C, h, w)

        U = alpha * U_neighbor + (1.0 - alpha) * U

        # anchor reliable seeds
        U = seed_mask * seed_pseudo + (1.0 - seed_mask) * U

        U = U / (U.sum(dim=1, keepdim=True) + eps)

    entropy = -(U * torch.log(U + eps)).sum(dim=1, keepdim=True)
    diffusion_conf = 1.0 - entropy / np.log(num_classes)

    if supervise_mode == 'seed_only':
        W = seed_mask
    elif supervise_mode == 'seed_plus_diffused':
        diffused_mask = ((diffusion_conf >= entropy_thresh).float() * (1.0 - seed_mask))
        W = seed_mask + expand_weight * diffusion_conf * diffused_mask
    else:
        raise ValueError('Unsupported supervise_mode: {}'.format(supervise_mode))

    return U.detach(), W.detach(), diffusion_conf.detach()
```

Required helper:

```python
def build_local_spatial_distance(kernel_size, device):
    radius = kernel_size // 2
    coords = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing='ij'
    ), dim=-1).float()
    dist = (coords[..., 0] ** 2 + coords[..., 1] ** 2).reshape(-1)
    return dist
```

---

# 8. Module 3: Scribble-Reliable Prototype Margin Learning

## 8.1 Motivation

Graph diffusion improves pseudo-label targets, but supervision is still mainly output-space. We also want feature-space separation:

```text
same-class anatomy features → compact
different-class anatomy features → separated
```

Use two trustworthy sources:

```text
1. Scribble-labeled pixels.
2. Reliable pseudo-label seed pixels.
```

Do not use all diffused pixels to update prototypes because diffused pseudo-labels may still contain noise.

## 8.2 Prototype source mask

At feature resolution `[h, w]`, downsample:

```python
label_low = F.interpolate(y.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()
seed_mask_low = F.interpolate(m_seed, size=(h, w), mode='nearest')
q_seed_low = F.interpolate(q_seed, size=(h, w), mode='bilinear')
q_seed_low = q_seed_low / (q_seed_low.sum(dim=1, keepdim=True) + eps)
pseudo_cls_low = torch.argmax(q_seed_low, dim=1)
```

Define prototype targets:

```text
For labeled scribble pixels:
    target_i = y_i

For reliable pseudo seed pixels:
    target_i = argmax(q_i)

Priority:
    if scribble label exists, use scribble label.
    else if reliable seed exists, use pseudo class.
```

Valid prototype mask:

```text
M_proto(i) = 1[y_i ≠ ignore_index] ∨ M_seed(i)
```

## 8.3 Prototype memory

Maintain a prototype memory:

```python
prototype_memory = torch.zeros(num_classes, feat_dim).cuda()
prototype_initialized = torch.zeros(num_classes).bool().cuda()
```

For each class `c`, collect features:

```text
Z_c = {z_i | M_proto(i)=1 and target_i=c}
```

Batch prototype:

```text
μ_c = mean_{i∈Z_c} normalize(z_i)
```

EMA update:

```text
P_c ← ρ P_c + (1 - ρ) μ_c
P_c ← normalize(P_c)
```

If prototype not initialized:

```text
P_c ← μ_c
```

Only update if:

```text
|Z_c| ≥ proto_min_pixels
```

## 8.4 Prototype contrastive loss

For each valid feature `z_i`, target class `a_i`:

```text
sim_i,c = cosine(z_i, P_c)
```

Contrastive loss:

```text
L_proto_con =
    - 1 / |Ω_proto|
      Σ_{i∈Ω_proto}
      log
      exp(sim_i,a_i / τ)
      /
      Σ_{c=0}^{C-1} exp(sim_i,c / τ)
```

Implementation:

```python
logits_proto = torch.matmul(z_valid, prototypes.T) / temperature
loss_proto_con = F.cross_entropy(logits_proto, target_valid)
```

where:

```text
z_valid: [N, D]
prototypes: [C, D]
target_valid: [N]
```

## 8.5 Prototype margin loss

Distance:

```text
d_i,c = 1 - cosine(z_i, P_c)
```

Positive distance:

```text
d_pos = d_i,a_i
```

Nearest negative distance:

```text
d_neg = min_{c≠a_i} d_i,c
```

Margin loss:

```text
L_proto_margin =
    1 / |Ω_proto|
    Σ_{i∈Ω_proto}
    max(0, δ + d_pos - d_neg)
```

Implementation:

```python
sim = torch.matmul(z_valid, prototypes.T)
dist = 1.0 - sim

pos_dist = dist.gather(1, target_valid.view(-1, 1)).squeeze(1)

mask = torch.ones_like(dist).bool()
mask.scatter_(1, target_valid.view(-1, 1), False)
neg_dist = dist.masked_fill(~mask, 1e6).min(dim=1)[0]

loss_margin = F.relu(margin + pos_dist - neg_dist).mean()
```

## 8.6 Final prototype loss

```python
if proto_loss_type == 'contrastive':
    loss_proto = loss_proto_con
elif proto_loss_type == 'margin':
    loss_proto = loss_margin
elif proto_loss_type == 'both':
    loss_proto = loss_proto_con + loss_margin
```

If no valid prototype pixels or prototypes are not sufficiently initialized:

```python
loss_proto = logits.new_tensor(0.0)
```

## 8.7 Implementation class

Create:

```python
class ClassPrototypeMemory:
    def __init__(self, num_classes, feat_dim, momentum=0.99, min_pixels=5):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.min_pixels = min_pixels
        self.prototypes = torch.zeros(num_classes, feat_dim).cuda()
        self.initialized = torch.zeros(num_classes).bool().cuda()

    @torch.no_grad()
    def update(self, features, targets, mask):
        # features: [B, D, h, w]
        # targets: [B, h, w]
        # mask: [B, 1, h, w] or [B, h, w]
        pass

    def compute_loss(self, features, targets, mask, temperature=0.2, margin=0.3, loss_type='contrastive'):
        # returns scalar loss
        pass
```

Important:

```text
Prototype memory update should use detached features.
Prototype loss should use non-detached features so gradients update the student.
```

Call order:

```python
prototype_memory.update(
    features=feature_s.detach(),
    targets=proto_targets,
    mask=proto_mask,
)

loss_proto = prototype_memory.compute_loss(
    features=feature_s,
    targets=proto_targets,
    mask=proto_mask,
    temperature=args.proto_temperature,
    margin=args.proto_margin,
    loss_type=args.proto_loss_type,
)
```

---

# 9. Loss functions

## 9.1 Partial CE

```python
ce_loss = CrossEntropyLoss(ignore_index=num_classes)
loss_pce = ce_loss(student_logits, label_batch.long())
```

Mathematical form:

```text
L_pCE =
    - 1 / |Ω_l|
      Σ_{i∈Ω_l} log p_s(i, y_i)
```

## 9.2 Masked soft CE

Add or keep:

```python
def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)
```

## 9.3 Graph-diffused pseudo-label loss

```python
loss_ghrd = masked_soft_ce_loss(
    logits=student_logits,
    target_prob=U_high.detach(),
    mask=W_high.detach(),
)
```

## 9.4 Prototype loss

```python
loss_proto = prototype_memory.compute_loss(...)
```

## 9.5 Total loss

```python
ramp = get_current_consistency_weight(epoch_num)

lambda_pseudo = ramp * args.pseudo_loss_weight

if args.use_proto and iter_num >= args.proto_warmup_iters:
    lambda_proto = ramp * args.proto_loss_weight
else:
    lambda_proto = 0.0

loss = loss_pce + lambda_pseudo * loss_ghrd + lambda_proto * loss_proto
```

---

# 10. Full training loop specification

Replace the current batch-training block with this pipeline.

```python
for epoch_num in iterator:
    for i_batch, sampled_batch in enumerate(trainloader):

        volume_batch = sampled_batch['image'].cuda()
        label_batch = sampled_batch['label'].cuda()

        # EMA teacher forward
        with torch.no_grad():
            teacher_out = model_ema(volume_batch, return_feature=True)
            teacher_logits, teacher_feature = unpack_logits_feature(teacher_out)
            teacher_prob = torch.softmax(teacher_logits, dim=1)

        # Student forward
        student_out = model(volume_batch, return_feature=True)
        student_logits, student_feature = unpack_logits_feature(student_out)
        student_prob = torch.softmax(student_logits, dim=1)

        # LpCE on scribble labels
        loss_pce = ce_loss(student_logits, label_batch.long())

        # Confidence agreement/disagreement pseudo-label seed
        pseudo_info = build_confidence_agree_disagree_pseudo(
            student_prob=student_prob,
            teacher_prob=teacher_prob,
            label=label_batch,
            agree_thresh=args.pseudo_agree_thresh,
            disagree_thresh=args.pseudo_disagree_thresh,
            margin_thresh=args.pseudo_margin_thresh,
            ignore_index=num_classes,
            pseudo_mask_mode=args.pseudo_mask_mode,
        )

        q_seed = pseudo_info['soft_pseudo_label']   # [B, C, H, W]
        m_seed = pseudo_info['reliable_mask']       # [B, 1, H, W]

        # Graph-harmonic pseudo-label diffusion
        if args.use_graph_diffusion:
            h, w = student_feature.shape[-2:]

            q_low = F.interpolate(q_seed, size=(h, w), mode='bilinear', align_corners=False)
            q_low = q_low / (q_low.sum(dim=1, keepdim=True) + 1e-8)

            m_low = F.interpolate(m_seed, size=(h, w), mode='nearest')

            u_low, w_low, diffusion_conf_low = graph_harmonic_diffusion(
                seed_pseudo=q_low,
                seed_mask=m_low,
                feature=student_feature.detach(),
                image=volume_batch,
                num_classes=num_classes,
                num_iters=args.graph_diffusion_iters,
                kernel_size=args.graph_kernel_size,
                alpha=args.graph_alpha,
                sigma_feat=args.graph_sigma_feat,
                sigma_spatial=args.graph_sigma_spatial,
                sigma_intensity=args.graph_sigma_intensity,
                use_intensity=bool(args.graph_use_intensity),
                expand_weight=args.graph_expand_weight,
                entropy_thresh=args.graph_entropy_thresh,
                supervise_mode=args.graph_supervise_mode,
            )

            u_high = F.interpolate(
                u_low,
                size=student_logits.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            u_high = u_high / (u_high.sum(dim=1, keepdim=True) + 1e-8)

            w_high = F.interpolate(
                w_low,
                size=student_logits.shape[-2:],
                mode='nearest',
            )

        else:
            u_high = q_seed
            w_high = m_seed
            diffusion_conf_low = None

        # Graph-diffused pseudo-label loss
        loss_ghrd = masked_soft_ce_loss(
            logits=student_logits,
            target_prob=u_high.detach(),
            mask=w_high.detach(),
        )

        # Prototype target construction
        if args.use_proto and iter_num >= args.proto_warmup_iters:
            proto_targets, proto_mask = build_scribble_reliable_proto_targets(
                label=label_batch,
                seed_pseudo=q_seed,
                seed_mask=m_seed,
                feature_size=student_feature.shape[-2:],
                ignore_index=num_classes,
            )

            prototype_memory.update(
                features=student_feature.detach(),
                targets=proto_targets,
                mask=proto_mask,
            )

            loss_proto = prototype_memory.compute_loss(
                features=student_feature,
                targets=proto_targets,
                mask=proto_mask,
                temperature=args.proto_temperature,
                margin=args.proto_margin,
                loss_type=args.proto_loss_type,
            )
        else:
            loss_proto = student_logits.new_tensor(0.0)

        # Total loss
        ramp = get_current_consistency_weight(epoch_num)
        lambda_pseudo = ramp * args.pseudo_loss_weight

        if args.use_proto and iter_num >= args.proto_warmup_iters:
            lambda_proto = ramp * args.proto_loss_weight
        else:
            lambda_proto = 0.0

        loss = (
            loss_pce
            + lambda_pseudo * loss_ghrd
            + lambda_proto * loss_proto
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema_optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num += 1
```

---

# 11. Helper functions

## 11.1 Unpack logits and feature

```python
def unpack_logits_feature(output):
    if isinstance(output, dict):
        return output['logits'], output['feature']

    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            return output[0], output[1]
        if len(output) > 2:
            return output[0], output[-1]

    raise ValueError(
        'Model must return (logits, feature) or dict with keys logits and feature when return_feature=True'
    )
```

## 11.2 Build prototype targets

```python
def build_scribble_reliable_proto_targets(
    label,
    seed_pseudo,
    seed_mask,
    feature_size,
    ignore_index=4,
):
    h, w = feature_size

    label_low = F.interpolate(
        label.unsqueeze(1).float(),
        size=(h, w),
        mode='nearest',
    ).squeeze(1).long()

    seed_mask_low = F.interpolate(
        seed_mask.float(),
        size=(h, w),
        mode='nearest',
    )

    seed_pseudo_low = F.interpolate(
        seed_pseudo,
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    )
    seed_pseudo_low = seed_pseudo_low / (
        seed_pseudo_low.sum(dim=1, keepdim=True) + 1e-8
    )

    pseudo_cls = torch.argmax(seed_pseudo_low, dim=1)

    labeled_mask = label_low != ignore_index
    pseudo_mask = seed_mask_low.squeeze(1) > 0.5

    targets = torch.full_like(label_low, fill_value=ignore_index)

    # priority 1: scribble labels
    targets[labeled_mask] = label_low[labeled_mask]

    # priority 2: reliable pseudo-labels where scribble is absent
    use_pseudo = (~labeled_mask) & pseudo_mask
    targets[use_pseudo] = pseudo_cls[use_pseudo]

    valid_mask = (targets != ignore_index).float().unsqueeze(1)

    return targets, valid_mask
```

---

# 12. Prototype memory implementation

```python
class ClassPrototypeMemory:
    def __init__(self, num_classes, feat_dim, momentum=0.99, min_pixels=5, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.min_pixels = min_pixels
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, features, targets, mask):
        B, D, h, w = features.shape

        feats = F.normalize(features, p=2, dim=1)
        feats = feats.permute(0, 2, 3, 1).reshape(-1, D)

        targets_flat = targets.reshape(-1)
        mask_flat = mask.squeeze(1).reshape(-1).bool()

        for c in range(self.num_classes):
            class_mask = mask_flat & (targets_flat == c)
            if class_mask.sum() < self.min_pixels:
                continue

            proto = feats[class_mask].mean(dim=0)
            proto = F.normalize(proto, p=2, dim=0)

            if not self.initialized[c]:
                self.prototypes[c] = proto
                self.initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1.0 - self.momentum) * proto
                )
                self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=0)

    def compute_loss(self, features, targets, mask, temperature=0.2, margin=0.3, loss_type='contrastive'):
        B, D, h, w = features.shape

        if self.initialized.sum() < self.num_classes:
            return features.new_tensor(0.0)

        feats = F.normalize(features, p=2, dim=1)
        feats = feats.permute(0, 2, 3, 1).reshape(-1, D)

        targets_flat = targets.reshape(-1)
        mask_flat = mask.squeeze(1).reshape(-1).bool()

        valid = mask_flat & (targets_flat >= 0) & (targets_flat < self.num_classes)

        if valid.sum() < 1:
            return features.new_tensor(0.0)

        z = feats[valid]
        y = targets_flat[valid].long()

        prototypes = F.normalize(self.prototypes, p=2, dim=1)

        sim = torch.matmul(z, prototypes.t())

        losses = []

        if loss_type in ['contrastive', 'both']:
            logits = sim / temperature
            loss_con = F.cross_entropy(logits, y)
            losses.append(loss_con)

        if loss_type in ['margin', 'both']:
            dist = 1.0 - sim

            pos_dist = dist.gather(1, y.view(-1, 1)).squeeze(1)

            neg_mask = torch.ones_like(dist).bool()
            neg_mask.scatter_(1, y.view(-1, 1), False)

            neg_dist = dist.masked_fill(~neg_mask, 1e6).min(dim=1)[0]

            loss_margin = F.relu(margin + pos_dist - neg_dist).mean()
            losses.append(loss_margin)

        if len(losses) == 0:
            raise ValueError('Unsupported proto loss type: {}'.format(loss_type))

        return sum(losses)
```

Initialize after first student forward:

```python
prototype_memory = None

if prototype_memory is None:
    feat_dim = student_feature.shape[1]
    prototype_memory = ClassPrototypeMemory(
        num_classes=num_classes,
        feat_dim=feat_dim,
        momentum=args.proto_momentum,
        min_pixels=args.proto_min_pixels,
        device=student_feature.device,
    )
```

---

# 13. Logging requirements

Add TensorBoard scalars:

```python
writer.add_scalar('info/lr', lr_, iter_num)
writer.add_scalar('loss/total', loss.item(), iter_num)
writer.add_scalar('loss/loss_pce', loss_pce.item(), iter_num)
writer.add_scalar('loss/loss_ghrd', loss_ghrd.item(), iter_num)
writer.add_scalar('loss/loss_proto', loss_proto.item(), iter_num)
writer.add_scalar('weight/lambda_pseudo', lambda_pseudo, iter_num)
writer.add_scalar('weight/lambda_proto', lambda_proto, iter_num)

writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)

writer.add_scalar('graph/w_high_ratio', (w_high > 0).float().mean().item(), iter_num)
writer.add_scalar('graph/w_high_mean', w_high.mean().item(), iter_num)

if diffusion_conf_low is not None:
    writer.add_scalar('graph/diffusion_conf_mean', diffusion_conf_low.mean().item(), iter_num)

if prototype_memory is not None:
    writer.add_scalar('proto/initialized_classes', prototype_memory.initialized.float().sum().item(), iter_num)
    writer.add_scalar('proto/prototype_norm_mean', prototype_memory.prototypes.norm(dim=1).mean().item(), iter_num)
```

Console logging every 200 iterations:

```python
logging.info(
    'iter %d | loss %.4f | pce %.4f | ghrd %.4f | proto %.4f | '
    'lambda_p %.4f | lambda_proto %.4f | rel %.4f | agree %.4f | disagree %.4f | '
    'graph_w %.4f'
    % (
        iter_num,
        loss.item(),
        loss_pce.item(),
        loss_ghrd.item(),
        loss_proto.item(),
        lambda_pseudo,
        lambda_proto,
        pseudo_info['reliable_ratio'].item(),
        pseudo_info['agreement_ratio'].item(),
        pseudo_info['disagreement_ratio'].item(),
        w_high.mean().item(),
    )
)
```

---

# 14. Validation and checkpointing

Use the student model for main validation:

```python
metric_i = test_single_volume_scribblevs(
    sampled_batch["image"],
    sampled_batch["label"],
    model,
    classes=num_classes,
)
```

Save student:

```python
torch.save(model.state_dict(), save_best)
```

Optional EMA validation:

```python
if args.eval_ema:
    eval_model = model_ema
else:
    eval_model = model
```

Recommended main paper result:

```text
Report student result by default.
Optionally report EMA teacher result in ablation/supplement.
```

---

# 15. Recommended implementation phases

## Phase 1: Baseline confidence fusion

Loss:

```text
L_total = L_pCE + λ_pseudo L_pseudo_seed
```

Where:

```text
L_pseudo_seed = masked soft CE on m_seed only.
```

Goal:

```text
Verify reliable pseudo-label selection works.
```

## Phase 2: Add graph diffusion

Loss:

```text
L_total = L_pCE + λ_pseudo L_GHRD
```

Goal:

```text
Show graph diffusion improves Dice and HD95 compared with seed-only pseudo-labels.
```

## Phase 3: Add prototype module

Loss:

```text
L_total = L_pCE + λ_pseudo L_GHRD + λ_proto L_proto
```

Goal:

```text
Show feature-level prototype regularization improves representation, stability, and boundary quality.
```

---

# 16. Ablation plan

Run these experiments:

## A0: Scribble baseline

```text
L_pCE only
```

## A1: Mean Teacher + confidence agree/disagree seed

```text
L_pCE + L_seed
```

## A2: RPG-MT without prototype

```text
L_pCE + L_GHRD
```

## A3: RPG-MT full

```text
L_pCE + L_GHRD + L_proto
```

## A4: Graph affinity ablation

```text
feature only
feature + spatial
feature + spatial + intensity
```

## A5: Diffusion supervision mode

```text
seed_only
seed_plus_diffused
```

## A6: Prototype loss type

```text
contrastive
margin
both
```

## A7: Threshold sensitivity

```text
τ_agree / τ_disagree:
0.6 / 0.7
0.7 / 0.8
0.8 / 0.9

τ_margin:
0.05
0.10
0.15
```

## A8: Diffusion iteration sensitivity

```text
K = 1, 3, 5, 7
```

Expected:

```text
Too small K:
    insufficient propagation.

Too large K:
    over-smoothing and boundary leakage.
```

---

# 17. Recommended default command

```bash
python train_rpg_mt.py \
  --root_path ../../data/ACDC \
  --data ACDC \
  --exp RPG_MT_ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet \
  --num_classes 4 \
  --batch_size 16 \
  --base_lr 0.01 \
  --max_iterations 60000 \
  --pseudo_agree_thresh 0.7 \
  --pseudo_disagree_thresh 0.8 \
  --pseudo_margin_thresh 0.1 \
  --pseudo_loss_weight 8.0 \
  --pseudo_mask_mode unlabeled \
  --use_graph_diffusion 1 \
  --graph_diffusion_iters 3 \
  --graph_alpha 0.7 \
  --graph_kernel_size 7 \
  --graph_sigma_feat 0.5 \
  --graph_sigma_spatial 4.0 \
  --graph_sigma_intensity 0.2 \
  --graph_use_intensity 1 \
  --graph_expand_weight 0.5 \
  --graph_entropy_thresh 0.6 \
  --graph_supervise_mode seed_plus_diffused \
  --use_proto 1 \
  --proto_loss_weight 0.1 \
  --proto_temperature 0.2 \
  --proto_margin 0.3 \
  --proto_momentum 0.99 \
  --proto_loss_type contrastive \
  --proto_warmup_iters 1000 \
  --gpu 0
```

---

# 18. Common implementation pitfalls

## Pitfall 1: Teacher receives gradient

Wrong:

```python
teacher_logits = model_ema(x)
```

Correct:

```python
with torch.no_grad():
    teacher_logits = model_ema(x)
```

## Pitfall 2: Pseudo-label target not detached

Correct:

```python
q_seed = q_seed.detach()
u_high = u_high.detach()
w_high = w_high.detach()
```

## Pitfall 3: Shape mismatch

Check:

```text
student_logits: [B, C, H, W]
q_seed:         [B, C, H, W]
m_seed:         [B, 1, H, W]

student_feature:[B, D, h, w]
q_low:          [B, C, h, w]
m_low:          [B, 1, h, w]
```

## Pitfall 4: Graph diffusion uses non-detached feature

Recommended:

```python
feature=student_feature.detach()
```

Reason:

```text
Graph affinity should not create unstable second-order gradient paths.
The graph module should generate targets, not become a trainable differentiable graph optimizer.
```

## Pitfall 5: Prototype memory update uses non-detached feature

Correct:

```python
prototype_memory.update(student_feature.detach(), ...)
```

Prototype loss uses non-detached feature:

```python
loss_proto = prototype_memory.compute_loss(student_feature, ...)
```

## Pitfall 6: Downsampling scribble label with bilinear

Wrong:

```python
F.interpolate(label.float(), mode='bilinear')
```

Correct:

```python
F.interpolate(label.float(), mode='nearest')
```

## Pitfall 7: Too strong prototype loss early

Use warm-up:

```python
if iter_num < proto_warmup_iters:
    loss_proto = 0
```

## Pitfall 8: Too much diffusion causes boundary leakage

Keep:

```text
K = 3
kernel_size = 7
alpha = 0.7
feature + spatial + intensity affinity
```

---

# 19. Acceptance criteria

The implementation is correct if:

## Basic training

- [ ] Code runs without undefined variables.
- [ ] Student is updated by gradient descent.
- [ ] EMA teacher is updated only by EMA.
- [ ] Validation still runs with `test_single_volume_scribblevs`.

## Module 1

- [ ] `m_agree` uses same predicted class and min confidence threshold.
- [ ] `m_disagree` uses different predicted classes, max confidence threshold, and confidence margin.
- [ ] `m_seed = m_agree | m_disagree`.
- [ ] `q_seed` is soft pseudo-label and detached.
- [ ] `m_seed` shape is `[B, 1, H, W]`.

## Module 2

- [ ] Graph diffusion works at feature resolution.
- [ ] Local affinity includes feature similarity.
- [ ] Optional spatial and intensity terms work.
- [ ] Seed pixels are anchored during diffusion.
- [ ] Diffused pseudo-label is normalized over classes.
- [ ] `W_high` shape is `[B, 1, H, W]`.
- [ ] `L_GHRD` uses `U_high` and `W_high`.

## Module 3

- [ ] Prototype targets combine scribble labels and reliable pseudo seeds.
- [ ] Scribble labels have priority over pseudo-labels.
- [ ] Prototype memory updates with detached features.
- [ ] Prototype loss uses non-detached features.
- [ ] Loss returns zero safely if not enough valid pixels or prototypes.

## Logging

- [ ] Logs `loss_pce`.
- [ ] Logs `loss_ghrd`.
- [ ] Logs `loss_proto`.
- [ ] Logs `reliable_ratio`, `agreement_ratio`, `disagreement_ratio`.
- [ ] Logs `graph/w_high_mean`.
- [ ] Logs prototype initialized class count.

---

# 20. Minimal code integration checklist

Add these components to `train_rpg_mt.py`:

```text
1. New argparse parameters.
2. Model forward returns logits + feature.
3. build_confidence_agree_disagree_pseudo.
4. masked_soft_ce_loss.
5. build_local_spatial_distance.
6. graph_harmonic_diffusion.
7. build_scribble_reliable_proto_targets.
8. ClassPrototypeMemory.
9. New training loop block.
10. TensorBoard logging.
11. Ablation flags.
```

---

# 21. Short conceptual summary for paper writing

RPG-MT is built on a Mean Teacher framework for scribble-supervised segmentation. The student and EMA teacher first generate confidence-calibrated pseudo-label seeds through agreement/disagreement reasoning. Instead of using these reliable pixels only as sparse output targets, RPG-MT treats them as graph anchors and propagates their soft labels through a local graph constructed from feature, spatial, and intensity affinities. This produces graph-harmonic pseudo-labels that expand supervision to unlabeled regions while preserving anatomical boundaries. In parallel, scribble-labeled pixels and reliable pseudo-labeled pixels are used to maintain class prototypes, and a prototype margin/contrastive loss encourages class-discriminative anatomical representations. The final objective combines partial CE on scribble labels, graph-diffused pseudo-label supervision, and prototype-level representation regularization.
