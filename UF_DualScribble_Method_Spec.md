# SPEC: UF-DualScribble — Uncertainty-Feedback Dual-Teacher Student Framework for Scribble-Supervised Medical Image Segmentation

## 0. Mục tiêu implementation

Tạo một training script mới cho proposed method:

```text
UF-DualScribble: Uncertainty-Feedback Dual-Teacher Student Framework
```

Method này dùng cho **scribble-supervised medical image segmentation** trên project hiện tại, ưu tiên chạy trước trên **ACDC** với setup giống codebase hiện có.

Ý tưởng chính:

1. Có **2 teachers** `T1`, `T2` và **1 student** `S`.
2. Hai teachers sinh pseudo-label cho vùng không có scribble label.
3. Dùng **evidential uncertainty map** để chia vùng non-scribble thành:
   - `Rc`: confident core region.
   - `Rb`: uncertain boundary region.
   - `Ru`: unreliable outlier region.
4. Student học thử từ từng vùng `Rc`, `Rb`, sau đó đo xem update đó có làm loss trên scribble anchors tốt hơn không.
5. Feedback scalar `delta_c`, `delta_b` được dùng để điều chỉnh lại teachers.
6. Vì scribble thiếu thông tin boundary, thêm một **boundary-aware refinement loss** nhẹ dựa trên boundary band từ pseudo-label.

Script cần được viết theo style codebase hiện tại:

- Dataset: `ACDCDataSets`, `RandomGenerator`.
- Network: `net_factory`.
- Validation: `test_single_volume_scribblevs`.
- Scribble label ignore index: `4`.
- Segmentation classes: mặc định `num_classes=4`.
- Optimizer mặc định: SGD.
- Logging: `tensorboardX.SummaryWriter` + `logging`.
- Checkpoint path: `../../checkpoints/{data}_{exp}`.

---

## 1. File cần tạo

Tạo file mới:

```text
code/train_uf_dualscribble_acdc.py
```

Không sửa trực tiếp các file train cũ. Chỉ reuse lại:

```python
from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from val import test_single_volume_scribblevs
```

Nếu cần helper function mới, ưu tiên đặt ngay trong file train mới để dễ debug. Sau khi chạy ổn mới tách ra `utils/`.

---

## 2. CLI arguments cần có

Giữ lại các arguments đang có trong codebase hiện tại:

```python
--root_path ../../data/ACDC
--exp UF_DualScribble
--data ACDC
--fold MAAGfold70
--sup_type scribble
--model unet
--num_classes 4
--max_iterations 60000
--batch_size 16
--deterministic 1
--base_lr 0.01
--patch_size [256, 256]
--seed 2022
--gpu 0
--consistency_rampup 40.0
--evidence_activation relu
```

Thêm arguments mới:

```python
# uncertainty
--uncertainty_type evidential
--tau_conf 0.35
--tau_uncertain 0.75
--pseudo_conf_thresh 0.60
--uncertainty_temp 0.5
--lambda_mutual_unc 0.0

# region losses
--lambda_pseudo_conf 1.0
--lambda_pseudo_boundary 0.5
--lambda_boundary 0.2
--lambda_cross_teacher 0.5
--lambda_feedback 0.05
--lambda_unc_cons 0.1

# feedback
--feedback_start_iter 3000
--feedback_interval 1
--feedback_lr 0.1
--feedback_clip 1.0
--normalize_trial_grad 1
--feedback_use_boundary_anchor 0
--lambda_feedback_boundary_anchor 0.1

# boundary
--boundary_radius 3
--boundary_mask_dilate 1
--use_boundary_region 1

# training stability
--teacher_diversity_mode aug_dropout
--teacher_ema_alpha 0.0
--save_debug_vis 1
--debug_vis_interval 1000
```

Ghi chú:

- `teacher_ema_alpha=0.0` nghĩa là teachers được train bằng optimizer bình thường. Chưa dùng EMA cho teachers ở version đầu.
- Nếu muốn dùng EMA sau này, implement optional nhưng default không bật.
- `lambda_feedback` phải nhỏ vì feedback loss có thể âm khi `delta < 0`.

---

## 3. Model architecture

Khởi tạo 3 models:

```python
teacher1 = create_model(num_classes=args.num_classes)
teacher2 = create_model(num_classes=args.num_classes)
student = create_model(num_classes=args.num_classes)
```

Function:

```python
def create_model(num_classes=4):
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    return net.cuda()
```

Optimizer:

```python
optimizer_t1 = optim.SGD(teacher1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
optimizer_t2 = optim.SGD(teacher2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
optimizer_s = optim.SGD(student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
```

Validation và checkpoint dùng **student** là final model chính.

Save thêm teachers để debug:

```text
unet_best_model.pth              # student
unet_teacher1_best_model.pth
unet_teacher2_best_model.pth
```

---

## 4. Label convention

ACDC scribble label có:

```text
0, 1, 2, 3 : labeled classes
4          : unlabeled / ignored pixels
```

Tạo masks:

```python
scribble_mask = (label_batch != 4).float().unsqueeze(1)   # [B,1,H,W]
unlabeled_mask = (label_batch == 4).float().unsqueeze(1)  # [B,1,H,W]
```

Partial CE trên scribble dùng:

```python
ce_loss = CrossEntropyLoss(ignore_index=4)
loss_scrib = ce_loss(logits, label_batch.long())
```

---

## 5. Reuse evidential uncertainty từ code hiện tại

Giữ đúng logic evidential uncertainty đang có:

```python
def logits_to_evidence(logits, activation='relu'):
    if activation == 'relu':
        return F.relu(logits)
    if activation == 'softplus':
        return F.softplus(logits)
    if activation == 'exp':
        return torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
    raise ValueError(...)


def evidential_uncertainty_from_logits(logits, num_classes, activation='relu', eps=1e-8):
    evidence = logits_to_evidence(logits, activation=activation)
    alpha = evidence + 1.0
    uncertainty = float(num_classes) / (torch.sum(alpha, dim=1, keepdim=True) + eps)
    return uncertainty
```

Shape:

```text
logits:      [B,C,H,W]
uncertainty: [B,1,H,W]
```

Interpretation:

```text
uncertainty gần 0: model rất chắc
uncertainty gần 1: model không chắc
```

Trong method mới, evidential uncertainty là reliability estimator chính.

---

## 6. Optional uncertainty components

Version đầu chỉ cần evidential uncertainty.

Nếu `lambda_mutual_unc > 0`, cộng thêm teacher discrepancy bằng symmetric KL hoặc JS divergence.

Function:

```python
def symmetric_kl(p, q, eps=1e-8):
    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    kl_pq = (p * (log_p - log_q)).sum(dim=1, keepdim=True)
    kl_qp = (q * (log_q - log_p)).sum(dim=1, keepdim=True)
    return 0.5 * (kl_pq + kl_qp)
```

Normalize KL per batch về `[0,1]`:

```python
def minmax_norm(x, eps=1e-8):
    return (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True) + eps)
```

Combined uncertainty:

```python
u1 = evidential_uncertainty_from_logits(logits_t1, C, args.evidence_activation)
u2 = evidential_uncertainty_from_logits(logits_t2, C, args.evidence_activation)
u_mean = 0.5 * (u1 + u2)

if args.lambda_mutual_unc > 0:
    u_kl = minmax_norm(symmetric_kl(prob_t1, prob_t2))
    u_ens = torch.clamp(u_mean + args.lambda_mutual_unc * u_kl, 0.0, 1.0)
else:
    u_ens = u_mean
```

---

## 7. Uncertainty-weighted pseudo-label ensemble

Teachers output:

```python
logits_t1 = unpack_model_output(teacher1(volume_batch))
logits_t2 = unpack_model_output(teacher2(volume_batch))
prob_t1 = torch.softmax(logits_t1, dim=1)
prob_t2 = torch.softmax(logits_t2, dim=1)
```

Reliability weights:

```python
w1 = torch.exp(-u1 / args.uncertainty_temp)
w2 = torch.exp(-u2 / args.uncertainty_temp)
w_sum = w1 + w2 + 1e-8
w1 = w1 / w_sum
w2 = w2 / w_sum
```

Pseudo soft label:

```python
pseudo_soft = (w1 * prob_t1 + w2 * prob_t2).detach()
pseudo_hard = torch.argmax(pseudo_soft, dim=1)  # [B,H,W]
pseudo_conf = torch.max(pseudo_soft, dim=1, keepdim=True)[0]
```

Important:

- `pseudo_soft` phải `.detach()` khi dùng để train student.
- Teacher feedback loss dùng target detached để tránh gradient loop không cần thiết.

---

## 8. Region partition bằng uncertainty

### 8.1 Confident core region `Rc`

```python
conf_mask = (
    (u_ens < args.tau_conf) &
    (pseudo_conf > args.pseudo_conf_thresh) &
    (unlabeled_mask > 0)
).float()
```

Shape: `[B,1,H,W]`.

Meaning:

- Vùng non-scribble mà teachers tương đối chắc.
- Dùng hard/soft pseudo-label supervision mạnh hơn.

### 8.2 Boundary candidate band

Tạo boundary band từ `pseudo_hard` bằng morphological gradient.

Function cần implement:

```python
def label_to_onehot(label, num_classes):
    # label: [B,H,W]
    # return: [B,C,H,W]
    return F.one_hot(label.long(), num_classes=num_classes).permute(0,3,1,2).float()


def boundary_from_label(label, num_classes, radius=3):
    # class-agnostic boundary band from multi-class label
    # label: [B,H,W]
    onehot = label_to_onehot(label, num_classes)  # [B,C,H,W]
    k = 2 * radius + 1
    dil = F.max_pool2d(onehot, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - onehot, kernel_size=k, stride=1, padding=radius)
    grad = (dil - ero).clamp(0, 1)
    boundary = grad.max(dim=1, keepdim=True)[0]
    return boundary
```

Boundary candidate:

```python
boundary_band = boundary_from_label(pseudo_hard, args.num_classes, args.boundary_radius).detach()
```

### 8.3 Uncertain boundary region `Rb`

```python
boundary_mask = (
    (u_ens >= args.tau_conf) &
    (u_ens <= args.tau_uncertain) &
    (boundary_band > 0) &
    (unlabeled_mask > 0)
).float()
```

Meaning:

- Vùng uncertainty cao hơn `Rc`, nhưng nằm gần predicted boundary.
- Không dùng hard CE quá mạnh.
- Dùng soft pseudo-label + boundary-aware consistency.

### 8.4 Unreliable outlier region `Ru`

```python
unreliable_mask = (
    (unlabeled_mask > 0) &
    (conf_mask <= 0) &
    (boundary_mask <= 0)
).float()
```

Không dùng hard pseudo-label ở vùng này. Có thể chỉ dùng uncertainty consistency nhẹ hoặc ignore.

---

## 9. Loss helper functions cần implement

### 9.1 Masked mean

```python
def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)
```

### 9.2 Soft cross entropy với mask

```python
def masked_soft_ce_loss(logits, target_prob, mask, eps=1e-8):
    # logits: [B,C,H,W]
    # target_prob: [B,C,H,W]
    # mask: [B,1,H,W]
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return masked_mean(ce_map, mask, eps)
```

### 9.3 Hard CE với mask

Có thể dùng soft CE với one-hot target để dễ mask.

```python
def masked_hard_ce_loss(logits, target_label, mask, num_classes, eps=1e-8):
    target_prob = label_to_onehot(target_label, num_classes)
    return masked_soft_ce_loss(logits, target_prob, mask, eps)
```

### 9.4 Boundary từ softmax prediction

Dùng differentiable morphological gradient trên probability map.

```python
def boundary_from_prob(prob, radius=3):
    # prob: [B,C,H,W]
    k = 2 * radius + 1
    dil = F.max_pool2d(prob, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - prob, kernel_size=k, stride=1, padding=radius)
    grad = (dil - ero).clamp(0, 1)
    boundary = grad.max(dim=1, keepdim=True)[0]
    return boundary
```

### 9.5 Boundary BCE loss

```python
def boundary_bce_loss(logits, target_boundary, mask=None, radius=3, eps=1e-8):
    prob = torch.softmax(logits, dim=1)
    pred_boundary = boundary_from_prob(prob, radius=radius)
    target_boundary = target_boundary.float().detach()
    loss_map = F.binary_cross_entropy(
        pred_boundary.clamp(1e-6, 1.0 - 1e-6),
        target_boundary,
        reduction='none'
    )
    if mask is None:
        return loss_map.mean()
    return masked_mean(loss_map, mask, eps)
```

Boundary mask có thể dùng:

```python
boundary_train_mask = torch.clamp(boundary_band + boundary_mask, 0, 1)
```

Version đầu có thể dùng `boundary_band` làm target và mask.

---

## 10. Student feedback mechanism

### 10.1 Anchor loss

Anchor loss dùng để đánh giá student trước/sau trial update.

Default:

```python
L_anchor = CE_scribble(student_logits, label_batch)
```

Nếu bật boundary anchor:

```python
L_anchor = CE_scribble + lambda_feedback_boundary_anchor * BoundaryBCE(student_logits, boundary_band, boundary_band)
```

Function:

```python
def compute_anchor_loss(student, volume_batch, label_batch, boundary_band=None):
    logits = unpack_model_output(student(volume_batch))
    loss = ce_loss(logits, label_batch.long())
    if args.feedback_use_boundary_anchor and boundary_band is not None:
        loss_bd = boundary_bce_loss(
            logits,
            target_boundary=boundary_band,
            mask=boundary_band,
            radius=args.boundary_radius
        )
        loss = loss + args.lambda_feedback_boundary_anchor * loss_bd
    return loss
```

Important:

- Anchor loss không update student trực tiếp trong feedback computation.
- Nó chỉ tạo scalar `delta`.

### 10.2 Region trial losses

Với `Rc`:

```python
L_trial_c = masked_soft_ce_loss(student_logits, pseudo_soft, conf_mask)
```

Với `Rb`:

```python
L_trial_b = masked_soft_ce_loss(student_logits, pseudo_soft, boundary_mask)
L_trial_b += args.lambda_boundary * boundary_bce_loss(student_logits, boundary_band, boundary_mask, args.boundary_radius)
```

Nếu mask rỗng thì return zero loss an toàn:

```python
if mask.sum() < 1:
    return torch.tensor(0.0, device=logits.device)
```

### 10.3 Safe virtual update implementation

Implement version đầu bằng clone-and-restore để dễ đúng.

Function signature:

```python
def compute_feedback_delta(
    student,
    volume_batch,
    label_batch,
    trial_loss_fn,
    boundary_band=None,
    feedback_lr=0.1,
    normalize_grad=True,
    clip_value=1.0
):
    ...
```

Pseudo-code:

```python
# 1. Anchor before
student.train()
with torch.no_grad():
    anchor_before = compute_anchor_loss(student, volume_batch, label_batch, boundary_band).detach()

# 2. Backup student params
backup = {k: v.detach().clone() for k, v in student.state_dict().items()}

# 3. Compute trial loss and grads
logits = unpack_model_output(student(volume_batch))
trial_loss = trial_loss_fn(logits)

if trial_loss requires no grad or mask empty:
    restore backup
    return 0 scalar

grads = torch.autograd.grad(trial_loss, [p for p in student.parameters() if p.requires_grad], retain_graph=False, create_graph=False, allow_unused=True)

# 4. Optional grad normalization
grad_norm = sqrt(sum(g^2))
step_scale = feedback_lr / (grad_norm + 1e-8) if normalize_grad else feedback_lr

# 5. Apply virtual update
with torch.no_grad():
    for p, g in zip(params, grads):
        if g is not None:
            p.add_(g, alpha=-step_scale)

# 6. Anchor after
with torch.no_grad():
    anchor_after = compute_anchor_loss(student, volume_batch, label_batch, boundary_band).detach()

# 7. Restore params
student.load_state_dict(backup)

# 8. Delta
delta = anchor_before - anchor_after
if clip_value > 0:
    delta = delta.clamp(-clip_value, clip_value)
return delta.detach()
```

Important implementation detail:

- Backup bằng `state_dict()` có cả BatchNorm buffers. Cần restore toàn bộ để tránh thay đổi student trong trial update.
- Nếu memory/time quá nặng, sau này optimize bằng `torch.func.functional_call`, nhưng version đầu clone-restore dễ debug hơn.

### 10.4 Feedback schedule

Chỉ compute feedback sau warmup:

```python
use_feedback = (
    iter_num >= args.feedback_start_iter and
    iter_num % args.feedback_interval == 0
)
```

Nếu chưa warmup:

```python
delta_c = 0.0
delta_b = 0.0
```

---

## 11. Feedback loss cho teachers

### 11.1 Theory sign

Feedback loss theo DualFete style:

```text
L_fb = delta * CE(teacher_logits, pseudo_target)
```

Vì:

```text
CE = -log P(target)
L_fb = -delta * log P(target) = delta * CE
```

Ý nghĩa:

- `delta > 0`: pseudo-supervision giúp student tốt hơn, minimize CE để teacher tăng likelihood target.
- `delta < 0`: pseudo-supervision làm student tệ hơn, minimize negative CE để teacher giảm likelihood target.

Vì negative CE có thể không ổn định, cần:

- clamp delta bằng `feedback_clip`.
- dùng `lambda_feedback` nhỏ, default `0.05`.
- chỉ bật sau warmup.

### 11.2 Teacher-specific receiver weight

Mỗi teacher nhận feedback theo mức đóng góp uncertainty-weighted của teacher đó:

```python
receiver_w1 = w1.detach()
receiver_w2 = w2.detach()
```

Masked CE có weight:

```python
def masked_soft_ce_loss_weighted(logits, target_prob, mask, weight=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    if weight is not None:
        mask = mask * weight
    return masked_mean(ce_map, mask, eps)
```

Teacher feedback:

```python
fb_t1_c = delta_c * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, conf_mask, receiver_w1)
fb_t1_b = delta_b * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, boundary_mask, receiver_w1)
loss_fb_t1 = fb_t1_c + fb_t1_b

fb_t2_c = delta_c * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, conf_mask, receiver_w2)
fb_t2_b = delta_b * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, boundary_mask, receiver_w2)
loss_fb_t2 = fb_t2_c + fb_t2_b
```

Nếu mask rỗng, corresponding loss = 0.

---

## 12. Cross-teacher loss với uncertainty weighting

Teacher 1 học từ Teacher 2 ở vùng Teacher 2 tự tin.
Teacher 2 học từ Teacher 1 ở vùng Teacher 1 tự tin.

```python
reliable_t1 = ((u1 < args.tau_conf) & (unlabeled_mask > 0)).float()
reliable_t2 = ((u2 < args.tau_conf) & (unlabeled_mask > 0)).float()

loss_cross_t1 = masked_soft_ce_loss_weighted(
    logits_t1,
    prob_t2.detach(),
    reliable_t2,
    weight=torch.exp(-u2).detach()
)

loss_cross_t2 = masked_soft_ce_loss_weighted(
    logits_t2,
    prob_t1.detach(),
    reliable_t1,
    weight=torch.exp(-u1).detach()
)
```

Purpose:

- Preserve dual-teacher mutual learning.
- Avoid full-image cross-supervision because scribble setting có nhiều vùng pseudo-label chưa chắc đúng.

---

## 13. Uncertainty consistency loss

Reuse function từ code hiện tại:

```python
def uncertainty_consistency_loss(student_unc, teacher_unc, loss_type='l1', mask=None):
    ...
```

Trong method mới, teacher uncertainty target là ensemble uncertainty:

```python
u_s = evidential_uncertainty_from_logits(logits_s, C, args.evidence_activation)
u_teacher = u_ens.detach()
loss_unc_s = uncertainty_consistency_loss(u_s, u_teacher, loss_type='l1', mask=unlabeled_mask)
```

Không bắt buộc cho teachers. Dùng cho student để student học reliability pattern từ teachers.

---

## 14. Student final training loss

Sau khi compute feedback, update student thật bằng refined pseudo-label.

Forward fresh:

```python
logits_s = unpack_model_output(student(volume_batch))
prob_s = torch.softmax(logits_s, dim=1)
```

Loss:

```python
loss_s_scrib = ce_loss(logits_s, label_batch.long())

loss_s_conf = masked_soft_ce_loss(logits_s, pseudo_soft, conf_mask)

loss_s_boundary_soft = masked_soft_ce_loss(logits_s, pseudo_soft, boundary_mask)
loss_s_boundary_bd = boundary_bce_loss(logits_s, boundary_band, boundary_mask, args.boundary_radius)
loss_s_boundary = loss_s_boundary_soft + args.lambda_boundary * loss_s_boundary_bd

loss_unc_s = uncertainty_consistency_loss(u_s, u_ens.detach(), loss_type='l1', mask=unlabeled_mask)

consistency_weight = get_current_consistency_weight(iter_num // 300)

loss_student = (
    loss_s_scrib
    + consistency_weight * args.lambda_pseudo_conf * loss_s_conf
    + consistency_weight * args.lambda_pseudo_boundary * loss_s_boundary
    + consistency_weight * args.lambda_unc_cons * loss_unc_s
)
```

Backprop:

```python
optimizer_s.zero_grad()
loss_student.backward()
optimizer_s.step()
```

---

## 15. Teacher final training losses

Teacher supervised loss:

```python
loss_t1_scrib = ce_loss(logits_t1, label_batch.long())
loss_t2_scrib = ce_loss(logits_t2, label_batch.long())
```

Boundary loss for teachers:

```python
loss_bd_t1 = boundary_bce_loss(logits_t1, boundary_band, boundary_mask, args.boundary_radius)
loss_bd_t2 = boundary_bce_loss(logits_t2, boundary_band, boundary_mask, args.boundary_radius)
```

Feedback loss computed as section 11.

Full objective:

```python
loss_teacher1 = (
    loss_t1_scrib
    + consistency_weight * args.lambda_cross_teacher * loss_cross_t1
    + args.lambda_feedback * loss_fb_t1
    + consistency_weight * args.lambda_boundary * loss_bd_t1
)

loss_teacher2 = (
    loss_t2_scrib
    + consistency_weight * args.lambda_cross_teacher * loss_cross_t2
    + args.lambda_feedback * loss_fb_t2
    + consistency_weight * args.lambda_boundary * loss_bd_t2
)
```

Update:

```python
optimizer_t1.zero_grad()
loss_teacher1.backward(retain_graph=True)
optimizer_t1.step()

optimizer_t2.zero_grad()
loss_teacher2.backward()
optimizer_t2.step()
```

Better implementation to avoid graph issues:

- Compute teacher losses from one forward pass.
- Cross targets and pseudo targets should be detached.
- If `loss_teacher1` and `loss_teacher2` share graph through `pseudo_soft`, make sure `pseudo_soft` is detached.
- Then no need `retain_graph=True` in most cases.

Recommended:

```python
optimizer_t1.zero_grad()
optimizer_t2.zero_grad()
(loss_teacher1 + loss_teacher2).backward()
optimizer_t1.step()
optimizer_t2.step()
```

Then update student separately with fresh student forward.

---

## 16. Full training step order

Recommended order per iteration:

```text
1. Load image and scribble label.
2. Forward teacher1 and teacher2.
3. Compute probability maps, evidential uncertainty maps.
4. Generate uncertainty-weighted pseudo_soft and pseudo_hard.
5. Build masks: conf_mask, boundary_mask, unreliable_mask.
6. If feedback enabled:
      compute delta_c via student virtual update on Rc
      compute delta_b via student virtual update on Rb
   else:
      delta_c = 0, delta_b = 0
7. Compute teacher losses:
      scribble CE
      uncertainty-weighted cross-teaching
      feedback loss
      boundary loss
8. Update teacher1 and teacher2.
9. Forward student fresh.
10. Compute student losses:
      scribble CE
      confident pseudo-label loss
      soft boundary pseudo-label loss
      uncertainty consistency loss
11. Update student.
12. Update LR for all optimizers.
13. Log scalars and debug visualizations.
14. Validate student every 200 iterations.
15. Save best student checkpoint.
```

Important:

- Teacher update should use pseudo targets computed before teacher update.
- Student update should use detached pseudo targets computed before teacher update. This is okay for stability.
- For later refinement, can recompute pseudo targets after teacher update, but not needed in version 1.

---

## 17. Learning rate schedule

Reuse existing polynomial decay:

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
for optimizer in [optimizer_t1, optimizer_t2, optimizer_s]:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
```

---

## 18. Logging requirements

Log these scalars:

```python
writer.add_scalar('info/lr', lr_, iter_num)
writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

writer.add_scalar('loss/student_total', loss_student.item(), iter_num)
writer.add_scalar('loss/student_scrib', loss_s_scrib.item(), iter_num)
writer.add_scalar('loss/student_conf', loss_s_conf.item(), iter_num)
writer.add_scalar('loss/student_boundary', loss_s_boundary.item(), iter_num)
writer.add_scalar('loss/student_unc', loss_unc_s.item(), iter_num)

writer.add_scalar('loss/teacher1_total', loss_teacher1.item(), iter_num)
writer.add_scalar('loss/teacher2_total', loss_teacher2.item(), iter_num)
writer.add_scalar('loss/cross_t1', loss_cross_t1.item(), iter_num)
writer.add_scalar('loss/cross_t2', loss_cross_t2.item(), iter_num)
writer.add_scalar('loss/fb_t1', loss_fb_t1.item(), iter_num)
writer.add_scalar('loss/fb_t2', loss_fb_t2.item(), iter_num)

writer.add_scalar('feedback/delta_c', float(delta_c), iter_num)
writer.add_scalar('feedback/delta_b', float(delta_b), iter_num)

writer.add_scalar('unc/u1_mean', u1.mean().item(), iter_num)
writer.add_scalar('unc/u2_mean', u2.mean().item(), iter_num)
writer.add_scalar('unc/u_ens_mean', u_ens.mean().item(), iter_num)

writer.add_scalar('mask/conf_ratio', conf_mask.mean().item(), iter_num)
writer.add_scalar('mask/boundary_ratio', boundary_mask.mean().item(), iter_num)
writer.add_scalar('mask/unreliable_ratio', unreliable_mask.mean().item(), iter_num)
```

Console log every 200 iterations:

```text
iteration X : loss_s=..., loss_t1=..., loss_t2=..., dice=..., dc=..., db=..., conf=..., boundary=..., unc=...
```

---

## 19. Debug visualization

Nếu `--save_debug_vis 1`, mỗi `debug_vis_interval` iterations save PNG vào:

```text
{snapshot_path}/debug_vis/iter_xxxx.png
```

Visualization nên có ít nhất 8 panels:

```text
1. input image
2. scribble label, show ignore=4 as black/transparent
3. teacher1 prediction
4. teacher2 prediction
5. pseudo_hard
6. evidential uncertainty ensemble u_ens
7. confident mask Rc
8. boundary mask Rb
```

Optional thêm:

```text
9. boundary band
10. unreliable mask Ru
```

Function signature:

```python
def save_debug_visualization(volume, label, pred_t1, pred_t2, pseudo, u_ens, conf_mask, boundary_mask, boundary_band, save_path):
    ...
```

Use `matplotlib.use("Agg")`.

---

## 20. Validation

Reuse existing validation:

```python
metric_i = test_single_volume_scribblevs(
    sampled_batch["image"], sampled_batch["label"], student, classes=num_classes
)
```

Evaluate every 200 iterations like current code.

Save best by mean Dice:

```python
save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
torch.save(student.state_dict(), save_best)
```

Also save teachers when best:

```python
torch.save(teacher1.state_dict(), os.path.join(snapshot_path, '{}_teacher1_best_model.pth'.format(args.model)))
torch.save(teacher2.state_dict(), os.path.join(snapshot_path, '{}_teacher2_best_model.pth'.format(args.model)))
```

---

## 21. Numerical stability checklist

### 21.1 Empty masks

Every masked loss must handle `mask.sum() == 0`.

Implement helper:

```python
def zero_if_empty(mask, device):
    return torch.tensor(0.0, device=device)
```

Inside each masked loss:

```python
if mask.sum() < 1:
    return torch.tensor(0.0, device=logits.device)
```

### 21.2 Negative feedback instability

Because `delta < 0` makes teacher maximize CE, keep:

```python
lambda_feedback <= 0.05
feedback_clip <= 1.0
feedback_start_iter >= 3000
```

Add logging for negative feedback count:

```python
writer.add_scalar('feedback/delta_c_negative', float(delta_c < 0), iter_num)
writer.add_scalar('feedback/delta_b_negative', float(delta_b < 0), iter_num)
```

### 21.3 Gradient explosion

Clip gradients optionally:

```python
--grad_clip 0.0
```

If enabled:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

### 21.4 BatchNorm during virtual update

For `compute_feedback_delta`, set student train mode but restore state dict after trial. This restores BatchNorm buffers.

If feedback computation still unstable, temporarily switch BN eval during virtual update:

```python
--feedback_bn_eval 1
```

Optional for later.

---

## 22. Minimal runnable command

From project train folder:

```bash
cd /path/to/project/code/train
python train_uf_dualscribble_acdc.py \
  --root_path ../../data/ACDC \
  --exp UF_DualScribble \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet \
  --num_classes 4 \
  --max_iterations 60000 \
  --batch_size 16 \
  --base_lr 0.01 \
  --gpu 0 \
  --tau_conf 0.35 \
  --tau_uncertain 0.75 \
  --pseudo_conf_thresh 0.60 \
  --lambda_feedback 0.05 \
  --lambda_boundary 0.2 \
  --feedback_start_iter 3000
```

Quick debug run:

```bash
python train_uf_dualscribble_acdc.py \
  --root_path ../../data/ACDC \
  --exp debug_UF_DualScribble \
  --max_iterations 1000 \
  --batch_size 4 \
  --gpu 0 \
  --save_debug_vis 1 \
  --debug_vis_interval 100
```

---

## 23. Expected checkpoint structure

```text
../../checkpoints/ACDC_UF_DualScribble/
├── log.txt
├── log/
│   └── tensorboard files
├── debug_vis/
│   ├── iter_1000.png
│   ├── iter_2000.png
│   └── ...
├── iter_3000.pth
├── iter_6000.pth
├── unet_best_model.pth
├── unet_teacher1_best_model.pth
└── unet_teacher2_best_model.pth
```

---

## 24. Ablation plan cần code support bằng flags

Agent nên viết code sao cho có thể bật/tắt các component sau:

| Variant | Flags |
|---|---|
| Scribble CE only | `lambda_pseudo_conf=0`, `lambda_pseudo_boundary=0`, `lambda_cross_teacher=0`, `lambda_feedback=0`, `lambda_boundary=0`, `lambda_unc_cons=0` |
| + pseudo confident core | `lambda_pseudo_conf=1`, others 0 |
| + uncertainty partition | enable `tau_conf`, `tau_uncertain`, `pseudo_conf_thresh` |
| + dual teacher cross | `lambda_cross_teacher=0.5` |
| + boundary refinement | `lambda_boundary=0.2`, `lambda_pseudo_boundary=0.5` |
| + feedback | `lambda_feedback=0.05`, `feedback_start_iter=3000` |
| no boundary region | `use_boundary_region=0` |
| feedback only confident | `lambda_pseudo_boundary=0` or skip `delta_b` |
| feedback with boundary anchor | `feedback_use_boundary_anchor=1` |

Need ensure script still runs if any lambda = 0.

---

## 25. Recommended implementation milestones

### Milestone 1: Baseline three-model training

Implement:

- 2 teachers + 1 student.
- Scribble CE for all 3.
- Validation student.
- Checkpoint save.

Goal: script runs and gets non-zero Dice.

### Milestone 2: Evidential uncertainty + masks

Implement:

- `u1`, `u2`, `u_ens`.
- `pseudo_soft`, `pseudo_hard`.
- `conf_mask`, `boundary_mask`, `unreliable_mask`.
- TensorBoard mask ratio logs.
- Debug visualization.

Goal: masks visually reasonable.

### Milestone 3: Student pseudo-label training

Implement:

- `loss_s_conf`.
- `loss_s_boundary`.
- `loss_unc_s`.

Goal: student improves over scribble-only.

### Milestone 4: Cross-teacher loss

Implement:

- `loss_cross_t1`, `loss_cross_t2`.

Goal: teachers improve and pseudo-labels become less noisy.

### Milestone 5: Feedback

Implement:

- virtual update function.
- `delta_c`, `delta_b`.
- feedback losses for teachers.

Goal: deltas are finite, not always zero, not exploding.

### Milestone 6: Stability tuning

Tune:

- `tau_conf` in `[0.25, 0.45]`.
- `tau_uncertain` in `[0.65, 0.85]`.
- `lambda_feedback` in `[0.01, 0.05, 0.1]`.
- `lambda_boundary` in `[0.1, 0.2, 0.5]`.

---

## 26. Key design decisions and rationale

### Why not use agreement/disagreement?

In scribble-supervised segmentation, two teachers may agree on a wrong boundary because sparse scribble does not constrain boundary pixels. Therefore agreement is not necessarily reliability. Evidential uncertainty is a more direct signal for pseudo-label reliability.

### Why keep uncertain boundary instead of ignoring it?

Boundary pixels are naturally uncertain but highly informative. Ignoring them improves pseudo-label precision but hurts boundary recall and HD95. Therefore uncertain boundary pixels should receive soft supervision and boundary consistency rather than hard CE or full ignore.

### Why feedback from student?

Pseudo-label confidence only tells whether teachers are confident. It does not tell whether learning from those pseudo-labels improves the model under scribble supervision. Student feedback estimates whether a region-level pseudo-supervision step aligns with scribble anchor supervision.

### Why boundary band from pseudo-label?

Scribble labels rarely annotate object boundaries. A boundary band from uncertainty-weighted pseudo-label gives an approximate boundary candidate. The loss is weak and should be combined with uncertainty filtering, not used as full ground truth.

---

## 27. Pseudo-code summary

```python
for epoch in range(max_epoch):
    for batch in trainloader:
        image, label = batch['image'].cuda(), batch['label'].cuda()
        scribble_mask = (label != 4).float().unsqueeze(1)
        unlabeled_mask = (label == 4).float().unsqueeze(1)

        # -----------------------------
        # 1. Teacher forward
        # -----------------------------
        logits_t1 = unpack_model_output(teacher1(image))
        logits_t2 = unpack_model_output(teacher2(image))
        prob_t1 = torch.softmax(logits_t1, dim=1)
        prob_t2 = torch.softmax(logits_t2, dim=1)

        u1 = evidential_uncertainty_from_logits(logits_t1, C, args.evidence_activation)
        u2 = evidential_uncertainty_from_logits(logits_t2, C, args.evidence_activation)
        u_ens = 0.5 * (u1 + u2)

        w1 = exp(-u1 / temp) / (exp(-u1 / temp) + exp(-u2 / temp) + eps)
        w2 = 1.0 - w1
        pseudo_soft = (w1 * prob_t1 + w2 * prob_t2).detach()
        pseudo_hard = pseudo_soft.argmax(dim=1)
        pseudo_conf = pseudo_soft.max(dim=1, keepdim=True)[0]

        boundary_band = boundary_from_label(pseudo_hard, C, args.boundary_radius).detach()
        conf_mask = ((u_ens < tau_conf) & (pseudo_conf > conf_thresh) & (unlabeled_mask > 0)).float()
        boundary_mask = ((u_ens >= tau_conf) & (u_ens <= tau_uncertain) & (boundary_band > 0) & (unlabeled_mask > 0)).float()

        # -----------------------------
        # 2. Student feedback
        # -----------------------------
        if use_feedback:
            delta_c = compute_feedback_delta(student, image, label, trial_conf_loss_fn, boundary_band)
            delta_b = compute_feedback_delta(student, image, label, trial_boundary_loss_fn, boundary_band)
        else:
            delta_c = torch.tensor(0.0, device=image.device)
            delta_b = torch.tensor(0.0, device=image.device)

        # -----------------------------
        # 3. Update teachers
        # -----------------------------
        loss_t1_scrib = ce_loss(logits_t1, label.long())
        loss_t2_scrib = ce_loss(logits_t2, label.long())

        loss_cross_t1 = masked_soft_ce_loss_weighted(logits_t1, prob_t2.detach(), reliable_t2, exp(-u2).detach())
        loss_cross_t2 = masked_soft_ce_loss_weighted(logits_t2, prob_t1.detach(), reliable_t1, exp(-u1).detach())

        loss_fb_t1 = delta_c * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, conf_mask, w1.detach()) \
                   + delta_b * masked_soft_ce_loss_weighted(logits_t1, pseudo_soft, boundary_mask, w1.detach())
        loss_fb_t2 = delta_c * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, conf_mask, w2.detach()) \
                   + delta_b * masked_soft_ce_loss_weighted(logits_t2, pseudo_soft, boundary_mask, w2.detach())

        loss_bd_t1 = boundary_bce_loss(logits_t1, boundary_band, boundary_mask, args.boundary_radius)
        loss_bd_t2 = boundary_bce_loss(logits_t2, boundary_band, boundary_mask, args.boundary_radius)

        loss_teacher1 = loss_t1_scrib + cw * lambda_cross * loss_cross_t1 + lambda_fb * loss_fb_t1 + cw * lambda_bd * loss_bd_t1
        loss_teacher2 = loss_t2_scrib + cw * lambda_cross * loss_cross_t2 + lambda_fb * loss_fb_t2 + cw * lambda_bd * loss_bd_t2

        optimizer_t1.zero_grad(); optimizer_t2.zero_grad()
        (loss_teacher1 + loss_teacher2).backward()
        optimizer_t1.step(); optimizer_t2.step()

        # -----------------------------
        # 4. Update student
        # -----------------------------
        logits_s = unpack_model_output(student(image))
        u_s = evidential_uncertainty_from_logits(logits_s, C, args.evidence_activation)

        loss_s_scrib = ce_loss(logits_s, label.long())
        loss_s_conf = masked_soft_ce_loss(logits_s, pseudo_soft, conf_mask)
        loss_s_boundary = masked_soft_ce_loss(logits_s, pseudo_soft, boundary_mask) \
                        + lambda_bd * boundary_bce_loss(logits_s, boundary_band, boundary_mask, args.boundary_radius)
        loss_s_unc = uncertainty_consistency_loss(u_s, u_ens.detach(), 'l1', unlabeled_mask)

        loss_student = loss_s_scrib + cw * lambda_pseudo_conf * loss_s_conf \
                     + cw * lambda_pseudo_boundary * loss_s_boundary \
                     + cw * lambda_unc_cons * loss_s_unc

        optimizer_s.zero_grad()
        loss_student.backward()
        optimizer_s.step()
```

---

## 28. Acceptance criteria

The implementation is considered successful if:

1. Script runs for at least 1000 iterations without NaN.
2. TensorBoard logs show non-zero `conf_ratio` and `boundary_ratio` after warmup.
3. Debug visualization correctly displays uncertainty map and masks.
4. Validation runs using student model.
5. Best checkpoint is saved as `unet_best_model.pth`.
6. Disabling feedback with `lambda_feedback=0` still works.
7. Disabling boundary region with `use_boundary_region=0` still works.
8. `delta_c` and `delta_b` are finite after `feedback_start_iter`.
9. Training memory is acceptable on GPU used for current ACDC experiments.

---

## 29. Common bugs to avoid

1. **Backprop into pseudo target**: always detach `pseudo_soft`, `pseudo_hard`, `boundary_band`.
2. **Mask shape mismatch**: masks should be `[B,1,H,W]`, logits `[B,C,H,W]`.
3. **Ignore index in one-hot**: do not one-hot original scribble label with value 4. Only one-hot `pseudo_hard`, which is in `[0,C-1]`.
4. **Empty mask loss NaN**: every masked loss must guard `mask.sum() < 1`.
5. **Student virtual update accidentally kept**: always restore state dict after feedback delta computation.
6. **Teacher/student graph entanglement**: detach pseudo targets and cross-teacher targets.
7. **Feedback too early**: start feedback after warmup, e.g. 3000 iterations.
8. **Negative feedback instability**: clamp delta and keep lambda small.

---

## 30. Initial hyperparameter recommendation

Use this first:

```text
tau_conf = 0.35
tau_uncertain = 0.75
pseudo_conf_thresh = 0.60
uncertainty_temp = 0.5
lambda_pseudo_conf = 1.0
lambda_pseudo_boundary = 0.5
lambda_boundary = 0.2
lambda_cross_teacher = 0.5
lambda_feedback = 0.03
lambda_unc_cons = 0.1
feedback_start_iter = 3000
feedback_lr = 0.1
feedback_clip = 1.0
boundary_radius = 3
```

If NaN or unstable:

```text
lambda_feedback -> 0.01
feedback_start_iter -> 6000
tau_conf -> 0.25
pseudo_conf_thresh -> 0.70
```

If boundary mask too small:

```text
tau_uncertain -> 0.85
boundary_radius -> 5
pseudo_conf_thresh -> 0.50
```

If too many unreliable pixels:

```text
tau_conf -> 0.45
tau_uncertain -> 0.85
```

---

## 31. Final conceptual summary

Implement a dual-teacher single-student scribble-supervised segmentation framework where:

- Teachers produce pseudo-labels and evidential uncertainty maps.
- Uncertainty partitions non-scribble pixels into confident core, uncertain boundary, and unreliable outlier regions.
- Student learns from confident and boundary regions, then provides feedback by measuring whether region-specific pseudo-supervision improves scribble-anchor loss.
- Feedback adjusts teacher pseudo-label likelihoods.
- Boundary-aware soft supervision compensates for missing boundary labels in scribble annotation.

The main contribution should stay simple:

```text
uncertainty-partitioned feedback + boundary-aware pseudo-label refinement
```

Do not add extra complex modules unless this minimal version is stable and gives improvement.
