# SPEC — Dual-Teacher Scribble Feedback for Scribble-Supervised Medical Image Segmentation

## 1. Goal

Implement a new training script for **scribble-supervised medical image segmentation** by adapting the **feedback-coupled teacher-student** idea from DualFete into the existing ACDC scribble codebase.

Proposed method name:

```text
Dual-Teacher Scribble Feedback (DTSF)
```

Recommended script name:

```text
train_scribble_dual_feedback.py
```

Core idea:

> Each sample has sparse scribble labels. The two teachers generate dense pseudo-labels only for non-scribble pixels. Reliable pseudo-label pixels are selected by teacher agreement/disagreement confidence. The student performs a virtual update using these pseudo-labels, and the change of the student's partial cross-entropy loss on scribble pixels is used as feedback. Positive feedback reinforces the corresponding teacher pseudo-label likelihood; negative feedback suppresses it.

Inference uses **student only**.

---

## 2. Source code mapping

### 2.1. From `train_scribble.py`

Keep the ACDC scribble training infrastructure:

```python
from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from val import test_single_volume_scribblevs
```

Keep the dataset setup:

```python
db_train = ACDCDataSets(
    base_dir=args.root_path,
    split="train",
    transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    fold=args.fold,
    sup_type=args.sup_type
)
db_val = ACDCDataSets(base_dir=args.root_path, fold=args.fold, split="val")
```

Keep scribble partial CE:

```python
ce_loss = CrossEntropyLoss(ignore_index=4)
```

In this codebase:

```text
label != 4  -> scribble-labeled pixels
label == 4  -> unlabeled / non-scribble pixels
```

Therefore:

```python
scribble_mask = (label_batch != 4).float().unsqueeze(1)  # [B,1,H,W]
unlabeled_mask = (label_batch == 4).float().unsqueeze(1) # [B,1,H,W]
```

### 2.2. From `train_semi_feedback.py`

Reuse the dual-teacher reliable pseudo-label mechanism:

```python
math_mask_lowconf = (math_pl == ling_pl) & (math_prob < ling_prob)
ling_mask_lowconf = (math_pl == ling_pl) & (math_prob >= ling_prob)

math_mask_highconf = (math_pl != ling_pl) & (math_prob > ling_prob)
ling_mask_highconf = (math_pl != ling_pl) & (math_prob <= ling_prob)

math_mask = math_mask_lowconf | math_mask_highconf
u_pl = torch.where(math_mask == 1, math_pl, ling_pl)

agreement = math_mask_lowconf | ling_mask_lowconf
disagreement = math_mask_highconf | ling_mask_highconf
```

Adaptation to scribble:

- Use all batch samples as scribble-supervised samples.
- Do **not** split batch into labeled/unlabeled images.
- Split pixels instead:
  - `scribble_mask` is the feedback anchor.
  - `unlabeled_mask` is the pseudo-label learning region.
- Pseudo-labels and feedback masks must be restricted to `label_batch == 4`.

---

## 3. Method overview

Use three networks:

```python
teacher_1 = create_model(num_classes=args.num_classes)
teacher_2 = create_model(num_classes=args.num_classes)
student = create_model(num_classes=args.num_classes)
```

All three are trainable.

Training objectives:

```text
Student:
    L_student = L_scribble_student + lambda_pseudo * rampup * L_pseudo_student

Teachers:
    L_teacher = L_scribble_teacher_1
              + L_scribble_teacher_2
              + lambda_fb * L_feedback
              + lambda_cross * rampup * L_cross_teacher
```

Where:

- `L_scribble_student`: partial CE on sparse scribble pixels.
- `L_pseudo_student`: pseudo-label loss on reliable non-scribble pixels.
- `L_scribble_teacher_*`: partial CE on sparse scribble pixels.
- `L_feedback`: feedback-coupled teacher loss from student virtual updates.
- `L_cross_teacher`: optional but recommended cross-teacher consistency on reliable pseudo pixels.

---

## 4. Key adaptation from semi-supervised DualFete to scribble-supervised setting

Original DualFete computes feedback by checking whether a pseudo-label-induced student update improves the student on labeled data.

For scribble supervision, full masks are unavailable. Replace labeled data with scribble pixels:

```text
Original:
    delta = L_l(student_before; full_label) - L_l(student_after; full_label)

Scribble version:
    delta = PartialCE(student_before; scribble_pixels)
          - PartialCE(student_after; scribble_pixels)
```

So:

```python
delta = scribble_loss_before - scribble_loss_after
```

Interpretation:

```text
delta > 0:
    pseudo-label update improves the student on scribble pixels.
    Reinforce the responsible teacher likelihood.

delta < 0:
    pseudo-label update hurts the student on scribble pixels.
    Suppress the responsible teacher likelihood.
```

---

## 5. CLI arguments to add

Add these args to `train_scribble_dual_feedback.py`:

```python
parser.add_argument('--lambda_pseudo', type=float, default=1.0,
                    help='weight for student pseudo-label loss')

parser.add_argument('--lambda_fb', type=float, default=0.1,
                    help='weight for dual-teacher feedback loss')

parser.add_argument('--lambda_cross', type=float, default=0.5,
                    help='weight for cross-teacher pseudo supervision')

parser.add_argument('--pseudo_agree_thresh', type=float, default=0.7,
                    help='minimum min-confidence for agreement pseudo pixels')

parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.8,
                    help='minimum max-confidence for disagreement pseudo pixels')

parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='minimum confidence margin for disagreement pseudo pixels')

parser.add_argument('--feedback_warmup', type=int, default=1000,
                    help='start feedback loss after this iteration')

parser.add_argument('--pseudo_warmup', type=int, default=500,
                    help='start pseudo supervision after this iteration')

parser.add_argument('--cross_warmup', type=int, default=500,
                    help='start cross-teacher loss after this iteration')

parser.add_argument('--feedback_lr_factor', type=float, default=1.0,
                    help='virtual student update step = current_lr * feedback_lr_factor')

parser.add_argument('--delta_clip', type=float, default=1.0,
                    help='clip feedback delta into [-delta_clip, delta_clip]')

parser.add_argument('--normalize_delta', type=int, default=1,
                    help='normalize delta by scribble loss before virtual update')

parser.add_argument('--step_normgrad', type=int, default=0,
                    help='normalize virtual update gradient in BackupModel')

parser.add_argument('--teacher_scribble_loss', type=int, default=1,
                    help='whether to train teachers with partial CE on scribbles')

parser.add_argument('--use_pseudo_dice', type=int, default=1,
                    help='whether to add pDLoss on pseudo labels')
```

---

## 6. Helper functions/classes to implement

### 6.1. `create_model`

Use the existing ACDC factory style.

```python
def create_model(num_classes=4):
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    return net.cuda()
```

Do **not** detach teacher parameters. Teachers must be trainable.

### 6.2. `BackupModel`

Reuse the idea from the semi-supervised feedback code.

```python
class BackupModel(object):
    def __init__(self, model, norm_grad=False):
        self.model = model
        self.backup = {}
        self.norm_grad = norm_grad

    def backup_param(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()

    def step(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                if self.norm_grad:
                    norm = torch.norm(grad)
                    if norm.item() > 0:
                        grad = grad / (norm + 1e-12)
                param.data.add_(grad, alpha=-epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].clone()
            if param.grad is not None:
                param.grad.data.zero_()
```

This avoids needing `torch.func.functional_call` and stays close to the original feedback implementation.

### 6.3. Masked hard CE

```python
def masked_hard_ce_loss(logits, target, mask, eps=1e-8):
    """
    logits: [B,C,H,W]
    target: [B,H,W]
    mask: [B,1,H,W] or [B,H,W]
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    ce_map = F.cross_entropy(logits, target.long(), reduction='none')
    return (ce_map * mask).sum() / (mask.sum() + eps)
```

### 6.4. Masked pseudo NLL for teacher feedback

```python
def masked_pseudo_nll_loss(logits, pseudo_hard, mask, eps=1e-8):
    """
    Computes -log P_teacher(pseudo_hard) on selected pixels.
    Equivalent to CE(logits, pseudo_hard) under a mask.
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    nll_map = F.cross_entropy(logits, pseudo_hard.detach().long(), reduction='none')
    return (nll_map * mask).sum() / (mask.sum() + eps)
```

### 6.5. Pseudo hard target with ignore index for Dice

```python
def build_ignore_target(pseudo_hard, mask, ignore_index=4):
    """
    pseudo_hard: [B,H,W]
    mask: [B,1,H,W] or [B,H,W]
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    target = pseudo_hard.clone()
    target[mask < 0.5] = ignore_index
    return target
```

---

## 7. Reliable pseudo-label selection from two teachers

Implement this function exactly.

```python
def build_dual_teacher_pseudo(
    logits_t1,
    logits_t2,
    label_batch,
    agree_thresh=0.7,
    disagree_thresh=0.8,
    margin_thresh=0.1,
    ignore_index=4,
):
    prob_t1 = torch.softmax(logits_t1, dim=1)
    prob_t2 = torch.softmax(logits_t2, dim=1)

    conf_t1, pl_t1 = torch.max(prob_t1, dim=1)  # [B,H,W]
    conf_t2, pl_t2 = torch.max(prob_t2, dim=1)

    unlabeled = (label_batch == ignore_index)  # [B,H,W]

    same = (pl_t1 == pl_t2)
    diff = ~same

    min_conf = torch.minimum(conf_t1, conf_t2)
    max_conf = torch.maximum(conf_t1, conf_t2)
    margin = torch.abs(conf_t1 - conf_t2)

    reliable_agree = same & (min_conf >= agree_thresh) & unlabeled
    reliable_disagree = diff & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & unlabeled

    choose_t1 = torch.where(same, torch.ones_like(same), conf_t1 > conf_t2)
    pseudo_hard = torch.where(choose_t1, pl_t1, pl_t2)

    reliable = reliable_agree | reliable_disagree

    # Receiver masks following DualFete:
    # agreement feedback -> lower-confidence teacher
    t1_lowconf = reliable_agree & (conf_t1 < conf_t2)
    t2_lowconf = reliable_agree & (conf_t1 >= conf_t2)

    # disagreement feedback -> higher-confidence teacher
    t1_highconf = reliable_disagree & (conf_t1 > conf_t2)
    t2_highconf = reliable_disagree & (conf_t1 <= conf_t2)

    return {
        'pseudo_hard': pseudo_hard.detach(),
        'reliable_mask': reliable.float().unsqueeze(1).detach(),
        'agreement_mask': reliable_agree.float().unsqueeze(1).detach(),
        'disagreement_mask': reliable_disagree.float().unsqueeze(1).detach(),
        't1_lowconf_mask': t1_lowconf.float().unsqueeze(1).detach(),
        't2_lowconf_mask': t2_lowconf.float().unsqueeze(1).detach(),
        't1_highconf_mask': t1_highconf.float().unsqueeze(1).detach(),
        't2_highconf_mask': t2_highconf.float().unsqueeze(1).detach(),
        'conf_t1': conf_t1.detach(),
        'conf_t2': conf_t2.detach(),
        'pl_t1': pl_t1.detach(),
        'pl_t2': pl_t2.detach(),
    }
```

Important:

- `reliable_agree` and `reliable_disagree` must be restricted to `label_batch == 4`.
- Scribble pixels must never be overwritten by pseudo-labels.
- Pseudo supervision only happens on reliable non-scribble pixels.

---

## 8. Feedback delta computation

```python
def compute_feedback_delta(
    student,
    backup_student,
    volume_batch,
    label_batch,
    pseudo_hard,
    feedback_mask,
    ce_loss,
    feedback_step,
    normalize_delta=True,
    delta_clip=1.0,
):
    """
    delta = CE_scribble(student_before) - CE_scribble(student_after_virtual_update)
    """
    if feedback_mask.sum() < 1:
        return volume_batch.new_tensor(0.0)

    backup_student.restore()

    with torch.no_grad():
        logits_before = student(volume_batch)
        if isinstance(logits_before, (tuple, list)):
            logits_before = logits_before[0]
        loss_before = ce_loss(logits_before, label_batch.long())

    logits_for_update = student(volume_batch)
    if isinstance(logits_for_update, (tuple, list)):
        logits_for_update = logits_for_update[0]

    tmp_loss = masked_hard_ce_loss(
        logits_for_update,
        pseudo_hard.detach(),
        feedback_mask
    )

    student.zero_grad()
    tmp_loss.backward()
    backup_student.step(epsilon=feedback_step)

    with torch.no_grad():
        logits_after = student(volume_batch)
        if isinstance(logits_after, (tuple, list)):
            logits_after = logits_after[0]
        loss_after = ce_loss(logits_after, label_batch.long())

    backup_student.restore()

    delta = (loss_before - loss_after).detach()

    if normalize_delta:
        delta = delta / (loss_before.detach() + 1e-8)

    if delta_clip > 0:
        delta = torch.clamp(delta, -delta_clip, delta_clip)

    return delta
```

Compute two feedback signals:

```python
backup_student.backup_param()

delta_agree = compute_feedback_delta(
    student=student,
    backup_student=backup_student,
    volume_batch=volume_batch,
    label_batch=label_batch,
    pseudo_hard=pseudo_info['pseudo_hard'],
    feedback_mask=pseudo_info['agreement_mask'],
    ce_loss=ce_loss,
    feedback_step=lr_ * args.feedback_lr_factor,
    normalize_delta=bool(args.normalize_delta),
    delta_clip=args.delta_clip,
)

delta_disagree = compute_feedback_delta(
    student=student,
    backup_student=backup_student,
    volume_batch=volume_batch,
    label_batch=label_batch,
    pseudo_hard=pseudo_info['pseudo_hard'],
    feedback_mask=pseudo_info['disagreement_mask'],
    ce_loss=ce_loss,
    feedback_step=lr_ * args.feedback_lr_factor,
    normalize_delta=bool(args.normalize_delta),
    delta_clip=args.delta_clip,
)
```

---

## 9. Student loss

```python
logits_s = unpack_model_output(student(volume_batch))
prob_s = torch.softmax(logits_s, dim=1)

loss_s_scribble = ce_loss(logits_s, label_batch.long())

loss_s_pseudo_ce = masked_hard_ce_loss(
    logits_s,
    pseudo_info['pseudo_hard'],
    pseudo_info['reliable_mask']
)

pseudo_target_ignore = build_ignore_target(
    pseudo_info['pseudo_hard'],
    pseudo_info['reliable_mask'],
    ignore_index=4
)

loss_s_pseudo_dice = dice_loss(
    prob_s,
    pseudo_target_ignore.unsqueeze(1)
)

if args.use_pseudo_dice:
    loss_s_pseudo = 0.5 * (loss_s_pseudo_ce + loss_s_pseudo_dice)
else:
    loss_s_pseudo = loss_s_pseudo_ce

if iter_num < args.pseudo_warmup:
    loss_s_pseudo = logits_s.new_tensor(0.0)

loss_student = loss_s_scribble + args.lambda_pseudo * consistency_weight * loss_s_pseudo
```

---

## 10. Teacher feedback loss

```python
nll_t1_low = masked_pseudo_nll_loss(
    logits_t1,
    pseudo_info['pseudo_hard'],
    pseudo_info['t1_lowconf_mask']
)

nll_t2_low = masked_pseudo_nll_loss(
    logits_t2,
    pseudo_info['pseudo_hard'],
    pseudo_info['t2_lowconf_mask']
)

nll_t1_high = masked_pseudo_nll_loss(
    logits_t1,
    pseudo_info['pseudo_hard'],
    pseudo_info['t1_highconf_mask']
)

nll_t2_high = masked_pseudo_nll_loss(
    logits_t2,
    pseudo_info['pseudo_hard'],
    pseudo_info['t2_highconf_mask']
)

loss_fb_agree = delta_agree * (nll_t1_low + nll_t2_low)
loss_fb_disagree = delta_disagree * (nll_t1_high + nll_t2_high)
loss_feedback = loss_fb_agree + loss_fb_disagree

if iter_num < args.feedback_warmup:
    loss_feedback = logits_t1.new_tensor(0.0)
```

Why this sign is correct:

```text
teacher_nll = -log P_teacher(pseudo_label)
L_fb = -delta * log P = delta * teacher_nll
```

So if `delta > 0`, minimizing feedback loss increases pseudo-label likelihood.
If `delta < 0`, minimizing feedback loss decreases pseudo-label likelihood.

---

## 11. Teacher scribble loss

```python
loss_t1_scribble = ce_loss(logits_t1, label_batch.long())
loss_t2_scribble = ce_loss(logits_t2, label_batch.long())

if args.teacher_scribble_loss:
    loss_t_scribble = loss_t1_scribble + loss_t2_scribble
else:
    loss_t_scribble = logits_t1.new_tensor(0.0)
```

This keeps the teachers grounded by real scribble annotations.

---

## 12. Cross-teacher reliable supervision

The semi-supervised code uses weak-to-strong consistency with strong views `s1_image` and `s2_image`. The ACDC scribble dataloader in `train_scribble.py` only provides `image` and `label`, so the minimal implementation should use same-image cross-teacher consistency first.

```python
loss_t1_cross = masked_hard_ce_loss(
    logits_t1,
    pseudo_info['pl_t2'],
    pseudo_info['reliable_mask']
)

loss_t2_cross = masked_hard_ce_loss(
    logits_t2,
    pseudo_info['pl_t1'],
    pseudo_info['reliable_mask']
)

loss_cross = loss_t1_cross + loss_t2_cross

if iter_num < args.cross_warmup:
    loss_cross = logits_t1.new_tensor(0.0)
```

Future upgrade:

- Modify `RandomGenerator` or dataset transform to return two strong augmented views.
- Then implement weak-to-strong consistency like the semi-supervised feedback code.

---

## 13. Teacher total loss

```python
loss_teacher = (
    loss_t_scribble
    + args.lambda_fb * loss_feedback
    + args.lambda_cross * consistency_weight * loss_cross
)
```

---

## 14. Training loop order

Use this order to avoid graph contamination:

```python
# 1. Forward both teachers
logits_t1 = unpack_model_output(teacher_1(volume_batch))
logits_t2 = unpack_model_output(teacher_2(volume_batch))

# 2. Build reliable pseudo-label and masks
pseudo_info = build_dual_teacher_pseudo(
    logits_t1=logits_t1.detach(),
    logits_t2=logits_t2.detach(),
    label_batch=label_batch,
    agree_thresh=args.pseudo_agree_thresh,
    disagree_thresh=args.pseudo_disagree_thresh,
    margin_thresh=args.pseudo_margin_thresh,
    ignore_index=4,
)

# 3. Compute feedback deltas using virtual student update
backup_student.backup_param()
delta_agree = compute_feedback_delta(...)
delta_disagree = compute_feedback_delta(...)

# 4. Forward student for real update
logits_s = unpack_model_output(student(volume_batch))
loss_student = ...

# 5. Update student
optimizer_student.zero_grad()
loss_student.backward()
optimizer_student.step()

# 6. Build teacher loss with trainable logits_t1/logits_t2 and detached pseudo_info/deltas
loss_teacher = ...

# 7. Update teachers
optimizer_teacher.zero_grad()
loss_teacher.backward()
optimizer_teacher.step()

# 8. LR schedule, logging, validation
```

Detach rules:

```text
pseudo_info['pseudo_hard'] must be detached.
all masks must be detached.
delta_agree and delta_disagree must be detached.
loss_student must not update teachers.
loss_teacher must not update student.
```

---

## 15. Optimizers

```python
optimizer_student = optim.SGD(
    student.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_teacher = optim.SGD(
    list(teacher_1.parameters()) + list(teacher_2.parameters()),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001
)
```

LR schedule:

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

for param_group in optimizer_student.param_groups:
    param_group['lr'] = lr_

for param_group in optimizer_teacher.param_groups:
    param_group['lr'] = lr_
```

---

## 16. Validation and checkpointing

Evaluate **student only**:

```python
student.eval()
metric_i = test_single_volume_scribblevs(
    sampled_batch["image"],
    sampled_batch["label"],
    student,
    classes=num_classes
)
student.train()
```

Save best student:

```python
torch.save(student.state_dict(), save_best)
```

Optional save teachers for debugging:

```python
torch.save(teacher_1.state_dict(), os.path.join(snapshot_path, f'{args.model}_best_teacher1.pth'))
torch.save(teacher_2.state_dict(), os.path.join(snapshot_path, f'{args.model}_best_teacher2.pth'))
```

---

## 17. Logging

Add TensorBoard scalars:

```python
writer.add_scalar('info/lr', lr_, iter_num)
writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

writer.add_scalar('loss/student_total', loss_student.item(), iter_num)
writer.add_scalar('loss/student_scribble', loss_s_scribble.item(), iter_num)
writer.add_scalar('loss/student_pseudo', loss_s_pseudo.item(), iter_num)

writer.add_scalar('loss/teacher_total', loss_teacher.item(), iter_num)
writer.add_scalar('loss/teacher_scribble', loss_t_scribble.item(), iter_num)
writer.add_scalar('loss/teacher_feedback', loss_feedback.item(), iter_num)
writer.add_scalar('loss/teacher_cross', loss_cross.item(), iter_num)

writer.add_scalar('feedback/delta_agree', delta_agree.item(), iter_num)
writer.add_scalar('feedback/delta_disagree', delta_disagree.item(), iter_num)

writer.add_scalar('mask/reliable_ratio', pseudo_info['reliable_mask'].mean().item(), iter_num)
writer.add_scalar('mask/agreement_ratio', pseudo_info['agreement_mask'].mean().item(), iter_num)
writer.add_scalar('mask/disagreement_ratio', pseudo_info['disagreement_mask'].mean().item(), iter_num)

writer.add_scalar('conf/t1_mean', pseudo_info['conf_t1'].mean().item(), iter_num)
writer.add_scalar('conf/t2_mean', pseudo_info['conf_t2'].mean().item(), iter_num)
```

---

## 18. Recommended command

```bash
python train_scribble_dual_feedback.py \
  --root_path ../../data/ACDC \
  --exp DTSF_ACDC \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet \
  --num_classes 4 \
  --max_iterations 60000 \
  --batch_size 16 \
  --base_lr 0.01 \
  --gpu 0 \
  --lambda_pseudo 1.0 \
  --lambda_fb 0.05 \
  --lambda_cross 0.5 \
  --pseudo_agree_thresh 0.7 \
  --pseudo_disagree_thresh 0.8 \
  --pseudo_margin_thresh 0.1 \
  --pseudo_warmup 500 \
  --feedback_warmup 1000 \
  --cross_warmup 500 \
  --feedback_lr_factor 1.0 \
  --delta_clip 0.5 \
  --normalize_delta 1 \
  --teacher_scribble_loss 1 \
  --use_pseudo_dice 1
```

If unstable:

```bash
--lambda_fb 0.01 --delta_clip 0.2 --pseudo_disagree_thresh 0.9
```

If too few pseudo pixels:

```bash
--pseudo_agree_thresh 0.6 --pseudo_disagree_thresh 0.7 --pseudo_margin_thresh 0.05
```

---

## 19. Ablation plan

### A. Scribble only

```bash
--lambda_pseudo 0 --lambda_fb 0 --lambda_cross 0
```

### B. Dual-teacher pseudo only

```bash
--lambda_pseudo 1.0 --lambda_fb 0 --lambda_cross 0
```

### C. Dual-teacher pseudo + cross teacher

```bash
--lambda_pseudo 1.0 --lambda_fb 0 --lambda_cross 0.5
```

### D. Full proposed DTSF

```bash
--lambda_pseudo 1.0 --lambda_fb 0.05 --lambda_cross 0.5
```

### E. Feedback only, no cross

```bash
--lambda_pseudo 1.0 --lambda_fb 0.05 --lambda_cross 0
```

Expected trend:

```text
Full DTSF
    > Dual-teacher pseudo + cross
    > Dual-teacher pseudo only
    > Scribble-only
```

---

## 20. Proposed method paragraph for paper/report

```text
We propose Dual-Teacher Scribble Feedback (DTSF), a feedback-coupled teacher-student framework for scribble-supervised medical image segmentation. Unlike semi-supervised settings where fully labeled samples are available, each training image in our setting contains only sparse scribble annotations. We therefore use scribble-labeled pixels as reliable anchors and non-scribble pixels as pseudo-label candidates. Two trainable teachers first predict dense probability maps. For non-scribble pixels, pseudo-labels are selected by a reliable dual-teacher rule: agreement pixels are retained only when both teachers are confident, while disagreement pixels are assigned to the higher-confidence teacher only when the confidence and margin are sufficiently large. The student is supervised by sparse scribbles and these reliable pseudo-labels. To prevent confirmation bias, we introduce a scribble-guided feedback signal. The student performs a one-step virtual update using pseudo-labels from agreement or disagreement regions, and the change in partial cross-entropy on scribble pixels is measured. Positive feedback reinforces the corresponding teacher likelihood, whereas negative feedback suppresses it. Agreement feedback is applied to the lower-confidence teacher, while disagreement feedback is applied to the higher-confidence teacher. This allows the teachers to refine dense pseudo-labels using only sparse scribble supervision, while reducing the risk of repeatedly reinforcing erroneous pseudo-labels.
```

---

## 21. Implementation checklist

- [ ] Create `train_scribble_dual_feedback.py`.
- [ ] Replace EMA model with two trainable teachers.
- [ ] Keep ACDC scribble dataloader and validation.
- [ ] Implement `BackupModel`.
- [ ] Implement `masked_hard_ce_loss`.
- [ ] Implement `masked_pseudo_nll_loss`.
- [ ] Implement `build_ignore_target`.
- [ ] Implement `build_dual_teacher_pseudo`.
- [ ] Implement `compute_feedback_delta`.
- [ ] Student loss = scribble CE + reliable pseudo CE/Dice.
- [ ] Teacher loss = teacher scribble CE + feedback + cross-teacher loss.
- [ ] Validate/save student only.
- [ ] Add logging for deltas, masks, confidence, and losses.
- [ ] Run ablations A-D.
