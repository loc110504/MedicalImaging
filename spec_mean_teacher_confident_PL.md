# SPEC: Implement Agreement/Disagreement Reliable Pseudo-labeling in Mean Teacher Scribble-Supervised Medical Image Segmentation

## 1. Goal

Implement the pseudo-label mechanism from `code/train/train_dmsps_2.py` into the provided **Mean Teacher** (reference Mean Teacher update in Scribble_DualTeachers/code/train/train_scribblevs_acdc.py) framework for scribble-supervised ACDC segmentation.

The final training objective must contain only two losses:

```text
L_total = L_pCE + lambda_pseudo * L_pseudo
```

Where:

```text
L_pCE:
    partial cross-entropy on scribble-labeled pixels only

L_pseudo:
    soft cross-entropy on reliable pseudo-label pixels selected by
    confidence-based agreement/disagreement between student and EMA teacher
```

Do **not** implement uncertainty consistency loss.  
Do **not** use evidential uncertainty maps.  
Do **not** use uncertainty-weighted fusion.

---

## 2. Current framework

The provided code uses Mean Teacher:

```python
model = create_model(ema=False, num_classes=num_classes)      # student
model_ema = create_model(ema=True, num_classes=num_classes)   # EMA teacher
```

The student is updated by backpropagation.

The EMA teacher is updated only by EMA:

```python
ema_optimizer.step()
```

Training data uses scribble labels. With ACDC and `num_classes = 4`, the ignored / unlabeled region is:

```python
ignore_index = num_classes  # 4
```

The existing partial CE is:

```python
ce_loss = CrossEntropyLoss(ignore_index=num_classes)
loss_pce = ce_loss(outputs, label_batch.long())
```

---

## 3. Method overview

For each training batch:

```text
Input image + scribble label
        ↓
Student forward
        ↓
EMA teacher forward with no_grad
        ↓
Student softmax probability
Teacher softmax probability
        ↓
Compute LpCE on scribble pixels
        ↓
Build soft pseudo-labels using confidence-based agree/disagree rule
        ↓
Select reliable pseudo-label pixels only
        ↓
Compute soft CE on reliable pseudo-label pixels
        ↓
L_total = LpCE + lambda_pseudo * L_pseudo
        ↓
Update student
        ↓
Update EMA teacher
```

The key rule:

```text
Pseudo-label loss must be applied only on:
reliable_agree OR reliable_disagree
```

Not on all unlabeled pixels.

---

## 4. Arguments to add / keep

Add these arguments:

```python
parser.add_argument('--pseudo_agree_thresh', type=float, default=0.7,
                    help='minimum confidence for both student and teacher when they agree')

parser.add_argument('--pseudo_disagree_thresh', type=float, default=0.8,
                    help='minimum confidence for the stronger prediction when student and teacher disagree')

parser.add_argument('--pseudo_margin_thresh', type=float, default=0.1,
                    help='minimum confidence margin between student and teacher when they disagree')

parser.add_argument('--pseudo_loss_weight', type=float, default=8.0,
                    help='weight for reliable pseudo-label supervision')

parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'all'],
                    help='where to apply pseudo-label supervision')
```

Remove or ignore these uncertainty-related arguments if they are no longer used:

```python
--lambda_unc_cons
--uncertainty_consistency
--evidence_activation
--detach_teacher_uncertainty
--uncertainty_mask_mode
--uncertainty_temp
--uncertainty_target
--use_uncertainty_consistency
```

The final method should not depend on any of the above.

---

## 5. Functions to remove or ignore

The final implementation does not need:

```python
logits_to_evidence(...)
evidential_uncertainty_from_logits(...)
uncertainty_weighted_fusion(...)
uncertainty_consistency_loss(...)
```

Remove them if the codebase allows.  
If removing them may break other imports or experiments, leave them unused but do not call them.

---

## 6. Keep this function: masked soft CE

Keep or add this function:

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

Expected tensor shapes:

```text
logits:      [B, C, H, W]
target_prob: [B, C, H, W]
mask:        [B, 1, H, W]
```

---

## 7. Function to add: Mean Teacher agreement/disagreement pseudo-label builder

Add this function near the loss utilities.

```python
def build_mt_confidence_pseudo_label(
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
    """
    Build reliable soft pseudo-labels using confidence-based agreement/disagreement
    between the student and EMA teacher.

    Args:
        student_prob:
            Tensor [B, C, H, W], softmax probability from student.
        teacher_prob:
            Tensor [B, C, H, W], softmax probability from EMA teacher.
        label:
            Tensor [B, H, W], scribble label.
        agree_thresh:
            Agreement threshold. If student and teacher predict the same class,
            both confidences must be at least this value.
        disagree_thresh:
            Disagreement threshold. If student and teacher predict different classes,
            the stronger prediction must be at least this value.
        margin_thresh:
            Confidence margin threshold for disagreement.
        ignore_index:
            Unlabeled / ignored scribble label index.
        pseudo_mask_mode:
            'unlabeled': select reliable pseudo-label pixels only from label == ignore_index.
            'all': select reliable pixels from all image pixels.
        eps:
            Numerical stability.

    Returns:
        dict:
            soft_pseudo_label: [B, C, H, W]
            reliable_mask: [B, 1, H, W]
            agreement_ratio: scalar tensor
            disagreement_ratio: scalar tensor
            reliable_ratio: scalar tensor
            pseudo_conf: [B, 1, H, W]
    """

    student_prob = student_prob.detach()
    teacher_prob = teacher_prob.detach()

    conf_s, pred_s = torch.max(student_prob, dim=1)   # [B, H, W]
    conf_t, pred_t = torch.max(teacher_prob, dim=1)   # [B, H, W]

    if pseudo_mask_mode == 'unlabeled':
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == 'all':
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError('Unsupported pseudo_mask_mode: {}'.format(pseudo_mask_mode))

    same_pred = pred_s == pred_t
    diff_pred = ~same_pred

    min_conf = torch.minimum(conf_s, conf_t)
    max_conf = torch.maximum(conf_s, conf_t)
    margin = torch.abs(conf_s - conf_t)

    # Case 1: agreement.
    # Student and teacher predict the same class, and both are confident.
    reliable_agree = (
        same_pred
        & (min_conf >= agree_thresh)
        & candidate_mask
    )

    # Case 2: disagreement.
    # Student and teacher predict different classes, but one side is clearly stronger.
    reliable_disagree = (
        diff_pred
        & (max_conf >= disagree_thresh)
        & (margin >= margin_thresh)
        & candidate_mask
    )

    # Agreement target:
    # If student and teacher agree, use the mean probability distribution.
    mean_pseudo = 0.5 * (student_prob + teacher_prob)

    # Disagreement target:
    # If they disagree but one is clearly more confident, use the stronger prediction.
    choose_student = (conf_s > conf_t).unsqueeze(1)  # [B, 1, H, W]
    high_conf_pseudo = torch.where(choose_student, student_prob, teacher_prob)

    # Use high-confidence distribution on disagreement regions,
    # otherwise use mean distribution.
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
        'soft_pseudo_label': soft_pseudo_label.detach(),
        'reliable_mask': reliable_mask.detach(),
        'reliable_agree': reliable_agree,
        'reliable_disagree': reliable_disagree,
        'agreement_ratio': reliable_agree.float().mean(),
        'disagreement_ratio': reliable_disagree.float().mean(),
        'reliable_ratio': reliable_mask.mean(),
        'pseudo_conf': pseudo_conf.detach(),
    }
```

---

## 8. Training loop modification

Inside the training loop, use only the following losses:

```text
1. loss_pce
2. loss_pseudo
```

### 8.1 Forward student and EMA teacher

```python
volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

# EMA teacher forward
with torch.no_grad():
    ema_output = unpack_model_output(model_ema(volume_batch))
    teacher_prob = torch.softmax(ema_output, dim=1)

# Student forward
outputs = unpack_model_output(model(volume_batch))
student_prob = torch.softmax(outputs, dim=1)
```

### 8.2 Partial CE on scribble pixels

```python
loss_pce = ce_loss(outputs, label_batch.long())
```

This loss only supervises labeled scribble pixels because:

```python
ce_loss = CrossEntropyLoss(ignore_index=num_classes)
```

### 8.3 Build reliable pseudo-labels

```python
pseudo_info = build_mt_confidence_pseudo_label(
    student_prob=student_prob,
    teacher_prob=teacher_prob,
    label=label_batch,
    agree_thresh=args.pseudo_agree_thresh,
    disagree_thresh=args.pseudo_disagree_thresh,
    margin_thresh=args.pseudo_margin_thresh,
    ignore_index=num_classes,
    pseudo_mask_mode=args.pseudo_mask_mode,
)
```

### 8.4 Pseudo-label loss on reliable pixels only

```python
loss_pseudo = masked_soft_ce_loss(
    logits=outputs,
    target_prob=pseudo_info['soft_pseudo_label'],
    mask=pseudo_info['reliable_mask'],
)
```

Important:

```text
Do not compute pseudo loss on all unlabeled pixels.
Use pseudo_info['reliable_mask'] only.
```

### 8.5 Total loss

Use a ramp-up weight if desired:

```python
pseudo_weight = get_current_consistency_weight(iter_num // len(trainloader)) * args.pseudo_loss_weight
loss = loss_pce + pseudo_weight * loss_pseudo
```

If you want the simplest version without ramp-up:

```python
loss = loss_pce + args.pseudo_loss_weight * loss_pseudo
```

Recommended default: keep ramp-up.

### 8.6 Optimization

The pasted code currently logs variables before a complete optimization block. Make sure this block exists:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
ema_optimizer.step()
```

Then update learning rate:

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
for param_group in optimizer.param_groups:
    param_group['lr'] = lr_

iter_num += 1
```

---

## 9. Full core training block

Replace the core batch-training section with:

```python
volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

# 1. EMA teacher forward
with torch.no_grad():
    ema_output = unpack_model_output(model_ema(volume_batch))
    teacher_prob = torch.softmax(ema_output, dim=1)

# 2. Student forward
outputs = unpack_model_output(model(volume_batch))
student_prob = torch.softmax(outputs, dim=1)

# 3. LpCE on scribble pixels
loss_pce = ce_loss(outputs, label_batch.long())

# 4. Confidence-based agreement/disagreement pseudo-label fusion
pseudo_info = build_mt_confidence_pseudo_label(
    student_prob=student_prob,
    teacher_prob=teacher_prob,
    label=label_batch,
    agree_thresh=args.pseudo_agree_thresh,
    disagree_thresh=args.pseudo_disagree_thresh,
    margin_thresh=args.pseudo_margin_thresh,
    ignore_index=num_classes,
    pseudo_mask_mode=args.pseudo_mask_mode,
)

# 5. Soft CE on reliable pseudo-label pixels only
loss_pseudo = masked_soft_ce_loss(
    logits=outputs,
    target_prob=pseudo_info['soft_pseudo_label'],
    mask=pseudo_info['reliable_mask'],
)

# 6. Final loss: only two losses
pseudo_weight = get_current_consistency_weight(iter_num // len(trainloader)) * args.pseudo_loss_weight
loss = loss_pce + pseudo_weight * loss_pseudo

# 7. Student update and EMA teacher update
optimizer.zero_grad()
loss.backward()
optimizer.step()
ema_optimizer.step()

# 8. Poly learning-rate decay
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
for param_group in optimizer.param_groups:
    param_group['lr'] = lr_

iter_num += 1
```

---

## 10. TensorBoard logging

Add:

```python
writer.add_scalar('info/lr', lr_, iter_num)
writer.add_scalar('info/total_loss', loss.item(), iter_num)
writer.add_scalar('info/loss_pce', loss_pce.item(), iter_num)
writer.add_scalar('info/loss_pseudo', loss_pseudo.item(), iter_num)
writer.add_scalar('info/pseudo_weight', pseudo_weight, iter_num)

writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
writer.add_scalar('pseudo/agreement_ratio', pseudo_info['agreement_ratio'].item(), iter_num)
writer.add_scalar('pseudo/disagreement_ratio', pseudo_info['disagreement_ratio'].item(), iter_num)
writer.add_scalar('pseudo/pseudo_conf', pseudo_info['pseudo_conf'].mean().item(), iter_num)
```

Update console logging:

```python
if iter_num % 200 == 0:
    logging.info(
        'iteration %d : loss=%f, loss_pce=%f, loss_pseudo=%f, '
        'pseudo_weight=%f, reliable=%f, agree=%f, disagree=%f, pseudo_conf=%f'
        % (
            iter_num,
            loss.item(),
            loss_pce.item(),
            loss_pseudo.item(),
            pseudo_weight,
            pseudo_info['reliable_ratio'].item(),
            pseudo_info['agreement_ratio'].item(),
            pseudo_info['disagreement_ratio'].item(),
            pseudo_info['pseudo_conf'].mean().item(),
        )
    )
```

Remove these old logging variables:

```python
loss_uncertainty
pseudo_mask
weight_student
weight_teacher
ws
wt
```

---

## 11. Validation and checkpointing

Keep the existing validation code unchanged.

Use the student model for validation:

```python
metric_i = test_single_volume_scribblevs(
    sampled_batch["image"],
    sampled_batch["label"],
    model,
    classes=num_classes,
)
```

Save the student model:

```python
torch.save(model.state_dict(), save_best)
```

Optional: evaluate EMA teacher only for ablation, not required for the main method.

---

## 12. Final objective

The final method should optimize:

```text
L_total = L_pCE + lambda_pseudo(t) * L_pseudo
```

Where:

```text
L_pCE:
    CrossEntropyLoss(ignore_index=num_classes)
    computed on scribble-labeled pixels.

L_pseudo:
    masked soft CE between student logits and fused soft pseudo-label.
    computed only on reliable pixels.

lambda_pseudo(t):
    ramp-up weight multiplied by args.pseudo_loss_weight.
```

In code:

```python
loss_pce = ce_loss(outputs, label_batch.long())

loss_pseudo = masked_soft_ce_loss(
    outputs,
    pseudo_info['soft_pseudo_label'],
    pseudo_info['reliable_mask'],
)

pseudo_weight = get_current_consistency_weight(iter_num // len(trainloader)) * args.pseudo_loss_weight

loss = loss_pce + pseudo_weight * loss_pseudo
```

---

## 13. Acceptance criteria

The implementation is correct if:

- [ ] There are exactly two loss terms in the final objective: `loss_pce` and `loss_pseudo`.
- [ ] No uncertainty consistency loss is used.
- [ ] No evidential uncertainty map is computed.
- [ ] No uncertainty-weighted fusion is used.
- [ ] Teacher prediction is computed under `torch.no_grad()`.
- [ ] Teacher is updated only by EMA.
- [ ] Pseudo-label target is detached.
- [ ] Pseudo loss is computed only on `reliable_agree | reliable_disagree`.
- [ ] `reliable_mask` shape is `[B, 1, H, W]`.
- [ ] Code runs without undefined variables in the logging block.
- [ ] TensorBoard logs `loss_pce`, `loss_pseudo`, `reliable_ratio`, `agreement_ratio`, and `disagreement_ratio`.

---

## 14. Recommended command

```bash
python train_mean_teacher_confidence_pseudo.py   --root_path ../../data/ACDC   --data ACDC   --exp ScribbleVS_MT_ConfidenceAgreeDisagree   --fold MAAGfold70   --sup_type scribble   --model unet   --num_classes 4   --batch_size 16   --base_lr 0.01   --max_iterations 60000   --pseudo_agree_thresh 0.7   --pseudo_disagree_thresh 0.8   --pseudo_margin_thresh 0.1   --pseudo_loss_weight 8.0   --pseudo_mask_mode unlabeled   --gpu 0
```

---

## 15. Short summary for coding agent

Implement the method as:

```text
Student + EMA Teacher
        ↓
Compare confidence and prediction class
        ↓
Agreement:
    same class + both confident
    target = mean(student_prob, teacher_prob)

Disagreement:
    different class + one side clearly more confident
    target = more confident probability

Reliable mask:
    reliable_agree OR reliable_disagree

Loss:
    L_total = LpCE_scribble + lambda * Lpseudo_reliable
```

Do not include uncertainty consistency in this version.
