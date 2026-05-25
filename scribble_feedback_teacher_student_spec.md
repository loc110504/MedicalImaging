# SPEC: Scribble-Guided Feedback Teacher-Student

## 1. Mục tiêu

Implement một method mới cho **scribble-supervised medical image segmentation** dựa trên ý tưởng feedback của DualFete, nhưng được đơn giản hóa cho bài toán **full scribble supervision**.

Trong setting này, mỗi sample đều có scribble annotation. Scribble annotation là sparse ground-truth: chỉ một số pixel/voxel có nhãn thật, các pixel còn lại là unlabeled/ignore.

Ý tưởng chính:

> Teacher tạo dense pseudo-label cho vùng không có scribble. Student học tạm từ pseudo-label đó. Nếu việc học pseudo-label làm student giảm loss trên scribble pixels, pseudo-label được xem là có ích và teacher được reinforce. Nếu làm scribble loss tăng, teacher bị suppress confidence tại pseudo-label đó.

Tên method đề xuất:

```text
Scribble Feedback Teacher-Student
```

Tên script mới:

```text
train_scribble_feedback.py
```

Có thể copy từ file training hiện tại rồi sửa theo spec này.

---

## 2. Codebase hiện tại cần giữ lại

Giữ lại các thành phần chính từ training file hiện tại:

```python
ACDCDataSets
RandomGenerator
net_factory
losses.pDLoss
test_single_volume_scribblevs
SummaryWriter
CrossEntropyLoss(ignore_index=4)
```

Các setting dataset giữ nguyên:

```python
--root_path ../../data/ACDC
--fold MAAGfold70
--sup_type scribble
--num_classes 4
--patch_size [256, 256]
```

Quy ước label trong codebase:

```text
0, 1, 2, 3 = valid class labels
4 = ignore / unlabeled pixel
```

Vùng scribble thật:

```python
scribble_mask = (label_batch != 4).float().unsqueeze(1)
```

Vùng không có scribble:

```python
unlabeled_mask = (label_batch == 4).float().unsqueeze(1)
```

---

## 3. Kiến trúc model

Không dùng `model_ema` kiểu EMA-only như code hiện tại, vì teacher cần nhận gradient từ feedback loss.

Thay bằng 2 model trainable:

```python
student = create_model(num_classes=num_classes)
teacher = create_model(num_classes=num_classes)
```

Khởi tạo teacher giống student:

```python
teacher.load_state_dict(student.state_dict())
```

Tạo 2 optimizer riêng:

```python
optimizer_student = optim.SGD(
    student.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_teacher = optim.SGD(
    teacher.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001
)
```

Không dùng `WeightEMA` trong version minimal.

Nếu muốn giữ tên tương thích với code cũ:

```python
model = student
model_teacher = teacher
```

Nhưng khuyến nghị đổi tên rõ ràng thành `student` và `teacher`.

---

## 4. CLI arguments cần thêm

Thêm các arguments sau:

```python
parser.add_argument('--lambda_pseudo', type=float, default=1.0,
                    help='weight for pseudo-label supervision on non-scribble pixels')

parser.add_argument('--lambda_fb', type=float, default=0.1,
                    help='weight for teacher feedback loss')

parser.add_argument('--feedback_lr', type=float, default=0.01,
                    help='virtual step size for student feedback estimation')

parser.add_argument('--feedback_conf_thresh', type=float, default=0.0,
                    help='confidence threshold for feedback mask; 0 disables filtering')

parser.add_argument('--feedback_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled', 'uncertain_unlabeled'],
                    help='region where teacher feedback loss is applied')

parser.add_argument('--pseudo_conf_thresh', type=float, default=0.7,
                    help='confidence threshold for pseudo-label supervision')

parser.add_argument('--feedback_warmup', type=int, default=1000,
                    help='start feedback loss after this iteration')

parser.add_argument('--pseudo_warmup', type=int, default=500,
                    help='start pseudo-label supervision after this iteration')

parser.add_argument('--delta_clip', type=float, default=1.0,
                    help='clip feedback delta into [-delta_clip, delta_clip]')

parser.add_argument('--normalize_delta', type=int, default=1,
                    help='whether to normalize delta by current scribble loss')

parser.add_argument('--teacher_scribble_loss', type=int, default=1,
                    help='whether teacher is also trained by scribble partial CE')

parser.add_argument('--feedback_interval', type=int, default=1,
                    help='compute feedback every N iterations; 1 means every iteration')
```

Có thể giữ lại các args cũ về uncertainty nếu chưa muốn xóa, nhưng trong method minimal này không cần dùng:

```python
lambda_unc_cons
uncertainty_consistency
evidence_activation
uncertainty_temp
uncertainty_target
```

---

## 5. Loss functions cần implement

### 5.1. Partial CE trên scribble

Dùng loss có sẵn:

```python
ce_loss = CrossEntropyLoss(ignore_index=4)
```

Student scribble loss:

```python
loss_scribble_student = ce_loss(student_logits, label_batch.long())
```

Teacher scribble loss nếu bật:

```python
loss_scribble_teacher = ce_loss(teacher_logits, label_batch.long())
```

---

### 5.2. Masked hard CE cho pseudo-label

Thêm function:

```python
def masked_hard_ce_loss(logits, target, mask, eps=1e-8):
    """
    logits: [B, C, H, W]
    target: [B, H, W], hard pseudo-label in [0, C-1]
    mask: [B, 1, H, W], float mask
    """
    ce_map = F.cross_entropy(logits, target.long(), reduction='none')  # [B, H, W]
    mask = mask.squeeze(1)

    if mask.sum() < 1:
        return logits.new_tensor(0.0)

    return (ce_map * mask).sum() / (mask.sum() + eps)
```

---

### 5.3. Teacher pseudo-label likelihood loss

Thêm function:

```python
def masked_pseudo_nll_loss(teacher_logits, pseudo_hard, mask, eps=1e-8):
    """
    Negative log likelihood of teacher generating its own pseudo-label.
    Equivalent to CE(teacher_logits, pseudo_hard) on selected mask.
    Used inside feedback loss.
    """
    log_prob = F.log_softmax(teacher_logits, dim=1)
    log_pseudo = torch.gather(
        log_prob,
        dim=1,
        index=pseudo_hard.unsqueeze(1)
    )  # [B, 1, H, W]

    if mask.sum() < 1:
        return teacher_logits.new_tensor(0.0)

    return -(log_pseudo * mask).sum() / (mask.sum() + eps)
```

---

## 6. Teacher pseudo-label generation

Forward teacher:

```python
teacher_logits = unpack_model_output(teacher(volume_batch))
teacher_prob = torch.softmax(teacher_logits, dim=1)
pseudo_conf, pseudo_hard = torch.max(teacher_prob.detach(), dim=1)  # [B, H, W]
pseudo_conf = pseudo_conf.unsqueeze(1)
```

Pseudo-label chỉ dùng ở vùng không có scribble:

```python
pseudo_mask = (label_batch == 4).float().unsqueeze(1)
```

Nếu bật confidence threshold:

```python
if args.pseudo_conf_thresh > 0:
    pseudo_mask = pseudo_mask * (pseudo_conf > args.pseudo_conf_thresh).float()
```

---

## 7. Feedback mask

Version đơn giản nhất:

```python
feedback_mask = (label_batch == 4).float().unsqueeze(1)
```

Nếu muốn chỉ feedback vùng uncertain:

```python
if args.feedback_mask_mode == 'uncertain_unlabeled':
    feedback_mask = (label_batch == 4).float().unsqueeze(1)
    feedback_mask = feedback_mask * (pseudo_conf < args.feedback_conf_thresh).float()
```

Khuyến nghị default:

```text
feedback_mask_mode = unlabeled
feedback_conf_thresh = 0.0
```

Sau khi code chạy ổn, thử thêm:

```text
feedback_mask_mode = uncertain_unlabeled
feedback_conf_thresh = 0.8
```

---

## 8. Feedback signal design

### 8.1. Mục tiêu

Tính:

```text
delta = L_scribble(S) - L_scribble(S')
```

Trong đó:

- `S`: student hiện tại.
- `S'`: student sau một bước virtual update bằng pseudo-label.
- Nếu `delta > 0`: pseudo-label giúp giảm scribble loss.
- Nếu `delta < 0`: pseudo-label làm tăng scribble loss.

---

### 8.2. Practical implementation bằng `torch.func.functional_call`

Import thêm:

```python
from torch.func import functional_call
from collections import OrderedDict
```

Helper:

```python
def get_param_dict(model):
    return OrderedDict((name, p) for name, p in model.named_parameters())


def get_buffer_dict(model):
    return OrderedDict((name, b) for name, b in model.named_buffers())
```

Tạo virtual update:

```python
student_params = get_param_dict(student)
student_buffers = get_buffer_dict(student)

pseudo_loss_for_feedback = masked_hard_ce_loss(
    student_logits,
    pseudo_hard.detach(),
    feedback_mask
)

grads = torch.autograd.grad(
    pseudo_loss_for_feedback,
    tuple(student_params.values()),
    create_graph=False,
    retain_graph=True,
    allow_unused=True
)

virtual_params = OrderedDict()
for (name, param), grad in zip(student_params.items(), grads):
    if grad is None:
        virtual_params[name] = param
    else:
        virtual_params[name] = param - args.feedback_lr * grad
```

Forward virtual student:

```python
student_virtual_logits = functional_call(
    student,
    {**virtual_params, **student_buffers},
    (volume_batch,)
)
student_virtual_logits = unpack_model_output(student_virtual_logits)
```

Tính delta:

```python
scribble_loss_before = ce_loss(student_logits.detach(), label_batch.long())
scribble_loss_after = ce_loss(student_virtual_logits, label_batch.long())

delta = scribble_loss_before - scribble_loss_after
delta = delta.detach()
```

Normalize delta:

```python
if args.normalize_delta:
    delta = delta / (scribble_loss_before.detach() + 1e-8)
```

Clip delta:

```python
if args.delta_clip > 0:
    delta = torch.clamp(delta, -args.delta_clip, args.delta_clip)
```

Lưu ý:

- `student_logits.detach()` trong `scribble_loss_before` để không tạo gradient thừa.
- `delta` phải detach trước khi dùng update teacher.
- Không backprop qua student thông qua `delta`.

---

## 9. Function `compute_scribble_feedback_delta`

Nên tạo function riêng:

```python
def compute_scribble_feedback_delta(
    student,
    volume_batch,
    label_batch,
    student_logits,
    pseudo_hard,
    feedback_mask,
    ce_loss,
    feedback_lr=0.01,
    normalize_delta=True,
    delta_clip=1.0,
):
    from torch.func import functional_call
    from collections import OrderedDict

    if feedback_mask.sum() < 1:
        return student_logits.new_tensor(0.0)

    pseudo_loss = masked_hard_ce_loss(
        student_logits,
        pseudo_hard.detach(),
        feedback_mask
    )

    params = OrderedDict((name, p) for name, p in student.named_parameters())
    buffers = OrderedDict((name, b) for name, b in student.named_buffers())

    grads = torch.autograd.grad(
        pseudo_loss,
        tuple(params.values()),
        create_graph=False,
        retain_graph=True,
        allow_unused=True
    )

    virtual_params = OrderedDict()
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            virtual_params[name] = param
        else:
            virtual_params[name] = param - feedback_lr * grad

    virtual_logits = functional_call(
        student,
        {**virtual_params, **buffers},
        (volume_batch,)
    )
    virtual_logits = unpack_model_output(virtual_logits)

    loss_before = ce_loss(student_logits.detach(), label_batch.long())
    loss_after = ce_loss(virtual_logits, label_batch.long())

    delta = (loss_before - loss_after).detach()

    if normalize_delta:
        delta = delta / (loss_before.detach() + 1e-8)

    if delta_clip > 0:
        delta = torch.clamp(delta, -delta_clip, delta_clip)

    return delta
```

Nếu `torch.func.functional_call` không chạy vì PyTorch version cũ, fallback có thể dùng package `higher`. Tuy nhiên ưu tiên `torch.func.functional_call`.

---

## 10. Feedback loss cho teacher

Teacher feedback loss theo công thức:

```text
L_fb = -delta * log P_T(pseudo_label)
```

Vì `teacher_nll = -log P`, code là:

```python
teacher_nll = masked_pseudo_nll_loss(
    teacher_logits,
    pseudo_hard.detach(),
    feedback_mask
)

loss_feedback = delta * teacher_nll
```

Giải thích dấu:

- Nếu `delta > 0`, minimize `delta * NLL` sẽ giảm NLL, tức tăng xác suất pseudo-label.
- Nếu `delta < 0`, minimize `delta * NLL` sẽ tăng NLL, tức giảm xác suất pseudo-label.

Warmup:

```python
if iter_num < args.feedback_warmup or feedback_mask.sum() < 1:
    loss_feedback = teacher_logits.new_tensor(0.0)
```

Nếu feedback loss âm quá mạnh gây instability:

```bash
--lambda_fb 0.01
```

hoặc:

```bash
--delta_clip 0.2
```

---

## 11. Student training loss

Student học từ scribble và pseudo-label:

```python
loss_scribble_student = ce_loss(student_logits, label_batch.long())

loss_pseudo_student = masked_hard_ce_loss(
    student_logits,
    pseudo_hard.detach(),
    pseudo_mask
)

if iter_num < args.pseudo_warmup:
    loss_pseudo_student = student_logits.new_tensor(0.0)

consistency_weight = get_current_consistency_weight(iter_num // 300)

loss_student = (
    loss_scribble_student
    + args.lambda_pseudo * consistency_weight * loss_pseudo_student
)
```

Version minimal chỉ dùng CE cho pseudo-label. Không thêm Dice ở giai đoạn đầu để tránh làm method phức tạp.

Nếu muốn giữ Dice pseudo sau khi code ổn:

```python
pseudo_hard_masked = pseudo_hard.clone()
pseudo_hard_masked[pseudo_mask.squeeze(1) < 0.5] = 4
loss_pseudo_dice = dice_loss(student_prob, pseudo_hard_masked.unsqueeze(1))

loss_student = (
    loss_scribble_student
    + 0.5 * args.lambda_pseudo * consistency_weight
    * (loss_pseudo_student + loss_pseudo_dice)
)
```

---

## 12. Teacher training loss

Teacher học từ scribble và feedback:

```python
loss_scribble_teacher = ce_loss(teacher_logits, label_batch.long())

teacher_nll = masked_pseudo_nll_loss(
    teacher_logits,
    pseudo_hard.detach(),
    feedback_mask
)

loss_feedback = delta * teacher_nll

if iter_num < args.feedback_warmup:
    loss_feedback = teacher_logits.new_tensor(0.0)

if args.teacher_scribble_loss:
    loss_teacher = loss_scribble_teacher + args.lambda_fb * loss_feedback
else:
    loss_teacher = args.lambda_fb * loss_feedback
```

Khuyến nghị bật:

```text
teacher_scribble_loss = 1
```

Vì nếu teacher chỉ học feedback thì dễ không ổn định.

---

## 13. Thứ tự update trong mỗi iteration

Implement đúng flow sau:

```python
student.train()
teacher.train()

volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

# 1. Forward teacher
teacher_logits = unpack_model_output(teacher(volume_batch))
teacher_prob = torch.softmax(teacher_logits, dim=1)
pseudo_conf, pseudo_hard = torch.max(teacher_prob.detach(), dim=1)
pseudo_conf = pseudo_conf.unsqueeze(1)
pseudo_hard = pseudo_hard.detach()
pseudo_conf = pseudo_conf.detach()

# 2. Build masks
scribble_mask = (label_batch != 4).float().unsqueeze(1)
unlabeled_mask = (label_batch == 4).float().unsqueeze(1)

pseudo_mask = unlabeled_mask.clone()
if args.pseudo_conf_thresh > 0:
    pseudo_mask = pseudo_mask * (pseudo_conf > args.pseudo_conf_thresh).float()

feedback_mask = unlabeled_mask.clone()
if args.feedback_mask_mode == 'uncertain_unlabeled':
    feedback_mask = feedback_mask * (pseudo_conf < args.feedback_conf_thresh).float()

# 3. Forward student
student_logits = unpack_model_output(student(volume_batch))
student_prob = torch.softmax(student_logits, dim=1)

# 4. Compute feedback delta using virtual student update
if args.feedback_interval > 0 and iter_num % args.feedback_interval == 0:
    delta = compute_scribble_feedback_delta(
        student=student,
        volume_batch=volume_batch,
        label_batch=label_batch,
        student_logits=student_logits,
        pseudo_hard=pseudo_hard,
        feedback_mask=feedback_mask,
        ce_loss=ce_loss,
        feedback_lr=args.feedback_lr,
        normalize_delta=bool(args.normalize_delta),
        delta_clip=args.delta_clip
    )
else:
    delta = student_logits.new_tensor(0.0)

# 5. Student loss
loss_scribble_student = ce_loss(student_logits, label_batch.long())
loss_pseudo_student = masked_hard_ce_loss(student_logits, pseudo_hard, pseudo_mask)

if iter_num < args.pseudo_warmup:
    loss_pseudo_student = student_logits.new_tensor(0.0)

consistency_weight = get_current_consistency_weight(iter_num // 300)
loss_student = loss_scribble_student + args.lambda_pseudo * consistency_weight * loss_pseudo_student

# 6. Teacher loss
loss_scribble_teacher = ce_loss(teacher_logits, label_batch.long())

teacher_nll = masked_pseudo_nll_loss(teacher_logits, pseudo_hard, feedback_mask)
loss_feedback = delta * teacher_nll

if iter_num < args.feedback_warmup:
    loss_feedback = teacher_logits.new_tensor(0.0)

if args.teacher_scribble_loss:
    loss_teacher = loss_scribble_teacher + args.lambda_fb * loss_feedback
else:
    loss_teacher = args.lambda_fb * loss_feedback

# 7. Backward student
optimizer_student.zero_grad()
loss_student.backward()
optimizer_student.step()

# 8. Backward teacher
optimizer_teacher.zero_grad()
loss_teacher.backward()
optimizer_teacher.step()
```

Important notes:

1. `pseudo_hard` phải detach khi dùng train student.
2. `delta` phải detach khi dùng train teacher.
3. `loss_student` không được backprop vào teacher.
4. `loss_teacher` không được backprop vào student.
5. Nếu gặp lỗi graph reuse, kiểm tra lại `retain_graph=True` trong `torch.autograd.grad` và đảm bảo `pseudo_hard`, `pseudo_conf`, `delta` đã detach.

---

## 14. LR schedule

Áp dụng cùng schedule cho cả hai optimizer:

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

for param_group in optimizer_student.param_groups:
    param_group['lr'] = lr_

for param_group in optimizer_teacher.param_groups:
    param_group['lr'] = lr_
```

---

## 15. Validation và checkpoint

Validation dùng **student**:

```python
metric_i = test_single_volume_scribblevs(
    sampled_batch['image'],
    sampled_batch['label'],
    student,
    classes=num_classes
)
```

Save best student:

```python
torch.save(student.state_dict(), save_best)
```

Optional save teacher:

```python
torch.save(
    teacher.state_dict(),
    os.path.join(snapshot_path, '{}_best_teacher.pth'.format(args.model))
)
```

Inference chính dùng student, không dùng teacher.

---

## 16. Logging cần thêm

Thêm TensorBoard scalars:

```python
writer.add_scalar('info/loss_student', loss_student.item(), iter_num)
writer.add_scalar('info/loss_teacher', loss_teacher.item(), iter_num)
writer.add_scalar('info/loss_scribble_student', loss_scribble_student.item(), iter_num)
writer.add_scalar('info/loss_scribble_teacher', loss_scribble_teacher.item(), iter_num)
writer.add_scalar('info/loss_pseudo_student', loss_pseudo_student.item(), iter_num)
writer.add_scalar('info/loss_feedback', loss_feedback.item(), iter_num)
writer.add_scalar('info/delta_feedback', delta.item(), iter_num)
writer.add_scalar('info/teacher_nll', teacher_nll.item(), iter_num)
writer.add_scalar('info/pseudo_conf_mean', pseudo_conf.mean().item(), iter_num)
writer.add_scalar('info/pseudo_mask_ratio', pseudo_mask.mean().item(), iter_num)
writer.add_scalar('info/feedback_mask_ratio', feedback_mask.mean().item(), iter_num)
```

Logging text mỗi 200 iterations:

```python
logging.info(
    'iteration %d : loss_student=%f, loss_teacher=%f, '
    'scribble_s=%f, scribble_t=%f, pseudo=%f, fb=%f, delta=%f, '
    'pseudo_conf=%f, pseudo_mask=%f, fb_mask=%f' %
    (
        iter_num,
        loss_student.item(),
        loss_teacher.item(),
        loss_scribble_student.item(),
        loss_scribble_teacher.item(),
        loss_pseudo_student.item(),
        loss_feedback.item(),
        delta.item(),
        pseudo_conf.mean().item(),
        pseudo_mask.mean().item(),
        feedback_mask.mean().item(),
    )
)
```

---

## 17. Commands chạy thử

### 17.1. Full proposed đơn giản

```bash
python train_scribble_feedback.py \
  --root_path ../../data/ACDC \
  --exp ScribbleFeedback_TS \
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
  --lambda_fb 0.1 \
  --feedback_lr 0.01 \
  --pseudo_conf_thresh 0.7 \
  --feedback_mask_mode unlabeled \
  --feedback_warmup 1000 \
  --pseudo_warmup 500 \
  --delta_clip 1.0 \
  --normalize_delta 1 \
  --teacher_scribble_loss 1
```

### 17.2. Feedback chỉ vùng uncertain

```bash
python train_scribble_feedback.py \
  --root_path ../../data/ACDC \
  --exp ScribbleFeedback_TS_uncertain \
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
  --lambda_fb 0.1 \
  --feedback_lr 0.01 \
  --pseudo_conf_thresh 0.7 \
  --feedback_mask_mode uncertain_unlabeled \
  --feedback_conf_thresh 0.8 \
  --feedback_warmup 1000 \
  --pseudo_warmup 500 \
  --delta_clip 1.0 \
  --normalize_delta 1 \
  --teacher_scribble_loss 1
```

---

## 18. Expected files to modify/create

Create:

```text
code/train_scribble_feedback.py
```

Copy từ current training file, rồi sửa theo spec.

Optional create:

```text
utils/scribble_feedback.py
```

Chứa:

```python
masked_hard_ce_loss
masked_pseudo_nll_loss
compute_scribble_feedback_delta
```

Nếu muốn ít thay đổi, có thể đặt luôn các function trong `train_scribble_feedback.py`.

---

## 19. Minimal ablation cần chạy

Để chứng minh proposed idea, chạy 4 setting.

### A. Scribble-only baseline

```text
loss = partial CE only
```

Command logic:

```bash
--lambda_pseudo 0 --lambda_fb 0
```

### B. Scribble + pseudo-label, no feedback

```bash
--lambda_pseudo 1.0 --lambda_fb 0
```

### C. Scribble + feedback teacher, no pseudo student

Không bắt buộc, nhưng có thể chạy để kiểm tra feedback riêng:

```bash
--lambda_pseudo 0 --lambda_fb 0.1
```

### D. Full proposed

```bash
--lambda_pseudo 1.0 --lambda_fb 0.1
```

Expected trend:

```text
Full proposed > Scribble + pseudo > Scribble-only
```

---

## 20. Method paragraph cho README/paper note

```text
We propose a Scribble-Guided Feedback Teacher-Student framework for scribble-supervised segmentation. A trainable teacher generates dense pseudo-labels for unlabeled pixels, while the student learns from both sparse scribble annotations and teacher pseudo-labels. To prevent misleading pseudo-labels from being repeatedly reinforced, we estimate a feedback signal from the student. Specifically, the student performs a one-step virtual update using teacher pseudo-labels on non-scribble regions, and we measure the change of its partial cross-entropy loss on scribble-annotated pixels. If the virtual update decreases the scribble loss, the teacher pseudo-labels are considered beneficial and their likelihood is reinforced. Otherwise, their likelihood is suppressed through a feedback loss. This uses sparse scribbles as reliable anchors to guide dense pseudo-label refinement without requiring full masks.
```

---

## 21. Important implementation notes

Điểm dễ bug nhất:

1. Teacher phải trainable, không phải EMA detached.
2. `pseudo_hard` phải `.detach()` khi dùng train student.
3. `delta` phải `.detach()` khi dùng train teacher.
4. `loss_student` không được backprop vào teacher.
5. `loss_teacher` không được backprop vào student.
6. Nếu feedback loss âm quá mạnh, giảm `--lambda_fb` xuống `0.01`.
7. Nếu delta dao động quá mạnh, giảm `--delta_clip` xuống `0.2`.
8. Nếu virtual update quá tốn memory, giảm batch size hoặc dùng `--feedback_interval 2`.
9. Validation và checkpoint dùng student.
10. Inference chỉ dùng student.

---

## 22. Fallback cực tối giản nếu agent bị lỗi `functional_call`

Nếu agent không chạy được `torch.func.functional_call`, có thể implement approximate feedback trước:

```python
delta_proxy = loss_scribble_student.detach() - loss_scribble_teacher.detach()
```

Nhưng đây không đúng hoàn toàn ý tưởng feedback, vì không đo student sau virtual update.

Version đúng cần ưu tiên:

```text
functional_call + virtual student update
```

Priority implementation:

```text
Priority 1: trainable teacher-student
Priority 2: pseudo-label on non-scribble pixels
Priority 3: virtual student update
Priority 4: delta-based teacher feedback loss
Priority 5: logging + ablation
```
