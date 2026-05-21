# SPEC: Sửa training code ScribbleVS để tạo pseudo-label bằng uncertainty-weighted fusion giữa Student và EMA Teacher

## 0. Mục tiêu chỉnh sửa

File hiện tại đang tạo pseudo-label từ EMA teacher và student/current model, sau đó **chọn target dựa trên so sánh supervised loss**:

```python
pseudo_label = process_pseudo_label(outputs_soft_ema, tau=args.tau)
pseudo_label_stu = process_pseudo_label(outputs_soft1, tau=args.tau)

loss_ce = ce_loss(outputs, label_batch[:].long())
loss_ce_pseudo = ce_loss(ema_output, label_batch[:].long())
if loss_ce > loss_ce_pseudo:
    loss_pseudo_ce = ce_loss(outputs, pseudo_label[:].long())
    loss_pseudo_dc = dice_loss(outputs_soft1, pseudo_label.unsqueeze(1))
else:
    loss_pseudo_ce = ce_loss(outputs, pseudo_label_stu[:].long())
    loss_pseudo_dc = dice_loss(outputs_soft1, pseudo_label_stu.unsqueeze(1))
```

Yêu cầu sửa: **bỏ hoàn toàn cơ chế chọn pseudo-label bằng loss**, thay bằng **fusion probability map của Student và EMA Teacher theo evidential uncertainty**.

Ý tưởng mới:

- Student/current model tạo probability map `P_s`.
- EMA Teacher tạo probability map `P_t`.
- Tính evidential uncertainty cho cả hai:
  - `U_s`
  - `U_t`
- Model nào uncertainty thấp hơn thì weight cao hơn.
- Tạo pseudo-label mềm:

\[
\tilde{P} = w_s P_s + w_t P_t
\]

- Từ `\tilde{P}` lấy pseudo-label hard nếu cần cho Dice:

\[
\hat{y} = \arg\max_c \tilde{P}_c
\]

- Student vẫn được train bằng:
  - supervised scribble loss trên scribble pixels;
  - pseudo-label loss trên unlabeled pixels;
  - uncertainty consistency loss như code hiện tại.

Code gốc đã có sẵn `evidential_uncertainty_from_logits`, `uncertainty_consistency_loss`, `WeightEMA`, `model`, `model_ema`, ACDC scribble dataloader, validation bằng `test_single_volume_scribblevs`. Giữ lại các phần này, chỉ sửa logic pseudo-label.

---

## 1. File cần sửa

Sửa trực tiếp file training hiện tại, ví dụ:

```text
train_method_acdc.py
# hoặc file đang chứa code user gửi
```

Không cần tạo framework 3 model mới. Bản chỉnh này vẫn dùng:

```text
model      = Student/current model
model_ema  = EMA Teacher
```

Trong paper/proposal có thể gọi là **Student–Teacher uncertainty-fusion pseudo-labeling**.

---

## 2. Các argument cần thêm/sửa

### 2.1. Giữ lại các argument hiện có

Giữ nguyên:

```python
--tau
--lambda_unc_cons
--uncertainty_consistency
--evidence_activation
--detach_teacher_uncertainty
--uncertainty_mask_mode
```

`--tau` có thể giữ để backward compatibility, nhưng sau khi sửa fusion thì không bắt buộc dùng nữa nếu pseudo-label lấy bằng `argmax(pseudo_soft)`.

### 2.2. Thêm các argument mới

Thêm vào parser:

```python
parser.add_argument('--uncertainty_temp', type=float, default=0.5,
                    help='temperature for uncertainty-weighted pseudo-label fusion')

parser.add_argument('--pseudo_conf_thresh', type=float, default=0.0,
                    help='optional confidence threshold for fused pseudo-label; 0 disables filtering')

parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['all', 'unlabeled'],
                    help='where to apply pseudo-label supervision')

parser.add_argument('--detach_student_fusion', type=int, default=1,
                    help='detach student branch when building fused pseudo label to avoid self-training gradient leakage')

parser.add_argument('--fusion_use_teacher_only_warmup', type=int, default=0,
                    help='if >0, use teacher-only pseudo label before this iteration, then switch to uncertainty fusion')

parser.add_argument('--uncertainty_target', type=str, default='teacher',
                    choices=['teacher', 'fused'],
                    help='target uncertainty map for uncertainty consistency')
```

Giải thích:

- `uncertainty_temp`: điều chỉnh độ sắc của weight. Nhỏ hơn thì source có uncertainty thấp sẽ được ưu tiên mạnh hơn.
- `pseudo_conf_thresh`: nếu muốn lọc pseudo-label yếu, chỉ train ở pixel có confidence của fused pseudo-label lớn hơn threshold.
- `pseudo_mask_mode='unlabeled'`: trong scribble setting, pseudo-label loss nên apply chủ yếu trên pixel `label == 4`, tránh override scribble label thật.
- `detach_student_fusion=1`: khi tạo pseudo-label từ chính output student, nên detach phần student probability trước khi dùng làm target để tránh gradient leakage/self-training collapse.
- `fusion_use_teacher_only_warmup`: optional warm-up. Nếu set `3000`, trước iter 3000 dùng teacher pseudo-label; sau đó mới fusion.
- `uncertainty_target`: chọn target uncertainty cho uncertainty consistency là EMA teacher hoặc fused uncertainty.

---

## 3. Hàm utility cần thêm

Thêm các hàm sau dưới `evidential_uncertainty_from_logits`.

### 3.1. Masked soft CE loss

Vì pseudo-label mới là soft probability map, cần soft CE:

```python
def masked_soft_ce_loss(logits, target_prob, mask=None, eps=1e-8):
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    if mask.sum() < 1:
        return torch.tensor(0.0, device=logits.device)

    return (ce_map * mask).sum() / (mask.sum() + eps)
```

Input/output shape:

```text
logits:      [B, C, H, W]
target_prob: [B, C, H, W]
mask:        [B, 1, H, W] hoặc None
```

### 3.2. Uncertainty-weighted fusion

Thêm hàm:

```python
def uncertainty_weighted_fusion(
    student_prob,
    teacher_prob,
    student_unc,
    teacher_unc,
    temperature=0.5,
    detach_student=True,
    detach_teacher=True,
    eps=1e-8,
):
    if detach_student:
        student_prob = student_prob.detach()
        student_unc = student_unc.detach()

    if detach_teacher:
        teacher_prob = teacher_prob.detach()
        teacher_unc = teacher_unc.detach()

    student_unc = student_unc.clamp(0.0, 1.0)
    teacher_unc = teacher_unc.clamp(0.0, 1.0)

    ws = torch.exp(-student_unc / temperature)
    wt = torch.exp(-teacher_unc / temperature)

    w_sum = ws + wt + eps
    ws = ws / w_sum
    wt = wt / w_sum

    fused_prob = ws * student_prob + wt * teacher_prob
    fused_prob = fused_prob / (fused_prob.sum(dim=1, keepdim=True) + eps)

    return fused_prob.detach(), ws.detach(), wt.detach()
```

Công thức:

\[
w_s = \frac{\exp(-U_s/T)}{\exp(-U_s/T)+\exp(-U_t/T)}
\]

\[
w_t = \frac{\exp(-U_t/T)}{\exp(-U_s/T)+\exp(-U_t/T)}
\]

\[
\tilde{P}=w_sP_s+w_tP_t
\]

Shape:

```text
student_prob: [B, C, H, W]
teacher_prob: [B, C, H, W]
student_unc:  [B, 1, H, W]
teacher_unc:  [B, 1, H, W]
fused_prob:   [B, C, H, W]
ws, wt:       [B, 1, H, W]
```

Lưu ý: user mô tả “hai teacher”, nhưng file code hiện tại chỉ có `model` và `model_ema`. Vì vậy trong bản sửa này:

```text
source 1 = student/current model
source 2 = EMA teacher
```

Nếu về sau muốn đúng two-teacher, agent có thể tổng quát hóa hàm này cho `P1`, `P2`.

---

## 4. Thay đổi chính trong training loop

### 4.1. Đoạn code cần xóa/thay

Tìm và xóa đoạn:

```python
pseudo_label = process_pseudo_label(outputs_soft_ema, tau=args.tau)
pseudo_label_stu = process_pseudo_label(outputs_soft1, tau=args.tau)

loss_ce = ce_loss(outputs, label_batch[:].long())
loss_ce_pseudo = ce_loss(ema_output, label_batch[:].long())
if loss_ce > loss_ce_pseudo:
    loss_pseudo_ce = ce_loss(outputs, pseudo_label[:].long())
    loss_pseudo_dc = dice_loss(outputs_soft1, pseudo_label.unsqueeze(1))
else:
    loss_pseudo_ce = ce_loss(outputs, pseudo_label_stu[:].long())
    loss_pseudo_dc = dice_loss(outputs_soft1, pseudo_label_stu.unsqueeze(1))
```

Sau khi sửa, không còn dùng:

```python
loss_ce_pseudo
if loss_ce > loss_ce_pseudo
pseudo_label_stu
```

---

### 4.2. Đoạn code mới thay thế

Sau khi có output của teacher và student:

```python
with torch.no_grad():
    ema_output = unpack_model_output(model_ema(volume_batch))
    outputs_soft_ema = torch.softmax(ema_output, dim=1)

outputs = unpack_model_output(model(volume_batch))
outputs_soft1 = torch.softmax(outputs, dim=1)
```

Tính uncertainty:

```python
student_unc = evidential_uncertainty_from_logits(
    outputs, num_classes=args.num_classes, activation=args.evidence_activation
)

teacher_unc = evidential_uncertainty_from_logits(
    ema_output, num_classes=args.num_classes, activation=args.evidence_activation
)

if args.detach_teacher_uncertainty:
    teacher_unc = teacher_unc.detach()
```

Tạo fused pseudo-label:

```python
if args.fusion_use_teacher_only_warmup > 0 and iter_num < args.fusion_use_teacher_only_warmup:
    pseudo_soft = outputs_soft_ema.detach()
    weight_student = torch.zeros_like(student_unc)
    weight_teacher = torch.ones_like(teacher_unc)
else:
    pseudo_soft, weight_student, weight_teacher = uncertainty_weighted_fusion(
        student_prob=outputs_soft1,
        teacher_prob=outputs_soft_ema,
        student_unc=student_unc,
        teacher_unc=teacher_unc,
        temperature=args.uncertainty_temp,
        detach_student=bool(args.detach_student_fusion),
        detach_teacher=True,
    )
```

Lấy hard pseudo-label và confidence:

```python
pseudo_conf, pseudo_hard = torch.max(pseudo_soft, dim=1)
pseudo_conf = pseudo_conf.unsqueeze(1)
```

Tạo mask cho pseudo-label loss:

```python
if args.pseudo_mask_mode == 'unlabeled':
    pseudo_mask = (label_batch == 4).float().unsqueeze(1)
elif args.pseudo_mask_mode == 'all':
    pseudo_mask = torch.ones_like(pseudo_conf)
else:
    raise ValueError('Unsupported pseudo_mask_mode: {}'.format(args.pseudo_mask_mode))

if args.pseudo_conf_thresh > 0:
    pseudo_mask = pseudo_mask * (pseudo_conf > args.pseudo_conf_thresh).float()
```

Tính supervised loss:

```python
loss_ce = ce_loss(outputs, label_batch.long())
```

Tính pseudo-label loss:

```python
loss_pseudo_ce = masked_soft_ce_loss(outputs, pseudo_soft, mask=pseudo_mask)
loss_pseudo_dc = dice_loss(outputs_soft1, pseudo_hard.unsqueeze(1))
```

Nếu Dice loss hard pseudo làm training bất ổn, có thể tạm tắt bằng cách set:

```python
loss_pseudo_dc = torch.tensor(0.0, device=outputs.device)
```

hoặc thêm argument `--use_pseudo_dice` ở bước mở rộng sau. Trong bản sửa tối thiểu, cứ giữ Dice như code cũ để ít thay đổi.

---

## 5. Sửa uncertainty consistency

Code hiện tại đã có `uncertainty_mask`. Giữ lại logic tạo mask:

```python
if args.uncertainty_mask_mode == 'all':
    uncertainty_mask = None
elif args.uncertainty_mask_mode == 'unlabeled':
    uncertainty_mask = (label_batch == 4).float().unsqueeze(1)
elif args.uncertainty_mask_mode == 'labeled':
    uncertainty_mask = (label_batch != 4).float().unsqueeze(1)
else:
    raise ValueError('Unsupported uncertainty mask mode: {}'.format(args.uncertainty_mask_mode))
```

Sau đó chọn uncertainty target:

```python
if args.uncertainty_target == 'teacher':
    uncertainty_target = teacher_unc.detach()
elif args.uncertainty_target == 'fused':
    uncertainty_target = (
        weight_student * student_unc.detach() + weight_teacher * teacher_unc.detach()
    ).detach()
else:
    raise ValueError('Unsupported uncertainty_target: {}'.format(args.uncertainty_target))
```

Tính loss:

```python
loss_uncertainty = uncertainty_consistency_loss(
    student_unc,
    uncertainty_target,
    loss_type=args.uncertainty_consistency,
    mask=uncertainty_mask
)
```

Khuyến nghị chạy mặc định:

```bash
--uncertainty_target teacher
```

Sau đó ablation thêm:

```bash
--uncertainty_target fused
```

---

## 6. Loss cuối cùng

Giữ tổng loss gần code gốc:

```python
consistency_weight = get_current_consistency_weight(iter_num // 300)
loss_pse_sup = (loss_pseudo_dc + loss_pseudo_ce) * 0.5 * consistency_weight
loss = loss_ce + loss_pse_sup + args.lambda_unc_cons * consistency_weight * loss_uncertainty
```

Điểm quan trọng:

- `loss_ce` vẫn là supervised scribble CE với `ignore_index=4`.
- `loss_pseudo_ce` là soft CE từ fused pseudo-label.
- `loss_pseudo_dc` là Dice với hard pseudo-label từ fused pseudo-label.
- `loss_uncertainty` giữ regularization uncertainty map.

---

## 7. TensorBoard logging cần thêm

Thêm các scalar:

```python
writer.add_scalar('info/pseudo_conf_mean', pseudo_conf.mean(), iter_num)
writer.add_scalar('info/pseudo_mask_ratio', pseudo_mask.mean(), iter_num)
writer.add_scalar('info/fusion_weight_student_mean', weight_student.mean(), iter_num)
writer.add_scalar('info/fusion_weight_teacher_mean', weight_teacher.mean(), iter_num)
writer.add_scalar('info/loss_pseudo_ce', loss_pseudo_ce, iter_num)
writer.add_scalar('info/loss_pseudo_dc', loss_pseudo_dc, iter_num)
```

Sửa logging mỗi 200 iter thành:

```python
logging.info(
    'iteration %d : loss=%f, loss_ce=%f, loss_pse_sup=%f, loss_uncertainty=%f, '
    'pseudo_conf=%f, mask_ratio=%f, ws=%f, wt=%f' %
    (
        iter_num,
        loss.item(),
        loss_ce.item(),
        loss_pse_sup.item(),
        loss_uncertainty.item(),
        pseudo_conf.mean().item(),
        pseudo_mask.mean().item(),
        weight_student.mean().item(),
        weight_teacher.mean().item(),
    )
)
```

---

## 8. Import cần xóa hoặc giữ

Sau khi sửa không còn cần:

```python
from utils.util import process_pseudo_label
```

Có thể xóa để code sạch.

Nếu muốn giữ tạm để không ảnh hưởng các nhánh khác thì cũng không gây lỗi, nhưng không được dùng lại trong logic mới.

---

## 9. Pseudo-code training loop sau khi sửa

```python
for i_batch, sampled_batch in enumerate(trainloader):
    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

    with torch.no_grad():
        ema_output = unpack_model_output(model_ema(volume_batch))
        prob_teacher = torch.softmax(ema_output, dim=1)

    output = unpack_model_output(model(volume_batch))
    prob_student = torch.softmax(output, dim=1)

    student_unc = evidential_uncertainty_from_logits(
        output, args.num_classes, args.evidence_activation
    )
    teacher_unc = evidential_uncertainty_from_logits(
        ema_output, args.num_classes, args.evidence_activation
    )

    if args.detach_teacher_uncertainty:
        teacher_unc = teacher_unc.detach()

    pseudo_soft, ws, wt = uncertainty_weighted_fusion(
        student_prob=prob_student,
        teacher_prob=prob_teacher,
        student_unc=student_unc,
        teacher_unc=teacher_unc,
        temperature=args.uncertainty_temp,
        detach_student=bool(args.detach_student_fusion),
        detach_teacher=True,
    )

    pseudo_conf, pseudo_hard = torch.max(pseudo_soft, dim=1)
    pseudo_conf = pseudo_conf.unsqueeze(1)

    if args.pseudo_mask_mode == 'unlabeled':
        pseudo_mask = (label_batch == 4).float().unsqueeze(1)
    else:
        pseudo_mask = torch.ones_like(pseudo_conf)

    if args.pseudo_conf_thresh > 0:
        pseudo_mask = pseudo_mask * (pseudo_conf > args.pseudo_conf_thresh).float()

    loss_ce = ce_loss(output, label_batch.long())
    loss_pseudo_ce = masked_soft_ce_loss(output, pseudo_soft, pseudo_mask)
    loss_pseudo_dc = dice_loss(prob_student, pseudo_hard.unsqueeze(1))

    if args.uncertainty_mask_mode == 'all':
        uncertainty_mask = None
    elif args.uncertainty_mask_mode == 'unlabeled':
        uncertainty_mask = (label_batch == 4).float().unsqueeze(1)
    elif args.uncertainty_mask_mode == 'labeled':
        uncertainty_mask = (label_batch != 4).float().unsqueeze(1)

    if args.uncertainty_target == 'teacher':
        uncertainty_target = teacher_unc.detach()
    else:
        uncertainty_target = (ws * student_unc.detach() + wt * teacher_unc.detach()).detach()

    loss_uncertainty = uncertainty_consistency_loss(
        student_unc,
        uncertainty_target,
        loss_type=args.uncertainty_consistency,
        mask=uncertainty_mask,
    )

    consistency_weight = get_current_consistency_weight(iter_num // 300)
    loss_pse_sup = 0.5 * (loss_pseudo_ce + loss_pseudo_dc) * consistency_weight
    loss = loss_ce + loss_pse_sup + args.lambda_unc_cons * consistency_weight * loss_uncertainty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema_optimizer.step()
```

---

## 10. Expected behavior sau khi sửa

Agent cần đảm bảo:

1. Không còn logic chọn pseudo-label bằng:

```python
if loss_ce > loss_ce_pseudo:
```

2. Pseudo-label được tạo từ:

```python
pseudo_soft = ws * outputs_soft1 + wt * outputs_soft_ema
```

3. Weight phụ thuộc vào evidential uncertainty:

```python
ws ∝ exp(-student_unc / T)
wt ∝ exp(-teacher_unc / T)
```

4. Model nào uncertainty thấp hơn thì weight cao hơn.

5. Pseudo loss mặc định apply trên unlabeled pixels:

```python
label_batch == 4
```

6. Code không NaN khi `pseudo_mask.sum() == 0`.

7. Validation, save checkpoint, EMA update giữ nguyên.

---

## 11. Test nhanh sau khi agent sửa

Chạy thử 1000 iterations:

```bash
python train_method_acdc.py \
  --root_path ../../data/ACDC \
  --exp ScribbleVS_uncertainty_fusion \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model unet \
  --num_classes 4 \
  --max_iterations 1000 \
  --batch_size 16 \
  --gpu 0 \
  --uncertainty_temp 0.5 \
  --pseudo_conf_thresh 0.0 \
  --pseudo_mask_mode unlabeled \
  --uncertainty_target teacher
```

Nếu muốn lọc pseudo-label yếu:

```bash
--pseudo_conf_thresh 0.6
```

Nếu muốn consistency theo fused uncertainty:

```bash
--uncertainty_target fused
```

Nếu muốn tránh fusion quá sớm:

```bash
--fusion_use_teacher_only_warmup 3000
```

---

## 12. Ablation nên chạy

| Setting | Args |
|---|---|
| Baseline cũ | code cũ |
| Fusion no threshold | `--pseudo_conf_thresh 0.0` |
| Fusion threshold 0.5 | `--pseudo_conf_thresh 0.5` |
| Fusion threshold 0.6 | `--pseudo_conf_thresh 0.6` |
| Temperature 1.0 | `--uncertainty_temp 1.0` |
| Temperature 0.5 | `--uncertainty_temp 0.5` |
| Temperature 0.25 | `--uncertainty_temp 0.25` |
| Uncertainty target teacher | `--uncertainty_target teacher` |
| Uncertainty target fused | `--uncertainty_target fused` |
| Fusion after warmup | `--fusion_use_teacher_only_warmup 3000` |

Metric chính:

```text
mean Dice
mean HD95
per-class Dice
per-class HD95
```

Log cần quan sát:

```text
fusion_weight_student_mean
fusion_weight_teacher_mean
pseudo_conf_mean
pseudo_mask_ratio
uncertainty_student_mean
uncertainty_teacher_mean
loss_pseudo_ce
loss_pseudo_dc
```

---

## 13. Các lỗi dễ gặp và cách xử lý

### Lỗi 1: Dice pseudo-label làm training bất ổn

Tạm bỏ Dice pseudo:

```python
loss_pse_sup = loss_pseudo_ce * consistency_weight
```

Sau đó chạy ablation riêng với Dice.

### Lỗi 2: Student tự reinforce vì dùng prediction của chính nó làm target

Đảm bảo:

```bash
--detach_student_fusion 1
```

Có thể thêm warm-up:

```bash
--fusion_use_teacher_only_warmup 3000
```

### Lỗi 3: Weight fusion collapse về một source

Tăng temperature:

```bash
--uncertainty_temp 1.0
```

Hoặc clamp uncertainty trong hàm fusion:

```python
student_unc = student_unc.clamp(0.0, 1.0)
teacher_unc = teacher_unc.clamp(0.0, 1.0)
```

### Lỗi 4: pseudo_mask quá ít pixel

Giảm hoặc tắt threshold:

```bash
--pseudo_conf_thresh 0.0
```

---

## 14. Tiêu chí hoàn thành

Agent sửa code xong cần đảm bảo:

- [ ] Code chạy được không lỗi import.
- [ ] Không còn dùng `process_pseudo_label` để chọn teacher/student target dựa trên loss.
- [ ] Không còn `if loss_ce > loss_ce_pseudo`.
- [ ] Có hàm `uncertainty_weighted_fusion`.
- [ ] Có hàm `masked_soft_ce_loss`.
- [ ] Có args `uncertainty_temp`, `pseudo_conf_thresh`, `pseudo_mask_mode`, `detach_student_fusion`, `fusion_use_teacher_only_warmup`, `uncertainty_target`.
- [ ] Pseudo-label fusion dựa trên `student_unc` và `teacher_unc`.
- [ ] `loss_pseudo_ce` dùng soft pseudo-label fusion.
- [ ] `loss_pseudo_dc` dùng hard pseudo-label từ fused probability hoặc được tắt rõ ràng.
- [ ] Uncertainty consistency vẫn hoạt động.
- [ ] TensorBoard có log weight fusion và pseudo confidence.
- [ ] Validation/save checkpoint giữ nguyên.
