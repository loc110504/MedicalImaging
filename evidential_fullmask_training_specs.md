# Spec: Fully-Supervised Evidential Training with Full Mask

> Mục tiêu: thêm một version train **fully-supervised với full mask** cho ACDC segmentation, có **evidential calibrated learning** theo phần Stage I của EUGIS.  
> Không làm prompt encoder, không sinh point, không interactive Stage II.

---

## 0. Tổng quan

### Goal

Thêm một training entrypoint mới cho repo hiện tại:

```text
train_evidential_fullmask.py
```

Version này cần:

- Train segmentation với **full mask**.
- Dùng `sup_type=label`.
- Dùng evidential learning để model sinh:
  - evidence,
  - Dirichlet alpha,
  - posterior probability,
  - uncertainty map.
- Loss tổng:

```text
LEcml = LCE + LCEU + λ1 * LKL + λ2 * LDice
```

### Không implement

Không làm các phần sau:

- point prompt generation,
- top-k uncertainty point sampling,
- positive / negative click,
- prompt encoder,
- prompt decoder,
- image-prompt interaction decoder,
- multi-mask selection,
- Stage II iterative prompt refinement,
- teacher-student training,
- EMA teacher,
- pseudo-label refinement,
- high / low feature consistency loss.

### Repo context hiện tại

Code hiện tại dùng:

```text
dataloader/acdc.py
  - ACDCDataSets
  - RandomGenerator

networks/net_factory.py
  - net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)

val.py
  - test_single_volume
```

Dataset layout giữ nguyên:

```text
../../data/ACDC/
  ACDC_training_slices/
  ACDC_training_volumes/
```

Checkpoint layout giữ nguyên:

```text
../../checkpoints/{args.data}_{args.exp}/
```

Ví dụ:

```text
../../checkpoints/ACDC_EvidentialFullMask/
```

---

## 1. Files cần tạo

Tạo 2 file mới:

```text
utils/evidential_losses.py
train_evidential_fullmask.py
```

Không sửa/xóa `train.py` cũ.

---

# 2. File: `utils/evidential_losses.py`

## 2.1. Purpose

File này chứa toàn bộ loss và helper cho evidential segmentation.

Input chính:

```python
logits: torch.Tensor  # [B, C, H, W]
target: torch.Tensor  # [B, H, W]
```

Output phụ cần có:

```python
evidence      # [B, C, H, W]
alpha         # [B, C, H, W]
prob          # [B, C, H, W]
uncertainty   # [B, 1, H, W]
belief        # [B, C, H, W]
```

---

## 2.2. Evidential formulation

Từ logits:

```python
evidence = relu(logits)
alpha = evidence + 1
S = alpha.sum(dim=1, keepdim=True)
prob = alpha / S
uncertainty = num_classes / S
belief = evidence / S
```

Trong đó:

- `evidence >= 0`
- `alpha` là tham số Dirichlet.
- `prob = alpha / S` là posterior probability.
- `uncertainty = C / S`.
- `belief = evidence / S`.

---

## 2.3. Required imports

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 2.4. Function: `evidence_from_logits`

```python
def evidence_from_logits(logits, activation="relu"):
    """
    Convert raw model logits to non-negative evidence.

    Args:
        logits: Tensor [B, C, H, W]
        activation: one of ["relu", "softplus", "exp"]

    Returns:
        evidence: Tensor [B, C, H, W]
    """
```

Implementation:

```python
if activation == "relu":
    return F.relu(logits)
elif activation == "softplus":
    return F.softplus(logits)
elif activation == "exp":
    return torch.exp(torch.clamp(logits, min=-10, max=10))
else:
    raise ValueError(f"Unsupported evidence activation: {activation}")
```

---

## 2.5. Function: `dirichlet_params_from_logits`

```python
def dirichlet_params_from_logits(logits, activation="relu", eps=1e-8):
    """
    Build evidence, Dirichlet parameters, posterior probability,
    uncertainty map and belief mass from segmentation logits.

    Args:
        logits: Tensor [B, C, H, W]
        activation: evidence activation
        eps: numerical stability value

    Returns:
        evidence: Tensor [B, C, H, W]
        alpha: Tensor [B, C, H, W]
        prob: Tensor [B, C, H, W]
        uncertainty: Tensor [B, 1, H, W]
        belief: Tensor [B, C, H, W]
    """
```

Implementation:

```python
num_classes = logits.shape[1]
evidence = evidence_from_logits(logits, activation=activation)
alpha = evidence + 1.0
S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
prob = alpha / S
uncertainty = float(num_classes) / S
belief = evidence / S
return evidence, alpha, prob, uncertainty, belief
```

---

## 2.6. Function: `one_hot_target`

```python
def one_hot_target(target, num_classes, ignore_index=None):
    """
    Convert target [B, H, W] to one-hot [B, C, H, W].

    Ignored pixels should become all-zero vectors.

    Args:
        target: Tensor [B, H, W]
        num_classes: int
        ignore_index: optional int

    Returns:
        y_onehot: Tensor [B, C, H, W]
        valid_mask: Tensor [B, 1, H, W]
    """
```

Implementation detail:

```python
if ignore_index is None:
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    safe_target = target
else:
    valid_mask = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid_mask] = 0

safe_target = safe_target.clamp(min=0, max=num_classes - 1)
y_onehot = F.one_hot(safe_target.long(), num_classes=num_classes)
y_onehot = y_onehot.permute(0, 3, 1, 2).float()

valid_mask = valid_mask.unsqueeze(1).float()
y_onehot = y_onehot * valid_mask

return y_onehot, valid_mask
```

---

## 2.7. Function: `evidential_ce_loss`

```python
def evidential_ce_loss(alpha, target, num_classes, ignore_index=None):
    """
    Evidential cross entropy:

    LCE = sum_c y_c * (digamma(S) - digamma(alpha_c))

    Args:
        alpha: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        num_classes: int
        ignore_index: optional int

    Returns:
        scalar Tensor
    """
```

Implementation:

```python
y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

S = alpha.sum(dim=1, keepdim=True)
loss_map = torch.sum(
    y_onehot * (torch.digamma(S) - torch.digamma(alpha)),
    dim=1,
    keepdim=True,
)

loss = (loss_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
return loss
```

---

## 2.8. Function: `evidential_dice_loss`

```python
def evidential_dice_loss(
    prob,
    target,
    num_classes,
    ignore_index=None,
    smooth=1e-5,
    include_background=False,
):
    """
    Soft Dice loss using posterior probability prob = alpha / S.

    Args:
        prob: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        include_background:
            False -> compute class 1..C-1
            True  -> compute class 0..C-1

    Returns:
        scalar Tensor
    """
```

Implementation:

```python
y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

if include_background:
    class_ids = range(num_classes)
else:
    class_ids = range(1, num_classes)

dice_losses = []

for c in class_ids:
    p = prob[:, c:c + 1] * valid_mask
    y = y_onehot[:, c:c + 1] * valid_mask

    intersection = (p * y).sum()
    denominator = p.sum() + y.sum()

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_losses.append(1.0 - dice)

if len(dice_losses) == 0:
    return prob.new_tensor(0.0)

return torch.stack(dice_losses).mean()
```

---

## 2.9. Function: `evidential_kl_loss`

```python
def evidential_kl_loss(
    alpha,
    target,
    num_classes,
    ignore_index=None,
    annealing_coef=1.0,
):
    """
    KL[D(p|alpha_tilde) || D(p|1)]

    alpha_tilde = y + (1 - y) * alpha

    This suppresses evidence for incorrect classes.

    Args:
        alpha: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        annealing_coef: float

    Returns:
        scalar Tensor
    """
```

Implementation:

```python
y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
S_alpha = alpha_tilde.sum(dim=1, keepdim=True)

lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)

lnB_uni = torch.lgamma(
    torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype)
)

digamma_sum = torch.digamma(S_alpha)
digamma_alpha = torch.digamma(alpha_tilde)

kl_map = lnB - lnB_uni + (
    (alpha_tilde - 1.0) * (digamma_alpha - digamma_sum)
).sum(dim=1, keepdim=True)

kl = (kl_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
return annealing_coef * kl
```

---

## 2.10. Function: `calibrated_evidential_uncertainty_loss`

```python
def calibrated_evidential_uncertainty_loss(
    logits,
    alpha,
    belief,
    uncertainty,
    target,
    num_classes,
    epoch,
    max_epoch,
    alpha0=1.0,
    ignore_index=None,
    eps=1e-8,
):
    """
    CEU loss.

    Correct pixels should become certain.
    Incorrect pixels should become uncertain.

    Args:
        logits: Tensor [B, C, H, W]
        alpha: Tensor [B, C, H, W]
        belief: Tensor [B, C, H, W]
        uncertainty: Tensor [B, 1, H, W]
        target: Tensor [B, H, W]
        epoch: current epoch
        max_epoch: total epoch count

    Returns:
        scalar Tensor
    """
```

Implementation:

```python
_, valid_mask = one_hot_target(target, num_classes, ignore_index)

pred = torch.argmax(logits, dim=1)

if ignore_index is None:
    valid_bool = torch.ones_like(target, dtype=torch.bool)
else:
    valid_bool = target != ignore_index

correct = ((pred == target) & valid_bool).unsqueeze(1)
incorrect = ((pred != target) & valid_bool).unsqueeze(1)

alpha_t = alpha0 * math.exp(-float(epoch) / max(float(max_epoch), 1.0))

u = uncertainty.clamp(min=eps, max=1.0 - eps)
belief_sum = belief.sum(dim=1, keepdim=True)

loss_correct = -alpha_t * belief_sum * torch.log(1.0 - u)
loss_incorrect = -(1.0 - alpha_t) * (1.0 - belief_sum) * torch.log(u)

loss_map = torch.zeros_like(u)
loss_map = loss_map + torch.where(correct, loss_correct, torch.zeros_like(loss_correct))
loss_map = loss_map + torch.where(incorrect, loss_incorrect, torch.zeros_like(loss_incorrect))

loss = (loss_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
return loss
```

Note:

Trong paper, phần diễn giải và công thức CEU có thể hơi dễ gây hiểu nhầm về term nào dominate ở early stage. Ở version này, implement đúng theo formula mặc định.

---

## 2.11. Class: `EvidentialSegmentationLoss`

```python
class EvidentialSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_kl=0.2,
        lambda_dice=1.0,
        ignore_index=None,
        activation="relu",
        alpha0=1.0,
        kl_annealing_epochs=50,
        include_background=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.lambda_dice = lambda_dice
        self.ignore_index = ignore_index
        self.activation = activation
        self.alpha0 = alpha0
        self.kl_annealing_epochs = kl_annealing_epochs
        self.include_background = include_background
```

Forward:

```python
def forward(self, logits, target, epoch, max_epoch):
    evidence, alpha, prob, uncertainty, belief = dirichlet_params_from_logits(
        logits,
        activation=self.activation,
    )

    loss_ce = evidential_ce_loss(
        alpha=alpha,
        target=target,
        num_classes=self.num_classes,
        ignore_index=self.ignore_index,
    )

    loss_ceu = calibrated_evidential_uncertainty_loss(
        logits=logits,
        alpha=alpha,
        belief=belief,
        uncertainty=uncertainty,
        target=target,
        num_classes=self.num_classes,
        epoch=epoch,
        max_epoch=max_epoch,
        alpha0=self.alpha0,
        ignore_index=self.ignore_index,
    )

    annealing_coef = min(1.0, float(epoch + 1) / float(max(self.kl_annealing_epochs, 1)))

    loss_kl = evidential_kl_loss(
        alpha=alpha,
        target=target,
        num_classes=self.num_classes,
        ignore_index=self.ignore_index,
        annealing_coef=annealing_coef,
    )

    loss_dice = evidential_dice_loss(
        prob=prob,
        target=target,
        num_classes=self.num_classes,
        ignore_index=self.ignore_index,
        include_background=self.include_background,
    )

    total_loss = loss_ce + loss_ceu + self.lambda_kl * loss_kl + self.lambda_dice * loss_dice

    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_ece": loss_ce.detach(),
        "loss_ceu": loss_ceu.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_dice": loss_dice.detach(),
        "uncertainty_mean": uncertainty.detach().mean(),
        "evidence_mean": evidence.detach().mean(),
    }

    aux_dict = {
        "prob": prob.detach(),
        "uncertainty": uncertainty.detach(),
        "evidence": evidence.detach(),
        "alpha": alpha.detach(),
    }

    return total_loss, loss_dict, aux_dict
```

---

# 3. File: `train_evidential_fullmask.py`

## 3.1. Purpose

Training script mới cho full-mask supervised evidential segmentation.

Script này giống style `train.py` hiện tại nhưng bỏ teacher/student và pseudo-label.

---

## 3.2. Required imports

```python
import argparse
import logging
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from val import test_single_volume
from utils.evidential_losses import EvidentialSegmentationLoss
```

---

## 3.3. Args

```python
parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../../data/ACDC')
parser.add_argument('--exp', type=str, default='EvidentialFullMask')
parser.add_argument('--data', type=str, default='ACDC')
parser.add_argument('--fold', type=str, default='MAAGfold70')
parser.add_argument('--sup_type', type=str, default='label')
parser.add_argument('--model', type=str, default='unet_hl')
parser.add_argument('--num_classes', type=int, default=4)

parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--lambda_kl', type=float, default=0.2)
parser.add_argument('--lambda_dice', type=float, default=1.0)
parser.add_argument('--alpha0', type=float, default=1.0)
parser.add_argument('--kl_annealing_epochs', type=int, default=50)
parser.add_argument(
    '--evidence_activation',
    type=str,
    default='relu',
    choices=['relu', 'softplus', 'exp'],
)
parser.add_argument('--include_background_dice', action='store_true')

parser.add_argument('--ignore_index', type=int, default=-1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.ignore_index < 0:
    args.ignore_index = None
```

---

## 3.4. Helper functions

```python
def unpack_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
```

Validation wrapper:

```python
class LogitsOnlyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return unpack_logits(output)
```

Worker seed:

```python
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
```

---

## 3.5. Train function skeleton

```python
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(
        net_type=args.model,
        in_chns=1,
        class_num=args.num_classes,
    )

    model.cuda()
    model.train()

    db_train = ACDCDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )

    db_val = ACDCDataSets(
        base_dir=args.root_path,
        fold=args.fold,
        split="val",
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    criterion = EvidentialSegmentationLoss(
        num_classes=args.num_classes,
        lambda_kl=args.lambda_kl,
        lambda_dice=args.lambda_dice,
        ignore_index=args.ignore_index,
        activation=args.evidence_activation,
        alpha0=args.alpha0,
        kl_annealing_epochs=args.kl_annealing_epochs,
        include_background=args.include_background_dice,
    )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled in enumerate(trainloader):
            image = sampled['image'].cuda()
            label = sampled['label'].cuda().long()

            output = model(image)
            logits = unpack_logits(output)

            loss, loss_dict, aux_dict = criterion(
                logits=logits,
                target=label,
                epoch=epoch_num,
                max_epoch=max_epoch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar("train/loss_total", loss_dict["loss_total"].item(), iter_num)
            writer.add_scalar("train/loss_ece", loss_dict["loss_ece"].item(), iter_num)
            writer.add_scalar("train/loss_ceu", loss_dict["loss_ceu"].item(), iter_num)
            writer.add_scalar("train/loss_kl", loss_dict["loss_kl"].item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dict["loss_dice"].item(), iter_num)
            writer.add_scalar("train/uncertainty_mean", loss_dict["uncertainty_mean"].item(), iter_num)
            writer.add_scalar("train/evidence_mean", loss_dict["evidence_mean"].item(), iter_num)
            writer.add_scalar("train/lr", lr_, iter_num)

            if iter_num % 400 == 0:
                logging.info(
                    "iteration %d : loss=%f, ce=%f, ceu=%f, kl=%f, dice=%f, uncertainty=%f"
                    % (
                        iter_num,
                        loss_dict["loss_total"].item(),
                        loss_dict["loss_ece"].item(),
                        loss_dict["loss_ceu"].item(),
                        loss_dict["loss_kl"].item(),
                        loss_dict["loss_dice"].item(),
                        loss_dict["uncertainty_mean"].item(),
                    )
                )

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                eval_model = LogitsOnlyModel(model)
                metric_list = 0.0

                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        eval_model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        'info/val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance

                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(
                            iter_num,
                            round(best_performance, 4),
                        ),
                    )

                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(args.model),
                    )

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f'
                    % (iter_num, performance, mean_hd95)
                )

                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path,
                    'iter_' + str(iter_num) + '.pth',
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
```

---

## 3.6. Main block

```python
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../../checkpoints/{}_{}".format(args.data, args.exp)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
```

---

# 4. Acceptance tests

## 4.1. Import test

```bash
python -c "from utils.evidential_losses import EvidentialSegmentationLoss; print('ok')"
```

Expected:

```text
ok
```

---

## 4.2. Loss shape test

```python
import torch
from utils.evidential_losses import EvidentialSegmentationLoss

criterion = EvidentialSegmentationLoss(num_classes=4)

logits = torch.randn(2, 4, 256, 256).cuda()
target = torch.randint(0, 4, (2, 256, 256)).cuda()

loss, loss_dict, aux_dict = criterion(
    logits,
    target,
    epoch=0,
    max_epoch=10,
)

assert torch.isfinite(loss)
assert aux_dict["prob"].shape == logits.shape
assert aux_dict["uncertainty"].shape == (2, 1, 256, 256)

print(loss.item())
```

Expected:

- không crash,
- loss finite,
- uncertainty shape đúng `[B,1,H,W]`.

---

## 4.3. Debug training 2 iterations

```bash
python train_evidential_fullmask.py \
  --root_path ../../data/ACDC \
  --exp Debug_EvidentialFullMask \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type label \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 2 \
  --batch_size 2 \
  --gpu 0
```

Expected:

- script chạy được,
- load data từ `ACDC_training_slices`,
- log loss,
- không NaN,
- dừng sau 2 iterations.

---

## 4.4. Validation compatibility test

```bash
python train_evidential_fullmask.py \
  --root_path ../../data/ACDC \
  --exp Debug_EvidentialFullMaskVal \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type label \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 401 \
  --batch_size 2 \
  --gpu 0
```

Expected:

- validation chạy ở iteration 400,
- `test_single_volume` hoạt động kể cả khi model return tuple,
- log mean Dice và mean HD95,
- save best checkpoint.

---

## 4.5. Checkpoint structure

Expected:

```text
../../checkpoints/ACDC_Debug_EvidentialFullMaskVal/
  log.txt
  log/
  unet_hl_best_model.pth
```

---

## 4.6. No interactive artifacts

Run:

```bash
grep -R "prompt" train_evidential_fullmask.py utils/evidential_losses.py
grep -R "teacher" train_evidential_fullmask.py utils/evidential_losses.py
grep -R "pseudo" train_evidential_fullmask.py utils/evidential_losses.py
```

Expected:

- không có prompt simulation code,
- không có teacher code,
- không có pseudo-label code.

---

# 5. Agent instruction ngắn gọn

Bạn đang modify một repo PyTorch medical image segmentation.

Hãy thêm một version train mới:

```text
Fully-supervised + full mask + evidential calibrated learning
```

## Add files

```text
utils/evidential_losses.py
train_evidential_fullmask.py
```

## Do not modify

Không xóa hoặc rewrite:

```text
train.py
dataloader/acdc.py
val.py
```

Chỉ sửa `val.py` nếu thực sự cần, nhưng ưu tiên dùng `LogitsOnlyModel` wrapper trong training script.

## Core logic

Use full mask:

```python
--sup_type label
```

Use only one model:

```python
model = net_factory(...)
```

If output is tuple/list:

```python
logits = output[0]
```

Compute evidential quantities:

```python
evidence = relu(logits)
alpha = evidence + 1
S = alpha.sum(dim=1, keepdim=True)
prob = alpha / S
uncertainty = num_classes / S
belief = evidence / S
```

Total loss:

```python
loss = LCE + LCEU + lambda_kl * LKL + lambda_dice * LDice
```

Default coefficients:

```python
lambda_kl = 0.2
lambda_dice = 1.0
alpha0 = 1.0
```

Training behavior:

- same dataset loader,
- same snapshot path format,
- same validation schedule,
- same best checkpoint saving,
- log all loss components.

---

# 6. Run commands

## Debug run

```bash
python train_evidential_fullmask.py \
  --root_path ../../data/ACDC \
  --exp Debug_EvidentialFullMask \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type label \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 2 \
  --batch_size 2 \
  --gpu 0
```

## Normal run

```bash
python train_evidential_fullmask.py \
  --root_path ../../data/ACDC \
  --exp EvidentialFullMask \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type label \
  --model unet_hl \
  --num_classes 4 \
  --max_iterations 30000 \
  --batch_size 8 \
  --base_lr 0.01 \
  --lambda_kl 0.2 \
  --lambda_dice 1.0 \
  --alpha0 1.0 \
  --gpu 0
```

---

# 7. Ghi chú method

Paper EUGIS có 2 stage:

1. **Stage I**: train evidential calibrated model để sinh uncertainty map.
2. **Stage II**: dùng uncertainty map để sinh point prompt cho interactive segmentation.

Task hiện tại chỉ implement **Stage I-style fully-supervised training** với full mask.

Không làm Stage II.
