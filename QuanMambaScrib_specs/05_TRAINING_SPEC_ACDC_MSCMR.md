# 05 - Training Script Specification for ACDC and MSCMR

Files:

```text
code/train/train_quanmambascrib_acdc.py
code/train/train_quanmambascrib_mscmr.py
```

Use the existing baseline training script as the starting point, but remove EMA teacher logic from the main method.

## Required CLI Arguments

```python
# Dataset/basic
--root_path
--exp QuanMambaScrib
--data ACDC or MSCMR
--fold MAAGfold70
--sup_type scribble
--num_classes 4
--max_iterations 40000
--batch_size 16
--base_lr 0.01
--patch_size [256,256]
--seed 2022
--gpu 0

# Model
--model quanmambascrib
--unet_type unet_hl
--mamba_variant vmunet
--in_chns 1

# QPIM
--qpim_backend torch_angle_fidelity
--qpim_dim 8
--qpim_tau 0.5
--qpim_momentum 0.99
--qpim_normalize_z 1
--qpim_detach_prototypes 0

# Loss weights
--lambda_q 1.0
--lambda_cps 1.0
--pseudo_loss_weight 8.0
--consistency_rampup 40.0

# Q-CPS thresholds
--warmup_iterations 5000
--tau_u_start 0.95
--tau_u_end 0.75
--tau_m_start 0.95
--tau_m_end 0.75
--tau_q_start 0.90
--tau_q_end 0.70

# Debug
--pseudo_metric_interval 200
--dense_label_key gt_label
--save_boundary_interval 400
--save_boundary_num_images 10
```

## Model Creation

```python
def create_model(num_classes, train_args):
    model = QuanMambaScrib(
        in_chns=1,
        class_num=num_classes,
        unet_type=train_args.unet_type,
        mamba_variant=train_args.mamba_variant,
        qpim_backend=train_args.qpim_backend,
        qpim_dim=train_args.qpim_dim,
        qpim_tau=train_args.qpim_tau,
        qpim_momentum=train_args.qpim_momentum,
        ignore_index=num_classes,
    ).cuda()
    return model
```

## Optimizer
Use one optimizer for all modules initially:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=0.0001,
)
```

If Mamba branch is unstable, allow parameter groups later:

```python
optimizer = optim.SGD([
    {"params": model.unet.parameters(), "lr": base_lr},
    {"params": model.mamba_unet.parameters(), "lr": base_lr},
    {"params": model.qpim.parameters(), "lr": base_lr * 0.1},
], momentum=0.9, weight_decay=0.0001)
```

Default should be the simple single optimizer.

## Training Loop Steps

For each mini-batch:

```python
volume_batch = sampled_batch['image'].cuda()
label_batch = sampled_batch['label'].cuda()

out = model(
    volume_batch,
    scribble_label=label_batch,
    update_memory=True,
    return_q=True,
)

logits_u = out['logits_u']
logits_m = out['logits_m']
prob_u = out['prob_u']
prob_m = out['prob_m']
Q_u = out['Q_u']
Q_m = out['Q_m']
```

Compute losses:

```python
loss_pce_u = partial_ce_loss(logits_u, label_batch, ignore_index=num_classes)
loss_pce_m = partial_ce_loss(logits_m, label_batch, ignore_index=num_classes)
loss_pce = loss_pce_u + loss_pce_m

loss_q_u = partial_prob_ce(Q_u, label_batch, ignore_index=num_classes)
loss_q_m = partial_prob_ce(Q_m, label_batch, ignore_index=num_classes)
loss_q = loss_q_u + loss_q_m
```

If after warmup:

```python
tau_u, tau_m, tau_q = get_current_thresholds(...)
pseudo_info = build_quantum_guided_reliable_masks(...)
loss_u_to_m = masked_hard_ce(logits_m, pseudo_info['pseudo_u'], pseudo_info['R_u'])
loss_m_to_u = masked_hard_ce(logits_u, pseudo_info['pseudo_m'], pseudo_info['R_m'])
loss_qcps = loss_u_to_m + loss_m_to_u
```

Otherwise:

```python
loss_qcps = logits_u.new_tensor(0.0)
```

Ramp pseudo weight:

```python
cps_weight = train_args.pseudo_loss_weight * ramps.sigmoid_rampup(
    iter_num // len(trainloader),
    train_args.consistency_rampup,
)
```

Final loss:

```python
loss = loss_pce + train_args.lambda_q * loss_q + cps_weight * train_args.lambda_cps * loss_qcps
```

Backprop:

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)  # optional but recommended
optimizer.step()
```

LR decay:

```python
lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
```

## Validation
Use `test_single_volume` with a wrapper that returns logits. Existing `test_single_volume` expects model(image) returns logits or tuple first item logits.

Create inference wrapper modes:

```python
class EnsembleInferenceWrapper(nn.Module):
    def __init__(self, model, mode='ensemble'):
        self.model = model
        self.mode = mode
    def forward(self, x):
        out = self.model(x, scribble_label=None, return_q=False)
        if self.mode == 'ensemble':
            prob = 0.5 * (out['prob_u'] + out['prob_m'])
            return torch.log(prob + 1e-8)
        if self.mode == 'mamba':
            return out['logits_m']
        if self.mode == 'unet':
            return out['logits_u']
```

Validate at least ensemble and Mamba-only:

```python
performance_ens, hd95_ens = validate(EnsembleInferenceWrapper(model, 'ensemble'), ...)
performance_m, hd95_m = validate(EnsembleInferenceWrapper(model, 'mamba'), ...)
```

Save best based on ensemble Dice by default.

## Checkpoint Save Format

```python
torch.save({
    'iter_num': iter_num,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_performance': best_performance,
    'args': vars(train_args),
}, save_path)
```

Also save a pure state_dict for compatibility:

```python
torch.save(model.state_dict(), save_best_state_dict)
```

## ACDC Differences
- default `num_classes=4`.
- `ignore_index=4`.
- likely input channel = 1.
- use existing `ACDCDataSets` and `RandomGenerator`.

## MSCMR Differences
Create `train_quanmambascrib_mscmr.py` by matching the existing MSCMR training conventions in the repository.

Checklist:
- import `MSCMRDataSets` from current dataloader if available;
- ensure `return_full_label=True` equivalent exists or disable dense debug;
- confirm `num_classes` and `ignore_index`;
- use dataset-specific validation/test script.

## Required Console Logs Every 200 Iterations

```text
iteration X : loss=..., loss_pce=..., loss_q=..., loss_qcps=..., cps_weight=...
thresholds: tau_u=..., tau_m=..., tau_q=...
reliable_u=..., reliable_m=..., qconf_u=..., qconf_m=...
```
