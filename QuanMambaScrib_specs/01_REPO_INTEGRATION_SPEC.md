# 01 - Repository Integration Specification

## Current Baseline Integration Points
The provided training code is a Mean-Teacher style script. QuanMambaScrib should **not** keep EMA teacher as the main algorithm. Replace the student/EMA teacher structure with two independent branches:

```text
U-Net branch       f_u(x; theta_u)
Mamba-UNet branch f_m(x; theta_m)
QPIM              quantum prototype verifier
```

Keep useful utilities from the existing code:
- seeding and deterministic setup;
- ACDC/MSCMR dataloaders;
- `return_full_label=True` debug option;
- TensorBoard logging;
- validation through `test_single_volume`;
- checkpoint saving style;
- pseudo-quality debug helpers if they are useful.

## New File Layout

```text
code/networks/
  quan_mamba_scrib.py       # main wrapper model
  mamba_unet_2d.py          # Mamba-UNet implementation or adapter
  qpim.py                   # Quantum Prototype Interaction Module
  mamba_blocks.py           # optional: SS2D/VSS blocks if not vendoring repo files

code/utils/
  quan_mamba_losses.py      # L_pCE, L_q, L_qcps, optional disabled losses
  quan_mamba_pseudo.py      # reliable mask construction and pseudo-label metrics

code/train/
  train_quanmambascrib_acdc.py
  train_quanmambascrib_mscmr.py

code/test/
  test_quanmambascrib_acdc.py
  test_quanmambascrib_mscmr.py
```

## `net_factory.py` Changes
Add a new option:

```python
elif net_type == "quanmambascrib":
    from networks.quan_mamba_scrib import QuanMambaScrib
    net = QuanMambaScrib(
        in_chns=in_chns,
        class_num=class_num,
        unet_type="unet_hl",
        mamba_variant="vmunet",
        qpim_backend="torch_angle_fidelity",
    )
```

Also allow standalone Mamba-UNet testing:

```python
elif net_type == "mamba_unet":
    from networks.mamba_unet_2d import MambaUNet2D
    net = MambaUNet2D(in_chns=in_chns, class_num=class_num)
```

## Main Model Forward Contract
`QuanMambaScrib.forward` must accept:

```python
def forward(
    self,
    x,
    scribble_label=None,
    update_memory=True,
    return_q=True,
):
    ...
```

During training, `scribble_label` is required for QPIM. During inference, it is `None`, and the wrapper should return only segmentation outputs.

Training output dictionary:

```python
{
    "logits_u": logits_u,       # [B, K, H, W]
    "logits_m": logits_m,       # [B, K, H, W]
    "prob_u": prob_u,           # [B, K, H, W]
    "prob_m": prob_m,           # [B, K, H, W]
    "feat_u": feat_u,           # [B, C_u, h, w]
    "feat_m": feat_m,           # [B, C_m, h, w]
    "Q_u": Q_u,                 # [B, K, H, W]
    "Q_m": Q_m,                 # [B, K, H, W]
    "q_logits_u": q_logits_u,   # [B, K, H, W]
    "q_logits_m": q_logits_m,   # [B, K, H, W]
    "proto_u": proto_u,         # [K, d]
    "proto_m": proto_m,         # [K, d]
}
```

Inference output dictionary:

```python
{
    "logits_u": logits_u,
    "logits_m": logits_m,
    "prob_u": prob_u,
    "prob_m": prob_m,
    "prob_ensemble": 0.5 * (prob_u + prob_m),
}
```

## Training Script Changes
The old baseline does:

```python
model = create_model()
model_ema = create_model(ema=True)
ema_output = model_ema(teacher_input)
outputs = model(volume_batch)
```

QuanMambaScrib training should do:

```python
outputs = model(
    volume_batch,
    scribble_label=label_batch,
    update_memory=True,
    return_q=True,
)

logits_u = outputs["logits_u"]
logits_m = outputs["logits_m"]
Q_u = outputs["Q_u"]
Q_m = outputs["Q_m"]
```

Then compute:

```python
loss_pce = pce_u + pce_m
loss_q = qce_u + qce_m
loss_qcps = qcps_u_to_m + qcps_m_to_u if after warmup else 0
loss = loss_pce + lambda_q * loss_q + lambda_cps * ramp * loss_qcps
```

## Checkpoint Naming
Save:

```text
checkpoints/ACDC_QuanMambaScrib/quanmambascrib_best_model.pth
checkpoints/ACDC_QuanMambaScrib/quanmambascrib_mamba_best_model.pth   # optional branch-only
checkpoints/MSCMR_QuanMambaScrib/quanmambascrib_best_model.pth
```

Store full wrapper `state_dict`, not only one branch, unless branch-only checkpoint is explicitly saved.

## TensorBoard Required Scalars

```text
info/lr
info/total_loss
info/loss_pce
info/loss_pce_u
info/loss_pce_m
info/loss_q
info/loss_q_u
info/loss_q_m
info/loss_qcps
info/loss_u_to_m
info/loss_m_to_u
pseudo/reliable_ratio_u
pseudo/reliable_ratio_m
pseudo/agreement_ratio_u
pseudo/agreement_ratio_m
pseudo/conf_u
pseudo/conf_m
quantum/q_conf_u
quantum/q_conf_m
quantum/prototype_norm_u
quantum/prototype_norm_m
info/val_mean_dice_ensemble
info/val_mean_dice_mamba
info/val_mean_hd95_ensemble
info/val_mean_hd95_mamba
```
