# 08 - Testing, Debugging, and Ablation Specification

## Smoke Tests
Create `code/test_debug/test_quanmambascrib_shapes.py` or run inside a notebook/script.

### Test 1: Model Shape

```python
model = QuanMambaScrib(in_chns=1, class_num=4).cuda()
x = torch.randn(2,1,256,256).cuda()
y = torch.full((2,256,256), 4, dtype=torch.long).cuda()
y[:, 50:60, 50:60] = 1
out = model(x, scribble_label=y)
assert out['logits_u'].shape == (2,4,256,256)
assert out['logits_m'].shape == (2,4,256,256)
assert out['Q_u'].shape == (2,4,256,256)
assert out['Q_m'].shape == (2,4,256,256)
assert torch.allclose(out['Q_u'].sum(dim=1), torch.ones_like(out['Q_u'][:,0]), atol=1e-4)
```

### Test 2: Loss Finite

```python
loss_pce = partial_ce_loss(out['logits_u'], y, 4) + partial_ce_loss(out['logits_m'], y, 4)
loss_q = partial_prob_ce(out['Q_u'], y, 4) + partial_prob_ce(out['Q_m'], y, 4)
loss = loss_pce + loss_q
assert torch.isfinite(loss)
loss.backward()
```

### Test 3: Empty Reliable Mask
Use thresholds `0.99999` and verify Q-CPS returns `0.0` without NaN.

### Test 4: Memory Save/Load

```python
state = model.state_dict()
model2 = QuanMambaScrib(in_chns=1, class_num=4).cuda()
model2.load_state_dict(state, strict=True)
```

## Debug Visualizations
Adapt existing boundary visualization utilities.

Save panels:

```text
1. input image
2. scribble label
3. U-Net prediction
4. Mamba prediction
5. Q_u prediction
6. Q_m prediction
7. reliable mask R_u
8. reliable mask R_m
9. dense label if available
```

Output folder:

```text
checkpoints/<DATASET>_QuanMambaScrib/images/iter_xxxxxx/
```

## Required Ablations
To defend the method, run these variants under the same training setting.

### Architecture Ablations

```text
A0: U-Net only + partial CE
A1: Mamba-UNet only + partial CE
A2: U-Net + Mamba-UNet co-training without QPIM
A3: QuanMambaScrib full
```

### Prototype Affinity Ablations

```text
B0: no prototype verifier
B1: cosine prototype verifier
B2: RBF prototype verifier
B3: MLP affinity verifier
B4: torch_angle_fidelity QPIM
B5: PennyLane strict quantum backend on sampled tokens, if feasible
```

### QPIM Hyperparameters

```text
proj_dim: 4, 6, 8
qpim_tau: 0.25, 0.5, 1.0
momentum: 0.9, 0.99
feature resolution: 16x16, 32x32, 64x64
```

### Threshold Ablations

```text
strict fixed: tau_u=tau_m=0.95, tau_q=0.90
relaxed fixed: tau_u=tau_m=0.75, tau_q=0.70
strict-to-relaxed schedule: default
```

## Metrics
Use existing validation metrics:

```text
Dice
HD95
```

Add debug metrics:

```text
pseudo selected ratio
pseudo label Dice against dense train mask if available
pseudo boundary loss
reliable ratio per class
```

## Failure Modes and Fixes

### Problem: `R_u` and `R_m` are always zero
Fix:
- lower thresholds;
- warm up longer;
- check QPIM output entropy;
- confirm labels use ignore_index correctly.

### Problem: QPIM predicts only background
Fix:
- class-balanced prototype update;
- ensure foreground scribbles are not lost during feature/label resizing;
- upsample projected features to full label size for prototype building.

### Problem: memory prototype is zero for absent class
Fix:
- initialize from first available scribble;
- skip absent-class logits by setting kernel to low value if memory not initialized;
- log missing classes.

### Problem: Mamba branch crashes due to CUDA selective scan
Fix:
- use VM-UNet vendored fallback;
- use simple VSS fallback for smoke testing;
- test Mamba branch independently before full training.

### Problem: validation wrapper fails
Fix:
- `test_single_volume` may expect raw logits tensor, not dict;
- use `EnsembleInferenceWrapper` that returns logits-like tensor.
