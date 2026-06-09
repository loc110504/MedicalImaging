# 00 - QuanMambaScrib Implementation Brief

## Goal
Final code name: **QuanMambaScrib**.

Paper/method aliases used in the draft:
- `QuanMambaScrib`: title-level method name.
- `QCo-Mamba`: internal framework name in the PDF.

Use `QuanMambaScrib` for class names, experiment names, checkpoint folders, and spec references unless an existing file already uses `QCo-Mamba`.

## Method Summary
QuanMambaScrib is a scribble-supervised medical image segmentation framework with:

1. **Dual-view co-training**
   - Branch A: conventional U-Net.
   - Branch B: Mamba-UNet.
   - U-Net captures local texture and boundary detail.
   - Mamba-UNet captures long-range anatomical context.

2. **Quantum Prototype Interaction Module (QPIM)**
   - Project U-Net and Mamba-UNet features into a compact dimension `d`.
   - Build class prototypes from scribble pixels.
   - Maintain EMA prototype memories.
   - Compute quantum prototype affinity between each dense token and each class prototype.
   - Output `Q_u` and `Q_m`, the quantum prototype predictions for U-Net and Mamba-UNet branches.

3. **Quantum prototype supervision**
   - Apply CE on `Q_u` and `Q_m` only at scribble pixels.

4. **Quantum-guided cross pseudo supervision (Q-CPS)**
   - U-Net pseudo-labels supervise Mamba-UNet only where U-Net prediction agrees with U-Net-side quantum prototype prediction.
   - Mamba-UNet pseudo-labels supervise U-Net only where Mamba prediction agrees with Mamba-side quantum prototype prediction.

5. **Final objective in the finalized PDF**
   - `L = L_pCE + lambda_q * L_q + lambda_cps * L_qcps`.

## Important Consistency Note
The abstract mentions dual-view consistency and prototype-affinity regularization, but Section 3.7 of the uploaded PDF only defines:

```text
L = L_pCE + lambda_q L_q + lambda_cps L_qcps
```

Therefore, the implementation **must default to the Section 3.7 objective**. Optional consistency/affinity losses may be added as disabled ablation hooks, but they must be off by default.

## Repository Constraints
Existing repository layout from `Agents.MD`:

```text
code/
  train/
  test/
  dataloader/
  networks/
  utils/
  val.py
checkpoints/
data/ACDC
data/MSCMR
```

The current training script uses:
- `ACDCDataSets(..., return_full_label=True)` for dense-label debug.
- scribble labels where unknown pixels use `ignore_index = num_classes`.
- `CrossEntropyLoss(ignore_index=num_classes)` for partial CE.
- `test_single_volume` in `code/val.py` for validation.
- TensorBoardX `SummaryWriter`.

## Non-goals
Do not implement a full quantum image segmentation network.
Do not encode the full image into a quantum circuit.
Do not claim or rely on quantum speedup.
QPIM is a compact prototype-affinity verifier operating on low-dimensional features.

## High-Level Files to Add

```text
code/networks/quan_mamba_scrib.py
code/networks/mamba_unet_2d.py
code/networks/qpim.py
code/utils/quan_mamba_losses.py
code/utils/quan_mamba_pseudo.py
code/train/train_quanmambascrib_acdc.py
code/train/train_quanmambascrib_mscmr.py
code/test/test_quanmambascrib_acdc.py
code/test/test_quanmambascrib_mscmr.py
```

Modify:

```text
code/networks/net_factory.py
code/train/run.sh
requirements.txt
```

## Primary Acceptance Criteria
1. `python code/train/train_quanmambascrib_acdc.py --max_iterations 2` runs without shape/device errors.
2. U-Net and Mamba-UNet both produce logits of shape `[B, K, H, W]`.
3. QPIM produces `Q_u` and `Q_m` of shape `[B, K, H, W]`, and each softmax distribution sums to 1 over class dimension.
4. Losses are finite when no reliable pseudo pixels exist.
5. Validation works using ensemble prediction and Mamba-only prediction.
6. Checkpoints are saved in `checkpoints/<DATASET>_QuanMambaScrib/`.
