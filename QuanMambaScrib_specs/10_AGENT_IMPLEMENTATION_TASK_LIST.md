# 10 - Agent Implementation Task List

Use this checklist to implement QuanMambaScrib in order. Do not start full training before all smoke tests pass.

## Phase 1 - Repository Preparation
- [ ] Create new files listed in `01_REPO_INTEGRATION_SPEC.md`.
- [ ] Add `quanmambascrib` to `code/networks/net_factory.py`.
- [ ] Add dependencies to `requirements.txt`.
- [ ] Create a debug script or temporary test cell for shape tests.

Completion criterion:
- `from networks.quan_mamba_scrib import QuanMambaScrib` works from `code/train`.

## Phase 2 - U-Net Feature Output
- [ ] Modify existing U-Net or wrap it to return `(logits, feature)`.
- [ ] Confirm original U-Net training/eval still works if `return_features=False`.
- [ ] Choose feature stage for QPIM.

Completion criterion:
- `logits_u, feat_u = unet(x, return_features=True)` works.

## Phase 3 - Mamba-UNet Branch
- [ ] Vendor/adapt VM-UNet or Mamba-UNet implementation.
- [ ] Create `MambaUNet2D` adapter.
- [ ] Ensure no hard-coded dataset paths or devices.
- [ ] Return logits and feature.

Completion criterion:
- Standalone Mamba-UNet forward/backward test passes.

## Phase 4 - QPIM
- [ ] Implement projection heads.
- [ ] Implement prototype construction from full-resolution scribble labels.
- [ ] Implement EMA prototype memory.
- [ ] Implement `torch_angle_fidelity`, `cosine`, `rbf`, `mlp_affinity` backends.
- [ ] Implement optional PennyLane backend.
- [ ] Return full-resolution `Q_u`, `Q_m`.

Completion criterion:
- QPIM shape/unit tests pass.

## Phase 5 - Wrapper Model
- [ ] Implement `QuanMambaScrib` wrapper.
- [ ] Ensure training forward returns required dict.
- [ ] Ensure inference forward works with `scribble_label=None`.
- [ ] Ensure `state_dict` save/load includes QPIM memories.

Completion criterion:
- Full wrapper random forward/backward works.

## Phase 6 - Losses and Pseudo Labels
- [ ] Implement `partial_ce_loss`.
- [ ] Implement `partial_prob_ce` for QPIM supervision.
- [ ] Implement quantum-guided reliable masks.
- [ ] Implement `L_u_to_m`, `L_m_to_u`, `L_qcps`.
- [ ] Implement threshold schedule.

Completion criterion:
- Losses finite under normal, empty-mask, and all-ignore cases.

## Phase 7 - ACDC Training Script
- [ ] Copy baseline training script to `train_quanmambascrib_acdc.py`.
- [ ] Remove EMA teacher logic from the main algorithm.
- [ ] Insert QuanMambaScrib forward/loss logic.
- [ ] Add TensorBoard logging.
- [ ] Add validation wrappers for ensemble and Mamba-only.
- [ ] Save best checkpoint.

Completion criterion:
- `--max_iterations 2` run completes.

## Phase 8 - MSCMR Training Script
- [ ] Create MSCMR version using current project dataloader.
- [ ] Adjust `num_classes` and ignore index.
- [ ] Verify validation/test script compatibility.

Completion criterion:
- MSCMR smoke run completes if data exists.

## Phase 9 - Testing Scripts
- [ ] Create ACDC test script for ensemble/mamba/unet modes.
- [ ] Create MSCMR test script.
- [ ] Ensure checkpoint loading supports both dict checkpoint and raw state_dict.

Completion criterion:
- Evaluation script runs on a saved checkpoint.

## Phase 10 - Ablations
- [ ] Add CLI `--qpim_backend`.
- [ ] Add `--disable_qpim` or set `lambda_q=0`, `lambda_cps` variants.
- [ ] Run cosine/RBF/MLP/torch-angle fidelity ablations.
- [ ] Log selected ratio and pseudo quality.

Completion criterion:
- Ablation commands run without changing source code.

## Phase 11 - Final Cleanup
- [ ] Remove debug prints.
- [ ] Keep imports grouped.
- [ ] Do not reformat unrelated files.
- [ ] Update `run.sh`.
- [ ] Write a short README section documenting commands.

Completion criterion:
- Training, validation, and testing can be launched from documented commands.
