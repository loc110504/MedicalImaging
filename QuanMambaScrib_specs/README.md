# QuanMambaScrib Implementation Specs

This folder contains detailed implementation specs for coding the finalized method:

**QuanMambaScrib: Quantum Prototype-Guided Dual-View Co-Training for Scribble-Supervised Medical Image Segmentation**.

Recommended reading order:

1. `00_IMPLEMENTATION_BRIEF.md`
2. `01_REPO_INTEGRATION_SPEC.md`
3. `02_DUAL_NETWORK_SPEC.md`
4. `03_QPIM_SPEC.md`
5. `04_LOSSES_AND_PSEUDO_SPEC.md`
6. `05_TRAINING_SPEC_ACDC_MSCMR.md`
7. `06_MAMBA_ADAPTATION_SPEC.md`
8. `07_QUANTUM_BACKENDS_AND_DEPENDENCIES.md`
9. `08_TESTING_DEBUG_ABLATION_SPEC.md`
10. `09_RUN_COMMANDS_AND_REQUIREMENTS.md`
11. `10_AGENT_IMPLEMENTATION_TASK_LIST.md`

Core implementation principle:

> Keep U-Net and Mamba-UNet as the actual segmentation models. Use QPIM only as a compact prototype-affinity verifier for reliable pseudo-label cross-supervision.
