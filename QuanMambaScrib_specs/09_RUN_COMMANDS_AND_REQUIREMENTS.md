# 09 - Run Commands and Requirements

## Environment
Current project environment from `Agents.MD`:

```bash
conda create -n sdt python=3.10.18
conda activate sdt
pip install -r requirements.txt
```

Add if not already present:

```bash
pip install pennylane pennylane-lightning einops timm
```

Try Mamba dependencies separately because CUDA compatibility can be sensitive:

```bash
pip install mamba-ssm causal-conv1d
```

If Mamba install fails, use the vendored VM-UNet implementation or fallback `simple_vss` variant until environment is fixed.

## ACDC Training
From repo root:

```bash
cd code/train
python train_quanmambascrib_acdc.py \
  --root_path ../../data/ACDC \
  --exp QuanMambaScrib \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model quanmambascrib \
  --num_classes 4 \
  --max_iterations 40000 \
  --batch_size 16 \
  --base_lr 0.01 \
  --patch_size 256 256 \
  --gpu 0 \
  --qpim_backend torch_angle_fidelity \
  --qpim_dim 8 \
  --qpim_tau 0.5 \
  --lambda_q 1.0 \
  --lambda_cps 1.0 \
  --pseudo_loss_weight 8.0 \
  --warmup_iterations 5000
```

## Quick Smoke Test

```bash
cd code/train
python train_quanmambascrib_acdc.py \
  --root_path ../../data/ACDC \
  --exp QuanMambaScrib_Debug \
  --data ACDC \
  --fold MAAGfold70 \
  --sup_type scribble \
  --model quanmambascrib \
  --num_classes 4 \
  --max_iterations 2 \
  --batch_size 2 \
  --gpu 0 \
  --qpim_backend torch_angle_fidelity \
  --warmup_iterations 1
```

## MSCMR Training
Adjust dataset-specific args based on current `code/dataloader/` implementation:

```bash
cd code/train
python train_quanmambascrib_mscmr.py \
  --root_path ../../data/MSCMR \
  --exp QuanMambaScrib \
  --data MSCMR \
  --sup_type scribble \
  --model quanmambascrib \
  --num_classes <SET_CORRECT_VALUE> \
  --max_iterations 40000 \
  --batch_size 16 \
  --base_lr 0.01 \
  --gpu 0
```

## Evaluation

```bash
cd code/test
python test_quanmambascrib_acdc.py \
  --root_path ../../data/ACDC \
  --model_path ../../checkpoints/ACDC_QuanMambaScrib/quanmambascrib_best_model.pth \
  --mode ensemble
```

Also evaluate Mamba-only:

```bash
python test_quanmambascrib_acdc.py \
  --root_path ../../data/ACDC \
  --model_path ../../checkpoints/ACDC_QuanMambaScrib/quanmambascrib_best_model.pth \
  --mode mamba
```

## Run Script Update
Append to `code/train/run.sh`:

```bash
python train_quanmambascrib_acdc.py --root_path ../../data/ACDC --exp QuanMambaScrib --data ACDC --fold MAAGfold70 --sup_type scribble --model quanmambascrib --num_classes 4 --gpu 0
```

## Checkpoint Directory
Expected:

```text
checkpoints/ACDC_QuanMambaScrib/
  log.txt
  log/
  images/
  quanmambascrib_best_model.pth
  iter_<iter>_dice_<score>.pth
```
