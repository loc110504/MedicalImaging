#!/bin/bash
# Run ACDC training

CUDA_VISIBLE_DEVICES=0 python train_acdc_lgdt.py \
    --root_path ../../data/ACDC \
    --exp LED_SDT \
    --data ACDC \
    --fold MAAGfold70 \
    --sup_type scribble \
    --model unet_lgdt \
    --num_classes 4 \
    --max_iterations 30000 \
    --batch_size 8 \
    --base_lr 0.01 \
    --patch_size 256 256 \
    --gpu 0




# Run MSCMR training
# CUDA_VISIBLE_DEVICES=0 python train_method_mscmr.py \
#   --root_path ../../data/MSCMR \
#   --exp DualTeacher \
#   --data MSCMR \
#   --sup_type scribble \
#   --model unet_hl \
#   --num_classes 4 \
#   --batch_size 8 \
#   --base_lr 0.01 \
#   --max_iterations 30000 \
#   --confidence_threshold 0.5 \
#   --consistency_rampup 40 \
#   --gpu 0 \
#   --seed 2022 \
#   --deterministic 1
