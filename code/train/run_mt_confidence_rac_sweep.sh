#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train_mt_confidence_rac.py \
    --root_path ../../data/ACDC \
    --exp MT_Confidence_RAC_Sweep \
    --data ACDC \
    --fold MAAGfold70 \
    --sup_type scribble \
    --model unet_hl \
    --num_classes 4 \
    --max_iterations 30000 \
    --batch_size 8 \
    --base_lr 0.01 \
    --gpu 0 \
    --seed 2022 \
    --deterministic 1 \
    --oracle_metric_logging 1 \
    --threshold_sweep_logging 1 \
    --threshold_sweep_interval 50 \
    --threshold_sweep_report_interval 1000
