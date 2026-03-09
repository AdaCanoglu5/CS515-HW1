#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs

COMMON_ARGS=(
  --mode both
  --seed 42
  --device cuda
  --epochs 10
  --batch_size 64
)

run_experiment() {
  local run_name="$1"
  shift

  local save_path="runs/${run_name}.pth"
  local log_csv="runs/${run_name}.csv"

  python main.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --run_name "$run_name" \
    --save_path "$save_path" \
    --log_csv "$log_csv"
}

# Baseline (B0)
run_experiment B0_baseline \
  --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre \
  --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0

# A) Architecture impact (depth/width)
run_experiment A1_depth1_w256 --hidden_sizes 256 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment A2_depth2_w256 --hidden_sizes 256,256 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment A3_depth3_w256 --hidden_sizes 256,256,256 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment A4_depth4_w256 --hidden_sizes 256,256,256,256 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment A5_depth3_w128 --hidden_sizes 128,128,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment A6_depth3_w512 --hidden_sizes 512,512,512 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0

# B) Activation functions (baseline is relu)
run_experiment B2_act_gelu --hidden_sizes 512,256,128 --activation gelu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment B3_act_tanh --hidden_sizes 512,256,128 --activation tanh --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0

# C) Dropout impact
run_experiment C1_drop_0p0 --hidden_sizes 512,256,128 --activation relu --dropout 0.0 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment C2_drop_0p1 --hidden_sizes 512,256,128 --activation relu --dropout 0.1 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment C4_drop_0p5 --hidden_sizes 512,256,128 --activation relu --dropout 0.5 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0

# D) Training framework (LR + scheduler)
run_experiment D1_lr_1e-2_step --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-2 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment D3_lr_3e-4_step --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 3e-4 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment D4_sched_none --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler none --weight_decay 1e-4 --l1_lambda 0
run_experiment D6_sched_cosine --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler cosine --tmax 10 --weight_decay 1e-4 --l1_lambda 0

# E) BatchNorm and position
run_experiment E2_bn_post --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position post --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment E3_bn_off --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 0 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0

# F) Regularization (L2 sweep + L1 sweep)
run_experiment F1_l2_0 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 0 --l1_lambda 0
run_experiment F2_l2_1e-5 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-5 --l1_lambda 0
run_experiment F3_l2_1e-4 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-4 --l1_lambda 0
run_experiment F4_l2_1e-3 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 1e-3 --l1_lambda 0
run_experiment F5_l1_1e-6 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 0 --l1_lambda 1e-6
run_experiment F6_l1_1e-5 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 0 --l1_lambda 1e-5
run_experiment F7_l1_1e-4 --hidden_sizes 512,256,128 --activation relu --dropout 0.3 --use_bn 1 --bn_position pre --lr 1e-3 --scheduler step --step_size 5 --gamma 0.5 --weight_decay 0 --l1_lambda 1e-4
