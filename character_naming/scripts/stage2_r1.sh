#!/bin/bash

source ~/anaconda3/bin/activate
conda activate video-r1

module load all/CUDA/12.4.0

# Paths (absolute, replace with your own)
output_dir="<path/to/stage2/output>"
stage1_checkpoint="<path/to/stage1/checkpoint>"
dataset_json="<path/to/stage_2_gpro_rl_aug.json>"
data_root="<path/to/video/root>"  # Directory that the dataset's video paths are relative to

cd r1

export DEBUG_MODE="false" # Set to "true" and set LOG_PATH to log training outputs
export LOG_PATH="" # Path to log file showing training outputs

export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
export WANDB_PROJECT=naming

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    grpo.py \
    --output_dir "$output_dir" \
    --model_name_or_path "$stage1_checkpoint" \
    --dataset_name "$dataset_json" \
    --data_root "$data_root" \
    --deepspeed ../scripts/zero3_grpo.json \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --report_to wandb \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 240704 \
    --num_train_epochs 1 \
    --run_name 20260329_qwen3vl_4b_thinking_train_stage2_grpo_rl_aug \
    --save_steps 100 \
    --beta 0.004 \
    --max_grad_norm 1 \
    --save_only_model false \
    --num_generations 4 \
    --ddp_timeout 1800
