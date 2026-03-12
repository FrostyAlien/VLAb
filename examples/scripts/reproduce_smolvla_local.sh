#!/usr/bin/env bash
set -euo pipefail

# Local (non-SLURM) reproduction entrypoint for SmolVLA pretraining
# with 512x512 inputs and LoRA on the full VLM (vision + text).
#
# Override examples:
#   POLICY=smolvla VLM_REPO_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct bash examples/scripts/reproduce_smolvla_local.sh
#   POLICY=smolvla2 VLM_REPO_ID=google/gemma-3-4b-pt bash examples/scripts/reproduce_smolvla_local.sh

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

# Ensure local package imports (lerobot) work even without editable install.
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

DATASET_LIST_FILE="${DATASET_LIST_FILE:-$ROOT_DIR/examples/all_datasets_relative.txt}"
if [[ ! -f "$DATASET_LIST_FILE" ]]; then
    echo "Dataset list file not found: $DATASET_LIST_FILE"
    exit 1
fi

REPO_IDS="${REPO_IDS:-$(cat "$DATASET_LIST_FILE")}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train_smolvla_local_$(date +%Y%m%d_%H%M%S)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT_DIR/accelerate_configs/single_gpu.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# Core training setup
POLICY="${POLICY:-smolvla2}"
VLM_REPO_ID="${VLM_REPO_ID:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:--1}"
USE_AMP="${USE_AMP:-true}"

# 512 image setup
MAX_NUM_IMAGES="${MAX_NUM_IMAGES:-2}"
MAX_IMAGE_DIM="${MAX_IMAGE_DIM:-512}"

# Dataset backend setup (aligned with hf_streaming pretraining)
DATASET_BACKEND="${DATASET_BACKEND:-hf_streaming}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
HF_STREAMING_SHUFFLE="${HF_STREAMING_SHUFFLE:-true}"
HF_STREAMING_SHUFFLE_BUFFER_SIZE="${HF_STREAMING_SHUFFLE_BUFFER_SIZE:-32}"
HF_STREAMING_NUM_SHARDS="${HF_STREAMING_NUM_SHARDS:-8}"
HF_STREAMING_MAX_OPEN_REPOS="${HF_STREAMING_MAX_OPEN_REPOS:-64}"
# Local root that should contain:
#   $DATA_ROOT/community_dataset_v1/<user>/<dataset>/meta/info.json
DATA_ROOT="${DATA_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}}"

# LoRA setup (full VLM so vision adapters are trained too)
PEFT_METHOD="${PEFT_METHOD:-lora}"
PEFT_TARGET_MODEL="${PEFT_TARGET_MODEL:-vlm}"
LORA_R="${LORA_R:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS:-true}"

# WandB setup
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-training}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NOTES="${WANDB_NOTES:-}"

echo "Running local SmolVLA reproduce training"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_image_dim=$MAX_IMAGE_DIM"
echo "accelerate_config=$ACCELERATE_CONFIG num_processes=$NUM_PROCESSES"
echo "dataset_backend=$DATASET_BACKEND video_backend=$VIDEO_BACKEND"
echo "hf_streaming_max_open_repos=$HF_STREAMING_MAX_OPEN_REPOS"
echo "data_root=$DATA_ROOT"
echo "peft_method=$PEFT_METHOD peft_target_model=$PEFT_TARGET_MODEL"
echo "freeze_vision_encoder=$FREEZE_VISION_ENCODER train_expert_only=$TRAIN_EXPERT_ONLY"
echo "wandb_enable=$WANDB_ENABLE wandb_project=$WANDB_PROJECT"

# Fast-fail validation for local hf_streaming layout.
FIRST_REPO_ID="${REPO_IDS%%,*}"
if [[ "$DATASET_BACKEND" == "hf_streaming" ]]; then
    if [[ ! -f "$DATA_ROOT/$FIRST_REPO_ID/meta/info.json" ]]; then
        echo "Missing local dataset metadata: $DATA_ROOT/$FIRST_REPO_ID/meta/info.json"
        echo "Set DATA_ROOT to the directory that contains community_dataset_v1/v2 folders."
        echo "Example: DATA_ROOT=\$HOME/.cache/huggingface/lerobot"
        exit 1
    fi
fi

# Guard against accidental multi-process launch on a single visible GPU.
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -z "$VISIBLE_GPU_COUNT" || "$VISIBLE_GPU_COUNT" -lt "$NUM_PROCESSES" ]]; then
            echo "Not enough visible GPUs for NUM_PROCESSES=$NUM_PROCESSES (detected: ${VISIBLE_GPU_COUNT:-0})."
            echo "Use NUM_PROCESSES=1 or expose more GPUs (e.g. CUDA_VISIBLE_DEVICES=0,1)."
            exit 1
        fi
    fi
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_PROCESSES" \
    src/lerobot/scripts/train.py \
    --policy.type="$POLICY" \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="$DATA_ROOT" \
    --dataset.backend="$DATASET_BACKEND" \
    --dataset.video_backend="$VIDEO_BACKEND" \
    --dataset.train_on_all_features=true \
    --dataset.features_version=2 \
    --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable=true \
    --dataset.max_num_images="$MAX_NUM_IMAGES" \
    --dataset.max_image_dim="$MAX_IMAGE_DIM" \
    --dataset.min_fps=30 \
    --dataset.max_fps=30 \
    --dataset.hf_streaming_shuffle="$HF_STREAMING_SHUFFLE" \
    --dataset.hf_streaming_shuffle_buffer_size="$HF_STREAMING_SHUFFLE_BUFFER_SIZE" \
    --dataset.hf_streaming_num_shards="$HF_STREAMING_NUM_SHARDS" \
    --dataset.hf_streaming_max_open_repos="$HF_STREAMING_MAX_OPEN_REPOS" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --eval_freq="$EVAL_FREQ" \
    --policy.use_amp="$USE_AMP" \
    --policy.repo_id="$VLM_REPO_ID" \
    --policy.vlm_model_name="$VLM_REPO_ID" \
    --policy.push_to_hub=false \
    --policy.max_action_dim=32 \
    --policy.max_state_dim=32 \
    --policy.optimizer_lr=5e-4 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps="$STEPS" \
    --policy.scheduler_decay_lr=1e-6 \
    --policy.num_vlm_layers=16 \
    --policy.expert_width_multiplier=1.0 \
    --policy.causal_action_attention_mask=true \
    --policy.self_attn_every_n_layers=2 \
    --policy.attention_mode=cross_attn \
    --policy.prefix_length=0 \
    --policy.load_vlm_weights="$LOAD_VLM_WEIGHTS" \
    --policy.resize_imgs_with_padding="[$MAX_IMAGE_DIM,$MAX_IMAGE_DIM]" \
    --policy.peft_method="$PEFT_METHOD" \
    --policy.peft_target_model="$PEFT_TARGET_MODEL" \
    --policy.peft_config.r="$LORA_R" \
    --policy.peft_config.target_modules="$LORA_TARGET_MODULES" \
    --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
    --wandb.enable="$WANDB_ENABLE" \
    --wandb.project="$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb.entity="$WANDB_ENTITY"} \
    ${WANDB_NOTES:+--wandb.notes="$WANDB_NOTES"} 

