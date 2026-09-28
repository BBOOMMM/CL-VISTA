#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
task="${1:-STAR}"
case "$task" in
  STAR) annotation=STAR/json_row/train_data_25k.json ;;
  Science) annotation=Science/json_row/train_data_30k.json ;;
  *) echo 'Usage: train.sh [STAR|Science]' >&2; exit 1 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$HIDE_PROJECT_ROOT/Video-LLaVA/HiDe"
exec deepspeed --include "localhost:${HIDE_GPUS:-0,1}" --master_port "${HIDE_PORT:-29571}" \
  videollava/train/train_mem.py \
  --deepspeed "$HIDE_PROJECT_ROOT/scripts/local_hide/zero3.json" \
  --model_name_or_path "$HIDE_PROJECT_ROOT/models/Video-LLaVA-7B" \
  --video_tower "$HIDE_PROJECT_ROOT/models/LanguageBind_Video_merge" \
  --text_tower "$HIDE_PROJECT_ROOT/models/clip-vit-large-patch14-336" \
  --data_path "$HIDE_PROJECT_ROOT/dataset/CL-VISTA/$annotation" \
  --video_folder "$HIDE_PROJECT_ROOT/dataset/CL-VISTA" \
  --output_dir "$HIDE_PROJECT_ROOT/outputs/hide-videollava-${task}-${HIDE_RUN_ID:-local}" \
  --lora_enable True --lora_r 256 --lora_alpha 512 --expert_num 8 --cur_task 0 \
  --mm_projector_lr 1e-5 --learning_rate 1e-4 --version v1 \
  --num_train_epochs 1 --max_steps "${HIDE_MAX_STEPS:-2}" \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad \
  --group_by_modality_length True --bf16 True --tf32 True \
  --evaluation_strategy no --save_strategy no \
  --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 \
  --model_max_length 2048 --tokenizer_model_max_length 3072 \
  --gradient_checkpointing True --dataloader_num_workers 2 --lazy_preprocess True \
  --report_to tensorboard --cache_dir "$HF_HOME"
