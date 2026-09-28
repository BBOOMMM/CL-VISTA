#!/usr/bin/env bash
source /cache/hpc_user_alden/cl-vista-5090-setup/activate.sh
export HIDE_PROJECT_ROOT=/cache/hpc_user_alden/CL-VISTA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-2}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_EXTENSIONS_DIR" /cache/hpc_user_alden/cl-vista-counting/reports
