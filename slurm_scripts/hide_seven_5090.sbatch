#!/usr/bin/env bash
# Executed after the local submitter restores its code snapshot.
set -eo pipefail
source /cache/hpc_user_alden/cl-vista-5090-setup/activate.sh
export HIDE_PROJECT_ROOT="$HIDE_SNAPSHOT_ROOT"
export PYTHONPATH="$HIDE_PROJECT_ROOT/Video-LLaVA/HiDe"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MAX_JOBS=4
cd "$HIDE_PROJECT_ROOT"
if [[ "${HIDE_ANCHOR_SMOKE:-0}" == 1 ]]; then
    exec python scripts/hide_5090/run_anchor_smoke.py
fi
exec python scripts/hide_5090/run_seven.py --mode "$HIDE_RUN_MODE" --execute
