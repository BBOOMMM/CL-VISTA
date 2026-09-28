#!/usr/bin/env bash
#SBATCH --job-name=hide_counting
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --time=24:00:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/cl-vista-counting/%x-%j.log
#SBATCH --error=/cache/hpc_user_alden/cl-vista-counting/%x-%j.err
set -eo pipefail
mode="${1:-smoke}"
[[ "$mode" == smoke || "$mode" == train ]] || { echo 'Usage: sbatch counting_5090.sbatch [smoke|train]'; exit 1; }
source /cache/hpc_user_alden/CL-VISTA/scripts/hide_5090/counting_env.sh
# Validate paths and CPU preprocessing before starting the distributed launcher.
CUDA_VISIBLE_DEVICES='' python "$HIDE_PROJECT_ROOT/scripts/hide_5090/check_counting.py"
exec python "$HIDE_PROJECT_ROOT/scripts/hide_5090/run_counting.py" --mode "$mode" --execute
