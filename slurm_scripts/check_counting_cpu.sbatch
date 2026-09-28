#!/usr/bin/env bash
#SBATCH --job-name=hide_counting_cpu
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/cl-vista-counting/%x-%j.log
#SBATCH --error=/cache/hpc_user_alden/cl-vista-counting/%x-%j.err
set -eo pipefail
source /cache/hpc_user_alden/CL-VISTA/scripts/hide_5090/counting_env.sh
export CUDA_VISIBLE_DEVICES=''
python "$HIDE_PROJECT_ROOT/scripts/hide_5090/check_counting.py" --load-weights
