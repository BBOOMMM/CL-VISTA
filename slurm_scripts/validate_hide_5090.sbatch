#!/usr/bin/env bash
#SBATCH --job-name=hide5090_validate
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --time=00:10:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/%x_%j.out
#SBATCH --error=/cache/hpc_user_alden/%x_%j.err
set -eo pipefail
source /cache/hpc_user_alden/cl-vista-5090-setup/activate.sh
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-2}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python /cache/hpc_user_alden/cl-vista-5090-setup/validate.py | tee /cache/hpc_user_alden/cl-vista-5090-setup/reports/final-validation.txt
python -m pip freeze > /cache/hpc_user_alden/cl-vista-5090-setup/reports/pip-freeze.txt
conda list --explicit > /cache/hpc_user_alden/cl-vista-5090-setup/reports/conda-explicit.txt
