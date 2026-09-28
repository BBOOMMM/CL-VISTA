#!/usr/bin/env bash
#SBATCH --job-name=hide5090_finish
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --time=04:00:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/%x_%j.out
#SBATCH --error=/cache/hpc_user_alden/%x_%j.err
set -euo pipefail
ROOT=/cache/hpc_user_alden
ENV_DIR="$ROOT/miniforge3/envs/clvista-hide-5090"
WORK="$ROOT/cl-vista-5090-setup"
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
mkdir -p "$WORK"/{tmp,cache,reports,wheels}
export XDG_CACHE_HOME="$WORK/cache" PIP_CACHE_DIR="$WORK/cache/pip"
export TORCH_EXTENSIONS_DIR="$WORK/cache/torch_extensions" HF_HOME="$WORK/cache/huggingface"
export CUDA_HOME=/usr/local/cuda-13.1
export DS_SKIP_CUDA_CHECK=1
export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=12 NVCC_THREADS=1 OMP_NUM_THREADS=4
export FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple PIP_DEFAULT_TIMEOUT=120
exec 9>"$WORK/setup.lock"
flock -n 9
# Reuse interrupted builds and avoid fetching a prebuilt wheel from GitHub.
if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3.post1"'; then
    shopt -s nullglob
    wheels=("$WORK"/wheels/flash_attn-2.8.3.post1-*.whl)
    sources=("$WORK"/tmp/pip-install-*/flash-attn_*/setup.py)
    if ((${#wheels[@]})); then
        python -m pip install --no-index --no-deps "${wheels[0]}"
    elif ((${#sources[@]} == 1)); then
        (
            cd "$(dirname "${sources[0]}")"
            PIP_NO_INDEX=1 python setup.py bdist_wheel > "$WORK/reports/flash-build.log" 2>&1
            cp dist/flash_attn-2.8.3.post1-*.whl "$WORK/wheels/"
        )
        python -m pip install --no-index --no-deps "$WORK"/wheels/flash_attn-2.8.3.post1-*.whl
    else
        echo 'No unique cached FlashAttention source or wheel; prepare one under WORK first.' >&2
        exit 1
    fi
fi
python -m pip freeze > "$WORK/reports/pip-freeze.txt"
export PYTHONPATH="$ROOT/CL-VISTA/Video-LLaVA/HiDe"
python - <<'PY' | tee "$WORK/reports/flash-check.txt"
import torch,flash_attn
from flash_attn import flash_attn_func
print('FlashAttention',flash_attn.__version__)
q=torch.randn(1,128,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=flash_attn_func(q,q,q,causal=True)
y.sum().backward();torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(q.grad).all()
print('FlashAttention forward/backward OK')
PY
python -c 'import videollava.train.train_mem; print("HiDe import OK")' > "$WORK/reports/hide-import.txt" 2>&1
python "$WORK/validate.py" | tee "$WORK/reports/final-validation.txt"
