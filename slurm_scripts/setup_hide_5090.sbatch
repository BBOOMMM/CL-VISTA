#!/usr/bin/env bash
#SBATCH --job-name=hide5090_setup
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
mkdir -p "$WORK"/{tmp,cache,reports}
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache" PIP_CACHE_DIR="$WORK/cache/pip"
export TORCH_EXTENSIONS_DIR="$WORK/cache/torch_extensions"
export HF_HOME="$WORK/cache/huggingface"
export CUDA_HOME=/usr/local/cuda-13.1
export MAX_JOBS=4 OMP_NUM_THREADS=4
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=120
exec 9>"$WORK/setup.lock"
flock -n 9 || { echo 'Setup already running'; exit 1; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
 "$ROOT/miniforge3/bin/conda" create -p "$ENV_DIR" --clone "$ROOT/miniforge3/envs/alden" --offline -y
fi
export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
python - <<'PY'
import torch
assert torch.__version__ == '2.11.0+cu130', torch.__version__
PY
cat > "$WORK/constraints.txt" <<'EOF'
torch==2.11.0+cu130
torchvision==0.26.0+cu130
torchaudio==2.11.0+cu130
numpy==1.26.4
transformers==4.31.0
tokenizers==0.13.3
peft==0.4.0
EOF
python -m pip install --no-deps torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c "$WORK/constraints.txt" \
 'numpy==1.26.4' 'transformers==4.31.0' 'tokenizers==0.13.3' 'peft==0.4.0' \
 'accelerate==1.15.0' 'huggingface-hub<1' 'sentencepiece==0.1.99' \
 decord imageio imageio-ffmpeg 'moviepy<2' 'opencv-python<4.12' \
 pytorchvideo==0.1.5 av pysubs2 shortuuid einops einops-exts timm==1.0.30 \
 tensorboard tensorboardX wandb packaging ninja wheel 'setuptools<81' \
 scipy scikit-learn pandas 'bitsandbytes==0.50.2' requests
DS_BUILD_OPS=0 python -m pip install -c "$WORK/constraints.txt" --no-build-isolation deepspeed==0.19.7
python - <<'PY'
import importlib.util
from pathlib import Path
p = Path(importlib.util.find_spec('pytorchvideo').origin).parent / 'transforms' / 'augmentations.py'
s = p.read_text()
old = 'import torchvision.transforms.functional_tensor as F_t'
if old in s:
    backup = p.with_suffix('.py.before_hide5090')
    if not backup.exists():
        backup.write_text(s)
    p.write_text(s.replace(old, 'import torchvision.transforms.functional as F_t'))
PY
python -m pip freeze > "$WORK/reports/pip-freeze.txt"
python - <<'PY' | tee "$WORK/reports/gpu-check.txt"
import torch,torchvision,torchaudio,transformers,peft,deepspeed,bitsandbytes,decord
print('torch',torch.__version__,'CUDA',torch.version.cuda)
print('GPU',torch.cuda.get_device_name(0),'capability',torch.cuda.get_device_capability(0))
print('architectures',torch.cuda.get_arch_list())
assert torch.cuda.get_device_capability(0)==(12,0)
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=(x@x).float().square().mean();y.backward();torch.cuda.synchronize()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
q=torch.randn(1,4,128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True).sum().backward()
torch.cuda.synchronize()
print('CUDA matmul/backward and SDPA/backward OK')
PY
echo 'BASE_ENV_READY'
