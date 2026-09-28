#!/usr/bin/env bash
# Source after `conda activate clvista-hide`.
HIDE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export TMPDIR="$HIDE_PROJECT_ROOT/.local_runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$HIDE_PROJECT_ROOT/.local_runtime/huggingface"
export XDG_CACHE_HOME="$HIDE_PROJECT_ROOT/.local_runtime/cache"
export TORCH_EXTENSIONS_DIR="$HIDE_PROJECT_ROOT/.local_runtime/torch_extensions"
export PIP_CACHE_DIR="$HIDE_PROJECT_ROOT/.local_runtime/pip"
export UV_CACHE_DIR="$HIDE_PROJECT_ROOT/.local_runtime/uv"
export CUDA_HOME="${CONDA_PREFIX:?Activate clvista-hide first}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$HIDE_PROJECT_ROOT/Video-LLaVA/HiDe${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=4
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
