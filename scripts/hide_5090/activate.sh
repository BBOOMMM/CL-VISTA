#!/usr/bin/env bash
# Source on xz313 after the independent environment has been prepared.
HIDE_NODE_ROOT=/cache/hpc_user_alden
source "$HIDE_NODE_ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$HIDE_NODE_ROOT/miniforge3/envs/clvista-hide-5090"
export CUDA_HOME=/usr/local/cuda-13.1
# DeepSpeed 0.19.7 lacks CUDA 13.x minor-version compatibility entries.
# The 13.1 toolkit / Torch cu130 combination is checked by validate.py.
export DS_SKIP_CUDA_CHECK=1
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$HIDE_NODE_ROOT/cl-vista-5090-setup/tmp" TMP="$HIDE_NODE_ROOT/cl-vista-5090-setup/tmp" TEMP="$HIDE_NODE_ROOT/cl-vista-5090-setup/tmp"
export XDG_CACHE_HOME="$HIDE_NODE_ROOT/cl-vista-5090-setup/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTHONPATH="$HIDE_NODE_ROOT/CL-VISTA/Video-LLaVA/HiDe${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=4 OMP_NUM_THREADS=4
