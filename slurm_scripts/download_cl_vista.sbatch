#!/usr/bin/env bash
#SBATCH --job-name=cl_vista_download
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz288
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/%x_%j.out
#SBATCH --error=/cache/hpc_user_alden/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=FT_USER_ALDEN

# Download only: GPUs are unnecessary. Training should request gpu:RTX5090:4.
# Self-contained because the submission host's project directory may not exist here.
set -euo pipefail
umask 022
trap 'echo "[ERROR] line=$LINENO node=$(hostname) time=$(date -Is)" >&2' ERR

NODE_ROOT="${NODE_ROOT:-/cache/$(id -un)}"
DATA_DIR="${DATA_DIR:-$NODE_ROOT/datasets/CL-VISTA}"
DOWNLOAD_ENV="${DOWNLOAD_ENV:-$NODE_ROOT/envs/cl-vista-download}"
DOWNLOAD_INCLUDE="${DOWNLOAD_INCLUDE:-}"
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_DEFAULT_TIMEOUT=120

# Keep all application caches and temporary downloads in the user's node directory.
# Do not change HOME or modify the existing training environment.
export XDG_CACHE_HOME="$NODE_ROOT/cl-vista-cache"
export XDG_CONFIG_HOME="$NODE_ROOT/cl-vista-config"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MODELSCOPE_CACHE="$XDG_CACHE_HOME/modelscope"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda/pkgs"
export CONDA_ENVS_PATH="$NODE_ROOT/envs"
export PYTHONUSERBASE="$NODE_ROOT/cl-vista-python-user"
export PYTHONNOUSERSITE=1
export TMPDIR="$NODE_ROOT/cl-vista-tmp/${SLURM_JOB_ID:-manual_$$}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$DATA_DIR" "$(dirname "$DOWNLOAD_ENV")" "$TMPDIR" \
    "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE" "$HF_HOME" "$CONDA_PKGS_DIRS" "$XDG_CONFIG_HOME"
cd "$NODE_ROOT"
exec 9>"$DATA_DIR/.download.lock"
flock -n 9 || { echo 'Another download is using this directory.' >&2; exit 1; }
echo "[start] job=${SLURM_JOB_ID:-manual} node=$(hostname) user=$(id -un)"
echo "[dataset] $DATA_DIR"
echo "[include] ${DOWNLOAD_INCLUDE:-entire repository}"
echo '[space] Full download is approximately 749 GB; extraction needs additional space.'
df -h "$DATA_DIR"

# Serialize setup even when jobs download to different dataset directories.
exec 8>"$NODE_ROOT/.cl-vista-env.lock"
flock -x 8
if [[ ! -x "$DOWNLOAD_ENV/bin/python" ]]; then
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        bootstrap_python="$PYTHON_BIN"
    elif [[ -x "$NODE_ROOT/miniforge3/bin/python" ]]; then
        bootstrap_python="$NODE_ROOT/miniforge3/bin/python"
    else
        # Same HDFS installer as alden-training/run.sbatch, with a dedicated prefix.
        bootstrap_dir="$NODE_ROOT/cl-vista-miniforge3"
        if [[ ! -x "$bootstrap_dir/bin/python" ]]; then
            export HADOOP_USER_NAME="${HADOOP_USER_NAME:-t0}"
            export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
            export HADOOP_HOME=/opt/hadoop
            export HADOOP_LOG_DIR="$NODE_ROOT/cl-vista-cache/hadoop/logs"
            export HADOOP_PID_DIR="$TMPDIR"
            export HADOOP_OPTS="${HADOOP_OPTS:-} -Djava.io.tmpdir=$TMPDIR"
            export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:$HADOOP_HOME/lib/native:${LD_LIBRARY_PATH:-}"
            mkdir -p "$HADOOP_LOG_DIR"
            installer="$TMPDIR/Miniforge3-Linux-x86_64.sh"
            "$HADOOP_HOME/bin/hdfs" dfs -get \
                /user/t0/tony/Miniforge3-Linux-x86_64.sh "$installer"
            bash "$installer" -b -p "$bootstrap_dir"
            rm -f "$installer"
        fi
        bootstrap_python="$bootstrap_dir/bin/python"
    fi
    "$bootstrap_python" -m venv "$DOWNLOAD_ENV"
fi
"$DOWNLOAD_ENV/bin/python" -m pip install --upgrade modelscope
"$DOWNLOAD_ENV/bin/python" -m pip show modelscope
flock -u 8

download_args=(download --dataset MLLM-CL/CL-VISTA --local_dir "$DATA_DIR")
if [[ -n "$DOWNLOAD_INCLUDE" ]]; then
    download_args+=(--include "$DOWNLOAD_INCLUDE")
fi

# Preserve SDK metadata and partial files so reruns can reuse downloads.
for attempt in 1 2 3; do
    echo "[download] attempt=$attempt time=$(date -Is)"
    if "$DOWNLOAD_ENV/bin/modelscope" "${download_args[@]}"; then
        echo "[success] node=$(hostname) data=$DATA_DIR time=$(date -Is)"
        du -sh "$DATA_DIR"
        echo 'Archives are downloaded, not extracted. Space videos require ScanNet access separately.'
        exit 0
    fi
    if [[ "$attempt" -lt 3 ]]; then
        echo '[retry] Waiting 30 seconds before retrying.' >&2
        sleep 30
    fi
done
echo '[failed] Resubmit on the SAME NODE with the SAME DATA_DIR to resume.' >&2
exit 1
