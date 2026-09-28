#!/usr/bin/env bash
# Usage: bash slurm_scripts/submit_download.sh [xz288|xz313] [sbatch options...]
set -euo pipefail
node="${1:-xz288}"
if [[ $# -gt 0 ]]; then shift; fi
case "$node" in
    xz288|xz313) partition=yzgpu ;;
    xz323)
        echo 'xz323 is currently reserved for another user; use xz288 or xz313.' >&2
        exit 1 ;;
    *) echo "Unsupported node: $node" >&2; exit 1 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
slurm_user="${SLURM_USER:-hpc_user_alden}"
node_root="${NODE_ROOT:-/cache/$slurm_user}"
data_dir="${DATA_DIR:-$node_root/datasets/CL-VISTA}"
export_args="ALL,NODE_ROOT=$node_root,DATA_DIR=$data_dir"
for name in DOWNLOAD_INCLUDE DOWNLOAD_ENV PYTHON_BIN; do
    if [[ -n "${!name:-}" ]]; then export_args+=",$name=${!name}"; fi
done
echo "[submit] user=$slurm_user node=$node partition=$partition data=$data_dir"
# Feed the script on stdin: the Slurm user need not read the project directory.
# Like alden-training/run.sbatch, NODE_ROOT must already exist on the target node.
sudo -n -u "$slurm_user" env SLURM_CONF_SERVER="${SLURM_CONF_SERVER:-xz01}" \
    sbatch --partition="$partition" --nodelist="$node" \
    --chdir="$node_root" \
    --output="$node_root/%x_%j.out" --error="$node_root/%x_%j.err" \
    --export="$export_args" "$@" < "$script_dir/download_cl_vista.sbatch"
