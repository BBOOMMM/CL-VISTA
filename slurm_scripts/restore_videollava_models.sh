#!/usr/bin/env bash
# Run on xz313 inside a CPU-only Slurm allocation; no model loading or training.
set -euo pipefail
[[ $(hostname -s) == xz313 ]] || { echo 'Run this on xz313'; exit 1; }
ROOT=/cache/hpc_user_alden
VERSION=20260926T060514Z
REMOTE=/user/t0/alden/models/cl-vista-videollava/$VERSION
WORK="$ROOT/model-transfer/$VERSION"
DEST="$ROOT/models"
STAGE="$DEST/.incoming-$VERSION"
ARCHIVE=videollava-hide-models.tar
EXPECTED=aca8896fcb09c8569996a8f3b2a6f0afe7b42fd956c2571e2ed1da18e7736770
mkdir -p "$WORK"/{tmp,hadoop-logs,hadoop-pids} "$DEST"
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export HADOOP_USER_NAME=t0 JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/opt/hadoop HADOOP_LOG_DIR="$WORK/hadoop-logs" HADOOP_PID_DIR="$WORK/hadoop-pids"
export HADOOP_OPTS="${HADOOP_OPTS:-} -Djava.io.tmpdir=$TMPDIR"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$HADOOP_HOME/lib/native:${LD_LIBRARY_PATH:-}"
HDFS="$HADOOP_HOME/bin/hdfs"
exec 9>"$ROOT/.videollava-models.lock"
flock -n 9 || { echo 'Another model preparation process holds the lock'; exit 1; }
for model in Video-LLaVA-7B LanguageBind_Video_merge clip-vit-large-patch14-336; do
    [[ ! -e "$DEST/$model" ]] || { echo "Destination exists, refusing overwrite: $DEST/$model"; exit 1; }
done
available=$(df -B1 --output=avail "$ROOT" | tail -1 | tr -d ' ')
((available > 45000000000)) || { echo 'At least 45 GB free space required'; exit 1; }
echo "[$(date -Is)] HDFS_DOWNLOAD_START $REMOTE"
for file in "$ARCHIVE" archive.sha256 MANIFEST.json SHA256SUMS README.txt UPLOAD_VERIFIED.json; do
    if [[ ! -f "$WORK/$file" ]]; then
        "$HDFS" dfs -get "$REMOTE/$file" "$WORK/$file"
    fi
done
cd "$WORK"
printf '%s  %s\n' "$EXPECTED" "$ARCHIVE" | sha256sum -c -
echo "[$(date -Is)] ARCHIVE_SHA256_OK"
mkdir -p "$STAGE"
tar -xf "$WORK/$ARCHIVE" -C "$STAGE" --no-same-owner --no-same-permissions
cd "$STAGE"
sha256sum -c SHA256SUMS
echo "[$(date -Is)] ALL_MODEL_FILES_SHA256_OK"
for model in Video-LLaVA-7B LanguageBind_Video_merge clip-vit-large-patch14-336; do
    mv -T -- "$STAGE/$model" "$DEST/$model"
done
# Keep manifests next to the downloaded archive and in staging for auditability.
"$ROOT/miniforge3/envs/clvista-hide-5090/bin/python" - "$WORK" "$DEST" "$EXPECTED" <<'PY'
import json, sys
from pathlib import Path
work, dest = map(Path, sys.argv[1:3])
manifest = json.loads((work / 'MANIFEST.json').read_text())
for item in manifest['files']:
    p = dest / item['path']
    assert p.is_file() and p.stat().st_size == item['bytes'], p
result = {'node': 'xz313', 'model_directory': str(dest), 'source': manifest['hdfs_destination'],
          'archive_sha256': sys.argv[3], 'per_file_sha256_verified': True,
          'model_files': len(manifest['files']), 'gpu_used': False, 'external_downloads': False,
          'configuration': 'Upstream config preserved; destination-specific offline configuration not applied',
          'training_started': False}
(work / 'RESTORE_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
PY
echo "[$(date -Is)] THREE_MODELS_RESTORED_AND_VERIFIED"
