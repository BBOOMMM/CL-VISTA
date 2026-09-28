#!/usr/bin/env bash
#SBATCH --job-name=cl_vista_extract
#SBATCH --partition=yzgpu
#SBATCH --nodelist=xz313
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --qos=fast
#SBATCH --chdir=/cache/hpc_user_alden
#SBATCH --output=/cache/hpc_user_alden/cl-vista-extract/extract-%j.log
#SBATCH --error=/cache/hpc_user_alden/cl-vista-extract/extract-%j.err
set -euo pipefail
ROOT=/cache/hpc_user_alden
DATA="$ROOT/datasets/CL-VISTA"
WORK="$ROOT/cl-vista-extract"
mkdir -p "$WORK"/{reports,tmp} "$DATA/.extract-state"
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
# Cooperate with the downloader as well as other extraction jobs.
exec 9>"$DATA/.download.lock"
flock -n 9 || { echo 'Dataset is in use by another preparation job'; exit 1; }
tasks=(STAR Science GUI Movie Traffic Sports Counting)
for task in "${tasks[@]}"; do
    parts=("$DATA/$task/$task.tar.part"*)
    [[ -f "${parts[0]}" ]] || { echo "Missing archive: $task"; exit 1; }
    for i in "${!parts[@]}"; do
        printf -v expected '%s/%s/%s.tar.part%04d' "$DATA" "$task" "$task" "$i"
        [[ "${parts[$i]}" == "$expected" && -s "$expected" ]] || { echo "Invalid parts: $task"; exit 1; }
    done
    signature=$(stat --printf='%n %s %Y\n' "${parts[@]}" | sha256sum | cut -d' ' -f1)
    marker="$DATA/.extract-state/$task.done"
    if [[ -f "$marker" && "$(cat "$marker")" == "$signature" ]]; then
        echo "[$(date -Is)] SKIP completed $task"
        continue
    fi
    bytes=$(stat --printf='%s\n' "${parts[@]}" | awk '{s+=$1} END {printf "%.0f",s}')
    available=$(df -B1 --output=avail "$DATA" | tail -1 | tr -d ' ')
    (( available > bytes + 10737418240 )) || { echo "Insufficient space: $task"; exit 1; }
    echo "[$(date -Is)] START $task archive_bytes=$bytes"
    cat "${parts[@]}" | tar --extract --verbose --file=- --directory="$DATA" \
        --no-same-owner --no-same-permissions --delay-directory-restore \
        --index-file="$WORK/reports/$task.members.txt" \
        --checkpoint=200000 --checkpoint-action=echo
    printf '%s\n' "$signature" > "$marker"
    echo "[$(date -Is)] DONE $task members=$(wc -l < "$WORK/reports/$task.members.txt")"
done
members=()
for task in "${tasks[@]}"; do members+=("json_row_unified/$task"); done
tar -xzf "$DATA/CL-VISTA_json.tar.gz" -C "$DATA" \
    --no-same-owner --no-same-permissions "${members[@]}"
echo "[$(date -Is)] Seven-task unified annotations extracted; Space excluded"
"$ROOT/miniforge3/envs/clvista-hide-5090/bin/python" "$WORK/verify_extracted.py" \
    "$DATA" "$WORK/reports/verification.json" --exclude-task Space --allow-unreferenced-empty
echo "[$(date -Is)] ALL_SEVEN_DATASETS_EXTRACTED_AND_VERIFIED"
