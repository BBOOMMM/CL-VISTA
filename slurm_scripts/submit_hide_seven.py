"""Submit a source snapshot from xz335 to xz313; never downloads data or models."""
import argparse
import base64
import datetime
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'smoke'], default='train')
    parser.add_argument('--anchor-smoke', action='store_true', help='Two microbatches/task with anchor assertions')
    parser.add_argument('--test-only', action='store_true', help='Slurm scheduling check only; no job starts')
    args = parser.parse_args()
    if args.anchor_smoke:
        args.mode = "smoke"
    # Archive only versioned HiDe sources plus the explicit new entry files.
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'Video-LLaVA/HiDe'], cwd=ROOT).decode().split('\0')
    names = [p for p in tracked if p and (ROOT/p).is_file()]
    names += ['Video-LLaVA/HiDe/videollava/model/task_anchors.py']
    names += ['scripts/hide_5090/run_anchor_smoke.py', 'scripts/hide_5090/anchor_smoke_train.py']
    names += ['scripts/hide_5090/run_counting.py', 'scripts/hide_5090/run_seven.py',
              'configs/xz313_hide_counting/model.json', 'configs/xz313_hide_seven/train.json',
              'configs/xz313_hide_seven/tasks.json', 'slurm_scripts/hide_seven_5090.sbatch']
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=True) as tar:
        for name in sorted(set(names)):
            tar.add(ROOT/name, arcname=name, recursive=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    script = '''#!/bin/bash
set -eo pipefail
export HIDE_SNAPSHOT_ROOT=/cache/hpc_user_alden/cl-vista-seven/code/job${SLURM_JOB_ID}
mkdir -p "$HIDE_SNAPSHOT_ROOT"
base64 -d > "$HIDE_SNAPSHOT_ROOT/source.tar.gz" <<'HIDE_SOURCE_BASE64'
''' + base64.encodebytes(payload).decode() + "HIDE_SOURCE_BASE64\n"
    script += f'echo "{digest}  $HIDE_SNAPSHOT_ROOT/source.tar.gz" | sha256sum -c -\n'
    script += 'tar -xzf "$HIDE_SNAPSHOT_ROOT/source.tar.gz" -C "$HIDE_SNAPSHOT_ROOT"\n'
    script += f'export HIDE_RUN_MODE={args.mode}\n'
    script += f'export HIDE_ANCHOR_SMOKE={int(args.anchor_smoke)}\n'
    script += 'exec bash "$HIDE_SNAPSHOT_ROOT/slurm_scripts/hide_seven_5090.sbatch"\n'
    work = ROOT/'.local_runtime/seven-submit'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    work.mkdir(parents=True)
    (work/'job.sbatch').write_text(script)
    (work/'source.sha256').write_text(digest+'\n')
    command = ['sudo', '-n', '-u', 'hpc_user_alden', 'env', 'SLURM_CONF_SERVER=xz01', 'sbatch',
        '--parsable', '--partition=yzgpu', '--nodelist=xz313', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=16', '--mem=128G', '--gres=gpu:RTX5090:4', '--qos=fast',
        '--time='+('24:00:00' if args.mode == 'train' else '01:00:00'),
        '--job-name='+('alden_train' if args.mode == 'train' else 'hide_seven_smoke'),
        '--chdir=/cache/hpc_user_alden',
        '--output=/cache/hpc_user_alden/%x-%j.log', '--error=/cache/hpc_user_alden/%x-%j.err']
    if not args.anchor_smoke:
        command += ['--mail-type=BEGIN,END,FAIL', '--mail-user=FT_USER_ALDEN']
    if args.test_only:
        command.append('--test-only')
    print(f'Source snapshot: {len(payload)} bytes; SHA256 {digest}', flush=True)
    result = subprocess.run(command, input=script, text=True, check=True, capture_output=True)
    (work/'submission.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr, end='')


if __name__ == '__main__':
    main()
