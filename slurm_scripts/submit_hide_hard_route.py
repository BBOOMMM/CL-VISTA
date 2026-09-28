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
    parser.add_argument('--dependency', help='Slurm dependency, e.g. afterok:JOBID')
    parser.add_argument('--phase', choices=['predict', 'judge'], default='predict')
    parser.add_argument('--scope', choices=['continual', 'final'], default='final')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--judge-python', help='Absolute path to a Qwen3-compatible Python on xz313')
    parser.add_argument('--train-root', default='/cache/hpc_user_alden/outputs/hide-videollava-seven/train-20260926T203107-job29435637')
    parser.add_argument('--output-root', default='/cache/hpc_user_alden/outputs/hide-hard-route-29435637-final')
    parser.add_argument('--test-only', action='store_true', help='Slurm scheduling check only; no job starts')
    args = parser.parse_args()
    if args.phase == 'judge' and not args.judge_python:
        parser.error('--phase judge requires --judge-python (Transformers with Qwen3 MoE support)')
    # Archive only versioned HiDe sources plus the explicit new entry files.
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'Video-LLaVA/HiDe'], cwd=ROOT).decode().split('\0')
    names = [p for p in tracked if p and (ROOT/p).is_file()]
    names += ['Video-LLaVA/HiDe/videollava/model/task_anchors.py']
    names += ['scripts/hide_hard_route/evaluate.py','scripts/hide_hard_route/inference.py','configs/xz313_hide_counting/model.json']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'configs/data_configs/CL-VISTA').glob('*.json')]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=True) as tar:
        for name in sorted(set(names)):
            tar.add(ROOT/name, arcname=name, recursive=False)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    script = '''#!/bin/bash
set -eo pipefail
export HIDE_SNAPSHOT_ROOT=/cache/hpc_user_alden/cl-vista-hard-route/code/job${SLURM_JOB_ID}
mkdir -p "$HIDE_SNAPSHOT_ROOT"
base64 -d > "$HIDE_SNAPSHOT_ROOT/source.tar.gz" <<'HIDE_SOURCE_BASE64'
''' + base64.encodebytes(payload).decode() + "HIDE_SOURCE_BASE64\n"
    script += f'echo "{digest}  $HIDE_SNAPSHOT_ROOT/source.tar.gz" | sha256sum -c -\n'
    script += 'tar -xzf "$HIDE_SNAPSHOT_ROOT/source.tar.gz" -C "$HIDE_SNAPSHOT_ROOT"\n'
    import shlex
    script += 'source /cache/hpc_user_alden/cl-vista-5090-setup/activate.sh\n'
    script += 'export LD_LIBRARY_PATH=/cache/hpc_user_alden/miniforge3/envs/clvista-hide-5090/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}\n'
    script += 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1\n'
    script += 'export PYTHONPATH="$HIDE_SNAPSHOT_ROOT/Video-LLaVA/HiDe" OMP_NUM_THREADS=4\n'
    python = args.judge_python if args.phase=='judge' else '/cache/hpc_user_alden/miniforge3/envs/clvista-hide-5090/bin/python'
    flags = ['--phase',args.phase,'--scope',args.scope,'--limit',str(args.limit),'--train-root',args.train_root,'--output-root',args.output_root]
    script += shlex.quote(python)+' "$HIDE_SNAPSHOT_ROOT/scripts/hide_hard_route/evaluate.py" '+shlex.join(flags)+'\n'
    work = ROOT/'.local_runtime/hard-route-submit'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    work.mkdir(parents=True)
    (work/'job.sbatch').write_text(script)
    (work/'source.sha256').write_text(digest+'\n')
    command = ['sudo', '-n', '-u', 'hpc_user_alden', 'env', 'SLURM_CONF_SERVER=xz01', 'sbatch',
        '--parsable', '--partition=yzgpu', '--nodelist=xz313', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=16', '--mem=128G', '--gres=gpu:RTX5090:4', '--qos=fast',
        '--time=24:00:00',
        '--job-name=alden_hard_route_'+args.phase,
        '--mail-type=BEGIN,END,FAIL', '--mail-user=FT_USER_ALDEN',
        '--chdir=/cache/hpc_user_alden',
        '--output=/cache/hpc_user_alden/%x-%j.log', '--error=/cache/hpc_user_alden/%x-%j.err']
    if args.dependency:
        command.append('--dependency='+args.dependency)
    if args.test_only:
        command.append('--test-only')
    print(f'Source snapshot: {len(payload)} bytes; SHA256 {digest}', flush=True)
    result = subprocess.run(command, input=script, text=True, check=False, capture_output=True)
    (work/'submission.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr, end='', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
