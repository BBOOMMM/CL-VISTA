"""Sequential HiDe training without Space; preview by default, no evaluation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from run_counting import training_arguments

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'smoke'], default='train')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    model = json.loads((ROOT/'configs/xz313_hide_counting/model.json').read_text())
    train = json.loads((ROOT/'configs/xz313_hide_seven/train.json').read_text())
    tasks = json.loads((ROOT/'configs/xz313_hide_seven/tasks.json').read_text())
    data_root = Path('/cache/hpc_user_alden/datasets/CL-VISTA')
    tag = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    output = Path(train['output_root']) / f'{args.mode}-{tag}-job{os.environ.get("SLURM_JOB_ID", "preview")}'
    if args.execute:
        assert os.environ.get('SLURM_JOB_GPUS'), 'Must run inside a Slurm GPU allocation'
        for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']:
            assert os.environ.get(name) == '1', name
        import torch
        assert torch.cuda.device_count() == train['gpu_num']
        for key in ['model_name', 'vision_tower', 'text_tower']:
            assert (Path(model[key])/'config.json').is_file(), model[key]
        # Check all required paths before starting any distributed process.
        counts = {}
        for name, rel in tasks:
            records = json.loads((data_root/rel).read_text())
            assert records, name
            for row in records:
                video = row.get('video')
                assert isinstance(video, str) and not video.startswith('Space/'), row.get('id')
                path = data_root/video
                assert path.is_file() and path.stat().st_size > 0, str(path)
            counts[name] = len(records)
        output.mkdir(parents=True, exist_ok=False)
        (output/'run.json').write_text(json.dumps(dict(mode=args.mode, tasks=tasks, train=train,
            model=model, counts=counts, source_root=str(ROOT), evaluation=False), indent=2)+'\n')
    previous = None
    for index, (name, rel) in enumerate(tasks):
        stage = output/f'{index:02d}-{name}'
        data = {'train_path': str(data_root/rel), 'train_folder': str(data_root)}
        if args.mode == 'smoke':
            subset = output/f'{index:02d}-{name}-128.json'
            if args.execute:
                records = json.loads((data_root/rel).read_text())
                subset.write_text(json.dumps([records[i*len(records)//128] for i in range(128)]))
            data['train_path'] = str(subset)
        settings = dict(train, cur_task=index)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            f'--nproc_per_node={train["gpu_num"]}', str(ROOT/'Video-LLaVA/HiDe/videollava/train/train_mem.py')]
        command += training_arguments(args.mode, stage, (model, data, settings))
        if previous:
            command += ['--previous_task_model_path', str(previous)]
        print(f'STAGE {index} {name}: {shlex.join(command)}', flush=True)
        if args.execute:
            stage.mkdir()
            (stage/'launch.json').write_text(json.dumps(dict(command=command, task=name,
                cur_task=index, previous=str(previous) if previous else None), indent=2)+'\n')
            # Preserve full stage logs and stop the chain on any failure.
            with (output/f'{index:02d}-{name}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT/'Video-LLaVA/HiDe', stdout=log,
                    stderr=subprocess.STDOUT, check=True)
            for filename in ['adapter_model.bin', 'non_lora_trainables.bin', 'trainer_state.json']:
                assert (stage/filename).is_file() and (stage/filename).stat().st_size > 0, filename
            state = json.loads((stage/'trainer_state.json').read_text())
            assert state['global_step'] > 0
            (stage/'COMPLETE.json').write_text(json.dumps({'global_step': state['global_step'],
                'history': state['log_history']}, indent=2)+'\n')
            print(f'STAGE_COMPLETE {name} steps={state["global_step"]}', flush=True)
        previous = stage
    if args.execute:
        (output/'COMPLETE.json').write_text(json.dumps({'tasks': [t[0] for t in tasks], 'evaluation': False})+'\n')
        print(f'TRAINING_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
