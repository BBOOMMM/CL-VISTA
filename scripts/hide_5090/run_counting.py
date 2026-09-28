"""Print or execute the prepared Counting command; printing is the default."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import sys

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / 'configs/xz313_hide_counting'


def configuration():
    return tuple(json.loads((CONFIG / f'{name}.json').read_text()) for name in ('model', 'data', 'train'))


def training_arguments(mode, output, configs=None):
    model, data, train = configuration() if configs is None else configs
    assert train['global_batch_size'] == train['gpu_num'] * train['batch_size'] * train['grad_acc']
    values = {
        'model_name_or_path': model['model_name'], 'video_tower': model['vision_tower'],
        'text_tower': model['text_tower'], 'data_path': data['train_path'],
        'video_folder': data['train_folder'], 'output_dir': str(output),
        'deepspeed': str(PROJECT / 'Video-LLaVA/HiDe/scripts/zero2_offload.json'),
        'lora_enable': True, 'lora_r': train['rank'], 'lora_alpha': train['lora_alpha'],
        'expert_num': train['expert_num'], 'cur_task': train['cur_task'],
        'learning_rate': train['lr'], 'mm_projector_lr': train['projector_lr'],
        'version': 'v1', 'num_train_epochs': train['epoch'],
        'max_steps': train['smoke_max_steps'] if mode == 'smoke' else -1,
        'per_device_train_batch_size': train['batch_size'], 'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': train['smoke_grad_acc'] if mode == 'smoke' else train['grad_acc'],
        'mm_projector_type': 'mlp2x_gelu', 'mm_vision_select_layer': -2,
        'mm_use_im_start_end': False, 'mm_use_im_patch_token': False, 'image_aspect_ratio': 'pad',
        'group_by_modality_length': True, 'bf16': True, 'tf32': True,
        'evaluation_strategy': 'no', 'save_strategy': 'no' if mode == 'smoke' else 'steps',
        'save_steps': 50000, 'weight_decay': 0, 'warmup_ratio': 0.03,
        'lr_scheduler_type': 'cosine', 'logging_steps': 1, 'model_max_length': 2048,
        'tokenizer_model_max_length': 3072, 'gradient_checkpointing': True,
        'dataloader_num_workers': 2, 'lazy_preprocess': True, 'report_to': 'tensorboard',
        'cache_dir': os.environ.get('HF_HOME', '/cache/hpc_user_alden/cl-vista-5090-setup/cache/huggingface')}
    return [part for key, value in values.items() for part in ('--' + key, str(value))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'train'), default='smoke')
    parser.add_argument('--execute', action='store_true', help='Requires an existing GPU Slurm allocation')
    args = parser.parse_args()
    model, data, train = configuration()
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    output = Path(train['output_root']) / f'{args.mode}-{stamp}-job{os.environ.get("SLURM_JOB_ID", "preview")}'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               f'--nproc_per_node={train["gpu_num"]}',
               str(PROJECT / 'Video-LLaVA/HiDe/videollava/train/train_mem.py')]
    command += training_arguments(args.mode, output)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS'):
        raise SystemExit('Use counting_5090.sbatch with a GPU allocation; no implicit GPU submission')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        assert os.environ.get(name) == '1', f'Source counting_env.sh first: {name}'
    import torch
    if torch.cuda.device_count() != train['gpu_num']:
        raise SystemExit(f'Expected {train["gpu_num"]} Slurm-visible GPUs, got {torch.cuda.device_count()}')
    assert (Path(model['model_name']) / 'config.json').is_file()
    assert Path(data['train_path']).is_file()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch.json').write_text(json.dumps({'command': command, 'model': model,
        'data': data, 'train': train, 'mode': args.mode}, indent=2) + '\n')
    os.chdir(PROJECT / 'Video-LLaVA/HiDe')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
