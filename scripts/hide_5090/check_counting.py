"""Offline CPU checks for Counting; optionally load all three model components."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--load-weights', action='store_true')
options = parser.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES empty'
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    assert os.environ.get(name) == '1', name
network_attempts = []


def blocked_connect(self, address):
    network_attempts.append(str(address))
    raise RuntimeError(f'Network connection forbidden during offline check: {address}')


socket.socket.connect = blocked_connect
import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser
from run_counting import PROJECT, configuration, training_arguments
from videollava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from videollava.model.multimodal_encoder.languagebind import LanguageBindVideoConfig, LanguageBindVideoProcessor
from videollava.model.multimodal_encoder.clip_encoder import CLIPTextTower
from videollava.train.train import ModelArguments, DataArguments, TrainingArguments, LazySupervisedDataset, DataCollatorForSupervisedDataset
from videollava.train.llava_trainer import LLaVATrainer
from videollava import conversation as conversation_lib

work = Path('/cache/hpc_user_alden/cl-vista-counting')
work.mkdir(exist_ok=True)
model_cfg, data_cfg, train_cfg = configuration()
model_path = Path(model_cfg['model_name'])
config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
assert config.mm_image_tower is None
assert config.mm_video_tower == model_cfg['vision_tower']
assert config.mm_text_tower == model_cfg['text_tower']
assert config.mm_text_select_layer == -1
video_config = LanguageBindVideoConfig.from_pretrained(model_cfg['vision_tower'], local_files_only=True)
assert video_config.vision_config.num_frames == 8
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True,
                                          padding_side='right', model_max_length=2048)
tokenizer.pad_token = tokenizer.unk_token
clip_tokenizer = AutoTokenizer.from_pretrained(model_cfg['text_tower'], local_files_only=True)
assert clip_tokenizer('Counting offline check', return_tensors='pt')['input_ids'].numel() > 0
processor = LanguageBindVideoProcessor(video_config)
data_root = Path(data_cfg['train_folder'])
report = {'node': socket.gethostname(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'model': model_cfg, 'train': train_cfg, 'image_tower_disabled': True,
          'frames_per_video': 8, 'annotations': {}, 'samples': [], 'weights_loaded_on_cpu': False}
evaluation_ids = {}
for key in ('train_path', 'test_q_path', 'test_a_path'):
    path = Path(data_cfg[key])
    records = json.loads(path.read_text())
    assert records, path
    refs = set()
    for row in records:
        if key == 'train_path':
            assert 'video' in row
        if key == 'test_q_path':
            # The evaluation loader appends a media suffix to video_name.
            name = row['video_name']
            candidates = [name + suffix for suffix in ('.mp4', '.avi', '.mov', '.mkv')]
            matched = [p for p in candidates if (data_root / p).is_file()]
            assert matched, f'Missing evaluation video: {name}'
            refs.add(matched[0])
        value = row.get('video', [])
        refs.update([value] if isinstance(value, str) else value)
        assert 'image' not in row, 'Counting entry is video-only'
    assert all(p.startswith('Counting/') for p in refs), 'Unexpected cross-task reference'
    assert all((data_root / p).is_file() and (data_root / p).stat().st_size > 0 for p in refs)
    report['annotations'][key] = {'records': len(records), 'unique_videos': len(refs),
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if key == 'train_path':
        train_rows = records
    else:
        ids = [row['question_id'] for row in records]
        assert len(set(ids)) == len(ids), f'Duplicate question IDs: {key}'
        evaluation_ids[key] = set(ids)
assert evaluation_ids['test_q_path'] == evaluation_ids['test_a_path'], 'Test question/answer IDs differ'
print('CONFIG_TOKENIZERS_AND_ALL_COUNTING_PATHS_OK', flush=True)

# Parse both real command lines, replacing only GPU/DeepSpeed options for CPU validation.
for mode in ('smoke', 'train'):
    args = training_arguments(mode, work / f'argument-check-{mode}')
    for key in ('--deepspeed',):
        index = args.index(key)
        del args[index:index + 2]
    for key, value in (('--bf16', 'False'), ('--tf32', 'False'), ('--report_to', 'none')):
        args[args.index(key) + 1] = value
    args += ['--no_cuda', 'True']
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args)
    assert model_args.cur_task == 0 and model_args.image_tower is None
    assert training_args.lora_r == 256 and model_args.expert_num == 8
    assert data_args.data_path == [data_cfg['train_path']]
    LLaVATrainer(model=torch.nn.Linear(2, 2), args=training_args)
print('SMOKE_AND_TRAIN_ARGUMENTS_AND_TRAINER_CPU_CONSTRUCTION_OK', flush=True)

data_args = DataArguments(data_path=[data_cfg['train_path']], video_folder=str(data_root),
                          is_multimodal=True, lazy_preprocess=True, image_aspect_ratio='pad', num_frames=8)
data_args.video_processor = processor
data_args.mm_use_im_start_end = False
conversation_lib.default_conversation = conversation_lib.conv_templates['v1']
class CheckedDataset(LazySupervisedDataset):
    """Fail on the original sample instead of allowing the dataset's random retry."""
    checking = False

    def __getitem__(self, index):
        if self.checking:
            raise RuntimeError('Dataset attempted to replace a failing sample; see original error above')
        self.checking = True
        try:
            return super().__getitem__(index)
        finally:
            self.checking = False


dataset = CheckedDataset(data_args.data_path, tokenizer, data_args)
for index in (0, len(dataset) // 2, len(dataset) - 1):
    # Direct decoding exposes errors that the training dataset's retry handler could hide.
    video_path = data_root / train_rows[index]['video']
    pixels = processor(str(video_path), return_tensors='pt')['pixel_values']
    assert tuple(pixels.shape) == (1, 3, 8, 224, 224), pixels.shape
    assert torch.isfinite(pixels).all()
    sample = dataset[index]
    assert (sample['labels'] != -100).any(), 'Sample has no supervised tokens'
    batch = DataCollatorForSupervisedDataset(tokenizer)([sample])
    assert batch['input_ids'].shape[0] == 1 and len(batch['images']) == 1
    report['samples'].append({'index': index, 'video': str(video_path),
                              'pixel_shape': list(pixels.shape),
                              'supervised_tokens': int((sample['labels'] != -100).sum())})
print('THREE_REAL_COUNTING_SAMPLES_DECODE_TOKENIZE_COLLATE_OK', flush=True)

if options.load_weights:
    print('LOADING_MAIN_MODEL_ON_CPU', flush=True)
    transformers.utils.logging.set_verbosity_error()
    model, info = LlavaLlamaForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, output_loading_info=True)
    allowed_missing = ('image_anchors.', 'text_anchors.', 'image_boundary.', 'text_boundary.')
    assert all(key.startswith(allowed_missing) for key in info['missing_keys']), info['missing_keys']
    assert all(key.startswith(('model.image_tower.', 'model.video_tower.'))
               for key in info['unexpected_keys']), info['unexpected_keys']
    assert not info.get('mismatched_keys') and not info.get('error_msgs'), info
    assert model.get_image_tower() is None
    model_args = ModelArguments(video_tower=model_cfg['vision_tower'], text_tower=model_cfg['text_tower'],
                                mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2)
    print('LOADING_VIDEO_AND_TEXT_TOWERS_ON_CPU', flush=True)
    model.get_model().initialize_vision_modules(model_args)
    model.get_model().initialize_text_modules(model_args)
    assert model.get_video_tower().is_loaded and model.get_text_tower().is_loaded
    assert model.get_video_tower().config.num_frames == 8
    assert not any(p.is_meta or p.device.type != 'cpu' for p in model.parameters())
    assert not any(p.requires_grad for p in model.get_video_tower().parameters())
    assert not any(p.requires_grad for p in model.get_text_tower().parameters())
    report.update(weights_loaded_on_cpu=True, parameters=sum(p.numel() for p in model.parameters()),
                  new_hide_parameters=info['missing_keys'],
                  ignored_source_tower_keys=len(info['unexpected_keys']))
    print('MAIN_MODEL_VIDEO_TOWER_TEXT_TOWER_OFFLINE_CPU_LOAD_OK', flush=True)

assert not network_attempts, network_attempts
report.update(network_attempts=network_attempts, training_started=False, passed=True,
              versions={n: __import__('importlib.metadata', fromlist=['version']).version(n)
                        for n in ('torch', 'transformers', 'accelerate', 'deepspeed')})
output = work / 'reports' / ('offline-cpu-load.json' if options.load_weights else 'preflight.json')
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, indent=2) + '\n')
print(f'COUNTING_OFFLINE_CPU_CHECK_PASSED: {output}', flush=True)
