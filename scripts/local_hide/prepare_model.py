"""Localize the downloaded Video-LLaVA configuration for offline HiDe training."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
model = root / 'models' / 'Video-LLaVA-7B'
config_path = model / 'config.json'
original = config_path.read_text()
backup = model / 'config.upstream.json'
if not backup.exists():
    backup.write_text(original)
config = json.loads(original)
config.update(
    mm_image_tower=str(root / 'models' / 'LanguageBind_Image'),
    mm_video_tower=str(root / 'models' / 'LanguageBind_Video_merge'),
    mm_text_tower=str(root / 'models' / 'clip-vit-large-patch14-336'),
    mm_text_select_layer=-1,
)
config_path.write_text(json.dumps(config, indent=2) + '\n')
generation_path = model / 'generation_config.json'
if generation_path.exists():
    generation = json.loads(generation_path.read_text())
    for key in ('temperature', 'top_p'):
        generation.pop(key, None)
    generation_path.write_text(json.dumps(generation, indent=2) + '\n')
print('Configured local model:', model)
