"""Create a video-only config view; preserve the uploaded source checkpoint."""
import fcntl
import hashlib
import json
from pathlib import Path
import socket

assert socket.gethostname().split('.')[0] == 'xz313', 'Run on xz313'
root = Path('/cache/hpc_user_alden')
project = root / 'CL-VISTA'
settings = json.loads((project / 'configs/xz313_hide_counting/model.json').read_text())
source = root / 'models/Video-LLaVA-7B'
target = Path(settings['model_name'])

with (root / '.videollava-models.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    original = (source / 'config.json').read_bytes()
    config = json.loads(original)
    config.update(mm_image_tower=None, mm_video_tower=settings['vision_tower'],
                  mm_text_tower=settings['text_tower'], mm_text_select_layer=-1)
    for key in ('vision_tower', 'text_tower'):
        assert (Path(settings[key]) / 'config.json').is_file(), settings[key]
    target.mkdir(exist_ok=True)
    for path in source.iterdir():
        if not path.is_file() or path.name.startswith('config') or path.name == 'generation_config.json':
            continue
        link = target / path.name
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == path.resolve(), link
        else:
            link.symlink_to(path)
    def write_json(name, value):
        temporary = target / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(target / name)
    write_json('config.json', config)
    generation = json.loads((source / 'generation_config.json').read_text())
    if not generation.get('do_sample', False):
        generation.pop('temperature', None)
        generation.pop('top_p', None)
    write_json('generation_config.json', generation)
    provenance = {'source_model': str(source), 'configured_model': str(target),
                  'source_config_sha256': hashlib.sha256(original).hexdigest(),
                  'image_tower_disabled': True, 'weights': 'Symlinks to unchanged source weights'}
    write_json('offline-view.json', provenance)
    assert (source / 'config.json').read_bytes() == original
print(json.dumps(provenance, indent=2))
