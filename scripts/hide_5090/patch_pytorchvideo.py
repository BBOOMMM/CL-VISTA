"""Replace the removed torchvision import in PyTorchVideo 0.1.5."""
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('pytorchvideo')
path = Path(spec.origin).parent / 'transforms' / 'augmentations.py'
old = 'import torchvision.transforms.functional_tensor as F_t'
new = 'import torchvision.transforms.functional as F_t'
source = path.read_text()
if old in source:
    backup = path.with_suffix('.py.before_hide5090')
    if not backup.exists():
        backup.write_text(source)
    path.write_text(source.replace(old, new))
elif new not in source:
    raise RuntimeError(f'Unexpected PyTorchVideo source: {path}')
print('PyTorchVideo compatibility import ready:', path)
