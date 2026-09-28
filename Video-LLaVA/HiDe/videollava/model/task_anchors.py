"""Task prototypes are distributed statistics, never optimizer parameters."""
import warnings

import torch
from torch import nn
import torch.distributed as dist

FAMILIES = ('image_anchors', 'text_anchors', 'image_boundary', 'text_boundary',
            'image_anchor_sums', 'text_anchor_sums')


class StatisticList(nn.Module):
    """Indexable persistent buffers retaining their precision across model.half/to."""
    def __init__(self, count, shape, dtype):
        super().__init__()
        for i in range(count):
            self.register_buffer(str(i), torch.zeros(shape, dtype=dtype))

    def __getitem__(self, index):
        return self._buffers[str(index)]

    def __iter__(self):
        return iter(self._buffers.values())

    def __len__(self):
        return len(self._buffers)

    def _apply(self, fn, recurse=True):
        # Only adopt the destination device. Never round FP32 statistics through
        # the model's BF16/FP16 dtype, even temporarily.
        for name, value in self._buffers.items():
            destination = fn(torch.empty(0, device=value.device, dtype=value.dtype))
            self._buffers[name] = value.to(device=destination.device)
        return self


def initialize_task_anchors(model, slots=10, image_dim=1024, text_dim=768):
    for modality, dim in [('image', image_dim), ('text', text_dim)]:
        setattr(model, modality + '_anchors', StatisticList(slots, (1, dim), torch.float32))
        setattr(model, modality + '_anchor_sums', StatisticList(slots, (1, dim), torch.float32))
        setattr(model, modality + '_boundary', StatisticList(slots, (1,), torch.int64))


@torch.no_grad()
def update_task_anchors(model, task, image_features, text_features):
    # Called once during multimodal preprocessing, outside decoder checkpointing.
    for modality, features in [('image', image_features), ('text', text_features)]:
        total = features.detach().float().sum(dim=0, keepdim=True)
        count = torch.tensor([features.shape[0]], dtype=torch.int64, device=features.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        sums = getattr(model, modality + '_anchor_sums')[task]
        counts = getattr(model, modality + '_boundary')[task]
        sums.add_(total.to(sums.device))
        counts.add_(count.to(counts.device))
        getattr(model, modality + '_anchors')[task].copy_(sums / counts.clamp_min(1))


def migrate_legacy_task_anchors(model, state_dict, prefix):
    """Keep legacy routing values readable; corrupt historical statistics stay corrupt."""
    migrated = False
    for modality in ('image', 'text'):
        for i in range(len(getattr(model, modality + '_anchors'))):
            anchor = f'{prefix}{modality}_anchors.{i}'
            count = f'{prefix}{modality}_boundary.{i}'
            sums = f'{prefix}{modality}_anchor_sums.{i}'
            if anchor in state_dict and count in state_dict and sums not in state_dict:
                state_dict[sums] = state_dict[anchor].float() * state_dict[count].float()
                migrated = True
    if migrated:
        warnings.warn('Loaded legacy task anchors without FP32 sums. Existing anchor values are '
                      'preserved, but historical counter/anchor errors are not repaired; '
                      'recompute prototypes from training data before relying on routing.', UserWarning)


def task_anchor_state_dict(model):
    # Explicitly include buffers omitted by the non-LoRA trainable-parameter filter.
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if any(part in FAMILIES for part in name.split('.')[:-1])}
