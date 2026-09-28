"""Real training entry with per-optimizer-step anchor assertions."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from videollava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
from transformers import TrainerCallback
from videollava.model.task_anchors import task_anchor_state_dict
from videollava.train import train as training


class AnchorAudit(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.before = task_anchor_state_dict(model)
        self.task = int(os.environ['ANCHOR_SMOKE_TASK'])
        self.batch = int(os.environ['ANCHOR_SMOKE_GLOBAL_BATCH'])
        self.reports = []
        assert len(self.before) == 60
        assert not any('anchors.' in n or 'boundary.' in n or 'anchor_sums.' in n
                       for n, _ in model.named_parameters())

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current = task_anchor_state_dict(model)
        changed = []
        for key, value in current.items():
            index = int(key.rsplit('.', 1)[1])
            assert torch.isfinite(value).all(), key
            if 'boundary.' in key:
                assert value.dtype == torch.int64, (key, value.dtype)
                expected = self.before[key].item() + (state.global_step * self.batch if index == self.task else 0)
                assert value.item() == expected, (key, value.item(), expected)
            else:
                assert value.dtype == torch.float32, (key, value.dtype)
            if index != self.task:
                assert torch.equal(value, self.before[key]), ('old/future anchor changed', key)
            elif 'anchors.' in key:
                assert not torch.equal(value, self.before[key]), ('anchor not updated', key)
                sums = current[key.replace('_anchors.', '_anchor_sums.')]
                count = current[key.replace('_anchors.', '_boundary.')]
                torch.testing.assert_close(value, sums / count)
                changed.append(key)
        # Verify identical state on all ranks after the actual DeepSpeed step.
        for key, value in current.items():
            device = next(model.parameters()).device
            local = value.to(device)
            reference = local.clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(local, reference), ('ranks diverged', key)
        self.reports.append({'optimizer_step': state.global_step,
                             'current_task_count': state.global_step * self.batch,
                             'changed_anchors': changed, 'all_ranks_equal': True,
                             'old_and_future_tasks_unchanged': True})
        path = Path(args.output_dir) / f'anchor-audit-rank{dist.get_rank()}.json'
        path.write_text(json.dumps(self.reports, indent=2)+'\n')


original_init = training.LLaVATrainer.__init__
def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.add_callback(AnchorAudit())
training.LLaVATrainer.__init__ = init

training.train()
