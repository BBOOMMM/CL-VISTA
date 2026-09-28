"""CPU checks, including a real two-rank Gloo reduction."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('task_anchors', ROOT / 'Video-LLaVA/HiDe/videollava/model/task_anchors.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        a.initialize_task_anchors(self, slots=2, image_dim=2, text_dim=2)

    def _load_from_state_dict(self, state, prefix, metadata, strict, missing, unexpected, errors):
        a.migrate_legacy_task_anchors(self, state, prefix)
        super()._load_from_state_dict(state, prefix, metadata, strict, missing, unexpected, errors)


def distributed_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        m = Model().bfloat16()
        # Unequal rank batch sizes ensure we sum samples, not means of means.
        x = torch.tensor([[1., 3.]]) if rank == 0 else torch.tensor([[5., 7.], [9., 11.]])
        for _ in range(2):
            a.update_task_anchors(m, 0, x, x * 2)
        torch.testing.assert_close(m.image_anchors[0], torch.tensor([[5., 7.]]))
        torch.testing.assert_close(m.text_anchors[0], torch.tensor([[10., 14.]]))
        assert m.image_boundary[0].item() == 6
        assert m.image_boundary[1].item() == 0
    finally:
        dist.destroy_process_group()


class Tests(unittest.TestCase):
    def test_precision_optimizer_and_counter(self):
        m = Model()
        m.image_anchor_sums[1].fill_(1.000123)
        m.bfloat16().half().float()
        self.assertEqual(m.image_anchor_sums[1].dtype, torch.float32)
        self.assertEqual(m.image_anchor_sums[1][0, 0].item(), torch.tensor(1.000123).item())
        self.assertEqual(m.image_boundary[0].dtype, torch.int64)
        self.assertEqual(list(dict(m.named_parameters())), ['weight'])
        optimizer = torch.optim.AdamW(m.parameters())
        x = torch.tensor([[1., 3.], [3., 5.]], dtype=torch.bfloat16)
        for _ in range(600):
            a.update_task_anchors(m, 0, x, x)
        before = m.image_anchors[0].clone()
        m.weight.sum().backward(); optimizer.step()
        torch.testing.assert_close(before, m.image_anchors[0])
        self.assertEqual(m.image_boundary[0].item(), 1200)
        torch.testing.assert_close(before, torch.tensor([[2., 4.]]))

    def test_checkpoint_and_continuation(self):
        m = Model(); x = torch.tensor([[1., 3.]])
        a.update_task_anchors(m, 0, x, x)
        state = a.task_anchor_state_dict(m)
        self.assertEqual(len(state), 12)
        target = ROOT / '.local_runtime/anchor-tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as d:
            path = Path(d) / 'non_lora_trainables.bin'
            torch.save(state, path)
            n = Model().bfloat16()
            result = n.load_state_dict(torch.load(path, weights_only=True), strict=False)
            self.assertEqual(result.missing_keys, ['weight'])
        for name, value in state.items():
            self.assertTrue(torch.equal(value, dict(n.named_buffers())[name]))
        a.update_task_anchors(n, 0, x * 3, x * 3)
        torch.testing.assert_close(n.image_anchors[0], torch.tensor([[2., 6.]]))
        a.update_task_anchors(n, 1, x * 7, x * 7)
        torch.testing.assert_close(n.image_anchors[1], x * 7)
        torch.testing.assert_close(n.image_anchors[0], torch.tensor([[2., 6.]]))

    def test_legacy_load(self):
        m = Model(); state = m.state_dict()
        state = {k: v for k, v in state.items() if '_sums.' not in k}
        state['image_anchors.0'] = torch.tensor([[2., 4.]], dtype=torch.bfloat16)
        state['image_boundary.0'] = torch.tensor([512.], dtype=torch.bfloat16)
        with self.assertWarnsRegex(UserWarning, 'legacy task anchors'):
            m.load_state_dict(state, strict=True)
        torch.testing.assert_close(m.image_anchor_sums[0], torch.tensor([[1024., 2048.]]))
        self.assertEqual(m.image_boundary[0].dtype, torch.int64)

    def test_distributed(self):
        target = ROOT / '.local_runtime/anchor-tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as d:
            mp.spawn(distributed_worker, args=(str(Path(d) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
