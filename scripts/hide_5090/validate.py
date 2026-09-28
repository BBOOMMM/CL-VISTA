"""Dependency and RTX5090 kernel checks; does not load model weights or train."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import sqlite3, ssl, bz2, lzma, ctypes
import torch

names = ['torch','torchvision','torchaudio','transformers','tokenizers','peft',
         'accelerate','deepspeed','bitsandbytes','decord','pytorchvideo','flash-attn']
print(json.dumps({n: importlib.metadata.version(n) for n in names},indent=2))
assert torch.version.cuda == '13.0'
assert torch.cuda.get_device_capability(0) == (12, 0)
assert 'sm_120' in torch.cuda.get_arch_list()
print('GPU:',torch.cuda.get_device_name(0))
print('Python standard library imports OK')
for module in ['torchvision','torchaudio','deepspeed','peft','pytorchvideo.transforms',
               'CoIN.peft','videollava.train.train_mem']:
    importlib.import_module(module)
    print('Import OK:',module)
from flash_attn import flash_attn_func
for dtype in (torch.float16,torch.bfloat16):
    q = torch.randn(1,128,4,128,device='cuda',dtype=dtype,requires_grad=True)
    out = flash_attn_func(q,q,q,causal=True)
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
    print('FlashAttention forward/backward OK:',dtype)

# Exercise the actual HiDe attention adapter, including FlashAttention's unpad API.
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
config = LlamaConfig(hidden_size=256, intermediate_size=512, num_attention_heads=2,
                     num_key_value_heads=2, num_hidden_layers=1)
for padded in (False, True):
    attention = LlamaAttention(config).cuda().to(torch.bfloat16)
    x = torch.randn(2, 16, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(2, 16, device='cuda', dtype=torch.bool) if padded else None
    if padded:
        mask[0, -4:] = False
    y = attention(x, attention_mask=mask,
                  position_ids=torch.arange(16, device='cuda').expand(2, -1))[0]
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    print('HiDe attention forward/backward OK, padded:', padded)

import bitsandbytes.functional as bnb
x = torch.randn(256, 256, device='cuda', dtype=torch.float16)
packed, state = bnb.quantize_4bit(x, quant_type='nf4')
assert torch.isfinite(bnb.dequantize_4bit(packed, state)).all()
print('bitsandbytes CUDA NF4 OK')

from deepspeed.ops.adam import DeepSpeedCPUAdam
p = torch.nn.Parameter(torch.ones(8))
optimizer = DeepSpeedCPUAdam([p], lr=0.01)
p.grad = torch.ones_like(p)
optimizer.step()
assert torch.isfinite(p).all() and (p < 1).all()
print('DeepSpeed CPUAdam extension OK')

# Generate a tiny local clip so codec checks need neither data nor downloads.
import av
import numpy as np
from decord import VideoReader
with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
    video = Path(tmp) / 'codec-check.mp4'
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('mpeg4', rate=4)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 50, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader = VideoReader(str(video))
    # HiDe selects Decord's torch bridge during import; both bridges expose shape.
    assert tuple(reader.get_batch([0, 3]).shape) == (2, 32, 32, 3)
print('PyAV encode / Decord decode OK')
print('ENVIRONMENT_VALIDATED_NO_TRAINING')
