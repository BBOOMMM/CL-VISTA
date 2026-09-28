"""CPU tests for no-fusion math, per-sample reset and cached-token routing."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('variant',Path(__file__).resolve().parents[1]/'inference.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.eye(2));self.bias=None;self.fan_in_fan_out=False
        self.active_adapter='default';self.disable_adapters=False
        self.lora_A={'default':SimpleNamespace(loraA=[nn.Linear(2,2,bias=False) for _ in range(2)])}
        self.lora_B={'default':SimpleNamespace(loraB=[nn.Linear(2,2,bias=False) for _ in range(2)])}
        self.lora_dropout={'default':nn.Identity()};self.scaling={'default':0.5}
        for i in range(2):
            self.lora_A['default'].loraA[i].weight.data.copy_(torch.eye(2)*(i+1))
            self.lora_B['default'].loraB[i].weight.data.copy_(torch.eye(2)*(i+3))
        self.expert_weight=[]

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.adapters=nn.ModuleList([Layer() for _ in range(14)])
        self.model=SimpleNamespace(layers=[None,SimpleNamespace(self_attn=SimpleNamespace(q_proj=self.adapters[-1]))])
        self.expert_num=2;self.next_probs=[0.1,0.9]
    def prepare_inputs_labels_for_multimodal(self,ids,*a,**k):
        if ids.shape[1]>1:self.adapters[-1].expert_weight=self.next_probs
        return ids

class Test(unittest.TestCase):
    def test_route_and_math(self):
        fake=SimpleNamespace(CoINMOELoraLinear=Layer)
        with patch.dict(sys.modules,{'CoIN.peft.tuners.coinmoelora':fake}):m=v.install(Model())
        x=torch.tensor([[2.,3.]])
        with self.assertRaises(RuntimeError):m.adapters[0](x)
        m.prepare_inputs_labels_for_multimodal(torch.ones(1,5,dtype=torch.long))
        self.assertEqual(m.hard_route_state['expert'],1)
        for layer in m.adapters:torch.testing.assert_close(layer(x),x*(1+0.5*2*4))
        # Cached decode must retain the route despite stale/changed anchor scores.
        m.next_probs=[0.9,0.1]
        m.prepare_inputs_labels_for_multimodal(torch.ones(1,1,dtype=torch.long))
        self.assertEqual(m.hard_route_state['expert'],1)
        # A new sample selects expert 0 in every layer, never a sum of experts.
        m.prepare_inputs_labels_for_multimodal(torch.ones(1,5,dtype=torch.long))
        for layer in m.adapters:torch.testing.assert_close(layer(x),x*(1+0.5*1*3))
        with self.assertRaises(ValueError):m.prepare_inputs_labels_for_multimodal(torch.ones(2,5,dtype=torch.long))

if __name__=='__main__':unittest.main()
