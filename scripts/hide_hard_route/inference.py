"""Inference-only, one anchor-selected expert for every decoder layer.

Original HiDe source/checkpoints are untouched. Install instance-local methods
only in this worker process, after the original loader restores all experts.
"""
import json
from pathlib import Path
import sys
from types import MethodType
import torch
import torch.nn.functional as F


def routed_forward(self, x, **kwargs):
    if self.training:
        raise RuntimeError('Hard routing is inference-only')
    route = self._route_state['expert']
    if route is None:
        raise RuntimeError('Decoder reached before routing the current sample')
    weight = self.weight.T if self.fan_in_fan_out else self.weight
    result = F.linear(x, weight, self.bias)
    adapter = self.active_adapter
    if self.disable_adapters or adapter not in self.lora_A:
        raise RuntimeError('Hard routing requires enabled task adapters')
    a = self.lora_A[adapter].loraA[route]
    b = self.lora_B[adapter].loraB[route]
    value = x.to(a.weight.dtype)
    # No parameter sum, expert mixture, or replacement of saved weights.
    result = result + b(a(self.lora_dropout[adapter](value))) * self.scaling[adapter]
    return result.to(x.dtype)


def install(model):
    from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
    state = {'expert': None, 'probabilities': None}
    modules = [m for m in model.modules() if isinstance(m, CoINMOELoraLinear)]
    assert len(modules) == len(model.model.layers)*7, 'Unexpected decoder adapter coverage'
    for module in modules:
        module._route_state = state
        module.forward = MethodType(routed_forward, module)
    original = model.prepare_inputs_labels_for_multimodal

    def prepare(self, input_ids, *args, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('This variant supports exactly one sample per GPU')
        # Original preprocessing computes video/text features and anchor similarities
        # before the decoder. It contains no decoder-layer forward operations.
        if input_ids.shape[1] > 1:
            state.update(expert=None, probabilities=None)
        result = original(input_ids, *args, **kwargs)
        if state['expert'] is None:
            probabilities = self.model.layers[-1].self_attn.q_proj.expert_weight
            assert len(probabilities) == self.expert_num and len(probabilities)>0
            scores = torch.tensor(probabilities)
            assert torch.isfinite(scores).all()
            state.update(expert=int(scores.argmax()), probabilities=list(probabilities))
        return result

    model.prepare_inputs_labels_for_multimodal = MethodType(prepare, model)
    model.hard_route_state = state
    model.eval()
    return model


def main():
    from videollava.eval.video import run_inference_video_qa as official
    original_loader = official.load_pretrained_model
    original_output = official.get_model_output
    args = official.parse_args()
    trace_path = Path(args.output_dir)/(args.output_name+'.routes.jsonl')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open('w') as trace:
        def load(*args, **kwargs):
            tokenizer, model, processor, length = original_loader(*args, **kwargs)
            return tokenizer, install(model), processor, length

        def output(model, video_processor, tokenizer, video, question, args):
            model.hard_route_state.update(expert=None, probabilities=None)
            answer = original_output(model, video_processor, tokenizer, video, question, args)
            trace.write(json.dumps(dict(video=video, question=question,
                **model.hard_route_state))+'\n');trace.flush()
            return answer

        official.load_pretrained_model = load
        official.get_model_output = output
        official.run_inference(args)

if __name__ == '__main__':
    main()
