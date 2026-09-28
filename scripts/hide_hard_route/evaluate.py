"""Offline HiDe evaluation orchestration; original inference and Qwen scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = ['Counting', 'Traffic', 'Movie', 'GUI', 'Science', 'Sports', 'STAR']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['predict', 'judge'], required=True)
    p.add_argument('--train-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--scope', choices=['continual', 'final'], default='final')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    assert a.limit >= 0
    if a.phase == 'judge':
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('/cache/hpc_user_alden/models/Qwen3-30B-A3B-Instruct-2507', local_files_only=True)
        assert config.model_type == 'qwen3_moe'
    train, out = Path(a.train_root), Path(a.output_root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(train_root=str(train), scope=a.scope, limit=a.limit,
                    tasks=TASKS, variant='hard-route-all-layers-v1', sampling='official temperature=0.1 max_new_tokens=256')
    mf = out/'evaluation.json'
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest, 'Output directory belongs to a different evaluation'
    else:
        mf.write_text(json.dumps(manifest, indent=2))
    pairs = [(i,j) for i in (range(7) if a.scope=='continual' else [6]) for j in range(i+1)]
    base = Path('/cache/hpc_user_alden/datasets/CL-VISTA')
    models = json.loads((ROOT/'configs/xz313_hide_counting/model.json').read_text())
    inference = ROOT/'scripts/hide_hard_route/inference.py'
    judge = ROOT/'Video-LLaVA/HiDe/videollava/eval/video/eval_video_qa_qwen.py'
    judge_model = None
    for i,j in pairs:
        ckpt = train/f'{i:02d}-{TASKS[i]}'
        for f in ['adapter_model.bin','non_lora_trainables.bin','adapter_config.json','config.json','COMPLETE.json']:
            assert (ckpt/f).is_file(), str(ckpt/f)
        # The original loader selects LoRA via the checkpoint directory name.
        view = out/f'videollava-lora-stage{i}'
        if not view.exists(): view.symlink_to(ckpt, target_is_directory=True)
        folder = out/f'stage{i:02d}'/TASKS[j]; folder.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((ROOT/f'configs/data_configs/CL-VISTA/{TASKS[j].lower()}.json').read_text())
        qpath = Path(cfg['test_q_path'].replace('/your_data_path/CL-VISTA',str(base)))
        apath = Path(cfg['test_a_path'].replace('/your_data_path/CL-VISTA',str(base)))
        questions, answers = json.loads(qpath.read_text()), json.loads(apath.read_text())
        ids = [str(q['question_id']) for q in questions]
        amap = {str(x['question_id']): x for x in answers}
        assert len(set(ids))==len(ids) and len(amap)==len(answers) and set(ids)==set(amap)
        if a.limit: questions=questions[:a.limit]
        answers=[amap[str(q['question_id'])] for q in questions]
        expected=[str(q['question_id']) for q in questions]
        merged=folder/'merge.jsonl'
        if a.phase=='predict':
            for q in questions:
                assert any((base/(q['video_name']+ext)).is_file() for ext in ['.mp4','.avi','.mov','.mkv']), q['video_name']
            if merged.exists():
                rows=[json.loads(l) for l in merged.read_text().splitlines()]
                assert [str(r['id']) for r in rows]==expected and all('pred' in r for r in rows)
                print('SKIP_COMPLETE',folder,flush=True);continue
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
            assert len(visible)==4 and all(visible), 'Expected four Slurm GPUs'
            processes=[]; handles=[]; chunks=[]
            try:
                for rank,gpu in enumerate(visible):
                    lo=rank*len(questions)//4;hi=(rank+1)*len(questions)//4
                    if lo==hi:continue
                    qf=folder/f'q{rank}.json';af=folder/f'a{rank}.json'
                    qf.write_text(json.dumps(questions[lo:hi]));af.write_text(json.dumps(answers[lo:hi]))
                    cmd=[sys.executable,str(inference),'--model_path',str(view),'--model_base',models['model_name'],
                         '--text-tower',models['text_tower'],'--num-task',str(i+1),'--video_dir',str(base),
                         '--gt_file_question',str(qf),'--gt_file_answers',str(af),'--output_dir',str(folder),
                         '--output_name',f'part{rank}','--cache_dir',os.environ['HF_HOME']]
                    log=(folder/f'part{rank}.log').open('w');handles.append(log)
                    processes.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT))
                    chunks.append(folder/f'part{rank}.json')
                codes=[p.wait() for p in processes]
                assert all(c==0 for c in codes), f'Inference failed: {codes}; see {folder}'
            finally:
                for proc in processes:
                    if proc.poll() is None:proc.terminate();proc.wait()
                for log in handles:log.close()
            rows=[json.loads(l) for part in chunks for l in part.read_text().splitlines()]
            assert [str(r['id']) for r in rows]==expected and all('pred' in r for r in rows), 'Incomplete prediction coverage'
            tmp=folder/'merge.partial';tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(merged)
        else:
            assert merged.is_file(), f'Run prediction first: {merged}'
            rows=[json.loads(l) for l in merged.read_text().splitlines()]
            assert [str(r['id']) for r in rows]==expected
            if judge_model is None:
                import importlib.util
                from transformers import AutoTokenizer, AutoModelForCausalLM
                spec = importlib.util.spec_from_file_location('official_judge', judge)
                scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
                model_path='/cache/hpc_user_alden/models/Qwen3-30B-A3B-Instruct-2507'
                judge_tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
                judge_tokenizer.padding_side='left'; judge_tokenizer.pad_token=judge_tokenizer.eos_token
                judge_model=AutoModelForCausalLM.from_pretrained(model_path,device_map='auto',dtype='auto',
                    attn_implementation='flash_attention_2',local_files_only=True)
                judge_model.eval()
                cfg=judge_model.generation_config
                cfg.do_sample=False;cfg.temperature=None;cfg.top_p=None;cfg.top_k=None;cfg.num_beams=1
            predictions={str(r['id']):dict(q=r['question'],a=r['answer'],pred=r['pred']) for r in rows}
            target=folder/'qwen3';target.mkdir(exist_ok=True)
            names=[k+'.json' for k in predictions]
            import contextlib
            start=time.monotonic()
            with (folder/'judge.log').open('w') as log, contextlib.redirect_stdout(log):
                for offset in range(0,len(names),8):
                    scorer.process_batch(names[offset:offset+8],predictions,str(target),judge_model,judge_tokenizer)
            elapsed=time.monotonic()-start
            judge_log=(folder/'judge.log').read_text(errors='replace')
            assert '[ERROR]' not in judge_log and "Response missing 'score'" not in judge_log, f'Invalid judge replies: {folder}'
            results={k:json.loads((target/(k+'.json')).read_text()) for k in predictions}
            (folder/'results.json').write_text(json.dumps(results))
            (folder/'judge-timing.json').write_text(json.dumps(dict(seconds=elapsed,count=len(rows))))
        print('COMPLETE',a.phase,i,TASKS[j],flush=True)
    if a.phase=='judge':
        matrix={}
        for i,j in pairs:
            f=out/f'stage{i:02d}'/TASKS[j]/'results.json'
            scores=[v[0]['score'] for v in json.loads(f.read_text()).values()]
            assert scores
            matrix[f'{i}:{TASKS[j]}']=20*sum(scores)/len(scores)
        (out/'scores.json').write_text(json.dumps(matrix,indent=2))
        final_scores = {name: matrix[f'6:{name}'] for name in TASKS}
        (out/'final_metrics.json').write_text(json.dumps({'tasks': final_scores, 'MFN': sum(final_scores.values())/7}, indent=2))
    (out/f'{a.phase}-COMPLETE.json').write_text(json.dumps({'pairs':len(pairs)}))

if __name__=='__main__': main()
