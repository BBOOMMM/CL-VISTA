"""Seven stages, exactly two distributed microbatches/optimizer steps per task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from run_counting import training_arguments

ROOT = Path(__file__).resolve().parents[2]
model = json.loads((ROOT/'configs/xz313_hide_counting/model.json').read_text())
train = json.loads((ROOT/'configs/xz313_hide_seven/train.json').read_text())
tasks = json.loads((ROOT/'configs/xz313_hide_seven/tasks.json').read_text())
assert torch.cuda.device_count() == 4
for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']:
    assert os.environ[key] == '1'
train.update(grad_acc=1, smoke_grad_acc=1, smoke_max_steps=2, global_batch_size=8)
out = Path('/cache/hpc_user_alden/outputs') / ('hide-anchor-smoke-job'+os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
base = Path('/cache/hpc_user_alden/datasets/CL-VISTA')
# Prepare/validate all subsets before any distributed process starts.
for name, rel in tasks:
    records = json.loads((base/rel).read_text())
    subset = [records[i*len(records)//16] for i in range(16)]
    for row in subset:
        assert (base/row['video']).is_file(), row['video']
    (out/(name+'.json')).write_text(json.dumps(subset))
(out/'run.json').write_text(json.dumps({'tasks':tasks,'train':train,'model':model,'source_root':str(ROOT),
    'samples_per_task':16,'microbatches_per_rank':2,'optimizer_steps_per_task':2},indent=2))
previous = None
reports = []
for i,(name,_) in enumerate(tasks):
    stage=out/f'{i:02d}-{name}';stage.mkdir()
    data={'train_path':str(out/(name+'.json')),'train_folder':str(base)}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(ROOT/'scripts/hide_5090/anchor_smoke_train.py')]
    command+=training_arguments('smoke',stage,(model,data,dict(train,cur_task=i)))
    if previous:command+=['--previous_task_model_path',str(previous)]
    (stage/'launch.json').write_text(json.dumps(command,indent=2))
    print('START',name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command,cwd=ROOT/'Video-LLaVA/HiDe',env=dict(os.environ,
            ANCHOR_SMOKE_TASK=str(i),ANCHOR_SMOKE_GLOBAL_BATCH='8'),stdout=log,stderr=subprocess.STDOUT,check=True)
    saved=torch.load(stage/'non_lora_trainables.bin',map_location='cpu',weights_only=False)
    stats={k:v for k,v in saved.items() if any(x in k for x in ['anchors.','boundary.','anchor_sums.'])}
    assert len(stats)==60
    for k,v in stats.items():
        slot=int(k.rsplit('.',1)[1])
        if 'boundary.' in k:assert v.dtype==torch.int64 and v.item()==(16 if slot<=i else 0),(k,v)
        else:assert v.dtype==torch.float32
        if 'anchors.' in k:
            assert bool(v.abs().sum()>0)==(slot<=i),k
        if previous and slot!=i:assert torch.equal(v,old[k]),k
    for rank in range(4):
        audit=json.loads((stage/f'anchor-audit-rank{rank}.json').read_text())
        assert [r['current_task_count'] for r in audit]==[8,16]
    state=json.loads((stage/'trainer_state.json').read_text());assert state['global_step']==2
    torch.save(stats,stage/'anchors.pt')
    (stage/'anchors.json').write_text(json.dumps({k:{'dtype':str(v.dtype),'values':v.tolist()} for k,v in stats.items()}))
    reports.append({'task':name,'steps':2,'count':16,'saved_state_verified':True,'all_ranks_verified':True})
    (out/'summary.json').write_text(json.dumps(reports,indent=2))
    print('PASS',name,'count=16; updated anchors survive optimizer/save; old anchors unchanged',flush=True)
    old=stats;previous=stage
(out/'COMPLETE.json').write_text(json.dumps({'passed':True,'tasks':reports},indent=2))
print('ALL_SEVEN_PASSED',out,flush=True)
