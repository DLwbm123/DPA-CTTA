"""Frozen R10 matrix and independent task identities."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SPEC=json.loads((ROOT/'docs/review/r10_use_write_rl/input/R10_SPEC_AND_MATRIX.json').read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
SPEC_SHA=digest(SPEC)
CAPS=dict(gpu_seconds=256*3600,disk_bytes=64*1024**3,model_forwards=12000000,backward_calls=2000000,optimizer_steps=1000000,vjp_calls=0)
PROFILE_CAPS=dict(gpu_seconds=8*3600,disk_bytes=64*1024**3,model_forwards=250000,backward_calls=20000,optimizer_steps=20000,vjp_calls=0)
RECOVERY_POLICY=dict(max_jobs=3,max_extra_attempts_per_job=1,target_score_shared=True)
METHODS=SPEC['training']['methods']


def graph():
    nodes=[dict(id='ASSET_BINDING',kind='bind',needs=[],resource='cpu'),dict(id='FREEZE_PROTOCOL',kind='freeze',needs=['ASSET_BINDING'],resource='cpu')]
    for j in SPEC['source_jobs']:
        nodes.append(dict(id=j['id'],kind='train',job=j,needs=list(dict.fromkeys(j['needs']+['FREEZE_PROTOCOL'])),resource='gpu'))
    for seed in (20260924,20260925):nodes.append(dict(id=f'D0_{seed}',kind='d0',seed=seed,needs=[f'WARM_{seed}'],resource='gpu'))
    discovery=[j['id'] for j in SPEC['source_jobs'] if j['kind']=='post_training' and j['seed'] in (20260924,20260925)]
    nodes.append(dict(id='FAMILY_SELECTION',kind='select',needs=discovery,resource='cpu'))
    nodes.append(dict(id='SOURCE_LOCK',kind='lock',needs=[j['id'] for j in SPEC['source_jobs']]+['FAMILY_SELECTION','D0_20260924','D0_20260925'],resource='cpu'))
    for j in SPEC['target_core_slots']+SPEC['target_final4000_slots_max']:
        nodes.append(dict(id=j['id']+'__online',kind='online',job=j,needs=['SOURCE_LOCK'],resource='gpu'))
        nodes.append(dict(id=j['id']+'__score',kind='score',job=j,needs=[j['id']+'__online'],resource='cpu'))
    ids={n['id'] for n in nodes}
    assert len(ids)==len(nodes) and all(set(n['needs'])<=ids for n in nodes)
    return nodes
