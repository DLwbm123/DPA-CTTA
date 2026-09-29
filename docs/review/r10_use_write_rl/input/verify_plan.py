"""Pure planning checks. No model/data/GPU execution and no runtime admission."""
from pathlib import Path
from collections import Counter
import json

root=Path(__file__).resolve().parent
s=json.loads((root/'R10_SPEC_AND_MATRIX.json').read_text())
source=s['source_jobs']; core=s['target_core_slots']; final=s['target_final4000_slots_max']
assert len(source)==25 and len({x['id'] for x in source})==25
assert sum(x['kind']=='warmup' for x in source)==5
assert sum(x['kind']=='post_training' for x in source)==20
assert len(core)==340 and len(final)==70
assert len({x['id'] for x in core+final})==410
stage=Counter(x['stage'] for x in core)
assert stage=={'DISCOVERY':28,'EXTENSION':12,'CONFIRMATION':30,'ABLATION':75,'BASELINE':125,'STRESS':70}
assert sum(x['arrivals'] for x in core)==1277905
assert sum(x['arrivals'] for x in core+final)==1414475
assert sum(x['principal_visits'] for x in core)==1110225
assert sum(x['principal_visits'] for x in core+final)==1228875
assert s['policy']['input_dim']==32+64+64+32+1
assert s['resources']['max_gpu_workers']==1
assert s['resources']['recovery']['max_jobs']==3
assert s['carrier']['job']=='B_FULL_20260924'
assert not s['carrier']['depends_on_R9_completion']
assert s['training']['methods']==['SUP_STATIC','SUP_SEQ','SUP_RET','GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA']
assert set(x['seed'] for x in source)==set(range(20260924,20260929))
assert sum(x.get('updates',0) for x in source)==10000
assert sum(x.get('collection_rounds',0)*x.get('optimizer_epochs',0) for x in source)==160000
# Pinned native-host call contracts. These are nominal counts, not measured timings.
physical={'N_SOURCE_EVAL':(1,0,0),'C0':(1,0,0),'VPTTA_NATIVE':(2,1,1),'C_CTTA':(8,1,1),'G_CTTA':(9,2,1)}
ops=Counter()
for row in core+final:
    f,b,o=physical.get(row['arm'],(2,0,0))
    ops.update(forwards=f*row['arrivals'],backward_calls=b*row['arrivals'],optimizer_steps=o*row['arrivals'])
# Only a partial deterministic lower bound. Source validation/profile/recovery must
# still be measured and expanded. No tautological or fabricated admission check.
assert ops['forwards'] < s['resources']['model_forwards']
assert ops['backward_calls'] + 170000 < s['resources']['backward_calls']
assert ops['optimizer_steps'] + 170000 < s['resources']['optimizer_steps']
result=dict(status='PASS_PLANNING_AND_NOMINAL_TARGET_COUNT_ONLY',stage_counts=dict(stage),
            counts=s['counts'],nominal_target_calls_no_alias=dict(ops),
            source_budget_warning='Profile must use the planned verified zero-observation cache and actual kernels; no admission is asserted.',
            real_profile=False,remote_execution=False)
(root/'PLAN_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
