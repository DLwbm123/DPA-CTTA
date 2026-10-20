"""Post-hoc closure of the frozen 48 windows; no labels, model or GPU."""
import csv,json,time
from pathlib import Path
START=time.time()
p=Path(__file__).parent
rows=list(csv.DictReader((p/'ALL_PROBES.csv').open()))
assert len(rows)==48
anchors=[]
for seed in (20260907,17011,29009):
 for order in (0,1):
  a=[r for r in rows if int(r['seed'])==seed and int(r['order'])==order];assert len(a)==8
  v=[float(r['R_K_raw_pp']) for r in a];z=[r['trigger']=='True' for r in a]
  rule=sum(-x*zz for x,zz in zip(v,z))/8;random=sum(z)/8*(-sum(v)/8)
  anchors.append(dict(seed=seed,order=order,oracle_local_pp=sum(max(-x,0) for x in v)/8,
                      G_rule_pp=rule,G_random_pp=random,G_identify_pp=rule-random))
gate=json.loads((p/'M1_GATE.json').read_text())
for name in ('G_rule_pp','G_identify_pp'):
 assert abs(sum(a[name] for a in anchors)/6-gate['gains'][name]['mean_pp'])<1e-14
out=dict(status='COMPLETE',analysis='POST_HOC_DESCRIPTION_NOT_REGISTERED_GATE',anchors=anchors,
 oracle_local_pp=sum(a['oracle_local_pp'] for a in anchors)/6,
 G_random_pp=sum(a['G_random_pp'] for a in anchors)/6,
 scope='Only original VV histories and these 48 16-arrival windows; not a full-policy upper bound',
 original_M1_gate_unchanged=True,new_model_calls=0,new_GPU_updates=0,new_image_reads=0,new_label_reads=0,
 CPU_script_wall_seconds=time.time()-START)
(p/'CPU_CLOSURE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
