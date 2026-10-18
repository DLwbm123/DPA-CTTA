"""Summarize already-scored private arrays; usage: python summarize.py RUN_ROOT."""
import csv,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
rows=json.loads((root/'scorer/SCREEN_o0.json').read_text()); rows=sorted(rows,key=lambda r:r['image_sha256'])
v=np.load(root/'scorer/scores.private.npy');domains=np.array([r['domain'] for r in rows]);cycles=np.array([r['cycle'] for r in rows]);names=sorted(set(domains))
assert v.shape==(192,2,2,2)
def write(name,records):
 with (root/'public'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
armrows=[];cells=[];means=[]
for cycle in (0,1):
 for b,arm in enumerate(('NATIVE','ENGRAM')):
  pick=cycles==cycle
  armrows.append(dict(cycle=cycle,arm=arm,n_contents=int(pick.sum()),macro_Dice_percent=float(np.mean([v[pick&(domains==d),:,b].mean() for d in names]))))
 for d in names:
  pick=(cycles==cycle)&(domains==d)
  for ch,label in enumerate(('OD','OC')):
   means.append(dict(cycle=cycle,domain=d,channel=label,n_contents=int(pick.sum()),native_Dice=float(v[pick,:,0,ch].mean()),engram_Dice=float(v[pick,:,1,ch].mean()),delta_pp=float((v[pick,:,1,ch]-v[pick,:,0,ch]).mean())))
   for order in (0,1):
    cells.append(dict(cycle=cycle,domain=d,order=order,channel=label,n_contents=int(pick.sum()),delta_pp=float((v[pick,order,1,ch]-v[pick,order,0,ch]).mean())))
write('CYCLE_ARM_SCORES.csv',armrows);write('CYCLE_CELLS.csv',cells);write('CYCLE_DOMAIN_CHANNEL.csv',means)
mech=[]
for order in (0,1):
 panel=json.loads((root/f'scorer/SCREEN_o{order}.json').read_text())
 for arm in ('NATIVE','ENGRAM'):
  traces=[json.loads(x) for x in (root/f'private/{arm}_o{order}.jsonl').read_text().splitlines()]
  for cycle in (0,1):
   t=[t for t,r in zip(traces,panel) if r['cycle']==cycle]
   mech.append(dict(arm=arm,order=order,cycle=cycle,n_contents=len(t),mean_recall_norm=float(np.mean([x['recall_norm'] for x in t])),mean_update_norm=float(np.mean([x['update_norm'] for x in t])),mean_key_shift=float(np.mean([x['key_shift'] for x in t])),final_rank=t[-1]['memory_rank'],growth_events=sum(x['grew'] for x in t)))
write('CYCLE_MECHANISM.csv',mech)
print(json.dumps(dict(cycle_arm_scores=armrows,cycle_cells=cells,mechanism=mech),indent=2))
