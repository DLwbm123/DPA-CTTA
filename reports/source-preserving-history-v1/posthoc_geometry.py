"""Posthoc CPU geometry only; reads existing bit predictions, never images/labels/weights."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

start=time.time();c=json.loads(Path(os.environ['DIAG_CONFIG']).read_text());root=Path(c['output_root'])
rows=json.loads((root/'private/predictions.json').read_text());assert len(rows)==2320
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ret=json.loads((root/'processes/score.json').read_text());assert not ret['active'] and ret['exit_code']==0
opened=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if p.is_relative_to(Path(c['source_root']).resolve()) or p.is_relative_to(root/'scorer') or p.suffix in ('.pt','.pth'):
        raise PermissionError('geometry analysis denies images, labels, weights and per-case scoring')
    if p.is_relative_to(root/'predictions'):opened.append(str(p))
sys.addaudithook(guard)
source={};grouped={};blocks={}
for row in rows:
    if time.time()-start>28:raise TimeoutError('posthoc 30s CPU cap')
    raw=(root/'predictions'/f"{row['index']:04d}.bits").read_bytes();assert len(raw)==65536
    area=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,-1).sum(-1)/262144
    if row['arm']=='SOURCE_EVAL':source[row['case'],row['condition']]=area
    base=source[row['case'],row['condition']]
    grouped.setdefault((row['arm'],row['policy'],row['stage']),[]).append((area,base))
    blocks.setdefault((row['arm'],row['policy'],row['stage'],row['block']),[]).append((area,base))
def summarize(data,block=None):
    out=[]
    for key,values in data.items():
        a,p,s=key[:3];assert len(values)==(80 if block is None else 16)
        v=np.stack([x[0] for x in values]);b=np.stack([x[1] for x in values])
        for ch,name in enumerate(('OD','OC')):
            item=dict(arm=a,policy=p,stage=s,channel=name,readouts=len(values),predicted_area_percent=float(v[:,ch].mean()*100),source_area_percent=float(b[:,ch].mean()*100),area_change_pp=float((v[:,ch]-b[:,ch]).mean()*100))
            if block is not None:item['block']=key[3]
            out.append(item)
    return out
for name,items in [('predicted_area_posthoc',summarize(grouped)),('block_area_posthoc',summarize(blocks,True))]:
    with (root/'public'/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
summary=dict(status='COMPLETE',posthoc=True,scope='Prediction area only, no true-label correctness inference',readouts=2320,prediction_bit_file_reads=len(opened),image_reads=0,label_reads=0,weight_reads=0,model_forwards=0,new_gpu_runs=0,cpu_cap_seconds=30,body_seconds=time.time()-start)
(root/'public/GEOMETRY_POSTHOC.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
