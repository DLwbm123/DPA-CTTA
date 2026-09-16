"""Additional current-host → independent scalar audit integration, procedural CPU only."""
import os,json
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
import torch
torch.set_num_threads(2)
from test_r6_host import new
from test_vptta_host import pixels
from dpa_ctta.r6_regional_consistency import analyze,evaluation
from dpa_ctta.r6_regional_consistency.loss import ARMS
results={}
with patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU only')),patch('dpa_ctta.r1.assets.checkpoint',side_effect=AssertionError('no checkpoint')),patch('dpa_ctta.r1.assets.target',side_effect=AssertionError('no target')):
    for arm in ARMS:
        h=new(arm);traces=[];labels=[];ordered=[]
        for visit in (1,2):
            z,t=h.step(pixels('fundus',visit-1));ms=evaluation.score(h.take_evaluation(),torch.zeros(1,2,512,512))
            entry=dict(sample_id=str(visit),group_id=str(visit),domain='procedural',subset='remaining_dev');ordered.append(entry)
            ident=dict(binding={'fixture':True},visit=visit,**entry)
            traces.append(dict(**ident,trace=t));labels.append(dict(**ident,evaluation=dict(metrics=ms,transaction_before_GT=True,evaluator_seconds=0.,pipeline_seconds=0.)))
        joined=analyze.join(traces,labels,ordered,dict(arm=arm,records=2),{'fixture':True})
        results[arm]=dict(visits=len(joined),physical=h.physical,n_fg=[r['n_fg'] for r in traces[-1]['trace']['channels']])
print(json.dumps(dict(status='PASS',cuda_initialized=torch.cuda.is_initialized(),actual=results),indent=2))
