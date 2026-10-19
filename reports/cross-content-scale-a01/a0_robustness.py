"""Posthoc robustness of sealed A0 scores; never reads labels or runs a model."""
import json
from pathlib import Path
import time
import numpy as np


def bootstrap(x):
    x=np.asarray(x,dtype=float)
    rng=np.random.default_rng(20261018)
    idx=rng.integers(0,len(x),(2000,len(x)))
    return np.quantile(x[idx].mean(1),[.025,.975]).tolist()


def analyze(rows):
    assert len(rows)==128 and len({(r['case'],r['condition'],r['arm']) for r in rows})==128
    cube={a:np.array([[[next(r for r in rows if (r['case'],r['condition'],r['arm'])==(i,k,a))['metrics'][ch]['dice_percent'] for ch in range(2)] for k in (0,1,2,3,0)] for i in range(8)]) for a in ('SOURCE','VAR_ONLY','RB_BN')}
    public=[];private=[]
    for name,a,b,ch in [('RB-VAR:OC','RB_BN','VAR_ONLY',1),('RB-SOURCE:OC','RB_BN','SOURCE',1),('RB-SOURCE:ALL','RB_BN','SOURCE',None)]:
        d=cube[a]-cube[b];x=(d if ch is None else d[:,:,ch]).reshape(8,-1).mean(1)
        loo=[dict(omitted_content=i,remaining_mean_pp=float(np.delete(x,i).mean()),ci95_pp=bootstrap(np.delete(x,i))) for i in range(8)]
        pos=float(np.maximum(x,0).sum())
        public.append(dict(comparison=name,full_mean_pp=float(x.mean()),full_ci95_pp=bootstrap(x),content_positive=int((x>0).sum()),content_negative=int((x<0).sum()),content_delta_min_pp=float(x.min()),content_delta_median_pp=float(np.median(x)),content_delta_max_pp=float(x.max()),largest_positive_share_of_positive_sum=float(np.maximum(x,0).max()/pos) if pos else None,loo_mean_min_pp=min(r['remaining_mean_pp'] for r in loo),loo_mean_max_pp=max(r['remaining_mean_pp'] for r in loo),loo_positive_means=sum(r['remaining_mean_pp']>0 for r in loo),loo_ci_low_min_pp=min(r['ci95_pp'][0] for r in loo),loo_ci_low_max_pp=max(r['ci95_pp'][0] for r in loo),loo_ci_excluding_zero_positive=sum(r['ci95_pp'][0]>0 for r in loo)))
        private.append(dict(comparison=name,per_content_mean_pp=x.tolist(),leave_one_content_out=loo))
    return public,private


def main(old,out):
    t=time.time();r=Path(old);p=Path(out)
    assert json.loads((r/'RESOURCE_LEDGER.json').read_text())['all_workers_retired']
    rows=json.loads((r/'scorer/individual.private.json').read_text());public,private=analyze(rows)
    (p/'private/a0_robustness.private.json').write_text(json.dumps(private,indent=2))
    result=dict(status='COMPLETE',posthoc=True,source_a0_sha='21f5ab6a6fcc47f28178ebba32897ab449b217c8',comparisons=public,bootstrap_replicates=2000,bootstrap_seed=20261018,weighting=[0,1,2,3,0],registered_contents=8,no_content_excluded=True,no_label_rgb_weight_reads=True,model_forwards=0,new_gpu_seconds=0,body_seconds=time.time()-t,scope='Posthoc sensitivity, not eight new tests, not updated A0 primary; per-content rows private only. Future support split fixed first4/last4, not based on these outcomes.')
    (p/'A0_ROBUSTNESS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
