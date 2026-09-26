"""CPU-only sealed-prediction metrics; retain undefined boundaries and soft Dice."""
import math
import statistics as st
from collections import defaultdict
from ..p1_analysis import evaluate
from .validation import dice


def evaluate_probability(probability,label):
    values=evaluate(probability,label,'fundus');_,soft=dice(probability,label)
    for i,m in enumerate(values):
        tp=m['intersection'];fp=m['pred_pixels']-tp;fn=m['gt_pixels']-tp
        m.update(soft_dice=soft[i],FP=fp,FN=fn,precision=tp/(tp+fp) if tp+fp else None,
                 recall=tp/(tp+fn) if tp+fn else None,mean_probability=float(probability[:,i].mean()),
                 predicted_foreground_fraction=m['pred_pixels']/m['total_pixels'])
    return values


def domain_macro(rows,metric='dice'):
    cells=defaultdict(list)
    for r in rows:
        if r['subset']=='remaining_dev':
            for m in r['metrics']:cells[r['domain'],m['channel']].append(m[metric])
    if len(cells)!=8 or any(not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v) for v in cells.values()):
        raise ValueError('complete finite four-domain two-channel metric required')
    return st.mean(st.mean(v) for v in cells.values())


def assd_pair(a,b):
    if len(a)!=len(b):raise ValueError('paired cohort mismatch')
    changes=[x-y for x,y in zip(a,b) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    return dict(common=len(changes),undefined=len(a)-len(changes),delta=st.mean(changes) if changes else None)
