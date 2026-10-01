"""Disjoint immutable scalar composition; never invent an online trajectory seal."""
import json
from pathlib import Path

def scalar_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def compose(full,parts):
    indexed={}
    for part in parts:
        for row in part:
            if row['content'] in indexed:raise ValueError('duplicate partition content')
            indexed[row['content']]=row
    if len(indexed)!=len(full) or set(indexed)!={r['group_id'] for r in full}:raise ValueError('partition coverage mismatch')
    result=[]
    for i,m in enumerate(full):
        r=indexed[m['group_id']]
        if (r['domain'],r['subset'])!=(m['domain'],m['subset']):raise ValueError('partition role mismatch')
        result.append(dict(r,visit=i+1,cycle=1,cycle_visit=i+1))
    return result

def hard_parity(short,full):
    ref={r['content']:r for r in full};maximum=0.
    for row in short:
        r=ref[row['content']]
        if (r['domain'],r['subset'])!=(row['domain'],row['subset']):raise ValueError('baseline role mismatch')
        for a,b in zip(row['metrics'],r['metrics']):
            if a['channel']!=b['channel'] or any(a[k]!=b[k] for k in ('intersection','pred_pixels','gt_pixels','total_pixels')):raise ValueError('baseline mask/count mismatch')
            maximum=max(maximum,abs(a['dice']-b['dice']))
    if maximum>1e-12:raise ValueError('baseline hard Dice mismatch')
    return dict(visits=len(short),max_hard_dice_delta=maximum,tolerance=1e-12)
