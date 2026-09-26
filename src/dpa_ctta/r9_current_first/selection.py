"""Source-only checkpoint/recipe/LR selection. No target path accepted."""
import math
import statistics as st
from ..r8_ba.schedule import CURRICULA
from .protocol import CHECKPOINTS, RECIPES


def balanced(rows, visits=32):
    if (len(rows)!=64 or {r['episode'] for r in rows}!=set(range(64)) or
        any(r.get('visits')!=visits for r in rows)):
        raise ValueError('64 complete source episodes required')
    values=[]
    for mode in CURRICULA:
        scores=[r['hard_Dice'] for r in rows if r['curriculum']==mode]
        if len(scores)!=16 or not all(math.isfinite(x) and 0<=x<=1 for x in scores):raise ValueError('source mode coverage')
        values.append(st.mean(scores))
    return .5*st.mean(values)+.5*min(values)


def checkpoint(rows_by_step):
    if set(rows_by_step)!=set(CHECKPOINTS):raise ValueError('all four fit points required')
    scores={s:balanced(rows) for s,rows in rows_by_step.items()}
    best=max(scores.values());step=min(s for s,v in scores.items() if best-v<=1e-8)
    return dict(step=step,score=scores[step],scores=scores)


def recipes(selected):
    expected={(route,recipe,mode,seed) for route in ('A','B') for recipe in RECIPES
              for mode in ('FULL','STATIC') for seed in (20260924,20260925)}
    if set(selected)!=expected:raise ValueError('all 24 A/B discovery source selections required')
    chosen={};scores={}
    for route in ('A','B'):
        values={recipe:st.mean(selected[route,recipe,mode,seed]['score']
                              for mode in ('FULL','STATIC') for seed in (20260924,20260925)) for recipe in RECIPES}
        if not all(math.isfinite(x) for x in values.values()):raise ValueError('nonfinite selection')
        best=max(values.values());chosen[route]=next(r for r in RECIPES if best-values[r]<=1e-8);scores[route]=values
    return dict(schema='R9_RECIPE_SELECTION_V1',recipes=chosen,scores=scores,mlp='MLP_SELF')


def learning_rate(rows_by_lr,grid):
    if set(rows_by_lr)!=set(grid):raise ValueError('all registered learning rates required')
    scores={lr:balanced(rows,visits=4) for lr,rows in rows_by_lr.items()}
    best=max(scores.values());return dict(lr=min(lr for lr,s in scores.items() if best-s<=1e-8),scores=scores)
