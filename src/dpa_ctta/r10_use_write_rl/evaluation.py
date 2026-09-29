"""Source-only checkpoint/family selection and paired write consequences."""
import copy
import statistics as st
import torch
from .math import Memory
from .controller import detached
from .learning import rng_state,restore_rng
from ..r9_current_first.validation import dice
from ..r7_shared.numerics import COUNTS


def score(rows):
    groups={m:[r['hard_Dice'] for r in rows if r['mode']==m] for m in set(r['mode'] for r in rows)}
    if len(rows)!=64 or len(groups)!=4 or any(len(v)!=16 for v in groups.values()):raise ValueError('complete balanced validation')
    means=[st.mean(v) for v in groups.values()];return .5*st.mean(means)+.5*min(means)


@torch.no_grad()
def episode(source,actor,seed,index,fold='val',static=False,stochastic=False,guard=lambda:None):
    state=Memory.zero();hard=[];soft=[];writes=[];gains=[]
    _,_,mode=source.schedule(fold,index,seed)
    for v in range(32):
        guard();item=source.item(fold,seed,index,v)
        if static:state=Memory.zero()
        use,state,a=source.controller.act(actor,item['observation'],state,torch.randn(10) if stochastic else None,.5 if static else None)
        logits=source.segmenter(item['image'],source.controller.carrier.basis@use)
        h,s=dice(logits.sigmoid(),item['label']);hard.append(h);soft.append(s)
        writes.append(.5 if static else float(a['raw_action'][9].sigmoid()));gains.append(float(a['raw_action'][0].sigmoid()))
    return dict(episode=index,mode=mode,hard_Dice=st.mean(x for r in hard for x in r),soft_Dice=st.mean(x for r in soft for x in r),hard_OD=st.mean(x[0] for x in hard),hard_OC=st.mean(x[1] for x in hard),write_mean=st.mean(writes),gain_mean=st.mean(gains),state_norm=float(state.m.norm()))


def validate(source,actor,seed,static,guard=lambda:None):
    saved=rng_state();counts=COUNTS.copy()
    try:
        a=copy.deepcopy(actor).eval();a.requires_grad_(False)
        rows=[episode(source,a,seed,i,static=static,guard=guard) for i in range(64)]
        return dict(rows=rows,S=score(rows))
    finally:restore_rng(saved);COUNTS.clear();COUNTS.update(counts)


def choose_checkpoint(points):
    if set(points)!={'500','1000','2000','4000'}:raise ValueError('four completed checkpoints')
    best=max(p['validation']['S'] for p in points.values())
    step=min(int(k) for k,p in points.items() if best-p['validation']['S']<=1e-8)
    return dict(step=step,S=points[str(step)]['validation']['S'],artifact=points[str(step)]['artifact'])


def choose_families(receipts):
    families={'SUP':['SUP_STATIC','SUP_SEQ','SUP_RET'],'GR':['GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA']};selected={};values={}
    for family,methods in families.items():
        values[family]={m:st.mean(receipts[f'FIT_{m}_{s}']['selection']['S'] for s in (20260924,20260925)) for m in methods if all(f'FIT_{m}_{s}' in receipts for s in (20260924,20260925))}
        if not values[family]:selected[family]=None;continue
        best=max(values[family].values());selected[family]=next(m for m in methods if m in values[family] and best-values[family][m]<=1e-8)
    return dict(selected=selected,scores=values)


def deployment_gap(source,actor,seed,static,guard=lambda:None):
    saved=rng_state();counts=COUNTS.copy()
    try:
        # cal's original schedule assigns mode by index modulo four.
        deterministic=[episode(source,actor,seed,i,'cal',static,False,guard) for i in range(16)]
        sampled=[[episode(source,actor,seed,i,'cal',static,True,guard) for i in range(16)] for _ in range(4)]
        return dict(deterministic=deterministic,sampled=sampled)
    finally:restore_rng(saved);COUNTS.clear();COUNTS.update(counts)


@torch.no_grad()
def d0(source,actor,seed,guard=lambda:None):
    saved=rng_state();counts=COUNTS.copy();rows=[]
    try:
        torch.manual_seed(seed)
        for index in range(64):
            t=4+4*(index%6);state=Memory.zero()
            for v in range(t):
                guard();item=source.item('val',seed,index,v)
                _,state,_=source.controller.act(actor,item['observation'],state,forced_write=.5)
            item=source.item('val',seed,index,t);probes=source.probes('val',seed,index,t)
            for candidate in range(4):
                epsilon=torch.randn(10);branches=[];use=None
                for w in (0.,1.):
                    u,m,_=source.controller.act(actor,item['observation'],state,epsilon,w)
                    if use is not None and not torch.equal(use,u):raise ValueError('write changed current use')
                    use=u;branches.append(m)
                current=source.segmenter(item['image'],source.controller.carrier.basis@use)
                future=[];revisit=[]
                for m in branches:
                    hs=[]
                    for v in range(t+1,t+5):
                        guard();n=source.item('val',seed,index,v)
                        u,m,_=source.controller.act(actor,n['observation'],m,forced_write=.5)
                        logits=source.segmenter(n['image'],source.controller.carrier.basis@u)
                        hs.append(dice(logits.sigmoid(),n['label'])[0])
                    future.append([st.mean(x[c] for x in hs) for c in range(2)])
                    ps=[]
                    for n in probes:
                        guard();u,_,_=source.controller.act(actor,n['observation'],m,forced_write=.5)
                        ps.append(dice(source.segmenter(n['image'],source.controller.carrier.basis@u).sigmoid(),n['label'])[0])
                    revisit.append([st.mean(x[c] for x in ps) for c in range(2)])
                rows.append(dict(episode=index,candidate=candidate,current_difference=0.,future=future,revisit=revisit,delta=st.mean(future[1])-st.mean(future[0])))
        return dict(schema='R10_D0_V1',seed=seed,rows=rows,mean_delta=st.mean(r['delta'] for r in rows),positive_fraction=st.mean(r['delta']>0 for r in rows),negative_fraction=st.mean(r['delta']<0 for r in rows),above_point002=st.mean(abs(r['delta'])>.002 for r in rows))
    finally:restore_rng(saved);COUNTS.clear();COUNTS.update(counts)
