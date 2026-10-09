"""Generated-state prospective HOLD and complete synthetic future scorer checks."""
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .method import PIVOTS, hold_step
from . import report
from ..r30_state_history.method import StateHost
from ..r32_history_origin.online import coupled_step
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import clone, equal


def mechanical():
    torch.set_num_threads(2);torch.manual_seed(17)
    h=StateHost(None,'C_CONT',17,'test','cpu',Tiny())
    x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    for i in range(3): coupled_step(h,x.roll(i,-1))
    state=h.snapshot();saved=clone(state);draws=[];expected=[]
    for i in range(3):
        z,_,pair=coupled_step(h,x.roll(i+3,-1));draws.append(pair);expected.append(z)
    h.restore(state)
    for i,pair in enumerate(draws):
        z,_,_=coupled_step(h,x.roll(i+3,-1),pair);assert torch.equal(z,expected[i])
    h.restore(state)
    for i,pair in enumerate(draws):
        from ..r30_state_history.method import readonly
        reference=readonly(h,x.roll(i+3,-1))
        z=hold_step(h,x.roll(i+3,-1),pair);assert torch.equal(reference,z)
    after=h.snapshot()
    assert h.visits==6 and h.native.steps==3
    for key in ('parameters','gradients','adam','grata','buffers','modes','steps'):
        assert equal(state[key],after[key])
    assert equal(h.native.rng,draws[-1][1]) and equal(state,saved)
    # Resumption is mechanically valid; no post-256 efficacy is claimed.
    coupled_step(h,x);assert h.visits==7 and h.native.steps==4
    h.check_frozen(True);h.close()
    assert len(PIVOTS)==15 and sum(min(256,1951-t) for t in PIVOTS)==3646


def scorer():
    rows=[dict(image_sha256=str(i),domain='D0' if i<1000 else 'D1') for i in range(1,1952)]
    split={str(i):('SEARCH' if 65<=i<=128 else 'CONTEXT' if i%5==0 else 'SEARCH' if i%2==0 else 'SEALED_REVIEW') for i in range(1,1952)}
    jobs=[dict(id=f's{s}_o{o}',seed=s,order=o) for s in (1,2,3) for o in (0,1)]
    observations=6*2*sum(split[str(v)]!='CONTEXT' for t in PIVOTS for v in range(t+1,min(t+256,1951)+1))
    original_open=Path.open;original_stat=Path.stat
    def fake_read(p):
        if p.name=='SPLIT.private.json': return split
        if p.name.startswith('FULL_'): return rows
        if p.parent.name=='processes': return dict(active=False,exit_code=0)
        if p.parent.name=='attempts': return dict(status='COMPLETE')
        if p.name=='COMPLETE.json': return dict(replay_arrivals=1951,reference_checks=1951,hold_future_arrivals=3646,windows=[{}]*15)
        raise AssertionError(p)
    def fake_open(p,*args,**kwargs):
        if p.suffix=='.bits':
            f=io.BytesIO();f.name=str(p);return f
        return original_open(p,*args,**kwargs)
    def fake_stat(p,*args,**kwargs):
        if p.suffix=='.bits':
            n=1951 if p.stem=='native' else min(256,1951-int(p.stem.split('_')[1]))
            return SimpleNamespace(st_size=n*report.WIDTH)
        return original_stat(p,*args,**kwargs)
    def fake_mask(f,offset):
        p=Path(f.name)
        if p.stem=='native': return (.6,.6)
        # Some windows favor hold jointly while hurting OC: oracle must share one action.
        return (.9,.5) if PIVOTS.index(int(p.stem.split('_')[1]))%2==0 else (.4,.4)
    def fake_metrics(values,label): return [dict(dice=x) for x in values]
    class Reader:
        def __init__(self,*args): pass
        def read(self,row): return None
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        root=Path(tmp)
        for d in ('scores','public'): (root/d).mkdir()
        with patch.object(report,'read',fake_read),patch.object(report,'TargetReader',Reader),patch.object(report,'mask_at',fake_mask),patch.object(report,'hard_metrics',fake_metrics),patch.object(Path,'open',fake_open),patch.object(Path,'stat',fake_stat):
            result=report.score(dict(output_root=tmp,target_root=tmp,jobs=jobs,expected_scored_observations=observations),lambda:None)
        p=result['primary']
        assert abs(p['hold_minus_update']-(-50/13))<1e-8
        assert abs(p['oracle_minus_best_fixed']-70/13)<1e-8
        assert p['conditional_headroom_gate'] and not p['fixed_action_gate']
        cells=list(csv.DictReader((root/'public/WINDOW_CELLS.csv').open()))
        q=next(x for x in cells if x['pivot']=='64' and x['channel']=='OC')
        assert q['oracle_action']=='HOLD' and float(q['oracle'])==50
        elig=list(csv.DictReader((root/'public/ELIGIBILITY.csv').open()))
        assert any(x['reason']=='NO_ELIGIBLE_CONTENT' for x in elig)
        assert sum(x['reason']=='TRUNCATED' for x in elig)==24
        assert (root/'public/ALL_NEGATIVE_CELLS.csv').stat().st_size>0
    cases=[]
    for seed in (1,2,3):
        for order in (0,1):
            cases.append(dict(role='SEARCH',horizon=256,seed=seed,order=order,update=60.,hold=61.,oracle=61.))
    _,es=report.summarize(cases)
    assert es[0]['fixed_action_gate'] and not es[0]['conditional_headroom_gate']
    assert es[0]['oracle_minus_best_fixed']==0


if __name__=='__main__':
    mechanical();scorer()
    print('PASS: exact UPDATE replay, HOLD state/clock/RNG, resume validity, donor isolation, full synthetic scorer, joint-action oracle, truncation/missing-role denominators and distinct allocation gates')
