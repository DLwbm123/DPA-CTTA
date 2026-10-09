"""Generated-input checks for paired histories, future policies and scoring."""
import copy
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ..r32_history_origin.online import validate_case, coupled_step, branch_step
from .method import exchange
from . import report
from ..r30_state_history.method import StateHost, FrozenHost
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


def mechanical():
    torch.set_num_threads(2);torch.manual_seed(17)
    m=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    h=StateHost(None,'C_CONT',17,'test','cpu',copy.deepcopy(m));source=h.snapshot()
    history_rng=[]
    for i in range(3):
        _,_,pair=coupled_step(h,x.roll(i,-1));history_rng.append(pair)
    state=h.snapshot();h.restore(source)
    for i,pair in enumerate(history_rng):coupled_step(h,x.roll(i,-1),pair)
    assert equal(state,h.snapshot())
    reference=[];draws=[]
    for i in range(3):
        z,_,pair=coupled_step(h,x.roll(i+3,-1));reference.append(z);draws.append(pair)
    after=h.snapshot();h.restore(state)
    for i,pair in enumerate(draws):
        pre,z,_=branch_step(h,x.roll(i+3,-1),'UPDATE',pair)
        assert torch.equal(z,reference[i])
        if i==0:first=pre
    assert equal(after['parameters'],h.snapshot()['parameters']) and equal(after['adam'],h.snapshot()['adam'])
    h.restore(state)
    for i,pair in enumerate(draws):
        pre,z,_=branch_step(h,x.roll(i+3,-1),'HOLD',pair);assert torch.equal(z,pre)
        if i==0:assert torch.equal(first,pre)
    assert h.native.steps==3 and h.visits==6
    assert all(equal(state[k],h.snapshot()[k]) for k in ('parameters','adam','buffers'))
    h.restore(source);f=FrozenHost(None,'S_BATCH',17,'cpu',copy.deepcopy(m))
    _,z,_=branch_step(h,x,'HOLD',draws[0]);assert torch.equal(z,f.step(x)[0])
    h.check_frozen(True);h.close();f.close()
    rows=[dict(image_sha256=str(i)) for i in range(256)]
    case=dict(same=list(range(65,129)),cross=list(range(1,65)),query=list(range(129,257)),pivot=128)
    validate_case(case,rows)
    rows[128]=rows[0]
    try:validate_case(case,rows)
    except ValueError:pass
    else:raise AssertionError('history/query identity overlap accepted')


def scorer():
    cases=[dict(id='d0',domain='D0',query=[129,130,131]),dict(id='d1',domain='D1',query=[301,302,303])]
    rows=[dict(image_sha256=str(i),domain='D0' if i<300 else 'D1') for i in range(1,1952)]
    split={str(i):role for c in cases for i,role in zip(c['query'],('SEARCH','SEALED_REVIEW','CONTEXT'))}
    jobs=[dict(id=f's{s}_o{o}',seed=s,order=o) for s in (1,2,3) for o in (0,1)]
    values=dict(SS=.80,SC=.78,CS=.76,CC=.77,SAME64_HOLD=.79,CROSS64_HOLD=.75,SOURCE_HOLD=.72,C_CONT=.79,ANCHOR=.80)
    original_open=Path.open;original_stat=Path.stat;original_text=Path.read_text
    def fake_read(p):
        if p.name=='SPLIT.private.json':return split
        if p.name=='PLAN.json':return {str(o):cases for o in (0,1)}
        if p.name.startswith('FULL_'):return rows
        if p.parent.name=='processes':return dict(active=False,exit_code=0)
        if p.name=='COMPLETE.json':
            return dict(arrivals=3) if p.parent.name.startswith('d') else dict(replay=dict(arrivals=1951),branches=[{}]*8,history_updates=256,diagonal_query_checks=2644,first_pre_invariance_checks=4)
        raise AssertionError(p)
    def fake_open(p,*args,**kwargs):
        if p.suffix=='.bits':
            f=io.BytesIO();f.name=str(p);return f
        return original_open(p,*args,**kwargs)
    def fake_stat(p,*args,**kwargs):
        if p.suffix=='.bits':return SimpleNamespace(st_size=0 if p.name=='pre.bits' and p.parent.name.endswith('HOLD') else 3*report.WIDTH)
        return original_stat(p,*args,**kwargs)
    def fake_text(p,*args,**kwargs):
        if p.name=='traces.jsonl':
            c=cases[int(p.parent.name[1])]
            return '\n'.join(json.dumps(dict(visit=i,diagnostics=dict(parameter_drift=.1))) for i in c['query'])
        return original_text(p,*args,**kwargs)
    def fake_mask(f,offset):
        p=Path(f.name);name=p.parent.name
        name=name.split('_',1)[1] if name.startswith('d') else name[2:].split('_s')[0]
        if name.endswith('_UPDATE'):name=name[:-7]
        return values[name]-(.001 if p.name=='pre.bits' else 0)
    def fake_metrics(value,label):
        return [dict(dice=value,channel=k,pred_empty=False,gt_empty=False) for k in ('OD','OC')]
    class Reader:
        def __init__(self,*args):pass
        def read(self,row):return None
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        root=Path(tmp)
        for d in ('scores','public'):(root/d).mkdir()
        with patch.object(report,'read',fake_read),patch.object(report,'TargetReader',Reader),patch.object(report,'mask_at',fake_mask),patch.object(report,'hard_metrics',fake_metrics),patch.object(Path,'open',fake_open),patch.object(Path,'stat',fake_stat),patch.object(Path,'read_text',fake_text):
            result=report.score(dict(output_root=tmp,target_root=tmp,snapshot_input_root=tmp,r32_target_root=tmp,jobs=jobs,expected_scored_observations=216),lambda:None)
        assert result['scored_observations']==216
        r=next(r for r in result['primary'] if r['candidate']=='SC' and r['baseline']=='SS')
        assert abs(r['delta_pp']+2)<1e-8 and r['negative_trajectories']==6
        r=next(r for r in result['primary'] if r['baseline']=='ALGEBRA')
        assert abs(r['delta_pp']+3)<1e-8 and r['negative_trajectories']==6
        assert (root/'public/ALL_NEGATIVE_CELLS.csv').stat().st_size>0
    assert report.bins(1,736)==['ALL','Q1','FIRST64'] and report.bins(736,736)==['ALL','Q4']


def exchange_check():
    torch.set_num_threads(2);torch.manual_seed(7)
    h=StateHost(None,'C_CONT',7,'exchange','cpu',Tiny());source=h.snapshot()
    x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    for i in range(3):coupled_step(h,x.roll(i,-1))
    same=h.snapshot();h.restore(source)
    for i in range(3):coupled_step(h,1-x.roll(i,-2))
    cross=h.snapshot()
    assert equal(exchange(same,same),same) and equal(exchange(cross,cross),cross)
    mixed=exchange(same,cross)
    assert equal(mixed['parameters'],same['parameters']) and equal(mixed['adam'],cross['adam'])
    from ..r30_state_history.method import readonly
    h.restore(same);pre=readonly(h,x);_,_,rng=coupled_step(h,x)
    h.restore(mixed);assert torch.equal(pre,readonly(h,x))
    mixed_before=copy.deepcopy(mixed)
    branch_step(h,x,'UPDATE',rng)
    assert h.native.steps==4 and equal(mixed,mixed_before)
    assert not equal(h.snapshot()['parameters'],same['parameters'])
    for bad in (dict(cross,steps=2),dict(cross,steps=0)):
        try:exchange(same,bad)
        except ValueError:pass
        else:raise AssertionError('mismatched optimizer age accepted')
    h.close()


if __name__=='__main__':
    mechanical();exchange_check();scorer()
    print('PASS: matched state exchange, diagonal identity, optimizer-independent pre-readout, no donor mutation, age rejection, RNG replay, full synthetic scorer and factorial contrasts')
