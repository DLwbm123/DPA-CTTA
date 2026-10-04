"""Cache action-independent source GraTa trajectories, then fit the full RL actor."""
import copy,time
from pathlib import Path
from collections import OrderedDict
import torch
from ..r10_carrier.source import data
from ..r10_use_write_rl.source import Source as OriginalSource
from ..r10_use_write_rl.controller import Actor
from ..r10_use_write_rl.math import Memory
from ..r10_use_write_rl.learning import Trainer,rng_state,restore_rng
from ..r9_current_first.validation import dice
import statistics as st
from ..r7_shared.source import simulate
from ..r10_12h_core.run import save,read,sha
from ..r9_current_first.storage import write_torch
from ..r16_evidence_correction.source import metric
from .method import SEED,Host,native,old,bn_state,set_bn,copy_bn,fixed_actor

EPISODES=tuple(i+16*m for i in (0,4,8,12) for m in range(4))


def image(src,fold,index,visit,group=None,anchor=None,probe_id=None):
    ep=EPISODES[index%16];ids,roles,mode=src.schedule(fold,ep,SEED);group=roles[visit][1] if group is None else group;anchor=ids[visit] if anchor is None else anchor
    row=src.data.get(group,fold)
    key=f'R10_QUERY|{fold}|{SEED}|{ep}|{visit}|{group}' if probe_id is None else f'R10_PROBE_QUERY|{fold}|{SEED}|{ep}|{visit}|{probe_id}|{group}'
    return simulate(row.image,src.banks[fold][anchor],key),row.label,mode,group


def build_cache(c,src,guard,fold,indices):
    root=Path(c['output_root'])/'private/bn_cache';root.mkdir(exist_ok=True);made=0
    for index in indices:
        p=root/f'{fold}_{index:02d}.pt'
        if p.exists():
            x=torch.load(p,weights_only=True)
            if x['config_sha256']!=sha(c) or len(x['rows'])!=32:raise ValueError('source cache identity')
            continue
        h,close=native(c,dict(sha256=sha(c)));meter=guard.meter.attach(h.native.model);rows=[]
        try:
            for v in range(32):
                guard();im,label,mode,group=image(src,fold,index,v);z,tr=h.step(im);copy_bn(h.native,src.segmenter.model)
                with torch.no_grad():
                    mirrored,raw,tokens=src.segmenter(im,observe=True)
                    if not torch.equal(z,mirrored):raise ValueError('source post-GraTa mirror differs')
                    obs=src.controller.observe(raw,tokens)
                rows.append(dict(bn=bn_state(h.native),observation=obs,group=group,mode=mode,native_metrics=metric(z,label)));guard.extra['image_accesses']+=1;made+=1
            torch.save(dict(config_sha256=sha(c),fold=fold,index=index,episode=EPISODES[index],rows=rows),p)
            save(Path(c['output_root'])/'SOURCE_CACHE_STATE.json',dict(status='RUNNING',complete_episodes=len(list(root.glob('*.pt'))),planned_episodes=32))
        finally:meter.remove();close()
    return made


class ScheduledSegmenter:
    def __init__(self,segmenter,guard):self.base=segmenter;self.model=segmenter.model;self.guard=guard
    def __call__(self,image,v=None,observe=False):
        self.guard.extra['image_accesses']+=1
        set_bn(self.model,image._r18_bn_state)
        return self.base(image,v,observe)


class CachedSource:
    def __init__(self,c,src,guard):
        self.c,self.original=c,src;self.data,self.oracles,self.controller=src.data,src.oracles,src.controller;self.banks=src.banks
        self.segmenter=ScheduledSegmenter(src.segmenter,guard);self.cache=OrderedDict();self.episodes=OrderedDict()
    def schedule(self,fold,episode,seed):
        if seed!=SEED:raise ValueError('source seed')
        return self.original.schedule(fold,EPISODES[episode%16],seed)
    def states(self,fold,index):
        key=(fold,index%16)
        if key not in self.episodes:self.episodes[key]=torch.load(Path(self.c['output_root'])/'private/bn_cache'/f'{fold}_{index%16:02d}.pt',weights_only=True)['rows']
        self.episodes.move_to_end(key)
        while len(self.episodes)>4:self.episodes.popitem(last=False)
        return self.episodes[key]
    def item(self,fold,seed,episode,visit,warm=False,group=None,anchor=None,probe_id=None):
        if seed!=SEED:raise ValueError('source seed')
        key=(fold,episode%16,visit,group,anchor,probe_id)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        im,label,mode,g=image(self.original,fold,episode%16,visit,group,anchor,probe_id)
        state=self.states(fold,episode)[min(visit+3,31) if probe_id is not None else visit]
        im._r18_bn_state=state['bn']
        if probe_id is None:
            if g!=state['group']:raise ValueError('source role identity')
            observation=state['observation']
        else:
            with torch.no_grad():_,raw,tokens=self.segmenter(im,observe=True);observation=self.controller.observe(raw,tokens)
        item=dict(image=im,label=label,observation=observation);self.cache[key]=item
        while len(self.cache)>96:self.cache.popitem(last=False)
        return item
    window=OriginalSource.window
    probes=OriginalSource.probes


@torch.no_grad()
def episode(source,actor,seed,index,fold='val',guard=lambda:None):
    memory=Memory.zero();hs=[];ss=[];writes=[];gains=[];_,_,mode=source.schedule(fold,index,seed)
    for visit in range(32):
        guard();item=source.item(fold,seed,index,visit);use,memory,a=source.controller.act(actor,item['observation'],memory)
        z=source.segmenter(item['image'],source.controller.carrier.basis@use);h,s=dice(z.sigmoid(),item['label']);hs.append(h);ss.append(s)
        writes.append(float(a['raw_action'][9].sigmoid()));gains.append(float(a['raw_action'][0].sigmoid()))
    return dict(episode=index,mode=mode,hard_Dice=st.mean(v for x in hs for v in x),soft_Dice=st.mean(v for x in ss for v in x),hard_OD=st.mean(x[0] for x in hs),hard_OC=st.mean(x[1] for x in hs),soft_OD=st.mean(x[0] for x in ss),soft_OC=st.mean(x[1] for x in ss),write_mean=st.mean(writes),gain_mean=st.mean(gains),state_norm=float(memory.m.norm()))


def validation(source,actor,guard,label):
    saved=rng_state()
    try:return [dict(condition=label,**episode(source,actor,SEED,i,guard=guard)) for i in range(16)]
    finally:restore_rng(saved)


def preflight(c,guard):
    root=Path(c['output_root']);profiles={};checks=[]
    with data(c,guard) as src:
        meter=guard.meter.attach(src.segmenter.model)
        try:
            for arm in ('RL_ORIGINAL','FIXED_GRATA'):
                actor=fixed_actor() if arm=='FIXED_GRATA' else None
                h=Host(c,arm,dict(sha256=sha(c)),actor);hs=[guard.meter.attach(h.seg.model)];h2=None
                if h.g:hs.append(guard.meter.attach(h.g.native.model))
                original,original_close=old(c,dict(sha256=sha(c))) if arm=='RL_ORIGINAL' else (None,lambda:None)
                if original:hs.append(guard.meter.attach(original.segmenter.model))
                try:
                    times=[]
                    for v in range(4):
                        im,label,_,_=image(src,'val',0,v);start=time.perf_counter();z,tr=h.step(im,force_zero=bool(h.g));torch.cuda.synchronize();times.append(time.perf_counter()-start);guard.extra['image_accesses']+=1
                        if original:
                            ref,_=original.step(im);guard.extra['image_accesses']+=1
                            if not torch.equal(ref,z) or any(not torch.equal(getattr(original.state,k),getattr(h.memory,k)) for k in ('m','q','h')):raise ValueError('original full RL parity failed')
                    snap=h.snapshot();h2=Host(c,arm,dict(sha256=sha(c)),actor);hs.append(guard.meter.attach(h2.seg.model))
                    if h2.g:hs.append(guard.meter.attach(h2.g.native.model))
                    h2.restore(snap);im,_,_,_=image(src,'val',0,4);z,_=h.step(im);zz,_=h2.step(im);guard.extra['image_accesses']+=2
                    if not torch.equal(z,zz) or any(not torch.equal(getattr(h.memory,k),getattr(h2.memory,k)) for k in ('m','q','h')):raise ValueError('full BN/Adam/RNG/memory continuation mismatch')
                    profiles[arm]=max(times);checks.append(dict(arm=arm,exact_parity=True,snapshot_next_image_exact=True))
                finally:
                    for x in hs:x.remove()
                    h.close();original_close()
                    if h2:h2.close()
            start=time.perf_counter();build_cache(c,src,guard,'fit',range(4));profiles['cache_visit']=(time.perf_counter()-start)/128
            cached=CachedSource(c,src,guard);actor=Actor(SEED)
            for method,total in (('WARM',1000),('GR_RET_EMA',1024)):
                trainer=Trainer(actor,src.controller,cached,SEED,method,dict(profile=True),actor,guard,total=total);times=[]
                for _ in range(2):
                    start=time.perf_counter();row=trainer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
                profiles[method]=max(times)
                if method=='GR_RET_EMA' and not all(p.grad is not None for p in actor.write.parameters()):raise ValueError('RL writer not connected')
            saved=rng_state();start=time.perf_counter()
            with torch.no_grad():
                for index in range(4):episode(cached,actor,SEED,index,fold='fit',guard=guard)
            restore_rng(saved);profiles['validation_visit']=(time.perf_counter()-start)/128
        finally:meter.remove()
    result=dict(passed=True,checks=checks,profiles=profiles,cache_fit_episodes_reused=4,target_accesses=0,all_RL_actions_and_memory_preserved=True,qualification_training_discarded=True)
    save(root/'SOURCE_PRECHECK.json',result);return result


def train(c,guard):
    root=Path(c['output_root']);results=[];hist=[]
    with data(c,guard) as src:
        meter=guard.meter.attach(src.segmenter.model)
        try:
            for fold in ('fit','val'):build_cache(c,src,guard,fold,range(16))
            save(root/'SOURCE_CACHE_STATE.json',dict(status='COMPLETE',complete_episodes=32,visits=1024))
            cached=CachedSource(c,src,guard);actor=Actor(SEED);artifacts={}
            for name,method,total in (('WARM','WARM',1000),('POST','GR_RET_EMA',1024)):
                trainer=Trainer(actor,src.controller,cached,SEED,method,dict(config_sha256=sha(c),phase=name),actor,guard,total=total)
                for step in range(total):
                    row=trainer.step();hist.append(dict(stage=name,**row))
                    if (step+1)%32==0 or step+1==total:save(root/'SOURCE_RL_PROGRESS.json',dict(stage=name,completed=step+1,total=total))
                p=root/'private'/f'actor_{name}.pt';digest=write_torch(p,dict(actor=actor.state_dict(),method=method,steps=total,config_sha256=sha(c)));artifacts[name]=dict(file=str(p),sha256=digest,steps=total)
                results.extend(validation(cached,copy.deepcopy(actor).eval().requires_grad_(False),guard,name+'_GRATA'))
            results.extend(validation(cached,fixed_actor(),guard,'FIXED_GRATA'))
            source_groups={fold:{x['group'] for i in range(16) for x in cached.states(fold,i)} for fold in ('fit','val')}
            if source_groups['fit']&source_groups['val']:raise ValueError('source fit/val overlap')
        finally:meter.remove()
    result=dict(status='COMPLETE',selected=artifacts,rows=results,history=hist,warm_updates=1000,RL_rounds=1024,RL_epochs_per_round=2,source_unique={k:len(v) for k,v in source_groups.items()},checkpoint_selection='fixed endpoints; no target/source best-checkpoint selection',policy_seed=SEED)
    save(root/'SOURCE_RL.json',result);return result
