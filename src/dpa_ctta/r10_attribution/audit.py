"""Source-only telemetry and exact endpoint audit; original objectives are reused."""
import copy,json,time,statistics
from pathlib import Path
import torch
from ..r10_12h_core.run import read,save,sha,OPS
from ..r10_use_write_rl.learning import Trainer,rng_state,restore_rng
from ..r10_use_write_rl.controller import Actor,Host
from ..r9_current_first.storage import load_torch

SEED=20260924
CONTEXTS=[16*m+i for m in range(4) for i in (0,2,4,6,8,10,12,14)]

def distribution(values):
    if not values:return dict(n=0,status='NOT_RECORDED')
    t=torch.tensor(values,dtype=torch.float64)
    return dict(n=len(values),mean=float(t.mean()),std=float(t.std(unbiased=False)),min=float(t.min()),p05=float(t.quantile(.05)),median=float(t.median()),p95=float(t.quantile(.95)),max=float(t.max()))

def gates(values):return dict(**distribution(values),max_abs_from_half=max((abs(v-.5) for v in values),default=None))

def actor_at(old,name):
    old=Path(old);r=read(old/'source'/name/'complete.json');p=r['artifact'];a=Actor(SEED)
    blob=load_torch(old/'source'/name/p['file'],p['sha256']);a.load_state_dict(blob['actor'])
    if blob['steps']!=r['steps'] or r['steps']!=dict(WARM=1000,POST=1024)[name]:raise ValueError('endpoint step identity')
    return a,r

class AuditTrainer(Trainer):
    def __init__(self,*a,updates=True,**kw):super().__init__(*a,**kw);self.updates=updates
    def step(self):
        self.branch_gradients=[];self.sampled_writes=[];self.deterministic_writes=[]
        r=super().step();r.update(branch_gradients=self.branch_gradients,sampled_writer=gates(self.sampled_writes),deterministic_probe_writer=gates(self.deterministic_writes));return r
    def prediction(self,actor,item,state,epsilon=None,forced_write=None):
        result=super().prediction(actor,item,state,epsilon,forced_write)
        if forced_write is None:
            (self.sampled_writes if epsilon is not None else self.deterministic_writes).append(float(result[2]['raw_action'][9].sigmoid().detach()))
        return result
    def update(self):
        row={}
        for name in ('use','write'):
            params=list(getattr(self.actor,name).parameters());row[name]=dict(trainable=sum(p.numel() for p in params if p.requires_grad),grad_norm=sum(float(p.grad.detach().double().square().sum()) for p in params if p.grad is not None)**.5,grad_tensors=sum(p.grad is not None for p in params))
        self.branch_gradients.append(row)
        return super().update() if self.updates else sum(v['grad_norm']**2 for v in row.values())**.5

def preflight(c,source,guard):
    from ..r10_use_write_rl import learning
    old=Path(c['previous_root']);root=Path(c['output_root']);warm,wr=actor_at(old,'WARM');post,pr=actor_at(old,'POST')
    result=dict(schema='R10_ATTRIBUTION_SOURCE_AUDIT_V1',warm=wr,post=pr,parameter_changes={},historical_training={},source_supplement=[],profiles={})
    for branch in ('use','write'):
        a=torch.cat([v.detach().reshape(-1).double() for v in getattr(warm,branch).parameters()]);b=torch.cat([v.detach().reshape(-1).double() for v in getattr(post,branch).parameters()]);d=b-a
        result['parameter_changes'][branch]=dict(count=a.numel(),L2=float(d.norm()),max_abs=float(d.abs().max()),relative_L2=float(d.norm()/a.norm().clamp_min(1e-30)),changed_elements=int((d!=0).sum()))
    meta=read(old/'source/POST/fit/latest.json');snap=load_torch(old/'source/POST/fit'/f"checkpoint.{meta['slot']}.pt",meta['sha256'])['snapshot']
    if snap['steps']!=1024 or snap['total']!=1024 or any(not torch.equal(v,snap['actor'][k]) for k,v in post.state_dict().items()):raise ValueError('POST endpoint/snapshot mismatch')
    tr=AuditTrainer(copy.deepcopy(post),source.controller,source,SEED,'GR_RET_EMA',{'supplement_only':True},warm,guard,total=1024,updates=False)
    optimizer_ids={id(p) for g in tr.opt.param_groups for p in g['params']}
    result['writer_optimizer']=dict(current_requires_grad=all(p.requires_grad for p in tr.actor.write.parameters()),current_in_optimizer=all(id(p) in optimizer_ids for p in tr.actor.write.parameters()),saved_optimizer_parameters=len(snap['optimizer']['param_groups'][0]['params']),reconstructed_optimizer_parameters=len(optimizer_ids),saved_optimizer_state_entries=len(snap['optimizer']['state']))
    if not all(result['writer_optimizer'][k] for k in ('current_requires_grad','current_in_optimizer')) or result['writer_optimizer']['saved_optimizer_parameters']!=len(optimizer_ids):raise ValueError('writer optimizer audit')
    logs=[json.loads(x) for x in (old/'source/POST/fit/physical.jsonl').read_text().splitlines()]
    for k in ('reward_std','zero_advantage_fraction','ema','KL'):result['historical_training'][k]=distribution([r[k] for r in logs])
    result['historical_training']['clip_fraction_by_epoch']=[distribution([r['clip_fraction'][i] for r in logs]) for i in range(2)]
    result['historical_training']['actual_advantage_norm']='NOT_RECORDED; supplementary fixed-source probe is separate'
    result['historical_training']['sampled_writer_distribution']='NOT_RECORDED; old controller means mix stochastic predictions and probes'
    result['target_deterministic_writer']={}
    for order in (0,1):
        vals=[json.loads(x)['write'] for x in (old/f'target/GR_RET_EMA_o{order}/visits.jsonl').read_text().splitlines()]
        result['target_deterministic_writer'][str(order)]=gates(vals)
    saved=rng_state();original=learning.group_advantage;advantages=[]
    def capture(*args,**kw):
        a,e=original(*args,**kw);advantages.append(dict(L2=float(a.norm()),max_abs=float(a.abs().max()),values=a.tolist()));return a,e
    try:
        learning.group_advantage=capture
        for k in (0,9,18,27):
            tr.steps=k;tr.ema=snap['ema'];advantages.clear();row=tr.step()
            result['source_supplement'].append(dict(schedule_round_zero_based=k,checkpoint='POST1024',optimizer_updates=0,frozen_copy=True,advantage=copy.deepcopy(advantages),row=row))
    finally:learning.group_advantage=original;restore_rng(saved)
    save(root/'WRITER_AUDIT.private.json',result)
    for method in ('SUP_RET','SUP_STATIC'):
        actor=copy.deepcopy(warm);t=AuditTrainer(actor,source.controller,source,SEED,method,dict(profile_only=True,method=method),warm,guard,total=1024)
        source.cache.clear();before=guard.meter.cost.copy();torch.cuda.synchronize();started=time.monotonic();rows=[]
        for _ in range(32):rows.append(t.step())
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        result['profiles'][method]=dict(complete_rounds=32,wall_seconds=elapsed,conservative_seconds_per_round=1.3*elapsed/32,safety_factor=1.3,cost={k:guard.meter.cost[k]-before[k] for k in OPS},peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),writer_grad_norm=distribution([g['write']['grad_norm'] for r in rows for g in r['branch_gradients']]),use_grad_norm=distribution([g['use']['grad_norm'] for r in rows for g in r['branch_gradients']]))
        save(root/'WRITER_AUDIT.private.json',result)
    # Check all new deployment branches only on one source image, never target pixels.
    from ..r10_use_write_rl.factory import construct
    from .run import resolve_job,source_lock
    image=source.item('val',SEED,0,0)['image'];checks=[]
    for arm in ('C0_CURRENT_STATS','B_PARENT_FULL','WARM_CONST_HALF','R10_RESET_ALL','R10_FORCE_WRITE'):
        job=dict(arm=arm,order=0,id=arm+'_o0',seed=SEED);lock=source_lock(c,job);host,close=construct(resolve_job(c,job),c,root,lock)
        model=host.segmenter.model;handle=guard.meter.attach(model)
        try:
            before=guard.meter.cost.copy();p,_=host.step(image);counts=tuple(guard.meter.cost[k]-before[k] for k in OPS)
            if counts!=((1,0,0,0) if arm=='C0_CURRENT_STATS' else (2,0,0,0)):raise ValueError('control source smoke counts')
            checks.append(dict(arm=arm,counts=counts,shape=list(p.shape)))
        finally:handle.remove();close()
    result['source_control_smoke']=checks;save(root/'WRITER_AUDIT.private.json',result);return result['profiles']

@torch.no_grad()
def counterfactual(c,source,guard):
    from ..r10_use_write_rl.math import Memory,use_and_write
    from ..r10_use_write_rl.controller import detached
    from ..r9_current_first.validation import dice
    root=Path(c['output_root'])
    if len(c['counterfactual_contexts'])!=32:raise ValueError('exactly 32 preselected source contexts required')
    actor,_=actor_at(c['previous_root'],'POST');actor.eval();actor.requires_grad_(False);rows=[];before=guard.meter.cost.copy();start=time.time()
    for index in c['counterfactual_contexts']:
        guard();state=Memory.zero();mode=source.schedule('val',index,SEED)[2]
        for v in range(12):
            item=source.item('val',SEED,index,v);_,state,_=source.controller.act(actor,item['observation'],state,forced_write=.5)
        current=source.item('val',SEED,index,12);proposed,features=source.controller.propose(current['observation'],state);raw=actor(features)
        # One observed current use and one current segmentation; writer changes only commit.
        commits={};uses={}
        for w in (0.,.5,1.):uses[w],commits[w]=use_and_write(proposed,current['observation']['d'],state,source.controller.scale,raw,forced_write=w)
        if not all(torch.equal(uses[.5],uses[w]) for w in (0.,1.)):raise ValueError('current use changed under writer intervention')
        for k in ('m','q','h'):
            if not torch.equal(getattr(state,k),getattr(commits[0.],k)):raise ValueError('zero writer failed atomic m/q/h invariant')
        current_logits=source.segmenter(current['image'],source.controller.carrier.basis@uses[.5]);assert torch.isfinite(current_logits).all()
        states={w:detached(s) for w,s in commits.items()};future=[]
        for v in range(13,29):
            guard();item=source.item('val',SEED,index,v);pred={};metrics={};codes={}
            for w in (0.,.5,1.):
                use,states[w],_=source.controller.act(actor,item['observation'],states[w],forced_write=.5);codes[w]=use
                p=source.segmenter(item['image'],source.controller.carrier.basis@use).sigmoid().detach().cpu();pred[w]=p;h,s=dice(p,item['label']);metrics[str(w)]=dict(hard=h,soft=s)
            changes={}
            for w in (0.,1.):
                d=pred[w]-pred[.5];changes[str(w)]=dict(probability_MAE=d.abs().mean((0,2,3)).tolist(),probability_max=float(d.abs().max()),mask_flip_fraction=((pred[w]>=.5)!=(pred[.5]>=.5)).float().mean((0,2,3)).tolist(),use_L2=float((codes[w]-codes[.5]).norm()))
            future.append(dict(visit=v,metrics=metrics,changes=changes))
        row=dict(episode=index,mode=mode,current_use_max_error=0.,current_prediction_equivalence='identical current input/use with frozen deterministic segmenter; one shared current prediction',current_forward_calls=1,commit_delta={str(w):{k:float((getattr(commits[w],k)-getattr(commits[.5],k)).norm()) for k in ('m','q','h')} for w in (0.,1.)},horizons={})
        for h in (4,16):
            chosen=future[:h];ms={w:{metric:[statistics.mean(r['metrics'][str(w)][metric][ch] for r in chosen) for ch in range(2)] for metric in ('hard','soft')} for w in (0.,.5,1.)}
            row['horizons'][str(h)]=dict(metrics={str(k):v for k,v in ms.items()},delta_vs_half={str(w):{metric:[ms[w][metric][ch]-ms[.5][metric][ch] for ch in range(2)] for metric in ('hard','soft')} for w in (0.,1.)},probability_and_masks={str(w):dict(probability_MAE=[statistics.mean(r['changes'][str(w)]['probability_MAE'][ch] for r in chosen) for ch in range(2)],mask_flip_fraction=[statistics.mean(r['changes'][str(w)]['mask_flip_fraction'][ch] for r in chosen) for ch in range(2)],probability_max=max(r['changes'][str(w)]['probability_max'] for r in chosen)) for w in (0.,1.)},finite_candidate_best_gain=max(statistics.mean(ms[w]['hard']) for w in ms)-statistics.mean(ms[.5]['hard']))
        rows.append(row);save(root/'SOURCE_COUNTERFACTUAL.private.json',dict(status='RUNNING',rows=rows))
    observed=tuple(guard.meter.cost[k]-before[k] for k in OPS)
    if observed!=(2496,0,0,0):raise ValueError('counterfactual actual forward/gradient contract: '+str(observed))
    if sorted(__import__('collections').Counter(r['mode'] for r in rows).values())!=[8]*4:raise ValueError('counterfactual mode coverage')
    result=dict(status='COMPLETE',contexts=32,future_segmentations=1536,current_segmentations=32,horizons=[4,16],rows=rows,cost={k:guard.meter.cost[k]-before[k] for k in OPS},wall_seconds=time.time()-start,label='finite-candidate source diagnostic; optimistic selection, not an upper bound or target performance')
    save(root/'SOURCE_COUNTERFACTUAL.private.json',result);return dict(contexts=len(rows),future_segmentations=1536)
