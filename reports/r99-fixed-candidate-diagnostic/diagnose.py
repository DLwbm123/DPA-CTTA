"""Frozen fixed-candidate diagnostic. Reads only existing scores and sealed signals."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,signal,resource
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.stats import spearmanr
ROOT=Path('/tmp/n99').resolve()
def read(p):return json.loads(Path(p).read_text())
def usage(p):return sum(max(x.stat().st_size,x.stat().st_blocks*512) for x in [p,*p.rglob('*')] if not x.is_symlink())
def save(name,obj):
    data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
    assert len(data)<131072 and usage(ROOT)+len(data)+16384<262144
    (ROOT/name).write_bytes(data)
def scalar(x):return float(x) if x is not None and np.isfinite(x) else None
def corr(x,y):
    return dict(n=len(x),pearson=scalar(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None,
                spearman=scalar(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None,
                same_direction=int(((x*y)>0).sum()),opposite_direction=int(((x*y)<0).sum()),U_ties=int((x==0).sum()),gain_ties=int((y==0).sum()),both_ties=int(((x==0)&(y==0)).sum()))
def main():
    start=time.time();cpu=time.process_time();budget=read(ROOT/'BUDGET.json')
    assert start<budget['deadline']-60
    signal.alarm(max(1,int(budget['deadline']-start-10)))
    resource.setrlimit(resource.RLIMIT_FSIZE,(131072,131072))
    r=Path('/tmp/n97').resolve();assert read(r/'public/CONTROLLER_EXIT.json')['status']=='COMPLETE'
    c=read(r/'private/config.json');old=read(c['prior_config']);r91=Path(old['output_root'])
    with np.load(r91/'scorer/values.private.npz',allow_pickle=False) as f:v=f['values'];dom=f['domains']
    assert v.shape==(192,6,6,2,2,11) and np.bincount(dom).tolist()==[22,57,57,56]
    keys=('U','entropyD','singleR','noAction','noHistory','Earea','Darea')
    scores={k:np.zeros((192,2)) for k in keys};z=np.zeros((192,2),bool);visit=np.zeros((192,2),int);elig=z.copy()
    for o in (0,1):
        s=read(r/'private'/f'signals_{o}.json');assert s['status']=='SEALED'
        assert s['online_label_reads']==s['online_score_reads']==0 and len(s['rows'])==192
        seen=set()
        for row in s['rows']:
            i=row['index'];assert i not in seen;seen.add(i);assert row['order']==o
            z[i,o]=row['z'];visit[i,o]=row['visit'];elig[i,o]=row['eligible']
            for k in keys:scores[k][i,o]=row[k]
        assert seen==set(range(192)) and sorted(visit[:,o])==list(range(192))
    assert np.array_equal(z,scores['U']>.5) and z.sum(axis=0).tolist()==[5,10]
    P0,P,K,Q=[v[:,:2,a] for a in (0,1,2,4)];diff=Q-K
    g=diff[:,:,:,:,0].mean(axis=3) # content, order, raw/post
    w=1/(4*np.bincount(dom)[dom]);assert abs(w.sum()-1)<1e-12
    def gains(sel):return [float(np.sum(sel*w[:,None]*g[:,:,vi])/2) for vi in (0,1)]
    assert abs(gains(z)[0]-read(r/'public/RESULT.json')['main_macro_gain_pp'])<1e-12
    # A: unconstrained budget oracle and independently constrained optimization.
    A=[]
    for o in (0,1):
        obj=w*g[:,o,0];blocks=np.array([visit[:,o]//32==b for b in range(6)],float);counts=blocks@z[:,o]
        direct={}
        for mode in ('unlimited','exact','at_most'):
            sel=np.zeros(192,bool)
            if mode=='unlimited':sel=g[:,o,0]>0
            else:
                for b in range(6):
                    ids=np.flatnonzero(blocks[b]);rank=sorted(ids,key=lambda i:(-obj[i],visit[i,o]));take=rank[:int(counts[b])]
                    if mode=='at_most':take=[i for i in take if obj[i]>0]
                    sel[take]=True
            direct[mode]=dict(raw_pp=float(obj@sel),post_pp=float((w*g[:,o,1])@sel),actions=int(sel.sum()))
        rows=[];names=[];finite=True
        for vi in (0,1):
            for d in range(4):
                ids=dom==d
                for h in (0,1):
                    for mi in (0,4,5,6):
                        if mi==6 and h==1:continue
                        if not(np.isfinite(K[ids,o,vi,h,mi]).all() and np.isfinite(Q[ids,o,vi,h,mi]).all()):finite=False
                        row=np.zeros(192);row[ids]=diff[ids,o,vi,h,mi]/ids.sum()*(1 if mi==0 else -1)
                        rows.append(row);names.append([vi,d,h,mi])
        mat=np.array(rows);results=[]
        for mode in ('unlimited','exact','at_most'):
            if not finite:results.append(dict(mode=mode,status='NOT_EVALUABLE_NONFINITE'));continue
            scales=np.maximum(np.max(np.abs(mat),axis=1),1e-12)
            constraints=[LinearConstraint(mat/scales[:,None],np.zeros(len(mat)),np.full(len(mat),np.inf))]
            if mode!='unlimited':constraints.append(LinearConstraint(blocks,counts if mode=='exact' else np.zeros(6),counts))
            t=time.time();res=milp(-obj,integrality=np.ones(192),bounds=Bounds(0,1),constraints=constraints,options={'time_limit':60.,'mip_rel_gap':0.})
            rr=dict(mode=mode,status=int(res.status),message=res.message,wall_seconds=time.time()-t,dual_upper_pp=scalar(-res.mip_dual_bound) if getattr(res,'mip_dual_bound',None) is not None else None,gap=scalar(getattr(res,'mip_gap',None)),nodes=scalar(getattr(res,'mip_node_count',None)),feasible=False)
            if res.x is not None:
                sel=np.rint(res.x).astype(bool);resid=mat@sel;bc=blocks@sel
                integer_error=float(np.max(np.abs(res.x-sel)))
                feasible=integer_error<=1e-7 and resid.min()>=-1e-7 and (mode=='unlimited' or (np.array_equal(bc,counts) if mode=='exact' else np.all(bc<=counts)))
                rr.update(feasible=bool(feasible),integer_error=integer_error,min_protection_residual=float(resid.min()),strict_negative_residual_cells=int((resid<0).sum()),raw_pp=float(obj@sel),post_pp=float((w*g[:,o,1])@sel),actions=int(sel.sum()),block_counts=bc.astype(int).tolist(),protection_residuals=[dict(variant=['raw','post'][n[0]],domain=n[1],head=n[2],metric=n[3],favorable_delta=float(x)) for n,x in zip(names,resid)])
            results.append(rr)
        A.append(dict(order=o,original_block_counts=counts.astype(int).tolist(),unconstrained=direct,all_candidates_finite=finite,constrained=results))
    save('A_ORACLES.json',A)
    # B: deterministic controls and matched random. Domain is offline only.
    def match(sc,strat):
        out=np.zeros_like(z)
        for o in (0,1):
            for b in range(6):
                for d in range(4) if strat else (None,):
                    ids=np.flatnonzero((visit[:,o]//32==b)&((dom==d) if d is not None else True))
                    n=int(z[ids,o].sum());rank=sorted(ids,key=lambda i:(-sc[i,o],visit[i,o]));out[rank[:n],o]=True
        return out
    def summarize(sel):return dict(raw_pp=gains(sel)[0],post_pp=gains(sel)[1],counts=sel.sum(0).tolist(),weighted_coverage_by_order=(w[:,None]*sel).sum(0).tolist(),weighted_coverage=float((w[:,None]*sel).sum()/2))
    controls={}
    for k in keys[1:]:controls[k]={label:summarize(match(scores[k],strat)) for label,strat in [('block',False),('block_domain',True)]}
    randoms={}
    for label,strat in [('block',False),('block_domain',True)]:
        rng=np.random.default_rng(20261021);a=np.array([gains(match(rng.random(z.shape),strat)) for _ in range(1000)])
        randoms[label]=dict(repeats=1000,seed=20261021,raw_mean=float(a[:,0].mean()),raw_p025=float(np.quantile(a[:,0],.025)),raw_p975=float(np.quantile(a[:,0],.975)),post_mean=float(a[:,1].mean()))
    strata=[]
    for o in (0,1):
        for b in range(6):
            for d in range(4):
                ids=(visit[:,o]//32==b)&(dom==d);n=int(ids.sum());m=int(z[ids,o].sum())
                strata.append(dict(order=o,block=b,domain=d,candidates=n,actions=m,exchangeable=bool(0<m<n)))
    B=dict(main=summarize(z),controls=controls,random=randoms,strata=strata,nonempty_strata=sum(s['candidates']>0 for s in strata),exchangeable_strata=sum(s['exchangeable'] for s in strata))
    for k in controls:assert controls[k]['block_domain']['counts']==[5,10] and abs(controls[k]['block_domain']['weighted_coverage']-B['main']['weighted_coverage'])<1e-12
    save('B_MATCHING.json',B)
    # C: endpoint arithmetic, all observations partitioned once.
    r0=(Q-P0)[:,:,:,:,0].mean(3);h=(P-P0)[:,:,:,:,0].mean(3);cur=(K-P)[:,:,:,:,0].mean(3)
    assert np.max(np.abs(g-(r0-h-cur)))<1e-10
    C=[]
    for vi,variant in enumerate(('raw','post')):
        gg=g[:,:,vi];parts={'selected_positive':z&(gg>0),'selected_negative':z&(gg<0),'selected_zero':z&(gg==0),'unselected_positive':(~z)&(gg>0),'unselected_nonpositive':(~z)&(gg<=0)}
        assert np.all(sum(parts.values())==1)
        groups=[]
        for name,mask in parts.items():
            vals={k:float(np.sum(w[:,None]*mask*a[:,:,vi])/2) for k,a in [('r0',r0),('h',h),('current',cur),('g',g)]}
            groups.append(dict(group=name,n=int(mask.sum()),weighted_contributions_pp=vals,identity_residual=vals['g']-(vals['r0']-vals['h']-vals['current'])))
        losses=-(w[:,None]*gg/2)[z&(gg<0)];losses=np.sort(losses)[::-1];total=float(losses.sum())
        C.append(dict(variant=variant,groups=groups,tail=[dict(k=k,loss_pp=float(losses[:k].sum()),fraction_of_gross_loss=float(losses[:k].sum()/total) if total else None) for k in (1,3,5)]))
    save('C_DECOMPOSITION.json',C)
    # D: exact-zero ties retained, no fitted rule or inferential p-value.
    du=scores['U'][:,1]-scores['U'][:,0];both=elig.all(1)
    D=dict(eligibility=dict(both=int(both.sum()),one=int((elig.sum(1)==1).sum()),neither=int((~elig.any(1)).sum())),action_patterns={f'{a}{b}':int(((z[:,0]==a)&(z[:,1]==b)).sum()) for a in (0,1) for b in (0,1)},variants=[])
    for vi,variant in enumerate(('raw','post')):
        dg=g[:,1,vi]-g[:,0,vi]
        D['variants'].append(dict(variant=variant,all=corr(du,dg),both_eligible=corr(du[both],dg[both])))
    save('D_PAIRED.json',D)
    save('RECEIPT.json',dict(status='COMPLETE',T0=budget['T0'],started_at=start,completed_at=time.time(),wall_seconds=time.time()-start,CPU_process_seconds=time.process_time()-cpu,elapsed_from_T0=time.time()-budget['T0'],deadline=budget['deadline'],new_model_calls=0,new_label_reads=0,rows=384,output_bytes_before_receipt=usage(ROOT),checks=['sealed signals and decision reproduction','main gain reproduction','score dimensions and domain counts','endpoint identity','all groups partition 384','matched counts and weighted coverage'],no_threshold_scan=True))
    print(json.dumps(dict(status='COMPLETE',main=B['main'],paired=D)))
if __name__=='__main__':main()
