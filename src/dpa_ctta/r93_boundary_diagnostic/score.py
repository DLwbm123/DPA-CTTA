"""Two-order one-seed offline mechanism screen; not a six-anchor method gate."""
import os,time
START=time.time();os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,runpy,signal
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from dpa_ctta.r85_migration.process import alive
from dpa_ctta.r7_target_screen.runner import TargetReader

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def macro(v,ds):return np.array([v[ds==d].mean((0,2)) for d in range(4)]).mean(0)
def gate(v,ds):
 rng=np.random.default_rng(20260907);ids=[np.flatnonzero(ds==d) for d in range(4)];boot=[np.mean([v[rng.choice(i,len(i),replace=True)].mean() for i in ids]) for _ in range(2000)];ci=np.quantile(boot,[.0125,.9875]);a=macro(v,ds)
 return dict(delta_pp=float(a.mean()),orders=a.tolist(),CI97_5_pp=ci.tolist(),passed=bool(a.mean()>=.1 and (a>0).all() and ci[0]>0))
def selfcheck():
 d=np.repeat(range(4),3);x=np.ones((12,2,2));assert gate(x,d)['passed'] and not gate(-x,d)['passed']

def run(c):
 root=Path(c['output_root']);pub=root/'public';campaign=Path(c['campaign_root']);source=Path(c['source_root'])
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU score cap')));signal.alarm(max(1,int(min(1740,c['deadline']-time.time()-180))))
 states=[]
 for name in ('qualify_worker0','formal_worker0','formal_worker1'):
  p=read(root/'processes'/f'{name}.json');s=read(root/'private'/f'{p["stage"]}_{p["slot"]}'/'RUN_STATE.json');assert p['exit_code']==0 and not p['active'] and not alive(p) and s['status']=='GPU_COMPLETE';states.append(s)
 totals={k:sum(s[k] for s in states) for k in ('physical_updates','completed_physical_updates','physical_forwards','training_backwards')};assert totals==dict(physical_updates=1161,completed_physical_updates=1161,physical_forwards=11610,training_backwards=1161)
 m=runpy.run_path(str(campaign/'m0.py'));post,metrics,boundary,csvout=[m[k] for k in ('post','metrics','boundary','csvout')];names=m['METRICS']
 files=[root/'jobs'/f'{arm}_o{o}'/'predictions.bits.gz' for arm in ('S','J') for o in (0,1)]
 for p in files:
  pp=p.with_name('post.bits.gz')
  with gzip.open(p,'rb') as f,gzip.GzipFile(filename=str(pp),mode='xb',compresslevel=1,mtime=0) as g:
   for _ in range(192):
    raw=f.read(131072);assert len(raw)==131072;a=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool);g.write(np.packbits(np.stack([post(z) for z in a])).tobytes())
   assert not f.read(1)
  p.chmod(0o444);pp.chmod(0o444)
 save(pub/'SEALED.json',dict(status='RAW_AND_POST_SEALED',streams=4,sealed_at=time.time(),label_reads_before_seal=0,all_GPU_exit0_absent=True))
 panels=[read(source/'scorer'/f'screen_o{o}.json') for o in (0,1)];idx={x['image_sha256']:i for i,x in enumerate(panels[0])};dn=sorted({x['domain'] for x in panels[0]});ds=np.array([dn.index(x['domain']) for x in panels[0]])
 old=np.load(campaign/'private/m0_values.private.npz')['values'];v=np.full((192,2,3,2,2,2,len(names)),np.nan);v[:,:,0]=old[:,:2,3]
 reader=TargetReader(c['target_root'],256*1024**2,'mask');truth=[];edges=[];dist=[]
 for row in panels[0]:
  t=reader.read(row).numpy()[0].astype(bool);e=np.stack([boundary(a) for a in t]);truth.append(t);edges.append(e);dist.append(np.stack([ndi.distance_transform_edt(~a).astype(np.float32) for a in e]))
 for ai,arm in enumerate(('S','J'),1):
  for o in (0,1):
   folder=root/'jobs'/f'{arm}_o{o}';assert read(root/'jobs'/f'V_o{o}'/'COMPLETE.json')['reference_exact']
   for vi,fn in enumerate(('predictions.bits.gz','post.bits.gz')):
    with gzip.open(folder/fn,'rb') as f:
     for row in panels[o]:
      i=idx[row['image_sha256']];raw=f.read(131072);assert len(raw)==131072;mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,2,512,512).astype(bool)
      for phase in (0,1):v[i,o,ai,phase,vi]=metrics(mask[phase],truth[i],edges[i],dist[i])
     assert not f.read(1)
 reader.after_check();np.savez_compressed(root/'scorer/values.private.npz',values=v,domains=ds)
 scores=[];contrasts=[]
 for ai,arm in enumerate(('V','S','J')):
  for phase,pn in enumerate(('PRE','K')):
   for vi,variant in enumerate(('raw','post')):
    for d in range(4):
     for o in (0,1):
      for ch,head in enumerate(('OD','OC')):
       for mi,name in enumerate(names):
        z=v[ds==d,o,ai,phase,vi,ch,mi];f=z[np.isfinite(z)];scores.append(dict(arm=arm,phase=pn,variant=variant,domain=f'D{d}',seed=20260907,order=o,channel=head,metric=name,mean=float(f.mean()) if len(f) else None,n_NA=int((~np.isfinite(z)).sum())))
 for vi,variant in enumerate(('raw','post')):
  for bi,bn in ((0,'V'),(2,'J')):
   delta=v[:,:,1,1,vi]-v[:,:,bi,1,vi]
   for d in range(4):
    for o in (0,1):
     for ch,head in enumerate(('OD','OC')):
      for mi,name in enumerate(names):
       z=delta[ds==d,o,ch,mi];z=z[np.isfinite(z)];contrasts.append(dict(contrast='S-'+bn,variant=variant,domain=f'D{d}',order=o,channel=head,metric=name,delta=float(z.mean()) if len(z) else None))
 csvout(pub/'ALL_SCORES.csv',scores);csvout(pub/'ALL_CONTRAST_CELLS.csv',contrasts)
 distribution=[]
 for vi,variant in enumerate(('raw','post')):
  for bi,bn in ((0,'V'),(2,'J')):
   delta=v[:,:,1,1,vi]-v[:,:,bi,1,vi]
   for d in range(4):
    for ch,head in enumerate(('OD','OC')):
     for mi,name in enumerate(names):
      x=delta[ds==d,:,ch,mi];count=np.isfinite(x).sum(1);a=np.divide(np.nansum(x,axis=1),count,out=np.full(len(x),np.nan),where=count>0);f=a[np.isfinite(a)]
      distribution.append(dict(contrast='S-'+bn,variant=variant,domain=f'D{d}',channel=head,metric=name,n_contents=len(x),n_NA=int((~np.isfinite(a)).sum()),mean=float(f.mean()) if len(f) else None,median=float(np.median(f)) if len(f) else None,p10=float(np.quantile(f,.1)) if len(f) else None,p90=float(np.quantile(f,.9)) if len(f) else None,minimum=float(f.min()) if len(f) else None,maximum=float(f.max()) if len(f) else None,positive_contents=int((f>0).sum()),negative_contents=int((f<0).sum())))
 csvout(pub/'CONTENT_DISTRIBUTIONS.csv',distribution)
 task=v[:,:,1,1,0,:,0]-v[:,:,0,1,0,:,0];specific=v[:,:,1,1,0,:,0]-v[:,:,2,1,0,:,0];main={'S-V':gate(task,ds),'S-J':gate(specific,ds)}
 d1=task[ds==1].mean((0,1));d3=float(task[ds==3,:,1].mean());nv=v[:,:,1,1,0,:,0]-old[:,:2,2,1,0,:,0];dh=np.array([nv[ds==d].mean((0,1)) for d in range(4)]);dho=np.array([nv[ds==d].mean(0) for d in range(4)])
 protect=bool((d1>=-.2).all() and d3>=.3 and (dh>=-.5).all() and (dho>=-2).all())
 pp=all(macro(v[:,:,1,1,1,:,0]-v[:,:,bi,1,1,:,0],ds).mean()>0 for bi in (0,2));structure=[]
 for vi,variant in enumerate(('raw','post')):
  for scope in ('overall','D1','D3'):
   ii=np.arange(192) if scope=='overall' else np.flatnonzero(ds==int(scope[1:]))
   for mi in (4,5,6):
    for ch in ((None,) if mi==6 or scope=='overall' else (0,1)):
     selected=[0] if mi==6 else ([0,1] if ch is None else [ch]);a=v[ii,:,1,1,vi][:,:,selected,mi];b=v[ii,:,0,1,vi][:,:,selected,mi];finite=np.isfinite(a)&np.isfinite(b);delta=a-b;groups=range(4) if scope=='overall' else [int(scope[1:])];pieces=[]
     for d in groups:
      z=delta[ds[ii]==d];ok=finite[ds[ii]==d];pieces.append(float(z[ok].mean()) if ok.any() else None)
     value=float(np.mean(pieces)) if all(x is not None for x in pieces) else None;tol=1e-9 if mi==6 else 1e-6
     structure.append(dict(variant=variant,scope=scope,metric=names[mi],channel='ratio' if mi==6 else ('both' if ch is None else ('OD','OC')[ch]),delta=value,S_NA=int((~np.isfinite(a)).sum()),V_NA=int((~np.isfinite(b)).sum()),passed=bool(value is not None and value<=tol and (~np.isfinite(a)).sum()<=(~np.isfinite(b)).sum())))
   for ch in (0,1):
    a=v[ii,:,1,1,vi,ch,9];b=v[ii,:,0,1,vi,ch,9];structure.append(dict(variant=variant,scope=scope,metric='empty',channel=('OD','OC')[ch],S_count=int((a>0).sum()),V_count=int((b>0).sum()),passed=bool((a>0).sum()<=(b>0).sum())))
 passed=all(g['passed'] for g in main.values()) and protect and pp and all(x['passed'] for x in structure)
 save(pub/'PERFORMANCE_GATE.json',dict(status='PASS' if passed else 'NOT_PASSED',main=main,D1_OD_OC_delta=d1.tolist(),D3_OC_delta=d3,NATIVE_domain_head=dh.tolist(),NATIVE_domain_order_head=dho.tolist(),protection_passed=protect,post_directions_positive=bool(pp),structure=structure,structure_passed=all(x['passed'] for x in structure),mechanism_screen_only=True,domain_metadata_used_offline=True,state_retrieval_identified=False,independent_validation=False))
 save(pub/'COMPLETION.json',dict(status='COMPLETE',**totals,seed=20260907,orders=2,arms=3,arrivals=1152,V_reference_exact=True,all_GPU_exit0_absent_before_labels=True,all_raw_post_sealed_before_labels=True,online_label_reads=0,source_image_reads=0,terminal_label_contents=192,CPU_body_seconds=time.time()-START,shared_total_updates=c['prior_updates']+1161,shared_total_forwards=c['prior_forwards']+11610))
if __name__=='__main__':selfcheck();run(read(Path(os.environ['CONFIG'])))
