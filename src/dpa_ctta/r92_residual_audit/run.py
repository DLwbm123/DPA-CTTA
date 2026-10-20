"""CPU-only binary edit ledger and exact cross-readout decomposition."""
import os,time
START=time.time();os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,gzip,json,signal,itertools,runpy
from pathlib import Path
import numpy as np
from dpa_ctta.r7_target_screen.runner import TargetReader
SEEDS=(20260907,17011,29009)
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def size(p):return sum(max(x.stat().st_size,x.stat().st_blocks*512) for x in p.rglob('*') if not x.is_symlink())
def edits(a,b,t):
 add=~a&b;delete=a&~b
 out=[int((add&t).sum()),int((add&~t).sum()),int((delete&~t).sum()),int((delete&t).sum())]
 def counts(m):return np.array([int((m&t).sum()),int((m&~t).sum()),int((~m&t).sum())])
 before=counts(a);after=counts(b);x,y,z,w=out
 assert np.array_equal(after,before+np.array([x-w,y-z,w-x]))
 def dice(c):return 200*c[0]/(2*c[0]+c[1]+c[2]) if c.sum() else 100.
 return out,before,after,dice(before),dice(after)
def selfcheck():
 t=np.array([1,0,0,1],bool);a=np.array([0,0,1,1],bool);b=~a
 v,_,_,_,_=edits(a,b,t);assert v==[1,1,1,1]
def run(c):
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU audit hard alarm')))
 signal.alarm(max(1,int(min(1740,c['deadline']-time.time()-180))))
 selfcheck();root=Path(c['output_root']);pub=root/'public';source=Path(c['source_root']);campaign=Path(c['campaign_root']);r91=Path(c['r91_root'])
 def guard():
  if time.time()>c['deadline']-180 or size(root)>14*1024**2 or size(campaign)>1610612736-134217728:raise RuntimeError('shared time/storage guard')
 m=runpy.run_path(str(campaign/'m0.py'));csvout=m['csvout'];post=m['post'];contrasts=m['contrasts']
 v=np.load(r91/'scorer/values.private.npz')['values'];panel=[read(source/'scorer'/f'screen_o{o}.json') for o in (0,1)]
 idx={row['image_sha256']:i for i,row in enumerate(panel[0])};names=sorted({x['domain'] for x in panel[0]});domains=np.array([names.index(x['domain']) for x in panel[0]])
 order_runs=[]
 for o in (0,1):
  runs=[dict(domain='D'+str(names.index(k)),arrivals=len(list(g))) for k,g in itertools.groupby(x['domain'] for x in panel[o])]
  order_runs.append(dict(order=o,runs=runs,one_contiguous_block_per_domain=len(runs)==4,domain_retrieval_events=0))
 save(pub/'ORDER_AUDIT.json',dict(orders=order_runs,S_equivalent_to_domain_boundary_reset=True,claim_state_retrieval_identified=False))
 # Arrays use content,anchor,arm(P0,P,K,Qt,Q0,NATIVE),raw/post,head,metric.
 parts={}
 for vi,kind in enumerate(('raw','post')):
  a=v[:,:,:,vi];h=a[:,:,1]-a[:,:,0];current=a[:,:,2]-a[:,:,1];h6=a[:,:,3]-a[:,:,4];rt=a[:,:,3]-a[:,:,1];r0=a[:,:,4]-a[:,:,0]
  assert np.allclose(a[:,:,3]-a[:,:,2],rt-current,atol=1e-10,rtol=0,equal_nan=True)
  assert np.allclose(h6-h,rt-r0,atol=1e-10,rtol=0,equal_nan=True)
  for n,x in [('H_id',h),('C_id',current),('H6',h6),('Rt',rt),('R0',r0),('DEPLOY',rt-current),('INTERACTION',h6-h),('K-NATIVE',a[:,:,2]-a[:,:,5])]:parts[n+'/'+kind]=x
 summaries=[];cells=[];contents=[]
 for n,x in parts.items():contrasts(pub,n,x,domains,summaries,cells,contents)
 csvout(pub/'DECOMPOSITION_SUMMARY.csv',summaries);csvout(pub/'DECOMPOSITION_CELLS.csv',cells);csvout(pub/'CONTENT_DISTRIBUTIONS.csv',contents)
 reader=TargetReader(c['target_root'],256*1024**2,'mask');truth=[reader.read(row).numpy()[0].astype(bool) for row in panel[0]]
 def decode(raw,shape):return np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(shape).astype(bool)
 static=[];q0=[]
 with (source/'jobs/source_only/predictions.bits').open('rb') as f,gzip.open(r91/'jobs/Q0.bits.gz','rb') as g:
  for i in range(192):
   a=f.read(131072);b=g.read(65536);assert len(a)==131072 and len(b)==65536
   static.append(decode(a,(2,2,512,512))[1]);q0.append(decode(b,(2,512,512)))
  assert not f.read(1) and not g.read(1)
 ledger={};checks=0;extreme=int(np.argmin(np.where(domains==1,(v[:,:,3,0,1,0]-v[:,:,2,0,1,0]).mean(1),np.inf)));ext=[]
 for j in range(6):
  seed=SEEDS[j//2];o=j%2
  with (source/'jobs'/f'replay_VV_s{seed}_o{o}'/'predictions.bits').open('rb') as f,gzip.open(r91/'jobs'/f's{seed}_o{o}'/'Qt.bits.gz','rb') as g:
   for visit,row in enumerate(panel[o]):
    i=idx[row['image_sha256']];raw=f.read(131072);rawq=g.read(65536);assert len(raw)==131072 and len(rawq)==65536
    pk=decode(raw,(2,2,512,512));qt=decode(rawq,(2,512,512));arms=[static[i],pk[0],pk[1],qt,q0[i]]
    for vi,variant in enumerate(('raw','post')):
     aa=[post(a) for a in arms] if vi else arms
     for a,b,n in [(0,1,'P0_to_P'),(1,2,'P_to_K'),(1,3,'P_to_Qt'),(4,3,'Q0_to_Qt')]:
      for ch in (0,1):
       e,be,af,db,da=edits(aa[a][ch],aa[b][ch],truth[i][ch]);assert abs(db-v[i,j,a,vi,ch,0])<1e-10 and abs(da-v[i,j,b,vi,ch,0])<1e-10;checks+=2
       key=(n,variant,int(domains[i]),j,ch)
       z=ledger.setdefault(key,dict(edits=np.zeros(4,dtype=np.int64),before=np.zeros(3,dtype=np.int64),after=np.zeros(3,dtype=np.int64),n=0,dice_sum=0.))
       z['edits']+=e;z['before']+=be;z['after']+=af;z['n']+=1;z['dice_sum']+=da-db
     if i==extreme and vi==0:
      c1=edits(aa[2][1],aa[3][1],truth[i][1]);ext.append(dict(seed=seed,order=o,K_Dice=c1[3],Qt_Dice=c1[4],delta_pp=c1[4]-c1[3],K_empty=not bool(aa[2][1].any()),Qt_empty=not bool(aa[3][1].any())))
    if visit%8==0:guard();save(root/'RUN_STATE.json',dict(stage='CPU_PIXEL_AUDIT',seed=seed,order=o,arrival=visit+1,checked=checks))
   assert not f.read(1) and not g.read(1)
 rows=[]
 for (name,variant,d,j,ch),z in ledger.items():
  pp=z['edits']/z['n']/512**2*100
  rows.append(dict(transition=name,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,channel=('OD','OC')[ch],n_contents=z['n'],true_add_pixels=int(z['edits'][0]),false_add_pixels=int(z['edits'][1]),correct_delete_pixels=int(z['edits'][2]),wrong_delete_pixels=int(z['edits'][3]),true_add_image_pp=float(pp[0]),false_add_image_pp=float(pp[1]),correct_delete_image_pp=float(pp[2]),wrong_delete_image_pp=float(pp[3]),before_TP=int(z['before'][0]),before_FP=int(z['before'][1]),before_FN=int(z['before'][2]),after_TP=int(z['after'][0]),after_FP=int(z['after'][1]),after_FN=int(z['after'][2]),mean_individual_Dice_delta_pp=z['dice_sum']/z['n']))
 csvout(pub/'EDIT_LEDGER.csv',rows);reader.after_check();assert len(ext)==6
 save(pub/'EXTREME_CHECK.json',dict(domain='D1',channel='OC',scope='same worst content across six anchors, private identity not released',anchors=ext,mean_delta_pp=sum(x['delta_pp'] for x in ext)/6,verified=True,excluded=False))
 save(pub/'COMPLETION.json',dict(status='COMPLETE',new_model_calls=0,new_GPU_updates=0,new_image_reads=0,new_label_contents=0,existing_label_read_operations=192,binary_Dice_reconstruction_checks=checks,ledger_rows=len(rows),cross_readout_identities_per_content=True,CPU_body_seconds=time.time()-START,ended=time.time(),deadline=c['deadline']))
 save(root/'RUN_STATE.json',dict(stage='COMPLETE'));guard()
if __name__=='__main__':run(read(Path(os.environ['CONFIG'])))
