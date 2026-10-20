"""Oracle-metadata offline boundary-reset diagnostic; never a deployed router."""
import copy,gzip,json,os,runpy,time
from pathlib import Path
import numpy as np
from dpa_ctta.r85_migration.process import child,finish_child
SEED=20260907

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def merge(bank,current):
 s=copy.deepcopy(bank)
 for k in ('visits','rng','native_rng','counts'):s['base'][k]=copy.deepcopy(current['base'][k])
 return s

def selfcheck():
 bank={'base':{'parameters':[1],'adam':[2],'steps':7,'visits':7,'rng':[3],'native_rng':[4],'counts':[5]}}
 current={'base':{'parameters':[9],'adam':[8],'steps':99,'visits':30,'rng':[31],'native_rng':[32],'counts':[33]}}
 x=merge(bank,current);assert x['base']['parameters']==[1] and x['base']['adam']==[2] and x['base']['steps']==7
 assert all(x['base'][k]==current['base'][k] for k in ('visits','rng','native_rng','counts'))
 x['base']['parameters'][0]=100;assert bank['base']['parameters']==[1]


def worker(c):
 import torch
 from dpa_ctta.r85_loss_aggregation.method import Host
 from dpa_ctta.r79_mechanism_control.method import style
 from dpa_ctta.r19_model_only.method import equal
 from dpa_ctta.r19_model_only.runtime import weights,image
 from dpa_ctta.r7_target_screen.runner import TargetReader
 from dpa_ctta.r9_current_first.assets import gpu_policy
 root=Path(c['output_root']);stage=os.environ['STAGE'];slot=int(os.environ['SLOT']);dest=root/'private'/f'{stage}_{slot}';dest.mkdir()
 helpers=runpy.run_path(str(root/'readout_helpers.py'));usage=helpers['usage']
 s=dict(status='RUNNING',started=time.time(),physical_updates=0,completed_physical_updates=0,physical_forwards=0,training_backwards=0,online_label_reads=0,source_image_reads=0)
 cap=9 if stage=='qualify' else 576;hosts=[];timings=[]
 def persist():save(dest/'RUN_STATE.json',s)
 def guard():
  if time.time()>c['deadline']-180:raise TimeoutError('original deadline')
 guard.extra={'image_accesses':0}
 def charge(k):
  guard();key={'attempt':'physical_updates','complete':'completed_physical_updates'}[k]
  if k=='attempt' and s[key]>=cap:raise RuntimeError('physical update cap')
  s[key]+=1
 def meter(h):
  def f(*_):
   guard()
   if s['physical_forwards']>=cap*10:raise RuntimeError('physical forward cap')
   s['physical_forwards']+=1
  def b(g):s['training_backwards']+=1;return g
  h.handles.append(h.native.model.register_forward_pre_hook(f));h.handles.append(h.native.params[0].register_hook(b))
 def make():
  h=Host(checkpoint,'VV',SEED,'qualification' if stage=='qualify' else 'formal',charge=charge);meter(h);hosts.append(h);return h
 def step(h,x,key):
  t=time.time();p=h.before(x);k,_=h.keyed_step(x,style(x,SEED,key));torch.cuda.synchronize();timings.append(time.time()-t);persist();return p,k
 def packed(p,k):return np.packbits(np.stack([(z[0]>=0).numpy() for z in (p,k)])).tobytes()
 persist()
 try:
  torch.set_num_threads(2);torch.set_num_interop_threads(2);gpu_policy(c['gpus'][slot]);torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
  m=runpy.run_path(str(root/'m1_helpers.py'));permit,opened,rejected=m['access_guard'](c);checkpoint=weights(c);reader=TargetReader(c['target_root'],256*1024**2,'image')
  canonical=read(root/'private/screen_o0.json');index={r['image_sha256']:i for i,r in enumerate(canonical)}
  if stage=='qualify':
   a=make();initial=a.snapshot();zs=[];states=[]
   for visit,row in enumerate(canonical[:2]):
    x=image(reader,row,permit,guard);p,k=step(a,x,index[row['image_sha256']]);zs.append(packed(p,k));states.append(a.snapshot())
   a.restore(initial)
   with Path(c['reference_predictions'][f'{SEED}_0']).open('rb') as f:
    for visit,row in enumerate(canonical[:2]):
     x=image(reader,row,permit,guard);p,k=step(a,x,index[row['image_sha256']]);assert packed(p,k)==zs[visit]==f.read(131072) and equal(a.snapshot(),states[visit])
   a.close();hosts.remove(a)
   b=make();ref=make();init=b.snapshot();banks={};ages={}
   for visit,bank in enumerate((0,1,0)):
    current=b.snapshot();b.restore(merge(banks.get(bank,init),current));entry=b.snapshot();local_age=ages.get(bank,0)
    assert b.native.steps==local_age and b.visits==visit
    x=image(reader,canonical[visit],permit,guard);p,k=step(b,x,index[canonical[visit]['image_sha256']]);banks[bank]=b.snapshot();ages[bank]=local_age+1
    if bank==0:
     ref.restore(merge(ref.snapshot(),entry));pp,kk=step(ref,x,index[canonical[visit]['image_sha256']]);assert packed(pp,kk)==packed(p,k) and equal(ref.snapshot(),b.snapshot())
   assert ages=={0:2,1:1} and b.visits==3 and b.native.steps==2
   save(root/'public/QUALIFICATION.json',dict(status='PASS',updates=9,forwards=90,training_backwards=9,original_VV_replay_reference_and_fullstate_exact=True,switched_bank_independent_replay_fullstate_exact=True,learning_pre_global_visits_RNG_counts_post=True,age_counts_exact=True,max_steady_transaction_seconds=max(timings[1:]),transaction_seconds=timings))
  else:
   assert read(root/'FORMAL_ADMISSION.json')['admitted'];rows=read(root/'private'/f'screen_o{slot}.json');routing=read(root/'private/routing.json')
   for arm in ('V','S','J'):
    h=make();init=h.snapshot();banks={};ages={};folder=root/'jobs'/f'{arm}_o{slot}';folder.mkdir()
    # V is re-executed for parity, but existing reference masks are not copied.
    output=None if arm=='V' else gzip.GzipFile(filename=str(folder/'predictions.bits.gz'),mode='xb',compresslevel=1,mtime=0)
    with Path(c['reference_predictions'][f'{SEED}_{slot}']).open('rb') as original:
     try:
      for visit,row in enumerate(rows):
       usage(c);key=index[row['image_sha256']];bank=0 if arm=='V' else routing[arm][key]
       if arm!='V':h.restore(merge(banks.get(bank,init),h.snapshot()))
       assert h.native.steps==ages.get(bank,0) and h.visits==visit
       x=image(reader,row,permit,guard);p,k=step(h,x,key);data=packed(p,k)
       if arm=='V':assert original.read(131072)==data,'V exact original reference'
       else:output.write(data);banks[bank]=h.snapshot()
       ages[bank]=ages.get(bank,0)+1;s.update(arm=arm,arrival=visit+1);persist()
      if arm=='V':assert not original.read(1)
     finally:
      if output is not None:output.close()
    assert sum(ages.values())==192 and h.visits==192
    if arm!='V':assert sorted(ages.values())==[22,56,57,57]
    h.check_frozen(True);h.close();hosts.remove(h)
    save(folder/'COMPLETE.json',dict(arrivals=192,updates=192,forwards=1920,bank_final_ages=ages,reference_exact=True if arm=='V' else None,global_visits=192))
  for h in hosts:h.check_frozen(True);h.close()
  hosts=[];assert s['physical_updates']==s['completed_physical_updates']==s['training_backwards']==cap and s['physical_forwards']==cap*10
  permit['reverify_images']={Path(p).resolve() for p in reader.used};assert permit['reverify_images']<=permit['opened_images']
  try:reader.after_check()
  finally:permit['reverify_images'].clear()
  save(dest/'ACCESS_AUDIT.json',dict(online_label_reads=0,source_image_reads=0,image_accesses=guard.extra['image_accesses'],opened_paths=sorted(opened),rejections=rejected))
  s.update(status='GPU_COMPLETE',ended=time.time());persist()
 except BaseException as e:s.update(status='FAILED',error=type(e).__name__,reason=str(e),ended=time.time());persist();raise
 finally:
  for h in hosts:h.close()

def controller(c):
 root=Path(c['output_root']);helpers=runpy.run_path(str(root/'readout_helpers.py'));attempts=[]
 def status(stage,**kw):
  save(root/'RUN_STATE.json',dict(stage=stage,shared_started=c['started'],deadline=c['deadline'],**kw));save(root/'RESOURCE_LEDGER.json',dict(attempts=attempts,GPU_process_seconds=sum(a['gpu_process_seconds'] for a in attempts),CPU_process_seconds=sum(a['cpu_process_seconds'] for a in attempts)))
 def group(mode,stage,slots):
  pending=list(slots);running=[];done=[]
  while pending or running:
   if time.time()>c['deadline']-180:raise TimeoutError('original deadline')
   helpers['usage'](c)
   if pending:
    try:running.append(child(c,mode,stage,pending[0]));pending.pop(0)
    except RuntimeError as e:
     if 'GPU UUID/free-memory' not in str(e):raise
     status('WAIT_RESOURCE')
   for item in running[:]:
    result=finish_child(c,item)
    if result is not None:attempts.append(result);done.append(result);running.remove(item);status(stage)
   if pending or running:time.sleep(5 if running else 30)
  return all(a['exit_code']==0 for a in done)
 try:
  status('QUALIFY')
  if not group('worker','qualify',[0]):raise RuntimeError('qualification NOT_EVALUABLE')
  q=read(root/'public/QUALIFICATION.json');projection=q['max_steady_transaction_seconds']*1152*1.5/2+2400
  ok=q['status']=='PASS' and projection<c['deadline']-time.time() and c['prior_updates']+1161<=5000 and c['prior_forwards']+11610<=50000
  save(root/'FORMAL_ADMISSION.json',dict(admitted=ok,projected_seconds=projection,prior_updates=c['prior_updates'],prior_forwards=c['prior_forwards'],expected_updates=1161,expected_forwards=11610))
  if not ok:raise RuntimeError('profile not admitted')
  status('FORMAL')
  if not group('worker','formal',[0,1]):raise RuntimeError('formal NOT_EVALUABLE')
  status('SCORING')
  if not group('score','final',[0]):raise RuntimeError('CPU NOT_EVALUABLE')
  status('COMPLETE');save(root/'CONTROLLER_EXIT.json',dict(status='COMPLETE',ended=time.time()))
 except BaseException as e:status('FAILED',reason=str(e));save(root/'CONTROLLER_EXIT.json',dict(status='FAILED',ended=time.time()));raise
if __name__=='__main__':
 c=read(Path(os.environ['CONFIG']));mode=os.environ['MODE']
 if mode=='watchdog':runpy.run_path(str(Path(c['output_root'])/'readout_helpers.py'))['watchdog'](c)
 elif mode=='controller':controller(c)
 elif mode=='worker':worker(c)
