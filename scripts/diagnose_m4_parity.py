"""Authorized repair diagnostic: two images only, no outer update or evaluation."""
import os,json,copy,sys,gc,time
from pathlib import Path
import torch
from dpa_ctta.m4_registration import load_registered
from dpa_ctta.m2_registration import inspect_history
from dpa_ctta.source_pilot_release import environment,source_unchanged
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.source_io import source_proxy
from dpa_ctta.m1_run import make_host,tensor_save,Observed
from dpa_ctta.m2_run import deterministic_smoke_pair
from dpa_ctta.offline.adaptation_dd import restore_history,cpu_tree
import dpa_ctta.offline.trajectory_dd as module
from test_vptta_host import pixels,proxy as fixture
from dpa_ctta.host_diagnostic_run import private_json
os.umask(0o077);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
out=Path(sys.argv[1]);task='fundus';started=time.monotonic()
overlay,reg=load_registered(Path(sys.argv[2]))
seed_all(20260907);env=environment()
r=reg['tasks'][task];source=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(r['proxy'],task)
history=inspect_history(overlay['tasks'][task]['history']['path'],task)[16]
queries=[(pixels(task,i),fixture(task).mask) for i in range(2)]
functional=[];actual=[];captured={}
original=module.native_adam

def capture(phi,gradient,state,lr):
 captured.clear();captured.update(phi=cpu_tree(phi),gradient=cpu_tree(gradient),input_adam=cpu_tree(state))
 result=original(phi,gradient,state,lr)
 captured.update(plus=cpu_tree(result[0]),adam=cpu_tree(result[1]))
 return result
module.native_adam=capture
with deterministic_smoke_pair():
 ep=module.TrajectoryEpisode(task,source,'cuda:0');S=real.pixel_rgb.to('cuda:0').clone().requires_grad_();state=ep.initial(history)
 for x,y in queries:
  loss,state,pred=ep.step('D4',S,real.mask,x,y,state)
  functional.append(dict(copy.deepcopy(captured),memory=cpu_tree(state['memory']),prediction=cpu_tree(pred)))
  state=module.detach_state(state);ep.memory.memory=dict(state['memory']);ep.clear_graphs();del loss,pred
 counts=copy.deepcopy(ep.counts);source_unchanged(ep.host,source);del ep,S,state;gc.collect()
 h=make_host(task,'R',source,real);h._initial_hook.remove();h._started=True;restore_history(h,history);watch=Observed(h)
 def before(opt,args,kwargs):
  p=h.prompt.data_prompt
  captured.clear();captured.update(phi=cpu_tree(p),gradient=cpu_tree(p.grad),input_adam=cpu_tree(opt.state[p]))
 handle=h.optimizer.register_step_pre_hook(before)
 for x,y in queries:
  pred=h.step(x);watch.verify();p=h.prompt.data_prompt
  actual.append(dict(copy.deepcopy(captured),plus=cpu_tree(p),adam=cpu_tree(h.optimizer.state[p]),memory={k:torch.from_numpy(v.copy()) for k,v in h.memory_bank.memory.items()},prediction=cpu_tree(pred)))
 source_unchanged(h,source);handle.remove();watch.release()
 actual_counts=copy.deepcopy(watch.counts)

def difference(a,b):
 d=(a-b).abs();return dict(max_abs=float(d.max()),unequal=int((a!=b).sum()),outside_tolerance=int((d>1e-5+1e-4*b.abs()).sum()),norm=float(d.norm()))
summary=[]
for i,(a,b) in enumerate(zip(functional,actual)):
 row=dict(position=17+i,fields={k:difference(a[k],b[k]) for k in ['phi','gradient','plus','prediction']},input_adam={k:difference(a['input_adam'][k],b['input_adam'][k]) for k in ['exp_avg','exp_avg_sq']},output_adam={k:difference(a['adam'][k],b['adam'][k]) for k in ['exp_avg','exp_avg_sq']},memory_key_order_same=list(a['memory'])==list(b['memory']))
 row['memory_max_abs']=max(float((a['memory'][k]-b['memory'][k]).abs().max()) for k in a['memory'])
 summary.append(row)
tensor_save(out/'diagnostic_tensors.pt',dict(functional=functional,actual=actual))
result=dict(status='DIAGNOSTIC_COMPLETE',scope='fixed two-image Fundus source-fixture mechanical parity only; no outer update, no new performance score',updates=dict(online=2,outer=0,inner=2),functional_counts=counts,native_counts=actual_counts,comparisons=summary,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),environment=env,exit_code=0)
private_json(out/'diagnostic.json',result)
print(json.dumps(result),flush=True)
