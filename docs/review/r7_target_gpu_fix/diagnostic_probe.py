import sys,os,json,time,copy
from pathlib import Path
import torch
sys.path.insert(0,os.environ['TEST_ROOT'])
from common import segmenter,pixels
from dpa_ctta.b1_host import Host
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.r7_source_prep.runner import configure_backend

torch.set_num_threads(2);configure_backend({'device':'cuda:0'});start=time.monotonic();runs=[]
for device in ('cpu','cuda:0'):
 seed_all(20260907);s=segmenter(full=True);s.close();h=Host('C',device=device,model=s.model)
 record=dict(inputs=[],outputs=[],initial={n:p.detach().cpu().clone() for n,p in h.model.named_parameters()})
 def hook(m,args,z):record['inputs'].append(args[0].detach().cpu().clone());record['outputs'].append(z[0].detach().cpu().clone())
 hd=h.model.register_forward_hook(hook)
 try:
  z,t=h.step(pixels());record.update(params={n:p.detach().cpu().clone() for n,p in h.model.named_parameters() if p.requires_grad},grads={n:p.grad.detach().cpu().clone() for n,p in h.model.named_parameters() if p.requires_grad},counts=h.counts.copy(),diagnostics=t['diagnostics'])
 finally:
  hd.remove()
  for x in h.handles:x.remove()
 del s,h;runs.append(record)
a,b=runs
def compare(x,y):
 d=(x-y).abs();bad=d>(.0003+.003*y.abs())
 return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),bad=int(bad.sum()),elements=x.numel())
r=dict(initial_exact=all(torch.equal(a['initial'][n],b['initial'][n]) for n in a['initial']),inputs_exact=[torch.equal(x,y) for x,y in zip(a['inputs'],b['inputs'])],forward_differences=[compare(x,y) for x,y in zip(a['outputs'],b['outputs'])],parameter_differences=sorted([(n,compare(a['params'][n],b['params'][n])) for n in a['params']],key=lambda x:x[1]['max_abs'],reverse=True)[:5],gradient_differences=sorted([(n,compare(a['grads'][n],b['grads'][n])) for n in a['grads']],key=lambda x:x[1]['max_abs'],reverse=True)[:5],counts=[x['counts'] for x in runs],diagnostics=[x['diagnostics'] for x in runs],wall_seconds=time.monotonic()-start)
Path(os.environ['DIAGNOSTIC_RESULT']).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
