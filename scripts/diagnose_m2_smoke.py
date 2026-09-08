"""Bounded procedural Polyp smoke diagnosis; no source/target scoring."""
import argparse,gc,json,os,sys,time,warnings
from pathlib import Path
import torch
from dpa_ctta.m2_registration import load_registered,inspect_history
from dpa_ctta.m2_run import m2_host
from dpa_ctta.m1_run import make_host,Observed,tensor_save
from dpa_ctta.offline.adaptation_dd import OfflineEpisode
from dpa_ctta.source_io import source_proxy
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.source_pilot_release import environment
from dpa_ctta.host_diagnostic import rng,restore,close,snapshot
from dpa_ctta.proxy_loss import FixedProxy,ProxyProvenance
from test_vptta_host import pixels,proxy as fixture
p=argparse.ArgumentParser();p.add_argument('registration',type=Path);p.add_argument('out',type=Path);p.add_argument('--mode',choices=['baseline','deterministic','warn'],default='baseline');a=p.parse_args();os.umask(0o077)
started=time.monotonic();overlay,reg=load_registered(a.registration);r=reg['tasks']['polyp'];seed_all(20260907)
state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(r['proxy'],'polyp')
result={'mode':a.mode,'environment':environment(),'online':0,'outer':0,'inner':0}
path=a.out/'temporary_polyp_D2_proxy.pt'
if not path.exists():
 library=inspect_history(overlay['tasks']['polyp']['history']['path'],'polyp');ep=OfflineEpisode('polyp',state,'cuda:0');S=torch.nn.Parameter(real.pixel_rgb.cuda().clone());opt=torch.optim.Adam([S],lr=.01)
 loss=ep.objective('D',S,real.mask,pixels('polyp'),fixture('polyp').mask,library[16]);g,=torch.autograd.grad(loss,S);assert torch.isfinite(g).all() and g.norm()>0;S.grad=g;opt.step();result['outer']=1
 with torch.no_grad():S.clamp_(0,1)
 tensor_save(path,S.detach().cpu());ep.clear_graphs();del ep,S,opt,loss,g,library;gc.collect()
payload=FixedProxy(torch.load(path,map_location='cpu',weights_only=True),real.mask,real.signed_distance,ProxyProvenance.SOURCE)
if a.mode!='baseline':torch.use_deterministic_algorithms(True,warn_only=a.mode=='warn')
direct=make_host('polyp','D',state,payload);wrapped=m2_host('polyp','D2',state,payload)
close(direct.model.state_dict(),wrapped.model.state_dict(),exact=True);close(snapshot(direct),snapshot(wrapped),exact=True);result['initial_state_exact']=True
traces=[]
for h in (direct,wrapped):
 trace={'live':[],'proxy':[],'gradient':[],'prompt_after':[]};traces.append(trace)
 def live(m,args,out,t=trace):t['live'].append((out[0] if isinstance(out,tuple) else out).detach().cpu().clone())
 def proxy(m,args,out,t=trace):t['proxy'].append((out[0] if isinstance(out,tuple) else out).detach().cpu().clone())
 def grad(g,t=trace):t['gradient'].append(g.detach().cpu().clone())
 def after(opt,args,kw,t=trace,h=h):t['prompt_after'].append(h.prompt.data_prompt.detach().cpu().clone())
 h.model.register_forward_hook(live);h._proxy_model.register_forward_hook(proxy);h.prompt.data_prompt.register_hook(grad);h.optimizer.register_step_post_hook(after)
initial=rng()
try:
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');x=direct.step(pixels('polyp'));result['online']+=1;after_rng=rng();restore(initial);y=wrapped.step(pixels('polyp'));result['online']+=1
  result['warnings']=list(dict.fromkeys(str(w.message) for w in caught))
 result['comparisons']={}
 for name in traces[0]:
  result['comparisons'][name]=[]
  for x,y in zip(traces[0][name],traces[1][name]):
   result['comparisons'][name].append({'max_abs':float((x-y).abs().max()),'norm_a':float(x.norm()),'norm_b':float(y.norm()),'exact':torch.equal(x,y),'within_tolerance':torch.allclose(x,y,rtol=1e-4,atol=1e-5)})
 try:close(snapshot(direct),snapshot(wrapped));close(after_rng,rng(),exact=True);result['state_rng_match']=True
 except Exception as e:result['state_rng_match']=False;result['state_error']=str(e)
except Exception as e:result['exception']=str(e)
result['seconds']=time.monotonic()-started;result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
(a.out/(a.mode+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
