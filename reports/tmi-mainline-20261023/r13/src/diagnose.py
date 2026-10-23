"""Frozen source-development diagnostics; never trains or reads target data."""
import os,json,time,signal,random,importlib.util,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_edt
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text());STOP=False
COUNTS={'counterfactual_outputs':0,'diagnosed_conditions':0}
def stop(*_):
 global STOP;STOP=True
def write(name,x):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def guard():
 if STOP:raise RuntimeError('SUPERVISOR_STOP')
 if sum(p.stat().st_blocks*512 for p in R.rglob('*') if p.is_file())>=3_000_000_000:raise RuntimeError('STORAGE_SOFT_STOP')
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def dice(p,y):
 p=p.double();y=y.double();return 100*(2*(p*y).sum((-2,-1))+1e-6)/((p+y).sum((-2,-1))+1e-6)
def rebuild(w,ps,res):
 wp=F.interpolate(w,size=(256,256),mode='bilinear',align_corners=False)
 return torch.logit((wp*(.5*ps+.5*ps[:,1:2])).sum(1).clamp(.001,.999))+res

def structure(prob,y):
 pred=(prob>.5).cpu().numpy();gt=(y>.5).cpu().numpy();d=dice(torch.from_numpy(pred),torch.from_numpy(gt)).tolist();boundary=[]
 for a,b in zip(pred,gt):
  if not a.any() and not b.any():boundary.append({'hd95':0.,'assd':0.,'empty_case':'both'});continue
  if not a.any() or not b.any():boundary.append({'hd95':None,'assd':None,'empty_case':'one'});continue
  sa=a^binary_erosion(a);sb=b^binary_erosion(b);dist=np.concatenate([distance_transform_edt(~sb)[sa],distance_transform_edt(~sa)[sb]])
  boundary.append({'hd95':float(np.percentile(dist,95)),'assd':float(dist.mean()),'empty_case':'neither'})
 def ratio(v):
  spans=[]
  for a in v:
   ids=np.where(a.any(1))[0];spans.append(int(ids[-1]-ids[0]+1) if len(ids) else 0)
  return spans[1]/max(spans[0],1)
 outside=int((pred[1]&~pred[0]).sum())
 return {'od':d[0],'oc':d[1],'dice':sum(d)/2,'vcdr_error':abs(ratio(pred)-ratio(gt)),'empty_heads':int(sum(not v.any() for v in pred)),'outside_pixels':outside,'outside_occurs':int(outside>0),'outside_fraction':outside/max(int(pred[1].sum()),1),'boundary':boundary}

def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 fusion=module('fusion',cfg['fusion_module']);core=module('core',cfg['expert_module']);signal.signal(signal.SIGTERM,stop)
 fit,dev=core.load_data(cfg);assert len(fit)==79 and len(dev)==20
 split=json.loads((R/'split.private.json').read_text());oldsplit=json.loads((Path(cfg['r12_root'])/'split.private.json').read_text());assert split['fit']==oldsplit['fit'] and split['dev']==oldsplit['dev']
 ref=torch.tensor(json.loads((Path(cfg['r12_root'])/'CANON_REFERENCE.json').read_text())['log_means']);newref=torch.cat([fusion.logmom(v[0][None]) for v in fit]).median(0).values;assert torch.allclose(ref,newref,atol=1e-7,rtol=0)
 sys.path.insert(0,cfg['backbone_code']);from networks.ResUnet import ResUnet
 base=ResUnet(resnet='resnet34',num_classes=2,pretrained=False).cuda();base.load_state_dict(torch.load(cfg['checkpoint'],weights_only=True,map_location='cpu'));base.eval().requires_grad_(False)
 write('ENVIRONMENT.json',{'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'target_reads':0,'training_updates':0,'fit':79,'dev':20})
 cache=R/'cache.private';cache.mkdir(exist_ok=True);repro=[];basics=[];labels512={}
 for j,row in enumerate(split['dev']):
  p=Path(cfg['data_root'])/row['mask']
  if not p.is_file():labels512[j]=None;continue
  y=np.array(Image.open(p).convert('L').resize((512,512),Image.Resampling.NEAREST));labels512[j]=torch.tensor(np.stack([y<255,y==0]))
 # Pass 1 validates the complete historical readout before any interpretation.
 with torch.no_grad():
  for seed in cfg['seeds']:
   expert=core.Model().cuda();expert.load_state_dict(torch.load(cfg['expert_checkpoints'][str(seed)],weights_only=True,map_location='cpu')['model']);expert.eval().requires_grad_(False)
   models={};expected={}
   for arm in ['full','control']:
    p=Path(cfg['r12_root'])/'runs'/f'{arm}_{seed}';m=fusion.Fusion().cuda();ck=torch.load(p/'best.pt',weights_only=True,map_location='cpu');m.load_state_dict(ck['model']);m.eval().requires_grad_(False);models[arm]=m
    rows=json.loads((p/'best_rows.private.json').read_text());expected[arm]={(v['content'],v['shift']):v for v in rows if v['order']==0 and v['arm']==('FULL' if arm=='full' else 'MATCHED_CONTROL')}
    # Existing duplicated sequence is checked without spending extra model calls.
    for v in rows:
     key=(v['content'],v['shift'])
     if v['order']==1 and v['arm']==('FULL' if arm=='full' else 'MATCHED_CONTROL'):assert abs(v['dice']-expected[arm][key]['dice'])<=.001
   ids=list(range(20));random.Random(8821).shuffle(ids)
   for kind in ['clean','gamma','color','blur']:
    for j in ids:
     guard();im,y,_=dev[j];x=core.corruption(im[None].cuda(),kind,.8);x,ps=fusion.experts(core,base,expert,x,ref);out,w,mix,res=models['full'](x,ps,False);ctrl=models['control'](x,ps,True)[0]
     assert torch.isfinite(out).all() and torch.isfinite(ctrl).all()
     for arm,z in [('full',out),('control',ctrl)]:
      dd=core.dice(z,y[None].cuda())[0].cpu();err=max(abs(dd[0].item()-expected[arm][j,kind]['od']),abs(dd[1].item()-expected[arm][j,kind]['oc']))
      repro.append(float(err));assert err<=.001,('REPRODUCTION_FAILED',seed,kind,j,arm,err)
     torch.save({'ps':ps[0].cpu(),'w':w[0].cpu(),'res':res[0].cpu(),'out':out[0].cpu(),'control':ctrl[0].cpu(),'y':y},cache/f'{seed}_{kind}_{j}.pt')
     write('RUN_STATE.json',{'stage':'REPRODUCING','seed':seed,'shift':kind,'completed_conditions':len(repro)//2,'counts':{**core.COUNTS,**fusion.COUNTS,**COUNTS}})
  write('REPRODUCTION.json',{'status':'PASS','max_head_dice_error_pp':max(repro),'conditions':160,'two_historical_orders_agree':True,'split_exact_rows_agree':True,'canonical_reference_agrees':True})
  creditrows=[];functional=[];raw=R/'actions.private.jsonl'
  # Stable file order, never selected by result.
  for f in sorted(cache.glob('*.pt')):
   guard();seed,kind,j=f.stem.split('_');j=int(j);c=torch.load(f,weights_only=True);ps=c['ps'].cuda();w=c['w'].cuda();res=c['res'].cuda();y=c['y'].cuda();out=c['out'].cuda()
   reconstructed=rebuild(w,ps,res);assert torch.allclose(reconstructed,out,atol=1e-5,rtol=1e-5);assert torch.equal(reconstructed>0,out>0)
   p0=out>0;d0=dice(p0,y);prop=.5*ps+.5*ps[:,1:2]
   preds={'FULL':out,'CONTROL':c['control'].cuda(),'SOURCE':torch.logit(ps[:,0].clamp(.001,.999)),'DIRECT':torch.logit(ps[:,1].clamp(.001,.999)),'CANON_SOURCE':torch.logit(ps[:,2].clamp(.001,.999)),'CANON_DIRECT':torch.logit(ps[:,3].clamp(.001,.999)),'MEAN':torch.logit(ps.mean(1).clamp(.001,.999)),'NO_RESIDUAL':out-res,'DIRECT_WEIGHTS':torch.logit(ps[:,1].clamp(.001,.999))+res}
   for name,z in preds.items():
    for resolution in [256,512]:
     if resolution==512 and labels512[j] is None:continue
     prob=z.sigmoid() if resolution==256 else F.interpolate(z.sigmoid()[None],size=(512,512),mode='bilinear',align_corners=False)[0]
     target=y if resolution==256 else labels512[j].cuda();basics.append({'seed':int(seed),'content':j,'shift':kind,'arm':name,'resolution':resolution,**structure(prob,target)})
   wp=F.interpolate(w,size=(256,256),mode='bilinear',align_corners=False);avg=wp.mean((-2,-1),keepdim=True).expand_as(w);global_z=rebuild(avg,ps,res)
   functional.append({'seed':int(seed),'content':j,'shift':kind,'mode':'GLOBAL','full_minus_readout':(d0-dice(global_z>0,y)).tolist(),'expert_mean_shift':(avg[:,:,0,0]-wp.mean((-2,-1))).abs().max().item()})
   for k in range(8):
    idx=torch.randperm(64,generator=torch.Generator().manual_seed(k)).cuda();ws=w.flatten(2)[:,:,idx].reshape_as(w);zz=rebuild(ws,ps,res);sp=F.interpolate(ws,size=(256,256),mode='bilinear',align_corners=False)
    functional.append({'seed':int(seed),'content':j,'shift':kind,'mode':'SHUFFLE','permutation':k,'full_minus_readout':(d0-dice(zz>0,y)).tolist(),'expert_mean_shift':(sp.mean((-2,-1))-wp.mean((-2,-1))).abs().max().item()})
   old=fusion.intervention_credit(ps[None],y[None])[0].double();g=torch.zeros(2,4,8,8,device='cuda',dtype=torch.float64);changed=torch.zeros_like(g,dtype=torch.int64)
   for h in range(2):
    for rr in range(64):
     guard();a,b=divmod(rr,8);ww=w[h:h+1].expand(4,-1,-1,-1).clone();ww[:,:,a,b]=torch.eye(4,device='cuda');up=F.interpolate(ww,size=(256,256),mode='bilinear',align_corners=False);zz=torch.logit((up*prop[h:h+1]).sum(1).clamp(.001,.999))+res[h];pp=zz>0;g[h,:,a,b]=dice(pp,y[h])-d0[h];changed[h,:,a,b]=(pp!=p0[h]).sum((-2,-1));COUNTS['counterfactual_outputs']+=4
     if rr in [0,27,63]:
      for e in [0,3]:
       wf=w.clone();wf[h,:,a,b]=torch.eye(4,device='cuda')[e];df=dice(rebuild(wf,ps,res)>0,y)[h]-d0[h];assert abs(float(df-g[h,e,a,b]))<=1e-4
   for h in range(2):
    o=old[h].cpu().numpy();v=g[h].cpu().numpy();ch=changed[h].cpu().numpy();pos=o>1e-6;neg=v< -1e-6;nonzero=ch>0;actualbest=np.maximum(0,v.max(0));oldbest=np.abs(o-o.max(0,keepdims=True))<=1e-6;actualset=np.abs(v-actualbest[None])<=1e-6
    # Average across all tied old maxima, with min/max regret also retained.
    regrets=actualbest[None]-v;regret=(regrets*oldbest).sum(0)/oldbest.sum(0)
    row={'seed':int(seed),'content':j,'shift':kind,'head':['OD','OC'][h],'old_positive':int(pos.sum()),'positive_to_negative':int((pos&neg).sum()),'old_positive_changed':int((pos&nonzero).sum()),'positive_to_negative_changed':int((pos&neg&nonzero).sum()),'unchanged':int((~nonzero).sum()),'changed_near_zero':int((nonzero&(np.abs(v)<=1e-6)).sum()),'nonzero_gain':int((np.abs(v)>1e-6).sum()),'topset_intersection':int((oldbest&actualset).any(0).sum()),'old_tied_regions':int((oldbest.sum(0)>1).sum()),'uninformative_regions':int((np.abs(v)<=1e-6).all(0).sum()),'mean_regret_pp':float(regret.mean()),'mean_best_tie_regret_pp':float(np.where(oldbest,regrets,np.inf).min(0).mean()),'mean_worst_tie_regret_pp':float(np.where(oldbest,regrets,-np.inf).max(0).mean()),'mean_positive_action_exec_pp':float(v[pos].mean()) if pos.any() else None}
    creditrows.append(row)
    with raw.open('a') as stream:stream.write(json.dumps({'seed':int(seed),'content':j,'shift':kind,'head':h,'old_credit':o.tolist(),'exec_credit':v.tolist(),'changed_pixels':ch.tolist()})+'\n')
   COUNTS['diagnosed_conditions']+=1
   write('functional.private.json',functional);write('credit.private.json',creditrows);write('metrics.private.json',basics)
   write('RUN_STATE.json',{'stage':'DIAGNOSTICS','completed_conditions':COUNTS['diagnosed_conditions'],'counts':{**core.COUNTS,**fusion.COUNTS,**COUNTS},'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
  write('FINISHED.json',{'status':'COMPLETE','counts':{**core.COUNTS,**fusion.COUNTS,**COUNTS},'missing_native_labels':sum(v is None for v in labels512.values()),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'target_reads':0,'outer_updates':0,'inner_grad':0})
if __name__=='__main__':
 try:main()
 except Exception as e:
  write('FAILURE.json',{'error':repr(e),'counts':COUNTS});raise
