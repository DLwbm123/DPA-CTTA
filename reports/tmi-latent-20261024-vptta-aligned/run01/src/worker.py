import os, sys, json, time, random, math
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['WORK_ROOT'])/'src'))
import core as c
import protocol as p
from metrics import calculate_metrics
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
R=c.ROOT; cfg=c.CFG

def check():
    if c.STOP: raise InterruptedError('Guard stop')

def load_fit(rows):
    fit=[]
    for row in rows:
        check()
        x=p.read_image(cfg['data_root'],row); y=p.read_label(cfg['data_root'],row).astype(np.float32)
        dt=np.stack([(distance_transform_edt(1-z)-distance_transform_edt(z)).clip(-16,16)/16 for z in y]).astype(np.float32)
        fit.append(tuple(torch.from_numpy(a) for a in (x,y,dt)))
    return fit

def loss(model,x,y,dt,prior,meta):
    xb,pars=c.paired(x)
    a,sa=model.encode(x); b,sb=model.encode(xb)
    da=model.head(a,x.shape[-2:]); db=model.head(b,x.shape[-2:])
    med=(c.segloss(da,y,dt)+c.segloss(db,y,dt))/2
    rec=(c.obs(x,model.render(a,sa,x.shape[-2:]))+c.obs(xb,model.render(b,sb,x.shape[-2:])))/2
    cross=(c.obs(xb,model.render(a,sb,x.shape[-2:]))+c.obs(x,model.render(b,sa,x.shape[-2:])))/2
    inv=sum(c.F.smooth_l1_loss(aa,bb) for aa,bb in zip(a,b))/3
    relative=c.F.mse_loss(model.relative_style(sb-sa),pars)
    total=med+.5*rec+.25*cross+.05*inv+.05*relative
    if meta:
        cc,ss,trace=c.infer(model,xb,b,sb,prior,True)
        dc,ds,tr=c.infer(model,x,a,sa,prior,True)
        outer=c.segloss(model.head(cc,x.shape[-2:]),y,dt)
        harm=c.F.relu(c.segloss(model.head(dc,x.shape[-2:]),y,dt)-c.segloss(da,y,dt).detach())
        total=total+.5*outer+.1*harm
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
    return total,med.detach().item()

def prior_save(pr):
    return {k:getattr(pr,k).detach().cpu() for k in ('means','var','weights','floor','anchor_mean','anchor_var')}

def prior_load(state):
    pr=c.Prior(state['means'].cuda(),state['var'].cuda(),state['weights'].cuda()).anchor()
    for k,v in state.items(): setattr(pr,k,v.cuda())
    return pr

def save(path,model,opt,epoch,prior):
    # Fixed checkpoint count, atomic replacement; no per-epoch accumulation.
    temp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'prior':prior_save(prior),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate()},temp)
    temp.replace(path)

def main():
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=p.manifest(cfg['data_root']); c.write(R/'manifest.private.json',data)
    # Formal full-objective preflight, no optimizer step or checkpoint reuse.
    torch.manual_seed(771); model=c.Model().cuda(); prior=c.Prior().anchor()
    fit=load_fit(data[p.DOMAINS[0]][:1]); x,y,dt=[a[None].cuda() for a in fit[0]]
    value,_=loss(model,x,y,dt,prior,True); value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    c.write(R/'PREFLIGHT.json',{'status':'PASS','full_meta_backward':True,'size':512,'peak_allocated':torch.cuda.max_memory_allocated(),'unix':time.time()})
    del model,prior,fit,x,y,dt,value; torch.cuda.empty_cache()
    for source in p.DOMAINS:
        check(); torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
        run=R/'runs'/source; run.mkdir(parents=True,exist_ok=True)
        model=c.Model().cuda(); prior=c.Prior().anchor()
        fit=load_fit(data[source]); opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001)
        for epoch in range(1,cfg['epochs']+1):
            check(); model.train(); meta=epoch>cfg['warm_epochs']; micro=1 if meta else 2
            lr=(cfg['meta_lr'] if meta else cfg['lr'])*(1-(epoch-1)/cfg['epochs'])**.9
            for group in opt.param_groups: group['lr']=lr
            ids=list(range(len(fit))); random.shuffle(ids); total=0.; medical=0.; n=0
            for offset in range(0,len(ids),cfg['batch']):
                check(); chosen=ids[offset:offset+cfg['batch']]; opt.zero_grad(set_to_none=True)
                for j in range(0,len(chosen),micro):
                    selected=chosen[j:j+micro]
                    x,y,dt=[torch.stack([fit[i][k] for i in selected]).cuda() for k in range(3)]
                    # Source-only spatial augmentation, identical across views and labels.
                    dims=[d for d in (-1,-2) if random.random()<.5]
                    if dims: x,y,dt=[v.flip(dims) for v in (x,y,dt)]
                    value,med=loss(model,x,y,dt,prior,meta)
                    (value*len(selected)/len(chosen)).backward()
                    total+=value.detach().item()*len(selected); medical+=med*len(selected); n+=len(selected)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); opt.step()
                del value,x,y,dt
            if epoch==cfg['warm_epochs'] or epoch==cfg['epochs']:
                model.eval()
                with torch.no_grad(): prior=c.calibrate(model,fit)
            state={'source':source,'epoch':epoch,'phase':'meta' if meta else 'warm','loss':total/n,'medical':medical/n,'lr':lr,'unix':time.time()}
            c.write(R/'RUN_STATE.json',state); print(json.dumps(state),flush=True)
            if epoch%10==0 or epoch==cfg['epochs']: save(run/'latest.pt',model,opt,epoch,prior)
        c.write(run/'TRAINED.json',{'epochs':cfg['epochs'],'source_n':len(fit),'seed':cfg['seed']})
        del fit,model,opt,prior; torch.cuda.empty_cache()
    c.write(R/'ALL_TRAINED.json',{'unix':time.time(),'sources':list(p.DOMAINS)})
    results={arm:{} for arm in ('ZERO','INDEPENDENT','CONTINUAL')}
    for source in p.DOMAINS:
        check(); run=R/'runs'/source
        saved=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        model=c.Model().cuda(); model.load_state_dict(saved['model']); model.eval()
        for v in model.parameters(): v.requires_grad_(False)
        base=prior_load(saved['prior']); del saved
        _,rows=p.scenario(data,source)
        for arm in results:
            current=base.copy().anchor(); count=[0]
            def predict(image):
                nonlocal current
                check(); x=torch.from_numpy(image[None]).cuda()
                if arm=='INDEPENDENT': current=base.copy().anchor()
                with torch.no_grad(): cc,ss=model.encode(x)
                if arm!='ZERO':
                    cc,ss,trace=c.infer(model,x,cc,ss,current)
                with torch.no_grad():
                    prediction=(-16*model.head(cc,x.shape[-2:])).sigmoid()[0].cpu().numpy()
                    if arm=='CONTINUAL' and any(t['accepted'] for t in trace): current.update(ss.detach())
                count[0]+=1
                if count[0]%100==0: c.write(R/'RUN_STATE.json',{'phase':'evaluation','source':source,'arm':arm,'processed':count[0],'total':len(rows),'unix':time.time()})
                return prediction
            summary,detail=p.evaluate(cfg['data_root'],rows,predict,calculate_metrics)
            c.write(run/(arm+'_detail.private.json'),detail); c.write(run/(arm+'_summary.json'),summary)
            results[arm][source]=summary
        del model,base,current; torch.cuda.empty_cache()
    c.write(R/'RESULTS.json',{'sources':results,'average':{a:p.aggregate(s) for a,s in results.items()},'seed':cfg['seed']})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE','unix':time.time(),'peak_allocated':torch.cuda.max_memory_allocated()})

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        c.write(R/'FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'unix':time.time()})
        raise
