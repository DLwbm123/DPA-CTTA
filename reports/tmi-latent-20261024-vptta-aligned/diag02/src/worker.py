import os,sys,json,time
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
sys.path.insert(0,str(Path(cfg['base_root'])/'src'))
import core as c
import protocol as p
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice(d,y):
    z=(-16*d).sigmoid()>=.5
    return ((2*(z*y).sum((-2,-1))+1e-6)/(z.sum((-2,-1))+y.sum((-2,-1))+1e-6))*100

def norm(gs): return sum(g.square().sum() for g in gs).sqrt()

def main():
    torch.set_num_threads(4); torch.manual_seed(0)
    data=p.manifest(cfg['data_root']); records=[]
    for source in p.DOMAINS:
        model=c.Model().cuda(); checkpoint=torch.load(Path(cfg['base_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        for par in model.parameters():par.requires_grad_(False)
        pp=checkpoint['prior']; prior=c.Prior(pp['means'].cuda(),pp['var'].cuda(),pp['weights'].cuda()).anchor()
        for k,v in pp.items(): setattr(prior,k,v.cuda())
        del checkpoint
        for index in np.linspace(0,len(data[source])-1,20,dtype=int):
            row=data[source][int(index)]
            x=torch.from_numpy(p.read_image(cfg['data_root'],row)[None]).cuda()
            y=torch.from_numpy(p.read_label(cfg['data_root'],row)[None].astype(np.float32)).cuda()
            dt=np.stack([(distance_transform_edt(1-z)-distance_transform_edt(z)).clip(-16,16)/16 for z in y[0].cpu().numpy()]).astype(np.float32)
            dt=torch.from_numpy(dt[None]).cuda()
            for condition in ('clean','fixed_appearance'):
                if c.STOP: raise InterruptedError('Guard stop')
                xx=x if condition=='clean' else (x.clamp_min(.001).pow(.7)*torch.tensor([1.15,.9,1.05],device='cuda')[None,:,None,None]).clamp(0,1)
                with torch.no_grad(): cc,ss=model.encode(xx); before=dice(model.head(cc,xx.shape[-2:]),y)
                cv=[v.detach().requires_grad_() for v in cc];sv=ss.detach().requires_grad_()
                obs=c.obs(xx,model.render(cv,sv,xx.shape[-2:])); supervised=c.segloss(model.head(cv,xx.shape[-2:]),y,dt)
                go=torch.autograd.grad(obs,cv+[sv],retain_graph=True)
                gy=torch.autograd.grad(supervised,cv)
                gp=torch.autograd.grad(cfg['lambda_prior']*prior.nll(sv),sv)[0]
                cosine=sum((a*b).sum() for a,b in zip(go[:3],gy))/(norm(go[:3])*norm(gy)).clamp_min(1e-20)
                ratio=gp.norm()/go[-1].norm().clamp_min(1e-20)
                ci,si,trace=c.infer(model,xx,cc,ss,prior)
                with torch.no_grad(): after=dice(model.head(ci,xx.shape[-2:]),y)
                records.append({'source':source,'source_index':int(index),'condition':condition,'zero_dice':before[0].cpu().tolist(),'adapted_dice':after[0].cpu().tolist(),'observation_vs_segmentation_gradient_cosine':cosine.item(),'weighted_prior_vs_observation_style_grad_ratio':ratio.item(),'trace':trace})
                c.write(R/'RUN_STATE.json',{'source':source,'completed':len(records),'total':200,'unix':time.time()})
        del model;torch.cuda.empty_cache()
    c.write(R/'records.private.json',records)
    result={}
    for condition in ('clean','fixed_appearance'):
        rr=[r for r in records if r['condition']==condition]
        delta=np.array([np.mean(r['adapted_dice'])-np.mean(r['zero_dice']) for r in rr])
        cos=np.array([r['observation_vs_segmentation_gradient_cosine'] for r in rr]); rat=np.array([r['weighted_prior_vs_observation_style_grad_ratio'] for r in rr])
        result[condition]={'n':len(rr),'dice_delta_mean':float(delta.mean()),'dice_delta_p10':float(np.quantile(delta,.1)),'harm_fraction':float((delta<0).mean()),'gradient_cosine_mean':float(cos.mean()),'negative_cosine_fraction':float((cos<0).mean()),'prior_grad_ratio_median':float(np.median(rat)),'prior_grad_ratio_p90':float(np.quantile(rat,.9)),'accepted_steps_mean':float(np.mean([sum(t['accepted'] for t in r['trace']) for r in rr]))}
    c.write(R/'RESULTS.json',result);c.write(R/'COMPLETE.json',{'unix':time.time(),'source_only':True,'records':200})
if __name__=='__main__':
    try: main()
    except BaseException as e:
        c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e)});raise
