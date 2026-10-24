import os,sys,json,time,random
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
sys.path.insert(0,str(Path(cfg['base_root'])/'src'))
import core as c
import protocol as p
from metrics import calculate_metrics
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class Energy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(5,32,5,2,2),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,1,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,1,1))
    def forward(self,x,q):
        v=torch.cat([F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False),F.interpolate(q,size=(128,128),mode='bilinear',align_corners=False)],1)
        return self.net(v).mean((1,2,3))

def check():
    if c.STOP:raise InterruptedError('Guard stop')

def read(row):
    x=p.read_image(cfg['data_root'],row);y=p.read_label(cfg['data_root'],row).astype(np.float32)
    dt=np.stack([(distance_transform_edt(1-z)-distance_transform_edt(z)).clip(-16,16)/16 for z in y]).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (x,y,dt))

def augment(x):
    gamma=torch.empty(len(x),1,1,1,device=x.device).uniform_(-.5,.5).exp();gain=torch.empty(len(x),3,1,1,device=x.device).uniform_(-.25,.25).exp()
    shade=F.interpolate(torch.randn(len(x),1,4,4,device=x.device)*.04,size=x.shape[-2:],mode='bilinear',align_corners=False)
    return (x.clamp_min(.001).pow(gamma)*gain*(1+shade)).clamp(0,1)

def negative(dt,step):
    if step%2:
        off=torch.empty(len(dt),2,1,1,device=dt.device).uniform_(.03,.12);sign=torch.randint(0,2,off.shape,device=dt.device)*2-1
        return dt+off*sign
    shifts=(torch.randint(3,13,(2,))* (torch.randint(0,2,(2,))*2-1)).tolist();return dt.roll(tuple(shifts),(-2,-1))

def base(source):
    model=c.Model().cuda();saved=torch.load(Path(cfg['base_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval()
    for v in model.parameters():v.requires_grad_(False)
    return model

def logits(model,x):
    with torch.no_grad():cs,_=model.encode(x);return -16*model.head(cs,x.shape[-2:])

def adapt(energy,x,z0):
    z=z0.detach();steps=0
    def objective(a):return energy(x,a.sigmoid()).mean()+.01*(a-z0).square().mean()
    for _ in range(3):
        z=z.detach().requires_grad_();value=objective(z);g=torch.autograd.grad(value,z)[0]
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite energy gradient')
        direction=.25*g/g.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8)
        accepted=False
        for shrink in (1.,.5):
            trial=torch.maximum(torch.minimum(z-shrink*direction,z0+1),z0-1).detach()
            with torch.no_grad():new=objective(trial)
            if new<=value.detach()+1e-7:z=trial;accepted=True;steps+=1;break
        if not accepted:break
    return z.detach(),steps

def dice(z,y):
    q=z.sigmoid()>=.5;return (((2*(q*y).sum((-2,-1))+1e-6)/(q.sum((-2,-1))+y.sum((-2,-1))+1e-6))*100)[0].cpu().tolist()

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;data=p.manifest(cfg['data_root']);splits={}
    for source,rows in data.items():
        ids=list(range(len(rows)));random.Random(92504).shuffle(ids);n=round(.2*len(ids));splits[source]={'fit':ids[n:],'dev':ids[:n]}
    c.write(R/'split.private.json',splits)
    for source in p.DOMAINS:
        check();torch.manual_seed(92504);rng=random.Random(92504);energy=Energy().cuda();opt=torch.optim.AdamW(energy.parameters(),lr=.0003,weight_decay=.0001)
        fit=[read(data[source][i]) for i in splits[source]['fit']];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1501):
            check();ii=rng.sample(range(len(fit)),4);x,_,dt=[torch.stack([fit[i][k] for i in ii]).cuda() for k in range(3)]
            xx=torch.cat([x,augment(x)]);dd=torch.cat([dt,dt]);q=(-16*dd).sigmoid(); zbad=(-16*negative(dd,step)).detach().requires_grad_(); bad=zbad.sigmoid()
            ep=energy(xx,q);en=energy(xx,bad)
            gradient=torch.autograd.grad(en.sum(),zbad,create_graph=True,retain_graph=True)[0]
            band=((dd.abs()<.25)|(zbad.detach().abs()<4)).float()
            g=(gradient*band).flatten(1); target=((zbad.detach()+16*dd).clamp(-4,4)*band).flatten(1)
            cosine=(g*target).sum(1)/(g.norm(dim=1)*target.norm(dim=1)).clamp_min(1e-12)
            direction_loss=(1-cosine).mean()
            loss=F.softplus(1+ep-en).mean()+.001*(ep.square()+en.square()).mean()+direction_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(energy.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:c.write(R/'RUN_STATE.json',{'phase':'training','source':source,'step':step,'loss':loss.item(),'direction_cosine':cosine.detach().mean().item(),'unix':time.time()})
            if step%300==0:
                tmp=run/'latest.tmp';torch.save({'energy':energy.state_dict(),'optimizer':opt.state_dict(),'step':step},tmp);tmp.replace(run/'latest.pt')
        del fit,energy,opt;torch.cuda.empty_cache()
    records=[];ranks={}
    for source in p.DOMAINS:
        check();torch.manual_seed(92504);model=base(source);energy=Energy().cuda().eval();energy.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['energy'])
        for v in energy.parameters():v.requires_grad_(False)
        dev=[read(data[source][i]) for i in splits[source]['dev']];correct=[]
        for i,(x,y,dt) in enumerate(dev):
            x=x[None].cuda();y=y[None].cuda();dt=dt[None].cuda();other=dev[(i+1)%len(dev)][0][None].cuda()
            with torch.no_grad():correct.append(bool((energy(x,(-16*dt).sigmoid())<energy(x,(-16*negative(dt,i+1)).sigmoid())).item()))
            for condition in ('clean','fixed_appearance','blur'):
                check();xx=x;oo=other
                if condition=='fixed_appearance':
                    gain=torch.tensor([1.15,.9,1.05],device='cuda')[None,:,None,None];xx=(x.clamp_min(.001).pow(.7)*gain).clamp(0,1);oo=(other.clamp_min(.001).pow(.7)*gain).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4);oo=F.avg_pool2d(other,9,1,4)
                z=logits(model,xx);zz,n=adapt(energy,xx,z);zm,_=adapt(energy,oo,z)
                records.append({'source':source,'index':i,'condition':condition,'ZERO':dice(z,y),'E':dice(zz,y),'MISMATCH':dice(zm,y),'accepted':n})
            if (i+1)%20==0:c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'case':i+1,'unix':time.time()})
        ranks[source]=float(np.mean(correct));del dev,model,energy;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records)
    means={s:{co:{a:float(np.mean([r[a] for r in records if r['source']==s and r['condition']==co])) for a in ('ZERO','E','MISMATCH')} for co in ('clean','fixed_appearance','blur')} for s in p.DOMAINS}
    delta=[np.mean([means[s][co]['E']-means[s][co]['ZERO'] for co in ('fixed_appearance','blur')]) for s in p.DOMAINS]
    clean=np.mean([means[s]['clean']['E']-means[s]['clean']['ZERO'] for s in p.DOMAINS]);mechanism=np.mean([means[s][co]['E']-means[s][co]['MISMATCH'] for s in p.DOMAINS for co in ('fixed_appearance','blur')]);tail=np.quantile([np.mean(r['E'])-np.mean(r['ZERO']) for r in records if r['condition']!='clean'],.1)
    passed=bool(min(ranks.values())>=.75 and np.mean(delta)>=.2 and mechanism>=.05 and min(delta)>=0 and clean>=-.1 and tail>=-1)
    c.write(R/'SOURCE_GATE.json',{'means':means,'ranking_accuracy':ranks,'perturbed_delta_by_source':dict(zip(p.DOMAINS,map(float,delta))),'perturbed_delta':float(np.mean(delta)),'matched_minus_mismatch':float(mechanism),'clean_delta':float(clean),'perturbed_delta_p10':float(tail),'gate_pass':passed})
    if passed:
        results={}
        for source in p.DOMAINS:
            check();model=base(source);energy=Energy().cuda().eval();energy.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['energy'])
            for v in energy.parameters():v.requires_grad_(False)
            count=[0]
            def predict(image):
                check();x=torch.from_numpy(image[None]).cuda();z,_=adapt(energy,x,logits(model,x));count[0]+=1
                if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'processed':count[0],'unix':time.time()})
                return z.sigmoid()[0].cpu().numpy()
            summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],predict,calculate_metrics);results[source]=summary;c.write(R/'runs'/source/'E_detail.private.json',detail)
            del model,energy;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':p.aggregate(results)})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
