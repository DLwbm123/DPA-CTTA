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

class Appearance(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,4));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)).tanh()*x.new_tensor([.25,.25,.25,.5])[None]

def inverse(x,pars):
    gain=pars[:,:3].exp()[:,:,None,None];power=(-pars[:,3]).exp()[:,None,None,None]
    return (x/gain).clamp(.001,1).pow(power).clamp(0,1)

def augment(x):
    pars=torch.empty(len(x),4,device=x.device).uniform_(-1,1)*x.new_tensor([.25,.25,.25,.5])[None]
    return (x.clamp_min(.001).pow(pars[:,3].exp()[:,None,None,None])*pars[:,:3].exp()[:,:,None,None]).clamp(0,1),pars

def check():
    if c.STOP:raise InterruptedError('Guard stop')

def read(row):
    x=p.read_image(cfg['data_root'],row);y=p.read_label(cfg['data_root'],row).astype(np.float32)
    dt=np.stack([(distance_transform_edt(1-z)-distance_transform_edt(z)).clip(-16,16)/16 for z in y]).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (x,y,dt))

def base(source):
    model=c.Model().cuda();saved=torch.load(Path(cfg['base_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval()
    for v in model.parameters():v.requires_grad_(False)
    return model

def predict(model,x):
    cs,_=model.encode(x);return model.head(cs,x.shape[-2:])

def dice(d,y):
    q=(-16*d).sigmoid()>=.5;return (((2*(q*y).sum((-2,-1))+1e-6)/(q.sum((-2,-1))+y.sum((-2,-1))+1e-6))*100)[0].cpu().tolist()

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;data=p.manifest(cfg['data_root']);splits={}
    for source,rows in data.items():
        ids=list(range(len(rows)));random.Random(92504).shuffle(ids);n=round(.2*len(ids));splits[source]={'fit':ids[n:],'dev':ids[:n]}
    c.write(R/'split.private.json',splits)
    for source in p.DOMAINS:
        check();torch.manual_seed(92504);rng=random.Random(92504);model=base(source);net=Appearance().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        fit=[read(data[source][i]) for i in splits[source]['fit']];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1501):
            check();ii=rng.sample(range(len(fit)),4);x,y,dt=[torch.stack([fit[i][k] for i in ii]).cuda() for k in range(3)];aug,pars=augment(x)
            xx=torch.cat([x,aug]);yy=torch.cat([y,y]);dd=torch.cat([dt,dt]);truth=torch.cat([torch.zeros_like(pars),pars]);estimated=net(xx);restored=inverse(xx,estimated)
            parameter=F.mse_loss(estimated,truth);reconstruction=F.l1_loss(restored,torch.cat([x,x]));medical=c.segloss(predict(model,restored),yy,dd);loss=parameter+.5*reconstruction+medical
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:c.write(R/'RUN_STATE.json',{'phase':'training','source':source,'step':step,'loss':loss.item(),'parameter_mse':parameter.item(),'reconstruction':reconstruction.item(),'unix':time.time()})
            if step%300==0:
                tmp=run/'latest.tmp';torch.save({'appearance':net.state_dict(),'optimizer':opt.state_dict(),'step':step},tmp);tmp.replace(run/'latest.pt')
        del model,net,opt,fit,loss,restored;torch.cuda.empty_cache()
    records=[]
    for source in p.DOMAINS:
        check();model=base(source);net=Appearance().cuda().eval();net.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['appearance'])
        for index in splits[source]['dev']:
            check();x,y,_=read(data[source][index]);x=x[None].cuda();y=y[None].cuda()
            for condition in ('clean','fixed_appearance','blur'):
                xx=x;truth=x.new_tensor([[np.log(1.15),np.log(.9),np.log(1.05),np.log(.7)]])
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    pars=net(xx);restored=inverse(xx,pars);rec={'source':source,'index':index,'condition':condition,'ZERO':dice(predict(model,xx),y),'A':dice(predict(model,restored),y)}
                    if condition=='fixed_appearance':rec.update(ORACLE=dice(predict(model,inverse(xx,truth)),y),parameter_mse=F.mse_loss(pars,truth).item(),zero_parameter_mse=truth.square().mean().item(),image_l1=F.l1_loss(restored,x).item(),uncorrected_l1=F.l1_loss(xx,x).item())
                records.append(rec)
        c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'cases':len(records),'unix':time.time()});del model,net;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records)
    means={s:{co:{a:float(np.mean([r[a] for r in records if r['source']==s and r['condition']==co])) for a in ('ZERO','A')} for co in ('clean','fixed_appearance','blur')} for s in p.DOMAINS}
    delta=[np.mean([means[s][co]['A']-means[s][co]['ZERO'] for co in ('fixed_appearance','blur')]) for s in p.DOMAINS];clean=np.mean([means[s]['clean']['A']-means[s]['clean']['ZERO'] for s in p.DOMAINS]);tail=np.quantile([np.mean(r['A'])-np.mean(r['ZERO']) for r in records if r['condition']!='clean'],.1)
    rr=[r for r in records if r['condition']=='fixed_appearance'];parameter_ratio=np.mean([r['parameter_mse'] for r in rr])/np.mean([r['zero_parameter_mse'] for r in rr]);image_ratio=np.mean([r['image_l1'] for r in rr])/np.mean([r['uncorrected_l1'] for r in rr])
    passed=bool(np.mean(delta)>=.2 and min(delta)>=0 and clean>=-.1 and tail>=-1 and parameter_ratio<=.8 and image_ratio<=.9)
    c.write(R/'SOURCE_GATE.json',{'means':means,'perturbed_delta_by_source':dict(zip(p.DOMAINS,map(float,delta))),'perturbed_delta':float(np.mean(delta)),'clean_delta':float(clean),'perturbed_delta_p10':float(tail),'parameter_error_ratio':float(parameter_ratio),'reconstruction_error_ratio':float(image_ratio),'oracle_fixed_appearance':{s:float(np.mean([r['ORACLE'] for r in rr if r['source']==s])) for s in p.DOMAINS},'gate_pass':passed})
    if passed:
        results={}
        for source in p.DOMAINS:
            check();model=base(source);net=Appearance().cuda().eval();net.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['appearance']);count=[0]
            def target(image):
                check();x=torch.from_numpy(image[None]).cuda()
                with torch.no_grad():out=(-16*predict(model,inverse(x,net(x)))).sigmoid()[0].cpu().numpy()
                count[0]+=1
                if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'processed':count[0],'unix':time.time()})
                return out
            summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],target,calculate_metrics);results[source]=summary;c.write(R/'runs'/source/'A_detail.private.json',detail);del model,net;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':p.aggregate(results)})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
