import os,sys,json,time,random,copy
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

class Corrector(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.ModuleList()
        for channels in (16,32,64):
            net=nn.Sequential(nn.Conv2d(2*channels,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,channels,1))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias);self.layers.append(net)
    def forward(self,cs,gs,with_gradient):
        out=[]
        for x,g,net in zip(cs,gs,self.layers):
            scale=x.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.1)
            gn=g/g.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-8) if with_gradient else torch.zeros_like(g)
            out.append(x+.1*scale*net(torch.cat([x/scale,gn],1)).tanh())
        return out

def check():
    if c.STOP:raise InterruptedError('Guard stop')

def encode_gradient(model,x):
    with torch.no_grad():cs,s=model.encode(x)
    cv=[v.detach().requires_grad_() for v in cs]
    obs=c.obs(x,model.render(cv,s,x.shape[-2:]))
    gs=torch.autograd.grad(obs,cv)
    return [v.detach() for v in cs],[g.detach() for g in gs]

def augment(x):
    gamma=torch.empty(len(x),1,1,1,device=x.device).uniform_(-.5,.5).exp()
    gain=torch.empty(len(x),3,1,1,device=x.device).uniform_(-.25,.25).exp()
    shade=F.interpolate(torch.randn(len(x),1,4,4,device=x.device)*.04,size=x.shape[-2:],mode='bilinear',align_corners=False)
    return (x.clamp_min(.001).pow(gamma)*gain*(1+shade)).clamp(0,1)

def read(row):
    x=p.read_image(cfg['data_root'],row);y=p.read_label(cfg['data_root'],row).astype(np.float32)
    dt=np.stack([(distance_transform_edt(1-z)-distance_transform_edt(z)).clip(-16,16)/16 for z in y]).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (x,y,dt))

def load_model(source):
    model=c.Model().cuda();saved=torch.load(Path(cfg['base_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval()
    for v in model.parameters():v.requires_grad_(False)
    return model

def dc(d,y):
    z=(-16*d).sigmoid()>=.5
    return ((2*(z*y).sum((-2,-1))+1e-6)/(z.sum((-2,-1))+y.sum((-2,-1))+1e-6))*100

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=p.manifest(cfg['data_root']);splits={}
    for source,rows in data.items():
        ids=list(range(len(rows)));random.Random(92503).shuffle(ids);n=round(len(ids)*.2)
        splits[source]={'fit':ids[n:],'dev':ids[:n]}
        assert not set(ids[n:])&set(ids[:n])
    c.write(R/'split.private.json',splits)
    for source in p.DOMAINS:
        check();torch.manual_seed(92503);rng=random.Random(92503)
        model=load_model(source);g=Corrector().cuda();d=copy.deepcopy(g)
        controllers={'G':g,'D':d};opts={k:torch.optim.AdamW(v.parameters(),lr=.001,weight_decay=.0001) for k,v in controllers.items()}
        fit=[read(data[source][i]) for i in splits[source]['fit']]
        run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1201):
            check();ii=rng.sample(range(len(fit)),2)
            x,y,dt=[torch.stack([fit[i][k] for i in ii]).cuda() for k in range(3)]
            xx=torch.cat([x,augment(x)]);yy=torch.cat([y,y]);dd=torch.cat([dt,dt])
            cs,gs=encode_gradient(model,xx);values={}
            for name,controller in controllers.items():
                corrected=controller(cs,gs,name=='G'); pred=model.head(corrected,xx.shape[-2:])
                match=sum(((z[2:]-ref[:2])/ref[:2].square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.1)).abs().mean() for z,ref in zip(corrected,cs))/3
                loss=c.segloss(pred,yy,dd)+.05*match
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                opts[name].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(controller.parameters(),5.,error_if_nonfinite=True);opts[name].step();values[name]=loss.item()
            if step==1 or step%100==0:
                c.write(R/'RUN_STATE.json',{'phase':'train','source':source,'step':step,'loss':values,'unix':time.time()});print(json.dumps({'source':source,'step':step,'loss':values}),flush=True)
            if step%200==0:
                temp=run/'latest.tmp';torch.save({'controllers':{k:v.state_dict() for k,v in controllers.items()},'optimizers':{k:v.state_dict() for k,v in opts.items()},'step':step,'rng':rng.getstate(),'cuda_rng':torch.cuda.get_rng_state()},temp);temp.replace(run/'latest.pt')
        del model,g,d,controllers,opts,fit,cs,gs,corrected,pred,loss;torch.cuda.empty_cache()
    records=[]
    for source in p.DOMAINS:
        check();model=load_model(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)
        controllers={k:Corrector().cuda().eval() for k in ('G','D')}
        for k,v in controllers.items():v.load_state_dict(saved['controllers'][k])
        del saved
        for index in splits[source]['dev']:
            x,y,_=read(data[source][index]);x=x[None].cuda();y=y[None].cuda()
            for condition in ('clean','fixed_appearance','blur'):
                check();xx=x
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*torch.tensor([1.15,.9,1.05],device='cuda')[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                cs,gs=encode_gradient(model,xx)
                with torch.no_grad():
                    vals={'ZERO':dc(model.head(cs,xx.shape[-2:]),y)[0].cpu().tolist()}
                    for k,v in controllers.items():vals[k]=dc(model.head(v(cs,gs,k=='G'),xx.shape[-2:]),y)[0].cpu().tolist()
                records.append({'source':source,'index':index,'condition':condition,'dice':vals})
        c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'cases':len(records),'unix':time.time()})
        del model,controllers;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records)
    means={s:{cond:{arm:float(np.mean([r['dice'][arm] for r in records if r['source']==s and r['condition']==cond])) for arm in ('ZERO','G','D')} for cond in ('clean','fixed_appearance','blur')} for s in p.DOMAINS}
    perturb=[np.mean([means[s][co]['G']-means[s][co]['ZERO'] for co in ('fixed_appearance','blur')]) for s in p.DOMAINS]
    mech=np.mean([means[s][co]['G']-means[s][co]['D'] for s in p.DOMAINS for co in ('fixed_appearance','blur')])
    clean=np.mean([means[s]['clean']['G']-means[s]['clean']['ZERO'] for s in p.DOMAINS])
    tail=np.quantile([np.mean(r['dice']['G'])-np.mean(r['dice']['ZERO']) for r in records if r['condition']!='clean'],.1)
    gate=bool(np.mean(perturb)>=.2 and mech>=.05 and min(perturb)>=0 and clean>=-.1 and tail>=-1)
    report={'means':means,'perturbed_delta_by_source':dict(zip(p.DOMAINS,map(float,perturb))),'perturbed_delta':float(np.mean(perturb)),'gradient_vs_direct':float(mech),'clean_delta':float(clean),'perturbed_delta_p10':float(tail),'gate_pass':gate,'base_model_seen_all_source':True}
    c.write(R/'SOURCE_GATE.json',report)
    if gate:
        results={k:{} for k in ('G','D')}
        for source in p.DOMAINS:
            check();model=load_model(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)
            for arm in ('G','D'):
                controller=Corrector().cuda().eval();controller.load_state_dict(saved['controllers'][arm]);count=[0]
                def predict(image):
                    check();xx=torch.from_numpy(image[None]).cuda();cs,gs=encode_gradient(model,xx)
                    with torch.no_grad():out=(-16*model.head(controller(cs,gs,arm=='G'),xx.shape[-2:])).sigmoid()[0].cpu().numpy()
                    count[0]+=1
                    if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'arm':arm,'processed':count[0],'unix':time.time()})
                    return out
                summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],predict,calculate_metrics)
                results[arm][source]=summary;c.write(R/'runs'/source/(arm+'_detail.private.json'),detail)
            del model,controller,saved;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':{k:p.aggregate(v) for k,v in results.items()}})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if gate else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':gate})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
