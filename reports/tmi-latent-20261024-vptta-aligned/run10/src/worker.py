import os,json,time,random,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('appearance_impl',Path(cfg['appearance_root'])/'src/worker.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
c,p,torch,np=a.c,a.p,a.torch,a.np
nn,F=torch.nn,a.F
class Gamma(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(115,64),nn.ReLU(),nn.Linear(64,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,pars):
        gamma=(pars[:,3:4]+.25*self.net(features).tanh()).clamp(-.5,.5)
        return torch.cat([pars[:,:3],gamma],1)
def features(net,x):
    with torch.no_grad():
        feat=net.net[:-1](F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False));v=F.interpolate(x,size=(64,64),mode='bilinear',align_corners=False).clamp_min(.001).log().flatten(2);q=torch.quantile(v,torch.linspace(.05,.95,17,device=x.device),dim=2).permute(1,2,0);q=q-q[:,:,8:9]
        return torch.cat([feat,q.flatten(1)],1)
def load(source):
    model=a.base(source);net=a.Appearance().cuda().eval();net.load_state_dict(torch.load(Path(cfg['appearance_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['appearance'])
    for v in net.parameters():v.requires_grad_(False)
    return model,net
def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text())
    for source in p.DOMAINS:
        a.check();torch.manual_seed(92510);rng=random.Random(92510);model,net=load(source);del model;head=Gamma().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);fit=[torch.from_numpy(p.read_image(cfg['data_root'],data[source][i])) for i in split[source]['fit']];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1201):
            a.check();x=torch.stack([fit[i] for i in rng.sample(range(len(fit)),8)]).cuda();aug,truth=a.augment(x);xx=torch.cat([x,aug]);truth=torch.cat([torch.zeros_like(truth),truth])
            with torch.no_grad():pars=net(xx);feat=features(net,xx)
            out=head(feat,pars)
            if step==1:assert torch.equal(out[:,:3],pars[:,:3]) and torch.equal(out[:,3],pars[:,3])
            loss=F.mse_loss(out[:,3],truth[:,3]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:c.write(R/'RUN_STATE.json',{'phase':'training_gamma','source':source,'step':step,'gamma_mse':loss.item(),'unix':time.time()})
            if step%300==0:
                tmp=run/'latest.tmp';torch.save({'head':head.state_dict(),'step':step},tmp);tmp.replace(run/'latest.pt')
        del net,head,opt,fit;torch.cuda.empty_cache()
    records=[]
    for source in p.DOMAINS:
        model,net=load(source);head=Gamma().cuda().eval();head.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['head'])
        for index in split[source]['dev']:
            a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
            for condition in ('clean','fixed_appearance','color','blur'):
                xx=x;truth=0.
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1);truth=float(np.log(.7))
                if condition=='color':xx=(x*x.new_tensor([.85,1.2,.95])[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    pars=net(xx);out=head(features(net,xx),pars);assert torch.equal(out[:,:3],pars[:,:3]);rec={'source':source,'index':index,'condition':condition,'ZERO':a.dice(a.predict(model,xx),y),'I':a.dice(a.predict(model,a.inverse(xx,pars)),y),'G':a.dice(a.predict(model,a.inverse(xx,out)),y)}
                    if condition!='blur':rec.update(I_mse=(pars[0,3].item()-truth)**2,G_mse=(out[0,3].item()-truth)**2)
                if not all(np.isfinite(rec[arm]).all() for arm in ('ZERO','I','G')):raise FloatingPointError('nonfinite source dice')
                records.append(rec)
        c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'cases':len(records),'unix':time.time()});del model,net,head;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records);means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ('ZERO','I','G')} for co in ('clean','fixed_appearance','color','blur')} for s in p.DOMAINS}
    delta={s:float(np.mean([means[s][co]['G']-means[s][co]['I'] for co in ('fixed_appearance','color')])) for s in p.DOMAINS};clean=float(np.mean([means[s]['clean']['G']-means[s]['clean']['I'] for s in p.DOMAINS]));tail=float(np.quantile([np.mean(r['G'])-np.mean(r['I']) for r in records if r['condition'] in ('fixed_appearance','color')],.1));rr=[r for r in records if 'I_mse' in r];ratio=float(np.mean([r['G_mse'] for r in rr])/np.mean([r['I_mse'] for r in rr]));passed=bool(np.mean(list(delta.values()))>=.1 and min(delta.values())>=0 and clean>=-.05 and tail>=-1 and ratio<=.8)
    c.write(R/'SOURCE_GATE.json',{'means':means,'gamma_minus_independent':float(np.mean(list(delta.values()))),'delta_by_source':delta,'clean_delta_vs_independent':clean,'delta_p10':tail,'gamma_mse_ratio':ratio,'gate_pass':passed})
    if passed:
        results={}
        for source in p.DOMAINS:
            model,net=load(source);head=Gamma().cuda().eval();head.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['head']);count=[0]
            def target(image):
                a.check();x=torch.from_numpy(image[None]).cuda()
                with torch.no_grad():pars=net(x);out=(-16*a.predict(model,a.inverse(x,head(features(net,x),pars)))).sigmoid()[0].cpu().numpy()
                count[0]+=1
                if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'processed':count[0],'unix':time.time()})
                return out
            summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],target,a.calculate_metrics);results[source]=summary;c.write(R/'runs'/source/'G_detail.private.json',detail);del model,net,head;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':p.aggregate(results)})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
