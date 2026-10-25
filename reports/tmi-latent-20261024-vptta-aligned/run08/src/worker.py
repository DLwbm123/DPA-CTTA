import os,json,time,random,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('filter_impl',Path(cfg['filter_root'])/'src/worker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a,c,p,torch,np=b.a,b.c,b.p,b.torch,b.np
nn=torch.nn;F=a.F
class Noise(nn.Module):
    def __init__(self,cov):
        super().__init__();self.net=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,4));nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor(np.log(np.diag(cov)),dtype=torch.float32))
    def forward(self,features):return self.net(features).clamp(-9,0)

def features(net,x):return net.net[:-1](F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False))
class Filter(b.Filter):
    def update(self,obs,var):
        std=np.sqrt(np.diag(self.R));corr=self.R/np.outer(std,std);rt=corr*np.sqrt(np.outer(var,var));pred=self.P+self.R/16;s=pred+rt;d=obs-self.m
        if d@np.linalg.solve(s,d)>13.3:self.m=obs.copy();self.P=rt.copy();self.resets+=1
        else:
            k=np.linalg.solve(s.T,pred.T).T;self.m+=k@d;self.P=(np.eye(4)-k)@pred;self.P=(self.P+self.P.T)/2
        return np.clip(self.m,[-.25,-.25,-.25,-.5],[.25,.25,.25,.5])

def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text());covs={s:np.array(v) for s,v in json.loads((Path(cfg['filter_root'])/'CALIBRATION.json').read_text())['covariances'].items()}
    test=Filter(np.eye(4)*.01);reference=b.Filter(np.eye(4)*.01)
    for _ in range(10):assert np.allclose(test.update(np.ones(4)*.1,np.ones(4)*.01),reference.update(np.ones(4)*.1))
    c.write(R/'SELF_CHECK.json',{'constant_variance_matches_reference':True})
    for source in p.DOMAINS:
        a.check();torch.manual_seed(92508);rng=random.Random(92508);model,net=b.load(source);del model
        for v in net.parameters():v.requires_grad_(False)
        head=Noise(covs[source]).cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);fit=[torch.from_numpy(p.read_image(cfg['data_root'],data[source][i])) for i in split[source]['fit']];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1201):
            a.check();x=torch.stack([fit[i] for i in rng.sample(range(len(fit)),8)]).cuda();aug,truth=a.augment(x);xx=torch.cat([x,aug]);truth=torch.cat([torch.zeros_like(truth),truth])
            with torch.no_grad():feat=features(net,xx);error=net(xx)-truth
            lv=head(feat);loss=.5*(error.square()*(-lv).exp()+lv).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite NLL')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:c.write(R/'RUN_STATE.json',{'phase':'training_uncertainty','source':source,'step':step,'nll':loss.item(),'unix':time.time()})
            if step%300==0:
                tmp=run/'latest.tmp';torch.save({'noise':head.state_dict(),'step':step},tmp);tmp.replace(run/'latest.pt')
        del net,head,opt,fit;torch.cuda.empty_cache()
    records=[];nlls=[]
    for source in p.DOMAINS:
        model,net=b.load(source);head=Noise(covs[source]).cuda().eval();head.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['noise']);state=Filter(covs[source])
        for condition in ('clean','fixed_appearance','color','blur'):
            for index in split[source]['dev']:
                a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda();xx=x;truth=np.zeros(4)
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1);truth=np.log([1.15,.9,1.05,.7])
                if condition=='color':xx=(x*x.new_tensor([.85,1.2,.95])[None,:,None,None]).clamp(0,1);truth=np.log([.85,1.2,.95,1.])
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    obs=net(xx)[0].cpu().numpy().astype(np.float64);var=head(features(net,xx)).exp()[0].cpu().numpy().astype(np.float64);filtered=state.update(obs,var);reset=Filter(covs[source]).update(obs,var)
                    vals={'ZERO':a.dice(a.predict(model,xx),y),'I':a.dice(b.apply(model,xx,obs),y),'F':a.dice(b.apply(model,xx,filtered),y),'RESET':a.dice(b.apply(model,xx,reset),y)}
                records.append({'source':source,'index':index,'condition':condition,**vals})
                if condition!='blur':
                    e=(obs-truth)**2;constant=np.diag(covs[source]);nlls.append({'source':source,'condition':condition,'learned':float(.5*np.mean(e/var+np.log(var))),'constant':float(.5*np.mean(e/constant+np.log(constant)))})
            c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'condition':condition,'cases':len(records),'unix':time.time()})
        del model,net,head;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records);c.write(R/'nll_records.private.json',nlls)
    means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ('ZERO','I','F','RESET')} for co in ('clean','fixed_appearance','color','blur')} for s in p.DOMAINS}
    delta=[np.mean([means[s][co]['F']-means[s][co]['I'] for co in ('fixed_appearance','color')]) for s in p.DOMAINS];resetdiff=np.mean([means[s][co]['F']-means[s][co]['RESET'] for s in p.DOMAINS for co in ('fixed_appearance','color')]);clean=np.mean([means[s]['clean']['F']-means[s]['clean']['ZERO'] for s in p.DOMAINS]);tail=np.quantile([np.mean(r['F'])-np.mean(r['I']) for r in records if r['condition'] in ('fixed_appearance','color')],.1);nll_delta=np.mean([r['learned']-r['constant'] for r in nlls])
    passed=bool(np.mean(delta)>=.1 and resetdiff>=.05 and min(delta)>=0 and clean>=-.1 and tail>=-1 and nll_delta<=-.05)
    c.write(R/'SOURCE_GATE.json',{'means':means,'filtered_minus_independent':float(np.mean(delta)),'filtered_minus_reset':float(resetdiff),'delta_by_source':dict(zip(p.DOMAINS,map(float,delta))),'clean_delta_vs_zero':float(clean),'delta_p10':float(tail),'learned_minus_constant_nll':float(nll_delta),'gate_pass':passed})
    if passed:
        results={arm:{} for arm in ('F','RESET')}
        for source in p.DOMAINS:
            model,net=b.load(source);head=Noise(covs[source]).cuda().eval();head.load_state_dict(torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['noise'])
            for arm in results:
                state=Filter(covs[source]);count=[0]
                def target(image):
                    a.check();x=torch.from_numpy(image[None]).cuda()
                    with torch.no_grad():
                        obs=net(x)[0].cpu().numpy().astype(np.float64);var=head(features(net,x)).exp()[0].cpu().numpy().astype(np.float64);pars=state.update(obs,var) if arm=='F' else Filter(covs[source]).update(obs,var);out=(-16*b.apply(model,x,pars)).sigmoid()[0].cpu().numpy()
                    count[0]+=1
                    if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'arm':arm,'processed':count[0],'unix':time.time()})
                    return out
                summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],target,a.calculate_metrics);results[arm][source]=summary;c.write(R/'runs'/source/(arm+'_detail.private.json'),detail)
            del model,net,head;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':{arm:p.aggregate(v) for arm,v in results.items()}})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
