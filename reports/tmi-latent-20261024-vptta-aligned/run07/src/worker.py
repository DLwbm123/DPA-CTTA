import os,sys,json,time,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('appearance_impl',Path(cfg['appearance_root'])/'src/worker.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
c,p,torch,np=a.c,a.p,a.torch,a.np
F=a.F

class Filter:
    def __init__(self,cov):self.R=cov.copy();self.m=np.zeros(4);self.P=4*cov.copy();self.resets=0
    def update(self,obs):
        pred=self.P+self.R/16;s=pred+self.R;d=obs-self.m
        if d@np.linalg.solve(s,d)>13.3:
            self.m=obs.copy();self.P=self.R.copy();self.resets+=1
        else:
            k=np.linalg.solve(s.T,pred.T).T;self.m=self.m+k@d;self.P=(np.eye(4)-k)@pred;self.P=(self.P+self.P.T)/2
        return np.clip(self.m,[-.25,-.25,-.25,-.5],[.25,.25,.25,.5])

def load(source):
    model=a.base(source);net=a.Appearance().cuda().eval();net.load_state_dict(torch.load(Path(cfg['appearance_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['appearance']);return model,net

def apply(model,x,pars):
    pp=torch.as_tensor(pars,dtype=x.dtype,device=x.device)[None];return a.predict(model,a.inverse(x,pp))

def main():
    test=Filter(np.eye(4)*.01);values=[test.update(np.ones(4)*.1)[0] for _ in range(20)]
    assert 0<values[0]<values[-1]<=.1
    jump=Filter(np.eye(4)*1e-4);assert np.allclose(jump.update(np.ones(4)*.2),.2) and jump.resets==1
    c.write(R/'SELF_CHECK.json',{'filter_convergence':True,'innovation_reset':True})
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text());covs={}
    for source in p.DOMAINS:
        a.check();torch.manual_seed(92507);model,net=load(source);fit=split[source]['fit'];indices=np.linspace(0,len(fit)-1,min(64,len(fit)),dtype=int);errors=[]
        with torch.no_grad():
            for j in indices:
                x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][fit[int(j)]])[None]).cuda()
                for _ in range(2):
                    xx,truth=a.augment(x);errors.append((net(xx)-truth)[0].cpu().numpy())
        ee=np.asarray(errors,dtype=np.float64);covs[source]=ee.T@ee/len(ee)+np.eye(4)*1e-4
        del model,net;torch.cuda.empty_cache()
    c.write(R/'CALIBRATION.json',{'covariances':{s:v.tolist() for s,v in covs.items()},'source_fit_only':True,'seed':92507})
    records=[]
    for source in p.DOMAINS:
        a.check();model,net=load(source);state=Filter(covs[source])
        for condition in ('clean','fixed_appearance','color','blur'):
            for index in split[source]['dev']:
                a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda();xx=x
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1)
                if condition=='color':xx=(x*x.new_tensor([.85,1.2,.95])[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    obs=net(xx)[0].cpu().numpy().astype(np.float64);filtered=state.update(obs);reset=Filter(covs[source]).update(obs)
                    vals={'ZERO':a.dice(a.predict(model,xx),y),'I':a.dice(apply(model,xx,obs),y),'F':a.dice(apply(model,xx,filtered),y),'RESET':a.dice(apply(model,xx,reset),y)}
                records.append({'source':source,'index':index,'condition':condition,**vals})
            c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'condition':condition,'cases':len(records),'unix':time.time()})
        del model,net;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records)
    means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ('ZERO','I','F','RESET')} for co in ('clean','fixed_appearance','color','blur')} for s in p.DOMAINS}
    delta=[np.mean([means[s][co]['F']-means[s][co]['I'] for co in ('fixed_appearance','color')]) for s in p.DOMAINS];resetdiff=np.mean([means[s][co]['F']-means[s][co]['RESET'] for s in p.DOMAINS for co in ('fixed_appearance','color')]);clean=np.mean([means[s]['clean']['F']-means[s]['clean']['ZERO'] for s in p.DOMAINS]);tail=np.quantile([np.mean(r['F'])-np.mean(r['I']) for r in records if r['condition'] in ('fixed_appearance','color')],.1)
    passed=bool(np.mean(delta)>=.1 and resetdiff>=.05 and min(delta)>=0 and clean>=-.1 and tail>=-1)
    c.write(R/'SOURCE_GATE.json',{'means':means,'filtered_minus_independent':float(np.mean(delta)),'filtered_minus_reset':float(resetdiff),'delta_by_source':dict(zip(p.DOMAINS,map(float,delta))),'clean_delta_vs_zero':float(clean),'delta_p10':float(tail),'gate_pass':passed})
    if passed:
        results={arm:{} for arm in ('F','RESET')}
        for source in p.DOMAINS:
            model,net=load(source)
            for arm in results:
                state=Filter(covs[source]);count=[0]
                def target(image):
                    a.check();x=torch.from_numpy(image[None]).cuda()
                    with torch.no_grad():
                        obs=net(x)[0].cpu().numpy().astype(np.float64);pars=state.update(obs) if arm=='F' else Filter(covs[source]).update(obs);out=(-16*apply(model,x,pars)).sigmoid()[0].cpu().numpy()
                    count[0]+=1
                    if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'arm':arm,'processed':count[0],'unix':time.time()})
                    return out
                summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],target,a.calculate_metrics);results[arm][source]=summary;c.write(R/'runs'/source/(arm+'_detail.private.json'),detail)
            del model,net;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':{arm:p.aggregate(v) for arm,v in results.items()}})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
