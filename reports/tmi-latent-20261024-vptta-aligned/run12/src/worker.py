import os,json,time,random,copy,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('gamma_impl',Path(cfg['gamma_root'])/'src/worker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a,c,p,torch,np=b.a,b.c,b.p,b.torch,b.np
F=a.F

def medical(model,x,pars,y):
    prob=(-16*a.predict(model,a.inverse(x,pars))).sigmoid()
    return (1-(2*(prob*y).sum((-2,-1))+1e-6)/(prob.sum((-2,-1))+y.sum((-2,-1))+1e-6)).mean(1)

def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text())
    for source in p.DOMAINS:
        a.check();torch.manual_seed(92512);rng=random.Random(92512);model,net=b.load(source);control=b.Gamma().cuda();risk=copy.deepcopy(control);opts=[torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for h in [control,risk]];fit=[(torch.from_numpy(p.read_image(cfg['data_root'],data[source][i])),torch.from_numpy(p.read_label(cfg['data_root'],data[source][i]).astype(np.float32))) for i in split[source]['fit']];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        for step in range(1,1201):
            a.check();ids=rng.sample(range(len(fit)),2);x=torch.stack([fit[i][0] for i in ids]).cuda();y=torch.stack([fit[i][1] for i in ids]).cuda();aug,truth=a.augment(x);xx=torch.cat([x,aug]);yy=torch.cat([y,y]);truth=torch.cat([torch.zeros_like(truth),truth])
            with torch.no_grad():pars=net(xx);feat=b.features(net,xx);reference=medical(model,xx,pars,yy)
            pc=control(feat,pars);pr=risk(feat,pars)
            if step==1:assert torch.equal(pc,pr) and torch.equal(pr,pars)
            lc=F.mse_loss(pc[:,3],truth[:,3]);lrisk=medical(model,xx,pr,yy);hinge=(lrisk-reference).relu().mean();lr=F.mse_loss(pr[:,3],truth[:,3])+.1*lrisk.mean()+hinge
            for opt,loss,head in zip(opts,[lc,lr],[control,risk]):
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:c.write(R/'RUN_STATE.json',{'phase':'paired_risk_training','source':source,'step':step,'control_mse':lc.item(),'risk_loss':lr.item(),'hinge':hinge.item(),'unix':time.time()})
            if step%300==0:
                tmp=run/'latest.tmp';torch.save({'C':control.state_dict(),'R':risk.state_dict(),'step':step},tmp);tmp.replace(run/'latest.pt')
        del model,net,control,risk,opts,fit,lc,lr,lrisk;torch.cuda.empty_cache()
    records=[]
    for source in p.DOMAINS:
        model,net=b.load(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False);heads={arm:b.Gamma().cuda().eval() for arm in ['C','R']}
        for arm in heads:heads[arm].load_state_dict(saved[arm])
        for index in split[source]['dev']:
            a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
            for condition in ('clean','fixed_appearance','color','blur'):
                xx=x;truth=0.
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1);truth=float(np.log(.7))
                if condition=='color':xx=(x*x.new_tensor([.85,1.2,.95])[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    pars=net(xx);feat=b.features(net,xx);rec={'source':source,'index':index,'condition':condition,'ZERO':a.dice(a.predict(model,xx),y),'I':a.dice(a.predict(model,a.inverse(xx,pars)),y)}
                    if condition!='blur':rec['I_mse']=(pars[0,3].item()-truth)**2
                    for arm,head in heads.items():
                        out=head(feat,pars);assert torch.equal(out[:,:3],pars[:,:3]);rec[arm]=a.dice(a.predict(model,a.inverse(xx,out)),y)
                        if condition!='blur':rec[arm+'_mse']=(out[0,3].item()-truth)**2
                if not all(np.isfinite(rec[arm]).all() for arm in ('ZERO','I','C','R')):raise FloatingPointError('nonfinite source dice')
                records.append(rec)
        c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'cases':len(records),'unix':time.time()});del model,net,heads;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records);means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ('ZERO','I','C','R')} for co in ('clean','fixed_appearance','color','blur')} for s in p.DOMAINS}
    delta={s:float(np.mean([means[s][co]['R']-means[s][co]['I'] for co in ('fixed_appearance','color')])) for s in p.DOMAINS};control=float(np.mean([means[s][co]['R']-means[s][co]['C'] for s in p.DOMAINS for co in ('fixed_appearance','color')]));clean=float(np.mean([means[s]['clean']['R']-means[s]['clean']['I'] for s in p.DOMAINS]));tail=float(np.quantile([np.mean(r['R'])-np.mean(r['I']) for r in records if r['condition'] in ('fixed_appearance','color')],.1));rr=[r for r in records if 'I_mse' in r];ratio=float(np.mean([r['R_mse'] for r in rr])/np.mean([r['I_mse'] for r in rr]));passed=bool(np.mean(list(delta.values()))>=.1 and min(delta.values())>=0 and control>=0 and clean>=-.05 and tail>=-.1 and ratio<=.8)
    c.write(R/'SOURCE_GATE.json',{'means':means,'risk_minus_independent':float(np.mean(list(delta.values()))),'risk_minus_control':control,'delta_by_source':delta,'clean_delta_vs_independent':clean,'delta_p10':tail,'gamma_mse_ratio':ratio,'gate_pass':passed})
    if passed:
        results={arm:{} for arm in ['R','C']}
        for source in p.DOMAINS:
            model,net=b.load(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)
            for arm in results:
                head=b.Gamma().cuda().eval();head.load_state_dict(saved[arm]);count=[0]
                def target(image):
                    a.check();x=torch.from_numpy(image[None]).cuda()
                    with torch.no_grad():pars=net(x);out=(-16*a.predict(model,a.inverse(x,head(b.features(net,x),pars)))).sigmoid()[0].cpu().numpy()
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
