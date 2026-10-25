import os,json,time,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('gamma_impl',Path(cfg['gamma_root'])/'src/worker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a,c,p,torch,np=b.a,b.c,b.p,b.torch,b.np

def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text());records=[]
    for source in p.DOMAINS:
        a.check();model,net=b.load(source);head=b.Gamma().cuda().eval();head.load_state_dict(torch.load(Path(cfg['gamma_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['head']);torch.manual_seed(92511)
        for index in split[source]['dev']:
            a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
            with torch.no_grad():
                for draw in range(4):
                    truth=torch.empty(1,4,device=x.device).uniform_(-1,1)*x.new_tensor([.25,.25,.25,.5])[None];raw=x.clamp_min(.001).pow(truth[:,3].exp()[:,None,None,None])*truth[:,:3].exp()[:,:,None,None];xx=raw.clamp(0,1);pars=net(xx);out=head(b.features(net,xx),pars);oracle=pars.clone();oracle[:,3]=truth[:,3]
                    rec={'source':source,'index':index,'draw':draw,'I_mse':float((pars[0,3]-truth[0,3]).square()),'G_mse':float((out[0,3]-truth[0,3]).square()),'high_clip_fraction':float((raw>1).float().mean()),'ZERO':a.dice(a.predict(model,xx),y)}
                    for arm,params in [('I',pars),('G',out),('O',oracle)]:rec[arm]=a.dice(a.predict(model,a.inverse(xx,params)),y)
                    if not all(np.isfinite(rec[arm]).all() for arm in ['ZERO','I','G','O']):raise FloatingPointError('Nonfinite dice')
                    records.append(rec)
            if len(records)%40==0:c.write(R/'RUN_STATE.json',{'phase':'random_source_diagnostic','source':source,'cases':len(records),'unix':time.time()})
        del model,net,head;torch.cuda.empty_cache()
    c.write(R/'records.private.json',records);summary={}
    for source in p.DOMAINS:
        rr=[r for r in records if r['source']==source];summary[source]={'n':len(rr),'means':{arm:float(np.mean([r[arm] for r in rr])) for arm in ['ZERO','I','G','O']},'gamma_mse_ratio':float(np.mean([r['G_mse'] for r in rr])/np.mean([r['I_mse'] for r in rr]))}
    delta=np.array([np.mean(r['G'])-np.mean(r['I']) for r in records]);improved=np.array([r['G_mse']<r['I_mse'] for r in records]);fraction=float(np.mean(delta[improved]<0)) if improved.any() else None
    c.write(R/'RESULTS.json',{'source_cases':len(records),'sources':summary,'G_minus_I_source_equal':float(np.mean([s['means']['G']-s['means']['I'] for s in summary.values()])),'O_minus_I_source_equal':float(np.mean([s['means']['O']-s['means']['I'] for s in summary.values()])),'pooled_p10_G_minus_I':float(np.quantile(delta,.1)),'pooled_decline_fraction':float(np.mean(delta<0)),'gamma_improved_count':int(improved.sum()),'gamma_improved_seg_decline_fraction':fraction,'risk_constraint_priority_rule_met':bool(fraction is not None and fraction>=.25),'target_evaluated':False})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE','unix':time.time(),'target_evaluated':False,'trained':False})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
