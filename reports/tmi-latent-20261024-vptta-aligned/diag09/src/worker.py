import os,json,time,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('appearance_impl',Path(cfg['appearance_root'])/'src/worker.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
c,p,torch,np=a.c,a.p,a.torch,a.np
F=a.F
ARMS=('ZERO','I','ORACLE','GAIN_ORACLE','GAMMA_ORACLE','CLEAN_REF')
def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text());records=[]
    for source in p.DOMAINS:
        a.check();model=a.base(source);net=a.Appearance().cuda().eval();net.load_state_dict(torch.load(Path(cfg['appearance_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['appearance'])
        for index in split[source]['dev']:
            a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
            with torch.no_grad():
                reference=a.dice(a.predict(model,x),y)
                for condition,truths in [('clean',[1,1,1,1]),('fixed_appearance',[1.15,.9,1.05,.7]),('color',[.85,1.2,.95,1])]:
                    truth=x.new_tensor([np.log(truths)]);raw=x if condition=='clean' else x.clamp_min(.001).pow(truth[:,3].exp()[:,None,None,None])*truth[:,:3].exp()[:,:,None,None];xx=raw.clamp(0,1);estimated=net(xx);gain=estimated.clone();gain[:,:3]=truth[:,:3];gamma=estimated.clone();gamma[:,3]=truth[:,3]
                    rec={'source':source,'index':index,'condition':condition,'CLEAN_REF':reference,'ZERO':reference if condition=='clean' else a.dice(a.predict(model,xx),y),'parameter_error':(estimated-truth)[0].cpu().tolist(),'high_clip_fraction':float((raw>1).float().mean()),'l1':{}}
                    for arm,pars in [('I',estimated),('ORACLE',truth),('GAIN_ORACLE',gain),('GAMMA_ORACLE',gamma)]:
                        restored=a.inverse(xx,pars);rec[arm]=a.dice(a.predict(model,restored),y);rec['l1'][arm]=F.l1_loss(restored,x).item()
                    if not all(np.isfinite(rec[k]).all() for k in ARMS):raise FloatingPointError('nonfinite dice')
                    records.append(rec)
            if len(records)%30==0:c.write(R/'RUN_STATE.json',{'phase':'source_component_diagnostic','source':source,'cases':len(records),'unix':time.time()})
        del model,net;torch.cuda.empty_cache()
    c.write(R/'records.private.json',records)
    means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ARMS} for co in ('clean','fixed_appearance','color')} for s in p.DOMAINS}
    components={}
    for arm in ('GAIN_ORACLE','GAMMA_ORACLE','ORACLE'):
        by={s:float(np.mean([means[s][co][arm]-means[s][co]['I'] for co in ('fixed_appearance','color')])) for s in p.DOMAINS};delta=[np.mean(r[arm])-np.mean(r['I']) for r in records if r['condition']!='clean'];components[arm]={'delta_by_source':by,'mean_source_equal_delta':float(np.mean(list(by.values()))),'pooled_p10':float(np.quantile(delta,.1)),'pooled_decline_fraction':float(np.mean(np.array(delta)<0)),'candidate_rule_met':bool(np.mean(list(by.values()))>=.1 and min(by.values())>=0)}
    condition_summary={}
    for co in ('clean','fixed_appearance','color'):
        rr=[r for r in records if r['condition']==co];err=np.array([r['parameter_error'] for r in rr]);condition_summary[co]={'pooled_parameter_bias':err.mean(0).tolist(),'pooled_parameter_mse':(err**2).mean(0).tolist(),'pooled_high_clip_fraction':float(np.mean([r['high_clip_fraction'] for r in rr])),'pooled_image_l1':{arm:float(np.mean([r['l1'][arm] for r in rr])) for arm in ('I','ORACLE','GAIN_ORACLE','GAMMA_ORACLE')},'oracle_minus_clean_reference_source_equal':float(np.mean([means[s][co]['ORACLE']-means[s][co]['CLEAN_REF'] for s in p.DOMAINS]))}
    c.write(R/'RESULTS.json',{'source_cases':len(records),'means':means,'components':components,'conditions':condition_summary,'source_only':True,'target_evaluated':False})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE','unix':time.time(),'target_evaluated':False,'trained':False})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
