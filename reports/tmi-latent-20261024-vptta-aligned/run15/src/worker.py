import os,json,time,importlib.util
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);cfg=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('gamma_impl',Path(cfg['gamma_root'])/'src/worker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a,c,p,torch,np=b.a,b.c,b.p,b.torch,b.np
nn,F=torch.nn,a.F
class Selector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(118,32),nn.ReLU(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)[:,0]
def load(source):
    model,net=b.load(source);head=b.Gamma().cuda().eval();head.load_state_dict(torch.load(Path(cfg['gamma_root'])/'runs'/source/'latest.pt',map_location='cpu',weights_only=False)['head']);return model,net,head
def candidates(model,net,head,x):
    feat=b.features(net,x);pars=net(x);new=head(feat,pars);di=a.predict(model,a.inverse(x,pars));dg=a.predict(model,a.inverse(x,new));ents=[]
    for d in [di,dg]:
        q=(-16*d).sigmoid().clamp(1e-6,1-1e-6);ents.append((-(q*q.log()+(1-q)*(1-q).log())).mean((1,2,3))[:,None])
    return torch.cat([feat,new[:,3:4]-pars[:,3:4],*ents],1),di,dg

def main():
    torch.set_num_threads(4);data=p.manifest(cfg['data_root']);split=json.loads((Path(cfg['appearance_root'])/'split.private.json').read_text())
    for source in p.DOMAINS:
        a.check();model,net,head=load(source);torch.manual_seed(92515);features=[];labels=[];run=R/'runs'/source;run.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            for index in split[source]['fit']:
                a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
                for draw in range(3):
                    xx=x if draw==0 else a.augment(x)[0];f,di,dg=candidates(model,net,head,xx);features.append(f[0].cpu());labels.append(float(np.mean(a.dice(dg,y))-np.mean(a.dice(di,y))))
                if len(labels)%60==0:c.write(R/'RUN_STATE.json',{'phase':'source_fit_collection','source':source,'cases':len(labels),'unix':time.time()})
        feats=torch.stack(features).cuda();target=torch.tensor(labels,device='cuda').clamp(-5,5)/5;mu=feats.mean(0);std=feats.std(0).clamp_min(.001);feats=(feats-mu)/std;torch.manual_seed(92515);selector=Selector().cuda();opt=torch.optim.AdamW(selector.parameters(),lr=.001,weight_decay=.0001)
        for step in range(1,1201):
            a.check();ii=torch.randint(len(feats),(64,),device='cuda');loss=F.mse_loss(selector(feats[ii]),target[ii]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(selector.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step==1 or step%300==0:c.write(R/'RUN_STATE.json',{'phase':'selector_training','source':source,'step':step,'loss':loss.item(),'unix':time.time()})
        torch.save({'selector':selector.state_dict(),'mu':mu.cpu(),'std':std.cpu(),'step':1200},run/'latest.pt')
        del model,net,head,features,labels,feats,target,selector,opt;torch.cuda.empty_cache()
    records=[]
    for source in p.DOMAINS:
        model,net,head=load(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False);selector=Selector().cuda().eval();selector.load_state_dict(saved['selector']);mu=saved['mu'].cuda();std=saved['std'].cuda()
        for index in split[source]['dev']:
            a.check();x=torch.from_numpy(p.read_image(cfg['data_root'],data[source][index])[None]).cuda();y=torch.from_numpy(p.read_label(cfg['data_root'],data[source][index])[None].astype(np.float32)).cuda()
            for condition in ('clean','fixed_appearance','color','blur'):
                xx=x
                if condition=='fixed_appearance':xx=(x.clamp_min(.001).pow(.7)*x.new_tensor([1.15,.9,1.05])[None,:,None,None]).clamp(0,1)
                if condition=='color':xx=(x*x.new_tensor([.85,1.2,.95])[None,:,None,None]).clamp(0,1)
                if condition=='blur':xx=F.avg_pool2d(x,9,1,4)
                with torch.no_grad():
                    feat,di,dg=candidates(model,net,head,xx);score=selector((feat-mu)/std).item();vi=a.dice(di,y);vg=a.dice(dg,y);rec={'source':source,'index':index,'condition':condition,'I':vi,'G':vg,'S':vg if score>0 else vi,'score':score,'choose_G':score>0}
                if not all(np.isfinite(rec[arm]).all() for arm in ['I','G','S','score']):raise FloatingPointError('Nonfinite selector output')
                records.append(rec)
        c.write(R/'RUN_STATE.json',{'phase':'source_validation','source':source,'cases':len(records),'unix':time.time()});del model,net,head,selector;torch.cuda.empty_cache()
    c.write(R/'source_records.private.json',records);means={s:{co:{arm:float(np.mean([r[arm] for r in records if r['source']==s and r['condition']==co])) for arm in ['I','G','S']} for co in ('clean','fixed_appearance','color','blur')} for s in p.DOMAINS};delta={s:float(np.mean([means[s][co]['S']-means[s][co]['I'] for co in ('fixed_appearance','color')])) for s in p.DOMAINS};against_g=float(np.mean([means[s][co]['S']-means[s][co]['G'] for s in p.DOMAINS for co in ('fixed_appearance','color')]));clean=float(np.mean([means[s]['clean']['S']-means[s]['clean']['I'] for s in p.DOMAINS]));tail=float(np.quantile([np.mean(r['S'])-np.mean(r['I']) for r in records if r['condition'] in ('fixed_appearance','color')],.1));passed=bool(np.mean(list(delta.values()))>=.1 and min(delta.values())>=0 and against_g>=0 and clean>=-.05 and tail>=-.1)
    c.write(R/'SOURCE_GATE.json',{'means':means,'selected_minus_I':float(np.mean(list(delta.values()))),'selected_minus_G':against_g,'delta_by_source':delta,'clean_delta':clean,'delta_p10':tail,'choose_G_fraction':float(np.mean([r['choose_G'] for r in records])),'gate_pass':passed})
    if passed:
        results={}
        for source in p.DOMAINS:
            model,net,head=load(source);saved=torch.load(R/'runs'/source/'latest.pt',map_location='cpu',weights_only=False);selector=Selector().cuda().eval();selector.load_state_dict(saved['selector']);mu=saved['mu'].cuda();std=saved['std'].cuda();count=[0]
            def target(image):
                a.check();x=torch.from_numpy(image[None]).cuda()
                with torch.no_grad():feat,di,dg=candidates(model,net,head,x);d=dg if selector((feat-mu)/std).item()>0 else di;out=(-16*d).sigmoid()[0].cpu().numpy()
                count[0]+=1
                if count[0]%100==0:c.write(R/'RUN_STATE.json',{'phase':'target_evaluation','source':source,'processed':count[0],'unix':time.time()})
                return out
            summary,detail=p.evaluate(cfg['data_root'],p.scenario(data,source)[1],target,a.calculate_metrics);results[source]=summary;c.write(R/'runs'/source/'S_detail.private.json',detail);del model,net,head,selector;torch.cuda.empty_cache()
        c.write(R/'TARGET_RESULTS.json',{'sources':results,'average':p.aggregate(results)})
    c.write(R/'COMPLETE.json',{'status':'COMPLETE' if passed else 'SOURCE_GATE_FAILED','unix':time.time(),'target_evaluated':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:c.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
