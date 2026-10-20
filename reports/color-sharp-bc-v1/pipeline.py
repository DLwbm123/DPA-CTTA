"""Readonly original model, fixed synthetic-qualified input policy, gated scoring."""
import os,sys,json,time,hashlib,io,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import base

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def audit(c,panel):
    root=Path(c['root']).resolve();old=Path(c['sealed_root']).resolve();data=Path(c['source_root']).resolve();models=Path(c['model_root']).resolve();checkpoint=Path(c['checkpoint_path']).resolve()
    allowed={Path(p['image_path']).resolve() for p in panel};bits={(old/'inputs'/f'{i}_{k}_{a}.bits').resolve() for i in range(8) for k in range(4) for a in ('SOURCE','TOPO_SMOOTH')}
    prefixes=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    def hook(event,args):
        if event.startswith('socket.'):raise PermissionError('network denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(data) and p not in allowed:raise PermissionError('unregistered data')
        if p.is_relative_to(models) and p!=checkpoint:raise PermissionError('unregistered model')
        if p.is_relative_to(old) and p not in bits and not p.is_relative_to(old/'code'):raise PermissionError('old metadata or evaluation')
        if p.is_relative_to(root/'scorer'):raise PermissionError('GT stage')
        if str(p).startswith('/remote-home') and not (p.is_relative_to(root) or p in allowed or p in bits or p==checkpoint or p.is_relative_to(old/'code') or any(p.is_relative_to(v) for v in prefixes)):raise PermissionError('outside assets')
        if args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and not p.is_relative_to(root):raise PermissionError('outside write')
    sys.addaudithook(hook)
    for p in [old/'scorer/panel.json',models/'forbidden.pth',data/'unregistered.png']:
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access gate')
    import socket
    try:socket.socket()
    except PermissionError:pass
    else:raise AssertionError('socket gate')
    return 4

def worker(c):
    begin=time.perf_counter()
    import torch
    from operator_a import transform
    root=Path(c['root']);old=Path(c['sealed_root']);mode=os.environ['AUDIT_MODE'];panel=read(root/'private/panel.json');frames=read(root/'private/cache.json');bykey={(r['case'],r['condition']):r['blocks'] for r in frames}
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261018);np.random.seed(20261018);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dummy=torch.zeros(1,requires_grad=True);torch.optim.Adam([dummy]);del dummy
    negatives=audit(c,panel);sys.path.insert(0,str(old/'code/src'))
    from dpa_ctta.integrations.ctta_suite import build_reference_model
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    state={k:v.cpu().clone() for k,v in model.state_dict().items()};norms=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)];assert len(norms)==41 and all(not m.training and m.track_running_stats for m in norms)
    stats={};hooks=[m.register_forward_hook(lambda mod,args,out,j=j:stats.__setitem__(j,args[0].mean((-2,-1)).double())) for j,m in enumerate(norms)]
    calls=0
    def forward(x):
        nonlocal calls
        stats.clear();z=model(x)[0];calls+=1
        assert z.shape==(1,2,512,512) and torch.isfinite(z).all() and len(stats)==41
        return z.double(),[stats[j] for j in range(41)]
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True)
    mine=[l for l in gpu.splitlines() if l.split(',')[0].strip()==str(os.getpid())];assert len(mine)==1 and c['gpu_uuid'] in mine[0] and 'wangbomin' not in mine[0] and 'color-sharp' not in mine[0]
    random_choices=np.random.default_rng(20261018).integers(0,2,32)
    storage=root/'private/measurement';storage.mkdir(exist_ok=True);supported=union=changed=qunion=qchanged=actions=maskchanges=0;matched=[True]*len(base.CONTROLS);parity=0
    for i,item in enumerate(panel):
        raw=Path(item['image_path']).read_bytes();expected=[v['image_sha256'] for v in c['rgb_identity'] if str(item['image_path']).endswith(v['image_relative'])];assert len(expected)==1 and hashlib.sha256(raw).hexdigest()==expected[0]
        with Image.open(io.BytesIO(raw)) as im:assert list(im.size)==item['image_size'];image=torch.from_numpy(np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()/255)[None]
        for k,view in enumerate(base.appearances(image)):
            span=view.max()-view.min();assert span>0;x=((view-view.min())/span)[0].cuda();original=x.clone()
            with torch.no_grad():zc,cleanmeans=forward(x[None]);q0=zc.sigmoid().clamp(1e-6,1-1e-6);cleanmeans=[v.detach() for v in cleanmeans]
            s=base.unpack(old/'inputs'/f'{i}_{k}_SOURCE.bits');b=base.unpack(old/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');assert np.array_equal(zc[0].ge(0).cpu().numpy(),s);parity+=1
            a=torch.zeros(2,device='cuda',dtype=torch.float64,requires_grad=True);m=torch.zeros(2,device='cuda',dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([{'params':[a],'lr':.001},{'params':[m],'lr':.01}],betas=(.9,.999),eps=1e-8,weight_decay=0);hist=[]
            for step in range(20):
                opt.zero_grad(set_to_none=True);terms=[]
                for j in range(2):
                    z,means=forward(transform(x,.1+1.8*m[j].sigmoid(),j)[None]);p=z.sigmoid().clamp(1e-6,1-1e-6)
                    ent=-(p*p.log()+(1-p)*(1-p).log()).mean();kl=(q0*(q0.log()-p.log())+(1-q0)*((1-q0).log()-(1-p).log())).mean();penalty=torch.stack([(v-u).square().mean() for v,u in zip(means,cleanmeans)]).mean();terms.append(ent+kl-penalty)
                loss=-(a.softmax(0)*torch.stack(terms)).sum();assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (a,m));assert all(v.grad is None for v in model.parameters());opt.step()
                assert torch.isfinite(a).all() and torch.isfinite(m).all();hist.append(np.r_[a.detach().cpu().numpy(),m.detach().cpu().numpy(),float(loss.detach())])
            with torch.no_grad():
                finals=[forward(transform(x,.1+1.8*m[j].sigmoid(),j)[None])[0][0].cpu().numpy() for j in range(2)];frozen=[forward(transform(x,x.new_tensor(1.),j)[None])[0][0].cpu().numpy() for j in range(2)];weights=a.softmax(0).cpu().numpy();zclean=zc[0].cpu().numpy();zp=weights[0]*finals[0]+weights[1]*finals[1];zu=(finals[0]+finals[1])/2;zf=weights[0]*frozen[0]+weights[1]*frozen[1];zr=finals[random_choices[i*4+k]];delta=zp-zclean;zs=zclean+np.roll(delta,257,axis=-1)
            assert torch.equal(x,original) and all(torch.equal(v.cpu(),state[n]) for n,v in model.state_dict().items()) and all(v.grad is None for v in model.parameters())
            g,n=base.components(s,b);allrows=bykey[i,k];flat=g.ravel();od=np.bincount(flat,weights=(s[0]!=b[0]).ravel(),minlength=n+1);oc=np.bincount(flat,weights=(s[1]!=b[1]).ravel(),minlength=n+1);assert {r['block_id'] for r in allrows}==set(np.flatnonzero((od==0)&(oc>0)))
            for r in allrows:assert np.array_equal(np.argwhere(g==r['block_id']),np.asarray(r['query'])) and r['size']==r['oc_pixels']==len(r['query'])
            rows=[r for r in allrows if r['common']];q=np.zeros((len(rows),len(base.ARMS)));u=np.zeros(len(rows),bool);d=s[1].astype(float)-b[1].astype(float)
            for j,r in enumerate(rows):
                region=g==r['block_id'];u[j]=bool((np.abs(delta[1,region])>1e-6).any())
                if not u[j]:continue
                for col,z in enumerate([zp,zclean,zu,zf,zr,zs]):q[j,col]=np.mean(d[region]*z[1,region])
                q[j,6]=r['controls']['LOGIT_Q'];q[j,7]=r['controls']['GEOMETRY_ONLY']
            assert np.isfinite(q).all();masks,match,selected=base.decisions(b,s,g,rows,q,u)
            payload=dict(input=x.cpu().numpy(),clean=zclean,final=np.stack(finals),frozen=np.stack(frozen),weights=weights,history=np.array(hist),q=q,support=u,masks=np.packbits(masks),matched=np.asarray(match))
            path=storage/f'{i}_{k}.npz'
            if mode=='worker1':np.savez_compressed(path,**payload)
            else:
                with np.load(path) as prev:
                    assert set(prev.files)==set(payload)
                    for name,v in payload.items():assert np.array_equal(prev[name],v),name
            support_delta=delta[1];shift_delta=np.roll(support_delta,257,axis=-1);nz=(np.abs(support_delta)>1e-12)|(np.abs(shift_delta)>1e-12);union+=int(nz.sum());changed+=int(((np.abs(support_delta-shift_delta)>1e-12)&nz).sum());nz=(np.abs(q[:,0])>1e-12)|(np.abs(q[:,5])>1e-12);qunion+=int(nz.sum());qchanged+=int(((np.abs(q[:,0]-q[:,5])>1e-12)&nz).sum());supported+=int(u.sum());actions+=selected;maskchanges+=not np.array_equal(masks[0],masks[5]);matched=[v and w for v,w in zip(matched,match)]
            write(root/'RUN_STATE.json',dict(stage=mode,completed_inputs=i*4+k+1,model_calls=calls))
            if mode=='worker1' and i==0 and k==0:
                torch.cuda.synchronize();first=time.perf_counter()-begin;elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)-c['origin_boottime'];estimate=first*63*1.5+elapsed+180;write(root/'public/COST_ADMISSION.json',dict(first_input_full_seconds=first,projected_total_seconds=estimate,passed=estimate<=3600));assert estimate<=3600,'readonly cost admission'
    for h in hooks:h.remove()
    assert calls==1440 and parity==32
    write(root/'public'/f'{mode.upper()}.json',dict(status='PASS',source_sha=c['source_sha'],registered_rgb_decodes=8,measured_inputs=32,model_calls=calls,source_bit_parity=parity,source_immutable=True,no_model_parameter_grad=True,policy_steps=640,normalization_layers=41,gpu_uuid_pass=True,neutral_nvidia_argv=True,independent_exact_replay=mode=='worker2',access_negative_checks=negatives,label_reads=0,supported_blocks=supported,sealed_selected_blocks=actions,matched_control_quotas=matched,shift_nonzero_union_positions=union,shift_changed_positions=changed,shift_nonzero_union_scores=qunion,shift_changed_scores=qchanged,shift_final_masks_changed=maskchanges))

def run(c):
    mode=os.environ['AUDIT_MODE']
    if mode=='check':base.check(c)
    elif mode in ('worker1','worker2'):worker(c)
    elif mode=='score':base.score(c)
    else:raise ValueError(mode)
