"""Original synthetic-only two-operation policy qualification. No trained model."""
import os,sys,json,time,socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def norm(x):
    span=x.max()-x.min()
    return (x-x.min())/span if span.item()>0 else x*0

def fields():
    y,x=torch.meshgrid(torch.linspace(0,1,32,dtype=torch.float64),torch.linspace(0,1,32,dtype=torch.float64),indexing='ij')
    return [norm(torch.stack([x,.2+.7*y,.3+.2*torch.sin(7*x+3*y)])),norm(torch.stack([torch.sin(5*x),torch.cos(4*y),torch.sin(3*x+7*y)])),norm(torch.stack([((x*7).floor()+(y*9).floor()).remainder(2),x*.4+y*.6,x*y])),torch.full((3,32,32),.4,dtype=torch.float64)*0]

def transform(x,v,j):
    if j==0:
        base=(x*torch.tensor([.2989,.5870,.1140],dtype=x.dtype)[:,None,None]).sum(0,keepdim=True).expand_as(x)
    else:
        k=torch.tensor([[1,1,1],[1,5,1],[1,1,1]],dtype=x.dtype)[None,None].repeat(3,1,1,1)/13
        base=F.conv2d(F.pad(x[None],(1,1,1,1),mode='replicate'),k,groups=3)[0]
    return norm((base+v*(x-base)).clamp(0,1))

def reference(x,v,j):
    z=x.numpy();b=np.empty_like(z)
    if j==0:
        for c in range(3):b[c]=.2989*z[0]+.5870*z[1]+.1140*z[2]
    else:
        padded=np.pad(z,((0,0),(1,1),(1,1)),mode='edge');k=((1,1,1),(1,5,1),(1,1,1))
        for c in range(3):
            for y in range(32):
                for xx in range(32):b[c,y,xx]=sum(k[dy][dx]*padded[c,y+dy,xx+dx] for dy in range(3) for dx in range(3))/13
    out=np.clip(b+v*(z-b),0,1);span=out.max()-out.min()
    return (out-out.min())/span if span>0 else out*0

def predict(x):
    r,g,b=x
    return torch.stack([1.2*r-.7*g+.2*b-.3,-.4*r+g+.5*b-.6]).sigmoid()

def objective(x,a,m):
    q=predict(x).detach();weights=a.softmax(0);values=[]
    for j in range(2):
        z=transform(x,.1+1.8*m[j].sigmoid(),j);p=predict(z)
        ent=-(p*p.log()+(1-p)*(1-p).log()).mean()
        kl=(q*(q.log()-p.log())+(1-q)*((1-q).log()-(1-p).log())).mean()
        values.append(ent+kl-(z.mean((1,2))-x.mean((1,2))).square().mean())
    return (weights*torch.stack(values)).sum()

def access(root):
    prefix=Path(sys.prefix).resolve()
    def hook(event,args):
        if event.startswith('socket.'):raise PermissionError('network forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if str(p).startswith('/remote-home') and not p.is_relative_to(root) and not p.is_relative_to(prefix):raise PermissionError('outside synthetic assets')
            if isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and not p.is_relative_to(root):raise PermissionError('outside write')
    sys.addaudithook(hook)
    for n in ['rgb.png','gt.png','weights.pth','scores.json']:
        try:Path('/remote-home/forbidden/'+n).read_bytes()
        except PermissionError:pass
        else:raise AssertionError('file access gate')
    try:socket.socket()
    except PermissionError:pass
    else:raise AssertionError('socket gate')
    return 5

def run(c):
    start=time.perf_counter();root=Path(c['root']).resolve();mode=os.environ['AUDIT_MODE']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    # Load Adam's optional Python imports before the asset restriction; no data access.
    dummy=torch.zeros(1,requires_grad=True);torch.optim.Adam([dummy]);del dummy
    negatives=access(root);arr={};checks=[];referr=graderr=0.;effects=[]
    def check(name,ok):
        checks.append({'check':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    for i,x in enumerate(fields()):
        original=x.clone();arr[f'input{i}']=x.numpy().copy()
        for j in range(2):
            check(f'identity{i}_{j}',(transform(x,1.,j)-x).abs().max().item()<=1e-12)
            v=.1+1.8*torch.tensor([-.8,-.4],dtype=torch.float64)[j].sigmoid()
            z=transform(x,v,j);r=reference(x,float(v),j)
            err=float(np.max(np.abs(z.numpy()-r)));referr=max(referr,err)
            check(f'reference{i}_{j}',err<=1e-12)
            check(f'range{i}_{j}',bool(torch.isfinite(z).all() and z.min()>=0 and z.max()<=1))
            arr[f'view{i}_{j}']=z.numpy();effects.append(float((z-x).abs().max()))
        if i<3:check(f'nontrivial{i}',max(effects[-2:])>1e-8)
        else:check('constantzero',not x.any())
        a=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True);m=torch.tensor([-.8,-.4],dtype=torch.float64,requires_grad=True)
        ga,gm=torch.autograd.grad(objective(x,a,m),(a,m));g=torch.cat([ga,gm]);vec=torch.cat([a,m]).detach()
        for k in range(4):
            d=torch.zeros(4,dtype=torch.float64);d[k]=1e-5
            hi,lo=vec+d,vec-d
            fd=(objective(x,hi[:2],hi[2:])-objective(x,lo[:2],lo[2:]))/2e-5
            err=abs(float(g[k]-fd));graderr=max(graderr,err);check(f'gradient{i}_{k}',err<=1e-6)
        a=torch.zeros(2,dtype=torch.float64,requires_grad=True);m=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        opt=torch.optim.Adam([{'params':[a],'lr':.001},{'params':[m],'lr':.01}],betas=(.9,.999),eps=1e-8,weight_decay=0)
        history=[]
        for step in range(20):
            opt.zero_grad(set_to_none=True);loss=-objective(x,a,m);loss.backward();opt.step()
            with torch.no_grad():
                v=.1+1.8*m.sigmoid();w=a.softmax(0)
                check(f'finite{i}_{step}',bool(torch.isfinite(loss) and torch.isfinite(a).all() and torch.isfinite(m).all() and (v>=.1).all() and (v<=1.9).all() and abs(float(w.sum())-1)<=1e-12))
                history.append(torch.cat([a,m,w,v,loss[None]]).numpy().copy())
        if i<3:check(f'learned{i}',m.abs().max().item()>1e-10)
        check(f'inputimmutable{i}',torch.equal(x,original));arr[f'history{i}']=np.array(history)
        check(f'controls{i}',bool(torch.isfinite(objective(x,torch.zeros(2,dtype=torch.float64),m.detach())) and torch.isfinite(objective(x,a.detach(),torch.zeros(2,dtype=torch.float64)))))
    target=root/'private/replay.npz'
    if mode=='synthetic1':np.savez_compressed(target,**arr)
    else:
        with np.load(target) as old:
            check('replaykeys',set(old.files)==set(arr))
            for key,value in arr.items():check('exact_'+key,np.array_equal(old[key],value))
    check('no_cuda',not torch.cuda.is_initialized())
    result=dict(status='PASS',source_sha=c['source_sha'],checks=checks,check_count=len(checks),inputs=4,optimizer_steps=80,independent_array_replay=mode=='synthetic2',reference_max_error=referr,gradient_max_error=graderr,read_negative_checks=negatives,real_data_reads=0,original_model_calls=0,gpu_calls=0,semantic_qualification=False,body_seconds=time.perf_counter()-start)
    (root/'public'/('QUALIFICATION1.json' if mode=='synthetic1' else 'QUALIFICATION2.json')).write_text(json.dumps(result,indent=2)+'\n')
