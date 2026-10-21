"""One source-frozen reciprocal-neighbor propagation in the deepest spatial grid."""
import numpy as np


def graph(v):
    # Stable ties keep the same graph across independent workers.
    norms=np.linalg.norm(v,axis=1,keepdims=True)
    unit=v/np.maximum(norms,1e-12)
    cosine=unit@unit.T
    np.fill_diagonal(cosine,-np.inf)
    ids=np.argsort(-cosine,axis=1,kind='stable')[:,:4]
    directed=np.zeros(cosine.shape,bool)
    np.put_along_axis(directed,ids,True,axis=1)
    mutual=directed&directed.T&(cosine>0)
    weights=mutual.astype(np.float32)
    degree=weights.sum(1,keepdims=True)
    weights=weights/np.maximum(degree,1)
    weights[degree[:,0]==0]=np.eye(len(v),dtype=np.float32)[degree[:,0]==0]
    return weights


def decode(m,deep,skips):
    import torch.nn.functional as F
    h=F.relu(deep)
    h=m.up1(h,skips[3]);h=m.up2(h,skips[2]);h=m.up3(h,skips[1]);h=m.up4(h,skips[0])
    return m.seg_head(F.relu(m.bnout(m.up5(h))))


def variants(m,x,verify=False):
    import torch
    captures=[]
    def capture(mod,args,out):captures.append((out[0].clone(),[f.clone() for f in out[1]]))
    hook=m.res.register_forward_hook(capture)
    try:z0=m(x)[0]
    finally:hook.remove()
    assert len(captures)==1
    deep,skips=captures[0]
    if verify:assert torch.equal(decode(m,deep,skips),z0),'native decoder interface mismatch'
    active=torch.relu(deep)
    v=active.flatten(2)[0].T.contiguous()
    n=len(v)
    assert n>4
    w=torch.from_numpy(graph(v.cpu().numpy())).to(v.device)
    # Conjugation retains row degrees and weight values, destroys spatial-semantic correspondence.
    permuted=torch.roll(w,(n//2,n//2),dims=(0,1))
    norm=torch.linalg.vector_norm(v,dim=1,keepdim=True)
    def propagate(weights):
        mixed=(v+weights@v)*.5
        mixed=mixed*norm/torch.linalg.vector_norm(mixed,dim=1,keepdim=True).clamp_min(1e-12)
        assert torch.allclose(torch.linalg.vector_norm(mixed,dim=1,keepdim=True),norm,rtol=1e-5,atol=1e-5)
        return mixed.T.reshape_as(active)
    g=decode(m,propagate(w),skips)
    u=decode(m,propagate(torch.full_like(w,1/n)),skips)
    p=decode(m,propagate(permuted),skips)
    zs=torch.cat((z0,g,u,p),dim=0).cpu().numpy()
    assert np.isfinite(zs).all()
    diag=np.array([float((g-u).abs().mean()),float((g-p).abs().mean()),float((w>0).sum(1).float().mean())])
    return zs,diag


def check():
    v=np.random.default_rng(17).uniform(size=(20,8)).astype(np.float32)
    w=graph(v)
    assert np.allclose(w.sum(1),1) and (w>=0).all()
    e=w>0;np.fill_diagonal(e,False)
    assert np.array_equal(e,e.T)
    assert np.array_equal(graph(np.zeros((8,3),np.float32)),np.eye(8,dtype=np.float32))
    assert np.array_equal(w,graph(v))
    p=np.roll(w,(10,10),axis=(0,1))
    assert np.allclose(p.sum(1),1) and np.array_equal(np.sort(w.ravel()),np.sort(p.ravel()))
