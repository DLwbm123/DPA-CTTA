"""Binary content-support host adaptation inspired by GAIN, fixed covariance."""
import numpy as np

class Memory:
    def __init__(self):self.ids=[];self.raw=[];self.shuffled=[]
    def add(self,i,proposals):
        assert i not in self.ids and len(proposals)==4
        self.ids.append(i)
        for j,target in enumerate((self.raw,self.shuffled)):
            target.append(tuple(np.mean([p[j][k] for p in proposals],axis=0) for k in range(2)))
    def get(self,shuffled=False):
        data=self.shuffled if shuffled else self.raw
        return tuple(np.sum([p[k] for p in data],axis=0) for k in range(2)) if data else None

def variants(m,x,memory):
    import torch
    import torch.nn.functional as F
    captures=[]
    hook=m.seg_head.register_forward_pre_hook(lambda mod,args:captures.append(args[0].clone()))
    try:z=m(x)[0]
    finally:hook.remove()
    assert len(captures)==1
    f=captures[0];w=m.seg_head.weight;b=m.seg_head.bias
    assert w.shape==(2,f.shape[1],1,1)
    assert torch.equal(F.conv2d(f,w,b),z)
    # Float64 analytic logits; all retained source logits are original float32.
    flat=f[0].double().flatten(1);weight=w[:,:,0,0].double();bias=b.double()
    centers=torch.stack((-weight/2,weight/2),dim=1)
    def readout(record):
        if record is None:return z.clone(),z.clone(),z.clone()
        n,u=[torch.from_numpy(v).to(x.device) for v in record]
        assert (n>=0).all() and (n<=len(memory.ids)+1e-10).all()
        kappa=1+n;mu=(centers+u)/kappa[:,:,None]
        # Fixed identity covariance, prior log odds b; do not infer pixel ESS.
        d=(flat[None,None,:,:]-mu[:,:,:,None]).square().sum(2)
        proposal=bias[:,None]-.5*(d[:,1]-d[:,0])
        h=1+1/kappa
        evaluator=bias[:,None]-.5*flat.shape[0]*torch.log(h[:,1,None]/h[:,0,None])-.5*(d[:,1]/h[:,1,None]-d[:,0]/h[:,0,None])
        source=z[0].double().flatten(1)
        main=torch.maximum(torch.minimum(source,proposal),torch.minimum(torch.maximum(source,proposal),evaluator))
        return tuple(v.reshape_as(z).float() for v in (main,proposal,evaluator))
    main,proposal,evaluator=readout(memory.get());null,_,_=readout(memory.get(True))
    if not memory.ids:assert torch.equal(main,z) and torch.equal(proposal,z) and torch.equal(evaluator,z) and torch.equal(null,z)
    source_prob=torch.sigmoid(z[0].double());adapted=torch.sigmoid(main[0].double())
    confidence=torch.where(adapted>=.5,source_prob,1-source_prob)
    omega=torch.stack((confidence*(1-adapted),confidence*adapted),dim=1)
    def summarize(v):
        mass=v.flatten(2).mean(-1)
        weighted=torch.einsum('hcp,dp->hcd',v.flatten(2),flat)/flat.shape[1]
        return mass.cpu().numpy(),weighted.cpu().numpy()
    own=summarize(omega);shuffled=summarize(torch.roll(omega,omega.shape[-1]//2,dims=-1))
    logits=torch.cat((z,main,proposal,evaluator,null),dim=0).cpu().numpy()
    assert np.isfinite(logits).all()
    diagnostic=np.array([len(memory.ids),float((main-proposal).abs().mean()),float((main-evaluator).abs().mean()),float((main-null).abs().mean()),float(own[0].max())])
    return logits,diagnostic,(own,shuffled)

def check():
    # Binary geometric probability path is linear in logits. Cross entropy is
    # strictly concave in this coordinate, maximized by interval clipping.
    for source,proposal,evaluator in ((-2.,3.,1.),(2.,-3.,4.),(-1.,-1.,3.),(-4.,-2.,0.)):
        chosen=np.clip(evaluator,min(source,proposal),max(source,proposal))
        q=1/(1+np.exp(-evaluator))
        utility=lambda x:q*x-np.logaddexp(0,x)
        grid=np.linspace(source,proposal,10001)
        assert utility(chosen)>=np.max(utility(grid))-1e-12
    w=np.array([.7,-1.2,2.]);f=np.array([2.,-.5,.3]);b=-.4
    odds=b-.5*(np.sum((f-w/2)**2)-np.sum((f+w/2)**2))
    assert abs(odds-(w@f+b))<1e-12
    item=(np.ones((2,2))*.25,np.ones((2,2,3))*.5)
    m=Memory();assert m.get() is None
    m.add(3,[(item,item)]*4);assert m.ids==[3]
    assert np.array_equal(m.get()[0],item[0]) # Four views do not count four times.
    try:m.add(3,[(item,item)]*4)
    except AssertionError:pass
    else:raise AssertionError('duplicate content')
