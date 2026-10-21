"""Content-causal pseudo-template readout with an immutable source model."""
import numpy as np


def combine(records):
    out=np.zeros_like(records[0][0]);valid=np.zeros((2,2),bool)
    for h in range(2):
        for c in range(2):
            xs=[x[h,c] for x,v in records if v[h,c]]
            if xs:out[h,c]=np.mean(xs,axis=0);valid[h,c]=True
    return out,valid


class Memory:
    def __init__(self):self.ids=[];self.raw=[];self.shuffled=[]
    def add(self,i,proposals):
        assert i not in self.ids and len(proposals)==4
        self.ids.append(i);self.raw.append(combine([p[0] for p in proposals]));self.shuffled.append(combine([p[1] for p in proposals]))
    def get(self,shuffled=False):
        records=self.shuffled if shuffled else self.raw
        return combine(records) if records else None


def variants(m,x,memory):
    import torch
    import torch.nn.functional as F
    captures=[]
    def capture(mod,args):captures.append(args[0].clone())
    hook=m.seg_head.register_forward_pre_hook(capture)
    try:z=m(x)[0]
    finally:hook.remove()
    assert len(captures)==1
    feature=captures[0];w=m.seg_head.weight;b=m.seg_head.bias
    assert w.shape==(2,feature.shape[1],1,1)
    assert torch.equal(F.conv2d(feature,w,b),z)
    unit=F.normalize(feature[0],dim=0);flat=unit.flatten(1)
    masks=torch.stack((z[0]<-np.log(19),z[0]>np.log(19)),dim=1)
    def prototypes(mask):
        means=torch.zeros((2,2,feature.shape[1]),device=x.device);valid=mask.flatten(2).any(-1)
        for h in range(2):
            for c in range(2):
                if valid[h,c]:means[h,c]=flat[:,mask[h,c].flatten()].mean(1)
        return means.cpu().numpy(),valid.cpu().numpy()
    own=prototypes(masks);shuffled=prototypes(torch.roll(masks,masks.shape[-1]//2,dims=-1))
    def readout(record):
        if record is None:return z.clone(),0
        means,valid=record;changed=0;virtual=w.clone()
        for h in range(2):
            if not valid[h].all():continue
            direction=torch.from_numpy(means[h,1]-means[h,0]).to(x.device)
            norm=torch.linalg.vector_norm(direction)
            if norm<=1e-12:continue
            virtual[h]=(.5*w[h].flatten()+.5*torch.linalg.vector_norm(w[h])*direction/norm).reshape_as(w[h]);changed+=1
        return F.conv2d(feature,virtual,b),changed
    main,eligible=readout(memory.get());current,_=readout(own);null,_=readout(memory.get(True))
    if not memory.ids:assert torch.equal(main,z) and torch.equal(null,z)
    logits=torch.cat((z,main,current,null),dim=0).cpu().numpy()
    assert np.isfinite(logits).all()
    diagnostic=np.array([len(memory.ids),eligible,float((main-null).abs().mean())])
    return logits,diagnostic,(own,shuffled)


def check():
    empty=(np.zeros((2,2,3),np.float32),np.zeros((2,2),bool))
    valid=(np.ones((2,2,3),np.float32),np.ones((2,2),bool))
    m=Memory();assert m.get() is None
    m.add(3,[(valid,empty)]*4);assert m.ids==[3]
    assert np.array_equal(m.get()[0],valid[0]) and m.get()[1].all()
    assert not m.get(True)[1].any()
    try:m.add(3,[(valid,empty)]*4)
    except AssertionError:pass
    else:raise AssertionError('duplicate content admitted')
