"""Fixed foreground-conditioned signed contrast; no anatomical causal claim."""
import numpy as np
class Memory:
    def __init__(self):self.ids=[]
    def add(self,i,proposals):
        assert i not in self.ids and len(proposals)==4
        self.ids.append(i)

def variants(m,x,memory):
    import torch
    z=m(x)[0];context=[];shift=[];counts=[]
    for h in range(2):
        mask=z[:,h:h+1]>=0
        background=~mask
        # Empty/full masks use the whole-image channel mean. No retries/tuning.
        if background.any():fill=(x*background).sum((2,3),keepdim=True)/background.sum((2,3),keepdim=True)
        else:fill=x.mean((2,3),keepdim=True)
        replaced=torch.where(mask,fill,x)
        shifted=torch.roll(mask,mask.shape[-1]//2,dims=-1)
        sham=torch.where(shifted,fill,x)
        assert torch.equal(replaced.masked_select(~mask.expand_as(x)),x.masked_select(~mask.expand_as(x)))
        assert int(mask.sum())==int(shifted.sum())
        context.append(m(replaced)[0][:,h:h+1]);shift.append(m(sham)[0][:,h:h+1]);counts.append(float(mask.float().mean()))
    c=torch.cat(context,1);s=torch.cat(shift,1)
    main=2*z-c;mean=(z+c)/2;reverse=2*c-z;null=2*z-s
    logits=torch.cat((z,main,mean,reverse,null),0).cpu().numpy()
    assert np.isfinite(logits).all()
    diag=np.array(counts+[float((main-z).abs().mean()),float((main-null).abs().mean())])
    return logits,diag,None

def check():
    # Same point can be flipped in either direction; this is not confidence scaling.
    z=np.array([.1,-.1,2.]);c=np.array([.5,-.5,2.]);main=2*z-c
    assert np.array_equal(main>=0,[False,True,True])
    assert np.array_equal(2*z-z,z)
    mask=np.arange(64).reshape(8,8)%3==0
    assert mask.sum()==np.roll(mask,4,axis=-1).sum()
