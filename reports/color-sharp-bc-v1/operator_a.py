"""Frozen A operators only; original numerical definitions reused."""
import torch
import torch.nn.functional as F
def norm(x):
    span=x.max()-x.min()
    return (x-x.min())/span if span.item()>0 else x*0

def transform(x,v,j):
    if j==0:
        base=(x*torch.tensor([.2989,.5870,.1140],dtype=x.dtype)[:,None,None]).sum(0,keepdim=True).expand_as(x)
    else:
        k=torch.tensor([[1,1,1],[1,5,1],[1,1,1]],dtype=x.dtype)[None,None].repeat(3,1,1,1)/13
        base=F.conv2d(F.pad(x[None],(1,1,1,1),mode='replicate'),k,groups=3)[0]
    return norm((base+v*(x-base)).clamp(0,1))

