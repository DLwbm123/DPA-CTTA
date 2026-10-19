"""Frozen four-phase translation ensemble; HRAC-inspired correspondence qualification."""
import torch
from torch.nn import functional as F
OFFSETS=((0,0),(0,1),(1,0),(1,1)) # (dy,dx), fixed native-pixel phases
ARMS=('SOURCE','PHASE_PROB','NOALIGN_PROB','PHASE_LOGIT','SOURCE_SMOOTH','SHIFT_X','SHIFT_Y','SHIFT_XY')

def shifted(x,dy,dx):
    if (dy,dx) not in OFFSETS:raise ValueError('fixed native phases only')
    if x.ndim!=4 or x.shape[:2]!=(1,3) or min(x.shape[-2:])<3 or not torch.isfinite(x).all():raise ValueError('finite single RGB')
    return F.pad(x,(dx,0,dy,0),mode='reflect')[...,:x.shape[-2],:x.shape[-1]] if dy or dx else x

@torch.no_grad()
def outputs(model,x):
    zz=[model(shifted(x,*p))[0].double() for p in OFFSETS]
    z=zz[0];h,w=z.shape[-2:]
    if z.shape[:2]!=(1,2) or (h,w)!=x.shape[-2:] or any(t.shape!=z.shape or not torch.isfinite(t).all() for t in zz):raise ValueError('native single image, independent heads')
    # Common real-image overlap; reflected/absent output locations are never installed.
    region=(slice(None),slice(None),slice(0,h-1),slice(0,w-1))
    aligned=[t[...,dy:dy+h-1,dx:dx+w-1] for t,(dy,dx) in zip(zz,OFFSETS)]
    raw=[t[region] for t in zz]
    p=z.sigmoid();prob=torch.stack([t.sigmoid() for t in aligned]).mean(0)
    noalign=torch.stack([t.sigmoid() for t in raw]).mean(0)
    out={'SOURCE':z}
    def install(v,is_probability=False):
        q=z.clone();q[region]=torch.logit(v.clamp(1e-12,1-1e-12)) if is_probability else v
        return q
    out['PHASE_PROB']=install(prob,True);out['NOALIGN_PROB']=install(noalign,True)
    out['PHASE_LOGIT']=install(torch.stack(aligned).mean(0))
    axis=z.new_tensor([1,2,1])/4;kernel=(axis[:,None]*axis[None,:]).expand(2,1,3,3)
    smooth=F.conv2d(F.pad(p,(1,1,1,1),mode='reflect'),kernel,groups=2)
    out['SOURCE_SMOOTH']=install(smooth[region],True)
    for a,t in zip(('SHIFT_X','SHIFT_Y','SHIFT_XY'),aligned[1:]):out[a]=install(t)
    for a,t in out.items():
        if not torch.isfinite(t).all() or not torch.equal(t[...,-1,:],z[...,-1,:]) or not torch.equal(t[...,:,-1],z[...,:,-1]):raise ValueError('invalid edge installation')
    diagnostics=dict(query_forwards=4,common_overlap_pixels=(h-1)*(w-1),source_fallback_pixels=h*w-(h-1)*(w-1),aligned_phase_probability_rms=float(torch.stack([(t.sigmoid()-p[region]).square().mean() for t in aligned[1:]]).mean().sqrt()),reflection_input_padding_only=True)
    return {a:out[a] for a in ARMS},diagnostics
