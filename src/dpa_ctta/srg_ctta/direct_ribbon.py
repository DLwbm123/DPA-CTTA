"""Training-free local image-evidence proposals, not the SRDF training system."""
import torch
from torch.nn import functional as F
from ..r78_cross_patient.objective import gradient,sample

ARMS=('SOURCE','RIBBON','IMAGE_ROLL','DELTA_ROLL','REVERSE','EXPAND_MATCH','SHRINK_MATCH')


@torch.no_grad()
def geometry(image,logits):
    if image.shape[:2]!=(1,3) or logits.shape[:2]!=(1,2) or not torch.isfinite(image).all() or not torch.isfinite(logits).all():raise ValueError('finite single RGB, independent OD/OC')
    rgb=F.interpolate(image,(256,256),mode='bilinear',align_corners=False)
    rgb=(rgb-rgb.mean((-2,-1),keepdim=True))/rgb.std((-2,-1),correction=0,keepdim=True).clamp_min(1e-6)
    ix,iy=gradient(rgb);edge=(ix.square()+iy.square()+1e-6).sqrt().mean(1,keepdim=True)
    descriptor=torch.cat((rgb,edge),1)[0]
    z=F.interpolate(logits,(256,256),mode='bilinear',align_corners=False)
    gx,gy=gradient(-z);h=(gx.square()+gy.square()+1e-6).sqrt().clamp_min(.1);d=-z/h
    dx,dy=gradient(d);length=(dx.square()+dy.square()).sqrt();nx=dx/length.clamp_min(1e-6);ny=dy/length.clamp_min(1e-6)
    q=(-d/.5).sigmoid();omega=4*q*(1-q);omega[:,1:2]*=(-d[:,:1]/.5).sigmoid()
    yy,xx=torch.meshgrid(torch.arange(256,device=z.device),torch.arange(256,device=z.device),indexing='ij')
    valid=(d.abs()<=1)&(length>1e-6)&(xx>=2.5)&(xx<=252.5)&(yy>=2.5)&(yy<=252.5)
    return dict(d=d,h=h,nx=nx,ny=ny,omega=omega,valid=valid,descriptor=descriptor)


@torch.no_grad()
def proposal(g,descriptor):
    e=torch.zeros_like(g['d']);eligible=torch.zeros_like(g['valid']);choices=[]
    for ch in range(2):
        yx=g['valid'][0,ch].nonzero()
        if len(yx)==0:choices.append(dict(eligible=0,moved=0));continue
        y,x=yx.unbind(1);normal=torch.stack((g['nx'][0,ch,y,x],g['ny'][0,ch,y,x]),-1);tangent=torch.stack((-normal[:,1],normal[:,0]),-1)
        centre=torch.stack((x,y),-1).to(e);x0=centre-g['d'][0,ch,y,x].clamp(-1,1)[:,None]*normal
        positions=[]
        for delta in (0,-1,1):
            positions.append(torch.stack([x0+(delta+n)*normal+t*tangent for n in (-1.5,-.5,.5,1.5) for t in (-1,0,1)],1))
        inside=torch.stack([((p>=0)&(p<=255)).all((-1,-2)) for p in positions]).all(0)
        y,x=y[inside],x[inside]
        if len(y)==0:choices.append(dict(eligible=0,moved=0));continue
        scores=[]
        for pp in positions:
            values=sample(descriptor,pp[inside]);sides=[]
            for side in (values[:,:,:6],values[:,:,6:]):
                mean=side.mean(-1);disp=((side-mean[...,None]).square().mean((0,2))+1e-6).sqrt();sides.append((mean,disp))
            between=((sides[0][0]-sides[1][0]).square().mean(0)+1e-6).sqrt();within=(sides[0][1]+sides[1][1])*.5
            scores.append(((between-within)/(between+within).clamp_min(1e-6)).clamp_min(0))
        scores=torch.stack(scores);ix=scores.argmax(0);rank=scores.sort(0,descending=True).values;gap=rank[0]-rank[1]
        shift=e.new_tensor([0,-1,1])[ix]*gap*g['omega'][0,ch,y,x]
        if not torch.isfinite(shift).all() or (shift.abs()>1+1e-6).any():raise ValueError('bounded local displacement failed')
        e[0,ch,y,x]=g['h'][0,ch,y,x]*shift;eligible[0,ch,y,x]=True
        choices.append(dict(eligible=len(y),moved=int((shift!=0).sum()),mean_abs_shift=float(shift.abs().mean()),mean_gap=float(gap.mean())))
    return e,eligible,choices


def match_injection(value,target):
    out=[];errors=[]
    for ch in range(2):
        want=target[:,ch].double().square().mean().sqrt();have=value[:,ch].double().square().mean().sqrt()
        if want==0:q=torch.zeros_like(value[:,ch])
        elif have==0:raise ValueError('nonzero proposal but destroyed control direction')
        else:q=(value[:,ch].double()*(want/have)).to(value)
        err=abs(float(q.double().square().mean().sqrt()-want))
        if err>1e-8+1e-5*float(want):raise ValueError('injected logit budget mismatch')
        out.append(q);errors.append(err)
    return torch.stack(out,1),max(errors)


@torch.no_grad()
def outputs(image,z):
    g=geometry(image,z);e,valid,info=proposal(g,g['descriptor']);other,valid2,otherinfo=proposal(g,g['descriptor'].roll(129,-1))
    if not torch.equal(valid,valid2):raise ValueError('image control changed geometric support')
    rolled=torch.zeros_like(e)
    for ch in range(2):
        values=e[0,ch][valid[0,ch]]
        if len(values):rolled[0,ch][valid[0,ch]]=values.roll(len(values)//2+1)
    up=lambda x:F.interpolate(x,z.shape[-2:],mode='bilinear',align_corners=False)
    target=up(e);dr,err1=match_injection(up(rolled),target);ex,err2=match_injection(up(e.abs()),target)
    fields=dict(SOURCE=torch.zeros_like(z),RIBBON=target,IMAGE_ROLL=up(other),DELTA_ROLL=dr,REVERSE=-target,EXPAND_MATCH=ex,SHRINK_MATCH=-ex)
    envelope=up(valid.float())>0
    for a,v in fields.items():
        if not torch.isfinite(v).all() or bool((v[~envelope]!=0).any()):raise ValueError('nonlocal/nonfinite intervention')
    pred={a:z+v if a!='SOURCE' else z for a,v in fields.items()}
    diagnostics=dict(eligible=sum(r['eligible'] for r in info),moved=sum(r['moved'] for r in info),max_injection_matching_error=max(err1,err2),envelope_pixels=int(envelope.sum()),image_control_matched=False)
    for ch,name in enumerate(('OD','OC')):
        diagnostics[name+'_eligible']=info[ch]['eligible'];diagnostics[name+'_moved']=info[ch]['moved'];diagnostics[name+'_image_roll_moved']=otherinfo[ch]['moved']
    for a,v in fields.items():diagnostics[a+'_injected_rms']=float(v.square().mean().sqrt())
    return pred,diagnostics
