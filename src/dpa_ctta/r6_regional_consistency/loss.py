"""Current-image positive BCE weights, with detached local logit-norm controls."""
import hashlib, math
import torch
import torch.nn.functional as F

ARMS=('C','R_BAL','R_SCALE','R_SHUFFLE')
PHYSICAL=dict(network_forwards=8,loss_backward_calls=1,adam_calls=1,jacobian_vjp_calls=0)
TOLERANCES={torch.float64:dict(rtol=1e-10,atol=1e-14),torch.float32:dict(rtol=1e-5,atol=1e-12)}


def finite(x):
    if isinstance(x,torch.Tensor):
        if not torch.isfinite(x).all():raise ValueError('R6 nonfinite tensor')
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
    elif isinstance(x,(float,int)) and not math.isfinite(x):raise ValueError('R6 nonfinite scalar')


def permutation_seed(visit,channel):
    if type(visit) is not int or visit<1 or type(channel) is not int or channel not in (0,1):raise ValueError('visit/channel')
    raw=f'R6_WEIGHT_PERM_V1|20260907|{visit}|{channel}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')&((1<<63)-1)


def partition(n_fg,n_bg):
    if any(type(n) is not int or n<0 for n in (n_fg,n_bg)) or n_fg+n_bg==0:raise ValueError('partition counts')
    if not n_fg or not n_bg:return dict(n_fg=n_fg,n_bg=n_bg,rho=1.,rho_clipped=False,w_fg=1.,w_bg=1.,fallback='ONE_PARTITION_EMPTY')
    ratio=n_bg/n_fg;rho=min(8.,max(1/8,ratio));bg=(n_fg+n_bg)/(rho*n_fg+n_bg)
    return dict(n_fg=n_fg,n_bg=n_bg,rho=rho,rho_clipped=rho!=ratio,w_fg=rho*bg,w_bg=bg,fallback=None)


def weights(q):
    if q.ndim!=4 or q.shape[:2]!=(1,2) or q.dtype not in TOLERANCES:raise ValueError('one two-channel probability grid')
    finite(q)
    if ((q<0)|(q>1)).any():raise ValueError('probability range')
    h=q.detach().cpu()>=.5;w=torch.empty(q.shape,dtype=torch.float64,device='cpu');rows=[]
    for c in range(2):
        n=int(h[0,c].sum());row=partition(n,h[0,c].numel()-n)
        # No Python-float torch.where: explicitly assign into double storage.
        w[0,c].fill_(row['w_bg']);w[0,c][h[0,c]]=row['w_fg'];rows.append(row)
    return w,h,rows


def scales(s0,sw,sp):
    finite((s0,sw,sp))
    if s0==0:
        if sw!=0 or sp!=0:raise ValueError('zero residual energy identity')
        return 1.,1.
    if min(s0,sw,sp)<=0:raise ValueError('nonpositive weighted residual energy')
    a,b=math.sqrt(sw/s0),math.sqrt(sw/sp);finite((a,b))
    return a,b


def objective(z,q,arm,visit):
    """One scalar objective, scalar audit, and its detached expected gradient norms.

    No network calls, autograd.grad or backward. Production inputs are float32;
    float64 is supported for the independent procedural algebra checks only.
    """
    if arm not in ARMS or z.shape!=q.shape or z.dtype!=q.dtype or z.device!=q.device:raise ValueError('loss contract')
    finite(z);ideal,h,rows=weights(q)
    w=ideal.to(dtype=z.dtype);wp=torch.empty_like(w)
    # Residual subtraction occurs in the actual loss dtype before CPU double.
    d=(z.detach().sigmoid()-q.detach()).cpu().double()
    raw=F.binary_cross_entropy_with_logits(z.detach(),q.detach(),reduction='none').cpu().double()
    a=[];b=[]
    for c,r in enumerate(rows):
        seed=permutation_seed(visit,c);g=torch.Generator(device='cpu').manual_seed(seed)
        index=torch.randperm(w[0,c].numel(),generator=g)
        wp[0,c]=w[0,c].flatten()[index].reshape(w[0,c].shape)
        wc,pc,dc=w[0,c].double(),wp[0,c].double(),d[0,c];mask=h[0,c]
        if not torch.equal(torch.sort(wc.flatten()).values,torch.sort(pc.flatten()).values):raise ValueError('permutation histogram')
        s0=float(dc.square().sum());sw=float((wc*dc).square().sum());sp=float((pc*dc).square().sum())
        ac,bc=scales(s0,sw,sp);a.append(ac);b.append(bc)
        fg=lambda t:float(t[mask].sum())
        bg=lambda t:float(t[~mask].sum())
        r.update(channel=('OD','OC')[c],applied_w_fg=float(torch.tensor(r['w_fg'],dtype=z.dtype)),applied_w_bg=float(torch.tensor(r['w_bg'],dtype=z.dtype)),
            mean_weight=float(wc.mean()),S0=s0,Sw=sw,Sperm=sp,a=ac,b=bc,seed=seed,
            bce_sum=float(raw[0,c].sum()),bce_fg_sum=fg(raw[0,c]),bce_bg_sum=bg(raw[0,c]),
            weighted_bce_sum=float((wc*raw[0,c]).sum()),shuffled_bce_sum=float((pc*raw[0,c]).sum()),
            residual_fg_sse=fg(dc.square()),residual_bg_sse=bg(dc.square()),weighted_fg_sse=fg((wc*dc).square()),weighted_bg_sse=bg((wc*dc).square()),
            residual_weighted_dot=float((wc*dc.square()).sum()),residual_shuffled_dot=float((pc*dc.square()).sum()),
            weight_residual_dot=float((wc*dc).sum()),weight_squared_sum=float(wc.square().sum()),
            permutation_histogram_preserved=True,permutation_changed_positions=int((wc!=pc).sum()),
            base_weight_sha256=hashlib.sha256(w[0,c].contiguous().numpy().tobytes()).hexdigest(),
            shuffled_weight_sha256=hashlib.sha256(wp[0,c].contiguous().numpy().tobytes()).hexdigest())
    aa=torch.tensor(a,dtype=z.dtype,device=z.device).reshape(1,2,1,1)
    bb=torch.tensor(b,dtype=z.dtype,device=z.device).reshape(1,2,1,1)
    wd=w.to(z.device);pd=wp.to(z.device)
    if arm=='C':
        # Exactly the published scalar BCE path, not an equivalent new reduction.
        loss=torch.nn.BCEWithLogitsLoss()(z,q);effective=torch.ones_like(wd)
    else:
        per=F.binary_cross_entropy_with_logits(z,q,reduction='none')
        if arm=='R_BAL':effective=wd;loss=(wd*per).mean()
        elif arm=='R_SCALE':effective=aa.expand_as(wd);loss=(per.mean(dim=(2,3))*aa.reshape(1,2)).mean()
        else:effective=bb*pd;loss=(effective*per).mean()
    finite(loss)
    expected=(effective.detach().cpu().double()*d/z.numel()).square().sum(dim=(2,3)).sqrt().flatten().tolist()
    for c,r in enumerate(rows):r['expected_logit_gradient_l2']=expected[c]
    return loss,rows
