"""Source-calibrated fractional feature history; independent GAIN-inspired variant."""
import torch

ARMS=('FEATURE_Q','LOGIT_Q','SHUFFLE_Q','FEATURE_R','FEATURE_CLIP','HALF_Q')
JOBS=(('FEATURE_Q','RESET'),('LOGIT_Q','RESET'),('FEATURE_Q','RETAIN'),('LOGIT_Q','RETAIN'),('SHUFFLE_Q','RETAIN'),('FEATURE_R','RETAIN'),('FEATURE_CLIP','RETAIN'),('HALF_Q','RETAIN'))


class History:
    def __init__(self,weight):
        if weight.ndim!=2 or weight.shape[0]!=2 or weight.shape[1] not in (1,32):raise ValueError('two heads with one or 32 features required')
        self.weight=weight.detach().double().clone();self.reset()

    def reset(self):
        self.mu=None;self.var=None;self.active=None;self.n=self.weight.new_zeros(2,2)
        self.s=self.weight.new_zeros(2,2,self.weight.shape[1]);self.images=0

    def initialize(self,features):
        # features [head, feature, pixel]; independent head geometry in 1D.
        if features.shape[:2]!=self.weight.shape:raise ValueError('history feature shape differs')
        self.mu=features.mean(-1);self.var=(features-self.mu[...,None]).square().mean(-1)
        self.active=self.var>0
        if not self.active.any(-1).all():raise ValueError('head has no variable feature direction')
        if not all(torch.isfinite(v).all() for v in (self.mu,self.var)):raise ValueError('nonfinite first-image geometry')

    @torch.no_grad()
    def add(self,features,source,shuffle=False):
        f=features.double();z=source.double()
        if f.ndim!=3 or z.shape!=(2,f.shape[-1]) or not torch.isfinite(f).all() or not torch.isfinite(z).all():raise ValueError('nonfinite or malformed current observations')
        if self.mu is None:self.initialize(f)
        p=z.sigmoid();mass=torch.stack((1-p,p),dim=1)*torch.maximum(p,1-p)[:,None]/f.shape[-1]
        if shuffle:
            # Deterministic toroidal row permutation; class masses stay exactly paired.
            mass=mass.roll(f.shape[-1]//2+1,dims=-1)
        n=mass.sum(-1)
        if not (n.sum(-1)<=1+1e-12).all():raise ValueError('image-unit support exceeded')
        self.n+=n
        u=(f-self.mu[...,None])*self.active[...,None]
        self.s+=torch.einsum('hcn,hdn->hcd',mass,u)
        self.images+=1
        if not torch.isfinite(self.s).all():raise ValueError('nonfinite accumulated moments')

    @torch.no_grad()
    def emit(self,features,source):
        if self.images==0:return source.clone(),source.clone()
        f=features.double();z=source.double();v=torch.where(self.active,self.var,1.)
        u=(f-self.mu[...,None])*self.active[...,None]
        signs=f.new_tensor([-.5,.5])[None,:,None]
        prior=signs*self.var[:,None]*self.weight[:,None]
        means=(3*prior+self.s)/(3+self.n[...,None])
        means=means*self.active[:,None]
        coef=means/v[:,None]
        dot=torch.einsum('hcd,hdn->hcn',coef,u)
        norm=(means*coef).sum(-1)
        gq=dot[:,1]-dot[:,0]-.5*(norm[:,1]-norm[:,0])[:,None]
        old=torch.einsum('hd,hdn->hn',self.weight,u)
        q=z+gq-old
        energy=(u.square()/v[...,None]).sum(1)
        distance=(energy[:,None]-2*dot+norm[...,None]).clamp_min(0)
        h=1+1/(3+self.n)
        logits=-.5*distance/h[...,None]-.5*self.active.sum(-1)[:,None,None]*h.log()[...,None]
        r=z-old+logits[:,1]-logits[:,0]
        if not all(torch.isfinite(x).all() for x in (q,r)):raise ValueError('nonfinite history readout')
        return q,r

    def snapshot(self):
        return dict(images=self.images,**{n:(v.clone() if v is not None else None) for n in ('mu','var','active','n','s') for v in (getattr(self,n),)})


class Stream:
    """One shared frozen-source stream with full/margin/shuffled and reset banks."""
    def __init__(self,weight):
        self.weight=weight.detach().double();self.full=History(weight);self.margin=History(torch.ones_like(weight[:,:1]));self.shuffled=History(weight)
        self.total_events=0

    def step(self,feature,source):
        if feature.shape[0]!=32 or source.shape[0]!=2:raise ValueError('frozen 32-feature/two-logit frame required')
        shape=source.shape;z=source.reshape(2,-1).double();f=feature.reshape(1,32,-1).expand(2,-1,-1).double()
        margin=torch.einsum('hd,hdn->hn',self.weight,f)[:,None]
        reset_full=History(self.weight);reset_margin=History(torch.ones_like(self.weight[:,:1]))
        def outputs():
            fq,fr=self.full.emit(f,z);lq,_=self.margin.emit(margin,z);sq,_=self.shuffled.emit(f,z)
            rq,_=reset_full.emit(f,z);rl,_=reset_margin.emit(margin,z)
            # Logit clipping equals probability clipping by monotonic sigmoid; avoids saturation.
            clip=torch.minimum(torch.maximum(fr,torch.minimum(z,fq)),torch.maximum(z,fq))
            return {('FEATURE_Q','RESET'):rq,('LOGIT_Q','RESET'):rl,('FEATURE_Q','RETAIN'):fq,('LOGIT_Q','RETAIN'):lq,('SHUFFLE_Q','RETAIN'):sq,('FEATURE_R','RETAIN'):fr,('FEATURE_CLIP','RETAIN'):clip,('HALF_Q','RETAIN'):(z+fq)/2}
        pre=outputs()
        for bank,x,shuffle in ((self.full,f,False),(self.margin,margin,False),(self.shuffled,f,True),(reset_full,f,False),(reset_margin,margin,False)):bank.add(x,z,shuffle)
        if not torch.allclose(self.full.n,self.margin.n,rtol=0,atol=1e-12) or not torch.allclose(self.full.n,self.shuffled.n,rtol=0,atol=1e-12) or not torch.allclose(reset_full.n,reset_margin.n,rtol=0,atol=1e-12):raise ValueError('shared source support differs')
        post=outputs();self.total_events+=5
        d=dict(shared_full_images=self.full.images,shared_full_active_dimensions=self.full.active.sum(-1).tolist(),shared_margin_active_dimensions=self.margin.active.sum(-1).tolist(),shared_full_support=self.full.n.tolist(),shared_source_support_paired=True,parameter_updates=0)
        return {k:v.reshape(shape) for k,v in pre.items()},{k:v.reshape(shape) for k,v in post.items()},d

    def snapshot(self):return {k:getattr(self,k).snapshot() for k in ('full','margin','shuffled')}
