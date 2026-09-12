"""Image-only host; inherited C device/augmentation order and one Adam step."""
import torch
from ..b1_host import Host as C,official,finite
from ..hosts.vptta import model_input_from_pixels
from ..host_diagnostic import rng,restore,close
from .recovery import TrendController,periodic_due,nested_erasure,dense_sensitivity
from .region_memory import Memory,grid

SUBSPACE_WEIGHT=.05
ARMS=('C','C_PER256','C_SENS','C_PCA_GLOBAL','C_PCA_SHUFFLED','C_PCA_REGION')


def segmentation(output):
    z=output[0] if isinstance(output,(tuple,list)) else output
    if not isinstance(z,torch.Tensor) or z.ndim!=4:raise ValueError('keep batch/channel')
    return z


class Host(C):
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ARMS:raise ValueError('frozen arm')
        super().__init__('C',state,device,model)
        if model is None and (len(self.names),sum(p.numel() for p in self.params))!=(82,19136):raise ValueError('Fundus BN inventory')
        self.variant=arm;self.visit=0;self.reset_count=0
        self.initial=[p.detach().clone() for p in self.params] if arm in ('C_SENS','C_PER256') else []
        self.controller=TrendController() if arm=='C_SENS' else None
        self.memory=Memory(arm.removeprefix('C_PCA_')) if arm.startswith('C_PCA_') else None
        self.pending={};self.phase=None;self.extra_handles=[]
        if self.memory:
            if not hasattr(self.model,'seg_head'):raise ValueError('exact segmentation head input required')
            self.extra_handles=[self.model.seg_head.register_forward_pre_hook(self._features),self.model.register_forward_hook(self._views)]
            original=self.opt.cal_consis_loss
            self.opt.cal_consis_loss=lambda data:original(data,criterion=self._criterion)
    def _features(self,module,args):
        i=self.pending.get('forward',0)
        if self.phase=='core' and i in (0,6):
            f=grid(args[0])
            if f.shape!=(1024,32):raise ValueError('actual 32D head_input')
            self.pending['f0' if i==0 else 'fs']=f.detach().cpu() if i==0 else f
    def _views(self,module,args,output):
        if self.phase!='core':return
        i=self.pending.get('forward',0)
        if i<6:
            z=segmentation(output)
            if i:z=official().Rotate_and_Flip().inverse(z,i-1)
            self.pending.setdefault('views',[]).append(grid(z.detach().cpu()).sigmoid())
        self.pending['forward']=i+1
    def _criterion(self,z,q):
        base=torch.nn.functional.binary_cross_entropy_with_logits(z,q)
        vectors,ids,audit=self.memory.prepare(self.pending['f0'],grid(q),torch.stack(self.pending['views']),self.visit)
        sub=self.memory.loss(self.pending['fs'],ids,self.pending['snapshots']);finite(sub)
        self.pending.update(vectors=vectors,memory_input=audit,subloss=float(sub.detach()),base_loss=float(base.detach()))
        return base+SUBSPACE_WEIGHT*sub
    @torch.no_grad()
    def recover(self):
        if not self.initial:raise ValueError('not a recovery arm')
        before=rng()
        for p,v in zip(self.params,self.initial):p.copy_(v);p.grad=None
        self.base.state.clear();self.base.zero_grad(set_to_none=True)
        self.base.param_groups[0].update(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.steps=0;self.reset_count+=1;close(before,rng(),exact=True)
    def step(self,pixels):
        x=model_input_from_pixels(pixels,'fundus')
        if len(x)!=1 or x.requires_grad:raise ValueError('one current RGB image only')
        self.visit+=1;before=self.counts.copy();control=None;reset=False;restore(self.rng)
        try:
            if self.variant=='C_PER256':reset=periodic_due(self.visit)
            elif self.controller:
                old_rng=rng();versions={n:(v,v._version) for n,v in self.model.state_dict(keep_vars=True).items()}
                views,valid,area=nested_erasure(x.to(self.device),self.visit)
                with torch.no_grad():probs=torch.cat([segmentation(self.model(v)).sigmoid() for v in views])
                s,groups=dense_sensitivity(probs,valid)
                if any(v._version!=n for v,n in versions.values()):raise ValueError('probe state changed')
                close(old_rng,rng(),exact=True)
                control=dict(sensitivity=None if s is None else float(s),regions=groups,area=area,trend=None)
                if s is not None:control['trend']=self.controller.observe(float(s));reset=control['trend']['trigger']
                del views,probs,valid,versions
            if reset:self.recover()
            self.pending={'snapshots':self.memory.snapshots(self.visit)} if self.memory else {}
            old_versions=[None if s is None else s[2] for s in self.pending.get('snapshots',[])]
            self.phase='core';prior=[p.detach().clone() for p in self.params]
            z,a=super().normalized_step(x);self.phase=None
            a.update(global_visit=self.visit,segment_age=self.steps,optimizer_steps_since_reset=self.steps,total_adam_calls=self.counts['base_adam'],reset_before_current=reset,reset_count=self.reset_count,controller=control,optimizer_update_l2=float(sum((p.detach()-v).double().square().sum() for p,v in zip(self.params,prior)).sqrt()),pca=None)
            if self.memory:
                self.memory.merge(self.pending['vectors'],self.visit)
                a['pca']=dict(input=self.pending['memory_input'],basis_versions_used=old_versions,subloss=self.pending['subloss'],base_loss=self.pending['base_loss'],banks=self.memory.audit())
            a['counts']={k:v-before[k] for k,v in self.counts.items()}
            if a['counts']!=dict(forwards=11 if self.controller else 8,backwards=1,base_adam=1,perturb=0,restore=0):raise ValueError('physical count mismatch')
            if self.counts['base_adam']!=self.visit:raise ValueError('one Adam per global visit')
            return z.detach(),a
        finally:self.phase=None;self.pending.clear()
    def finish(self,state):
        super().finish(state)
        for handle in self.extra_handles:handle.remove()
