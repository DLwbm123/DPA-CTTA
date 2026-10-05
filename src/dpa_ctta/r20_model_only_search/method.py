"""Native GraTa hosts with causal teacher, positive pixel weights and online modules.

The pinned optimizer still owns its original asymmetric entropy, six weak views,
strong augmentation and BN-only alignment. Modifications enter its criterion.
"""
import copy,random,math
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..b1_host import Host as Native,official,finite,configure
from ..r19_model_only.method import clone,equal
from ..host_diagnostic import rng,restore as restore_native_rng
from ..r8_ba.rng import capture,restore
from ..r7_shared.context import tensor_digest
from ..hosts.vptta import model_input_from_pixels
from ..integrations.ctta_suite import build_reference_model

FAMILIES={'teacher':'T','pixel_weight':'W','online_adapter':'A','target_prototype':'P'}


def components(config):
    if 'components' in config:return copy.deepcopy(config['components'])
    key=FAMILIES.get(config['family'])
    return {key:copy.deepcopy(config['params'])} if key else {}


def boundary(mask,radius=3):
    k=radius*2+1;x=mask.float();d=F.max_pool2d(x,k,1,radius)>0
    e=(1-F.max_pool2d(F.pad(1-x,(radius,)*4,value=1),k,1))>0
    return d&~e


def weights_for(predictions,tau,beta,enabled=True):
    q=predictions.mean(0)
    if not enabled:return torch.ones_like(q)
    variance=(predictions-q).square().mean(0)
    raw=(.25+.75*torch.exp(-variance/tau))*(1+beta*boundary(q>=.5,3))
    return (raw/(raw.mean((-2,-1),keepdim=True)+1e-8)).detach()


class Adapter(nn.Module):
    def __init__(self,channels,rank,device,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.V=nn.Conv2d(channels,rank,1);self.U=nn.Conv2d(rank,channels,1)
            nn.init.zeros_(self.U.weight);nn.init.zeros_(self.U.bias)
        self.to(device);self.last={}
    def forward(self,h):
        scale=(h.square().mean(1,keepdim=True)+1e-6).sqrt().detach()
        a=self.U(F.gelu(self.V(h/scale)));delta=.1*scale*torch.tanh(a)
        self.last=dict(relative_rms=float((delta.square().mean()/(h.square().mean()+1e-12)).sqrt().detach()),saturation=float((a.abs()>3).float().mean().detach()))
        return h+delta


class Host:
    def __init__(self,state,config,seed,identity,device='cuda:0',model=None):
        self.config=copy.deepcopy(config);self.modules=components(config);self.seed=seed;self.context={'sha256':identity};self.visits=0;self.diag={};self.weak=[];self.collect=False
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        arm='C' if config.get('host') in ('plain_consistency','C') else 'G'
        template=copy.deepcopy(model) if model is not None else None
        self.model_factory=(lambda:copy.deepcopy(template)) if model is not None else (lambda:build_reference_model('fundus')[0])
        self.native=Native(arm,state=state,device=device,model=model);n=self.native;self.device=n.device
        self.initial=state;self.handles=[];self.adapter=None;self.teacher=None;self.teacher_adapter=None;self.f0=None;self.proto={i:deque() for i in range(3)};self.current_proto={};self.q_teacher=None;self.proto_target=None;self.feature=None
        self.fixed_digest=tensor_digest([(k,v) for k,v in n.model.state_dict().items() if k not in n.names])
        saved=capture()
        if 'A' in self.modules and not self.modules['A'].get('disabled',False):
            if not hasattr(n.model,'up3') or not hasattr(n.model.up3,'bn'):raise ValueError('verified up3 BN channel definition required')
            self.channels=int(n.model.up3.bn.num_features)
            self.adapter=Adapter(self.channels,int(self.modules['A']['rank']),n.device,seed+1000003)
            self.handles.append(n.model.up3.register_forward_hook(lambda m,a,h:self.adapter(h)))
            n.base.add_param_group({'params':list(self.adapter.parameters()),'lr':1e-4*float(self.modules['A']['adapter_lr_multiplier'])})
            n.opt.param_groups=[n.base.param_groups[0]]  # Alignment, perturbation and norms remain BN-only.
            n.handles[2].remove()  # Replace BN-only Adam ownership check with an explicit two-group check.
            n.handles[2]=n.base.register_step_pre_hook(self._adapter_before_adam)
            aux=n.opt.cal_ent_loss
            def bn_entropy(*args,**kwargs):
                self.adapter.requires_grad_(False)
                try:return aux(*args,**kwargs)
                finally:self.adapter.requires_grad_(True)
            n.opt.cal_ent_loss=bn_entropy
        if 'T' in self.modules and float(self.modules['T']['teacher_mix'])!=0:
            self.teacher=self._new_model(state,adaptive=True)
            if self.adapter:
                self.teacher_adapter=copy.deepcopy(self.adapter).requires_grad_(False)
                self.handles.append(self.teacher.up3.register_forward_hook(lambda m,a,h:self.teacher_adapter(h)))
        if 'P' in self.modules and float(self.modules['P']['prototype_mix'])!=0:
            self.f0=self._new_model(state,adaptive=False)
            if not hasattr(self.f0,'up3'):raise ValueError('prototype feature up3 absent')
            self.handles.append(self.f0.up3.register_forward_hook(self._feature))
            self.f0_digest=tensor_digest(list(self.f0.state_dict().items()))
        restore(saved)
        self.handles.append(n.base.register_step_pre_hook(self._lr))
        self.handles.append(n.model.register_forward_hook(self._weak_hook))
        original=n.opt.cal_consis_loss
        def consis(data):
            self.weak=[];self.collect=True
            if self.adapter:
                for p in self.adapter.parameters():p.grad=None
            try:return original(data,criterion=self._criterion)
            finally:self.collect=False
        n.opt.cal_consis_loss=consis

    def _new_model(self,state,adaptive):
        m=self.model_factory()
        if state is not None:m.load_state_dict(state)
        m.to(self.device)
        if adaptive:configure(m)
        else:m.eval()
        return m.requires_grad_(False)

    def _feature(self,m,args,h):self.feature=h.detach()

    def _weak_hook(self,m,args,out):
        if not self.collect or len(self.weak)>=6:return
        z=out[0]
        if self.weak:z=official().Rotate_and_Flip().inverse(z,len(self.weak)-1)
        self.weak.append(z.detach().cpu())

    def _lr(self,opt,args,kwargs):
        factor=float(self.config.get('final_lr_multiplier',self.config.get('params',{}).get('final_lr_multiplier',1)))
        opt.param_groups[0]['lr']*=factor
        if self.adapter:opt.param_groups[1]['lr']=opt.param_groups[0]['lr']*float(self.modules['A']['adapter_lr_multiplier'])
        self.diag['BN_gradient_norm']=float(torch.stack([p.grad.detach().norm() for p in self.native.params]).norm())
        self.diag['BN_lr']=float(opt.param_groups[0]['lr'])

    def _adapter_before_adam(self,opt,args,kwargs):
        wanted=list(self.native.params)+list(self.adapter.parameters())
        if [id(p) for g in opt.param_groups for p in g['params']]!=[id(p) for p in wanted]:raise ValueError('Adam parameter ownership')
        if any(p.grad is None for p in wanted):raise ValueError('unconnected BN/adapter gradient')
        finite([p.grad for p in wanted]);self.diag['adapter_U_gradient']=float(self.adapter.U.weight.grad.norm());self.diag['adapter_V_gradient']=float(self.adapter.V.weight.grad.norm())

    def _criterion(self,z,q_native):
        q=q_native.detach();pieces=self.modules
        if len(self.weak)!=6:raise ValueError('six aligned native views required')
        predictions=torch.stack(self.weak).sigmoid()
        if not torch.equal(predictions.mean(0).to(q),q):raise ValueError('native pseudo-label capture mismatch')
        if 'T' in pieces and float(pieces['T']['teacher_mix'])!=0:
            mix=float(pieces['T']['teacher_mix']);q=(1-mix)*q+mix*self.q_teacher.to(q)
        if 'P' in pieces and self.proto_target is not None:
            mask=((q_native[:,0:1]-.5).abs()<=.2)|((q_native[:,1:2]-.5).abs()<=.2)
            q=q+float(pieces['P']['prototype_mix'])*mask*(self.proto_target.to(q)-q)
            self.diag['prototype_edit_coverage']=float(mask.float().mean());self.diag['prototype_target_change']=float((q-q_native).abs().mean())
        q=q.detach();finite(q);w=None
        if 'W' in pieces and not pieces['W'].get('disabled',False):
            p=pieces['W'];w=weights_for(predictions,float(p['variance_temperature']),float(p['boundary_boost']),not p.get('variance_disabled',False)).to(z)
            # Variance-disabled boundary ablation still retains its boundary factor.
            if p.get('variance_disabled',False):
                raw=1+float(p['boundary_boost'])*boundary(q_native>=.5,3);w=(raw/(raw.mean((-2,-1),keepdim=True)+1e-8)).detach().to(z)
            loss=(F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean()
            band=boundary(q_native>=.5,3);fg=(q_native>=.5)&~band;bg=(q_native<.5)&~band
            masks=[fg,band,bg]
            self.diag['weights']=[[float(w[0,c][mask[0,c]].mean()) if mask[0,c].any() else None for mask in masks] for c in range(2)]
            self.diag['weight_channel_means']=[float(a) for a in w.mean((-2,-1)).flatten()]
            def grad_record(g):
                self.diag['logit_gradient_regions']=[[float(g[0,c][mask[0,c]].norm()) if mask[0,c].any() else None for mask in masks] for c in range(2)]
            z.register_hook(grad_record)
        else:loss=F.binary_cross_entropy_with_logits(z,q)
        self.diag['pseudo_loss']=float(loss.detach());self.diag['pseudo_target_change']=float((q-q_native).abs().mean())
        self.diag['target_entropy']=float((-(q.clamp(1e-6,1-1e-6)*q.clamp(1e-6,1-1e-6).log()+(1-q).clamp(1e-6,1-1e-6)*(1-q).clamp(1e-6,1-1e-6).log())).mean())
        return loss

    @torch.no_grad()
    def _readonly(self,model,x):
        saved=capture();buffers={k:v.clone() for k,v in model.named_buffers()};modes={k:m.training for k,m in model.named_modules()}
        try:return model(model_input_from_pixels(x,'fundus').to(self.device))[0].detach().float()
        finally:
            for k,v in model.named_buffers():v.copy_(buffers[k])
            for k,m in model.named_modules():m.training=modes[k]
            restore(saved)

    def _prepare_teacher(self,x):
        if self.teacher is None:self.q_teacher=None;return
        if self.modules['T'].get('current_student',False):
            with torch.no_grad():
                for k,p in self.teacher.named_parameters():p.copy_(dict(self.native.model.named_parameters())[k])
                if self.teacher_adapter:self.teacher_adapter.load_state_dict(self.adapter.state_dict())
        with torch.no_grad():self.q_teacher=(self._readonly(self.teacher,x).sigmoid()+torch.flip(self._readonly(self.teacher,torch.flip(x,(-1,))).sigmoid(),(-1,)))*.5
        self.diag['teacher_prediction_entropy']=float((-(self.q_teacher.clamp(1e-6,1-1e-6)*self.q_teacher.clamp(1e-6,1-1e-6).log())).mean())

    def _prepare_prototypes(self,x):
        self.proto_target=None;self.current_proto={}
        if self.f0 is None:return
        p=self.modules['P'];t=self.visits+1;expired=0
        for bank in self.proto.values():
            while bank and t-bank[0]['time']>=int(p.get('expiry_arrivals',128)):bank.popleft();expired+=1
        with torch.no_grad():
            q0=self._readonly(self.f0,x).sigmoid();features=self.feature.clone();qh=torch.flip(self._readonly(self.f0,torch.flip(x,(-1,))).sigmoid(),(-1,))
            def classes(q):return [(q[:,0:1]<=.1)&(q[:,1:2]<=.1),(q[:,0:1]>=.9)&(q[:,1:2]<=.1),(q[:,0:1]>=.9)&(q[:,1:2]>=.9)]
            tokens=F.normalize(features,dim=1,eps=1e-8);counts=[]
            for i,(a,b) in enumerate(zip(classes(q0),classes(qh))):
                mask=F.adaptive_avg_pool2d((a&b).float(),features.shape[-2:])==1;count=int(mask.sum());counts.append(count)
                if count>=int(p.get('min_tokens_per_class',32)):
                    v=features[0,:,mask[0,0]].mean(1);v=F.normalize(v,dim=0,eps=1e-8)
                    if torch.isfinite(v).all() and v.norm()>0:self.current_proto[i]=dict(vector=v.detach().cpu(),time=t)
            banks={k:list(v) for k,v in self.proto.items()}
            if p.get('current_only',False):banks={k:([self.current_proto[k]] if k in self.current_proto else []) for k in range(3)}
            self.diag.update(prototype_tokens=counts,prototype_expired=expired,prototype_empty=not all(banks.values()),prototype_edit_coverage=0.,prototype_target_change=0.)
            if all(banks.values()):
                scores=[];ages=[]
                for k,items in banks.items():
                    vectors=torch.stack([a['vector'] for a in items]).to(features)
                    cosine=torch.einsum('kc,bchw->bkhw',vectors,tokens);scores.append(cosine.topk(min(2,len(items)),dim=1).values.mean(1));ages.extend(t-a['time'] for a in items)
                sim=torch.stack(scores,1)
                if torch.isfinite(sim).all():
                    pi=(sim/float(p['cosine_temperature'])).softmax(1);q=torch.stack((pi[:,1]+pi[:,2],pi[:,2]),1)
                    self.proto_target=F.interpolate(q,size=(512,512),mode='bilinear',align_corners=False).detach()
                self.diag['prototype_mean_age']=float(np.mean(ages))

    def step(self,x):
        if x.shape!=(1,3,512,512) or x.dtype!=torch.float32 or not torch.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('current legal image tensor')
        self.diag={};self._prepare_teacher(x);self._prepare_prototypes(x)
        before=[p.detach().clone() for p in self.native.params]
        steps=int(self.config.get('params',{}).get('online_steps_per_arrival',1));details=[]
        for _ in range(steps):z,d=self.native.step(x);details.append(d)
        out=z.detach().float()
        if self.config.get('params',{}).get('horizontal_output_weight',0):
            # Native state includes its diagnostic counters and gradients: restore every side effect.
            saved=self.snapshot();h=self._readonly(self.native.model,torch.flip(x,(-1,)))
            self.restore(saved);p=.75*out.sigmoid()+.25*torch.flip(h.sigmoid(),(-1,));out=torch.logit(p.clamp(1e-6,1-1e-6))
        if self.teacher is not None:
            alpha=float(self.modules['T']['ema_alpha'])
            with torch.no_grad():
                src=dict(self.native.model.named_parameters())
                for k,p in self.teacher.named_parameters():
                    if k in self.native.names:p.mul_(alpha).add_(src[k],alpha=1-alpha)
                if self.teacher_adapter:
                    for p,q in zip(self.teacher_adapter.parameters(),self.adapter.parameters()):p.mul_(alpha).add_(q,alpha=1-alpha)
        writes=evictions=0
        if self.f0 is not None:
            for k,v in self.current_proto.items():
                self.proto[k].append(v);writes+=1
                while len(self.proto[k])>int(self.modules['P'].get('capacity_per_class',8)):self.proto[k].popleft();evictions+=1
        self.visits+=1
        self.diag.update(BN_update_norm=float(torch.stack([(p-q).norm() for p,q in zip(self.native.params,before)]).norm()),prototype_writes=writes,prototype_evictions=evictions,prototype_bank_sizes=[len(x) for x in self.proto.values()])
        if self.adapter:self.diag.update(adapter=self.adapter.last)
        finite(self.diag)
        def scalar_finite(v):
            if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite diagnostic')
            if isinstance(v,dict):
                for x in v.values():scalar_finite(x)
            if isinstance(v,(list,tuple)):
                for x in v:scalar_finite(x)
        scalar_finite(self.diag)
        return out.cpu(),dict(visit=self.visits,native=details,diagnostics=copy.deepcopy(self.diag),state_committed=True)

    def snapshot(self):
        n=self.native;saved=capture();restore_native_rng(n.rng);nr=capture();restore(saved)
        return dict(context=self.context,visits=self.visits,parameters={k:clone(v) for k,v in n.model.named_parameters() if k in n.names},buffers={k:clone(v) for k,v in n.model.named_buffers()},gradients={k:None if p.grad is None else clone(p.grad) for k,p in n.model.named_parameters()},adam=clone(n.base.state_dict()),grata=clone(n.opt.state_dict()),counts=n.counts.copy(),last=copy.deepcopy(n.last),steps=n.steps,rng=saved,native_rng=nr,adapter=None if self.adapter is None else clone(self.adapter.state_dict()),adapter_gradients=None if self.adapter is None else [None if p.grad is None else clone(p.grad) for p in self.adapter.parameters()],teacher=None if self.teacher is None else {k:clone(p) for k,p in self.teacher.named_parameters() if k in n.names},teacher_adapter=None if self.teacher_adapter is None else clone(self.teacher_adapter.state_dict()),prototypes={k:clone(list(v)) for k,v in self.proto.items()},modes={k:m.training for k,m in n.model.named_modules()})

    def restore(self,s):
        if s['context']!=self.context:raise ValueError('snapshot identity')
        n=self.native;ps=dict(n.model.named_parameters());bs=dict(n.model.named_buffers())
        with torch.no_grad():
            for k,v in s['parameters'].items():ps[k].copy_(v.to(ps[k]))
            for k,v in s['buffers'].items():bs[k].copy_(v.to(bs[k]))
            if self.teacher:
                ts=dict(self.teacher.named_parameters())
                for k,v in s['teacher'].items():ts[k].copy_(v.to(ts[k]))
        for k,p in ps.items():p.grad=None if s['gradients'][k] is None else s['gradients'][k].to(p).clone()
        groups=n.base.param_groups;n.opt.load_state_dict(copy.deepcopy(s['grata']));n.base.param_groups=groups;n.base.load_state_dict(copy.deepcopy(s['adam']));n.opt.param_groups=[n.base.param_groups[0]] if self.adapter else n.base.param_groups
        if self.adapter:
            self.adapter.load_state_dict(s['adapter'])
            for p,g in zip(self.adapter.parameters(),s['adapter_gradients']):p.grad=None if g is None else g.to(p).clone()
        if self.teacher_adapter:self.teacher_adapter.load_state_dict(s['teacher_adapter'])
        n.steps=s['steps'];n.counts=s['counts'].copy();n.last=copy.deepcopy(s['last']);self.visits=s['visits'];self.proto={k:deque(clone(v)) for k,v in s['prototypes'].items()}
        for k,m in n.model.named_modules():m.training=s['modes'][k]
        restore(s['native_rng']);n.rng=rng();restore(s['rng'])

    def check_frozen(self,boundary=False):
        if not boundary:return
        n=self.native
        if tensor_digest([(k,v) for k,v in n.model.state_dict().items() if k not in n.names])!=self.fixed_digest:raise ValueError('unadapted weights/buffers changed')
        if self.f0 is not None and tensor_digest(list(self.f0.state_dict().items()))!=self.f0_digest:raise ValueError('prototype coordinate encoder changed')

    def close(self):
        for h in self.handles+self.native.handles:h.remove()
