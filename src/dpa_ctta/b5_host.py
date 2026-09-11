"""Three exact score-residual BN policies around the unchanged B4 step."""
import torch
from .b4_host import PolypC,C0

SCORE_HEAD_BN=('ra4_conv5.bn','ra3_conv4.bn','ra2_conv4.bn')
FIELDS=('weight','bias','running_mean','running_var','num_batches_tracked')


def policy(arm):
    if arm not in ('F','H','H0'):raise ValueError('unregistered head policy')
    return dict(arm=arm,source_statistics_heads=list(SCORE_HEAD_BN) if arm!='F' else [],frozen_affine_heads=list(SCORE_HEAD_BN),updating_BN_layers=0 if arm=='H0' else 150,updating_affine_scalars=0 if arm=='H0' else 66212,updating_parameter_tensors=0 if arm=='H0' else 300)


def protect(h,state,arm):
    h.policy=policy(arm);modules=dict(h.model.named_modules());h.heads={};h.source_heads={}
    for name in SCORE_HEAD_BN:
        m=modules[name]
        if type(m) is not torch.nn.BatchNorm2d or m.num_features!=1 or not m.affine:raise ValueError('exact score head structure')
        saved={k:state[name+'.'+k].detach().to(h.device).clone() for k in FIELDS}
        if any(saved[k].shape!=getattr(m,k).shape for k in ('weight','bias','num_batches_tracked')):raise ValueError('head state shape')
        h.heads[name]=m;h.source_heads[name]=saved;m.requires_grad_(False)
        if arm!='F':
            m.eval();m.track_running_stats=True
            m.running_mean=saved['running_mean'].clone();m.running_var=saved['running_var'].clone()
            m.num_batches_tracked=saved['num_batches_tracked'].clone()
    h.head_stats={};h.head_input_gradient={};h.policy_handles=[]
    h.policy_handles.append(h.model.register_forward_pre_hook(lambda m,x:check_policy(h)))
    for name,m in h.heads.items():
        def observed(m,args,out,name=name):
            h.head_stats[name]=dict(mean=float(out.detach().mean()),std=float(out.detach().std(unbiased=False)))
            if args[0].requires_grad:
                def grad(g):h.head_input_gradient[name]=float(g.detach().double().norm());return g
                args[0].register_hook(grad)
        h.policy_handles.append(m.register_forward_hook(observed))
    check_policy(h)


def check_policy(h):
    source=h.policy['arm']!='F';update=h.policy['arm']!='H0';count=0;scalars=0
    for name,m in h.model.named_modules():
        if type(m) is not torch.nn.BatchNorm2d:continue
        head=name in SCORE_HEAD_BN
        if head:
            s=h.source_heads[name]
            if m.weight.requires_grad or m.bias.requires_grad:raise ValueError('head affine trainable')
            for k in ('weight','bias','num_batches_tracked'):
                if not torch.equal(getattr(m,k),s[k]):raise ValueError('source head changed')
            if source:
                if m.training or not m.track_running_stats or any(not torch.equal(getattr(m,k),s[k]) for k in ('running_mean','running_var')):raise ValueError('source head statistics policy')
            elif not m.training or m.track_running_stats or m.running_mean is not None or m.running_var is not None:raise ValueError('F current head statistics policy')
        else:
            if not m.training or m.track_running_stats or m.running_mean is not None or m.running_var is not None or m.weight.requires_grad!=update or m.bias.requires_grad!=update:raise ValueError('feature BN policy')
            count+=1;scalars+=m.weight.numel()+m.bias.numel()
    if (count,scalars)!=(150,66212):raise ValueError('actual feature BN enumeration')


def observations(h,z):
    return dict(score_heads_final_original=h.head_stats.copy(),head_input_gradient_l2=h.head_input_gradient.copy(),prediction_foreground_fraction=float((z.sigmoid()>=.5).float().mean()))


class Host(PolypC):
    def __init__(self,arm,state,device='cpu'):
        if arm not in ('F','H'):raise ValueError('updating arm')
        super().__init__(state,device)
        if len(self.names)!=306 or sum(p.numel() for p in self.params)!=66218:raise ValueError('B4 complete BN inventory')
        protect(self,state,arm)
        pairs=[(n,p) for n,p in zip(self.names,self.params) if n.rsplit('.',1)[0] not in SCORE_HEAD_BN]
        self.names=[n for n,p in pairs];self.params=[p for n,p in pairs]
        # Adam is still empty; filter its one existing group before any step.
        if self.base.state or len(self.base.param_groups)!=1:raise ValueError('nonempty parent optimizer')
        self.base.param_groups[0]['params']=self.params
        self.frozen={n:(p,p._version) for n,p in self.model.named_parameters() if not p.requires_grad}
        for hook in self.bn_hooks:hook.remove()
        self.bn_hooks=[m.register_forward_pre_hook(lambda m,x,n=n:self.seen_bn.add(n)) for n,m in self.model.named_modules() if type(m) is torch.nn.BatchNorm2d and n not in SCORE_HEAD_BN]
    def step(self,pixels):
        check_policy(self);before=[p.detach().clone() for p in self.params];self.head_input_gradient={}
        z,m=super().step(pixels);check_policy(self)
        update=float(sum((p.detach()-b).double().square().sum() for p,b in zip(self.params,before)).sqrt())
        m.update(layer_policy=self.policy.copy(),head_state_checked=True,optimizer_update_l2=update,**observations(self,z))
        return z,m
    def finish(self,state):
        check_policy(self);super().finish(state)
        for hook in self.policy_handles:hook.remove()


class H0(C0):
    def __init__(self,state,device='cpu'):
        super().__init__('polyp',state,device);protect(self,state,'H0')
        self.versions={n:(p,p._version) for n,p in self.model.state_dict(keep_vars=True).items()}
    def step(self,pixels):
        z,m=super().step(pixels);check_policy(self)
        m.update(layer_policy=self.policy.copy(),head_state_checked=True,optimizer_update_l2=0.,**observations(self,z))
        return z,m
    def finish(self,state):
        check_policy(self);super().finish(state)
        for hook in self.policy_handles:hook.remove()


def make_host(arm,state,device='cpu'):
    if arm=='all-adapt':return PolypC(state,device)
    return H0(state,device) if arm=='H0' else Host(arm,state,device)
