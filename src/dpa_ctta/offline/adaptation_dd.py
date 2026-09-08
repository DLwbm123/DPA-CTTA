"""M1 source-only truncated adaptation objective, preserving native forward semantics."""
import copy
import torch
from torch.func import functional_call

from ..hosts.vptta import VPTTAHost
from ..medical_losses import medical_loss


class FiniteSqrt(torch.autograd.Function):
    """Exact sqrt forward; derivative zero at exactly zero, native derivative elsewhere.

    The explicit zero subgradient avoids 0 * infinity in zero Adam second moments.
    It changes no forward epsilon and does not detach the image-to-update path.
    """
    @staticmethod
    def forward(ctx,x):
        y=x.sqrt();ctx.save_for_backward(y);return y
    @staticmethod
    def backward(ctx,grad):
        y,=ctx.saved_tensors
        divisor=torch.where(y>0,2*y,torch.ones_like(y))
        return torch.where(y>0,grad/divisor,torch.zeros_like(grad))


def native_adam(phi,gradient,state,lr):
    """Functional equivalent of non-AMSGrad, non-decoupled native Adam, one prompt."""
    step=int(state.get('step',0))+1
    m=state.get('exp_avg',torch.zeros_like(phi)).lerp(gradient,1-.9)
    v=state.get('exp_avg_sq',torch.zeros_like(phi))*.99+gradient.square()*(1-.99)
    denominator=FiniteSqrt.apply(v)/(1-.99**step)**.5+1e-8
    updated=phi-(lr/(1-.9**step))*m/denominator
    return updated,dict(step=step,exp_avg=m,exp_avg_sq=v)


def preprocess(pixels,task):
    """Already resized RGB [0,1], differentiable on device; never FixedProxy/PIL."""
    if task=='fundus':
        lo=pixels.amin((1,2,3),keepdim=True);span=pixels.amax((1,2,3),keepdim=True)-lo
        if (span==0).any(): raise ValueError('constant Fundus pixels')
        return (pixels-lo)/span
    if task!='polyp': raise ValueError('unknown task')
    return (pixels-pixels.new_tensor([.485,.456,.406]).view(1,3,1,1))/pixels.new_tensor([.229,.224,.225]).view(1,3,1,1)


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def history_state(host):
    return dict(prompt=cpu_tree(host.prompt.state_dict()),adam=cpu_tree(host.optimizer.state_dict()),
        memory=copy.deepcopy(host.memory_bank.memory),
        counters={n:(m.sample_num,m.new_sample) for n,m in host.model.named_modules() if isinstance(m,host.adabn)})


def restore_history(host,state):
    host.prompt.load_state_dict(state['prompt']);host.optimizer.load_state_dict(copy.deepcopy(state['adam']))
    host.memory_bank.memory=copy.deepcopy(state['memory'])
    for n,m in host.model.named_modules():
        if isinstance(m,host.adabn): m.sample_num,m.new_sample=state['counters'][n]


class OfflineEpisode:
    """One reusable isolated live/clone pair. Immutable Base history is never updated."""
    def __init__(self,task,source_state,device='cpu'):
        self.host=VPTTAHost(task,source_state=source_state,device=device)
        h=self.host;h.audit_initial_state();h._initial_hook.remove();h._started=True
        h.model.requires_grad_(False)
        self.clone=copy.deepcopy(h.model).eval().requires_grad_(False)
        self.task=task;self.device=h.device;self.counts=dict(live_forwards=0,proxy_forwards=0,proxy_images=0,prompt_forwards=0,gradient_calls=0,differentiable_inner=0)

    def initialize(self,pixels,state):
        h=self.host;restore_history(h,state);h.model.eval();h.prompt.train()
        x=preprocess(pixels.to(self.device),self.task)
        if h.memory_bank.get_size()>=h.neighbor:
            with torch.no_grad(): _,low=h.prompt(x)
            self.counts['prompt_forwards']+=1
            retrieved=h.memory_bank.get_neighbours(low.cpu().numpy(),h.neighbor)
            init=retrieved[0] if self.task=='fundus' else retrieved
        else: init=torch.ones_like(h.prompt.data_prompt)
        h.prompt.update(init)
        phi=h.prompt.data_prompt.detach().clone().requires_grad_(True)
        opt=h.optimizer.state.get(h.prompt.data_prompt,{})
        opt={k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in opt.items()}
        counts={v[0] for v in state['counters'].values()}
        if len(counts)!=1: raise ValueError('history counter mismatch')
        self.count=counts.pop()+1
        return x,phi,opt

    def forward(self,model,x,phi,proxy=False):
        h=self.host
        for m in model.modules():
            if isinstance(m,h.adabn): m.sample_num=self.count;m.new_sample=False
        px,_=functional_call(h.prompt,{'data_prompt':phi},(x,))
        self.counts['prompt_forwards']+=1
        self.counts['proxy_forwards' if proxy else 'live_forwards']+=1
        if proxy:
            if len(x)!=4: raise ValueError('full K=4 required')
            self.counts['proxy_images']+=len(x)
        output=model(px)
        return output[0] if isinstance(output,tuple) else output

    def gradient(self,loss,parameter,create_graph=False):
        self.counts['gradient_calls']+=1
        g,=torch.autograd.grad(loss,parameter,create_graph=create_graph)
        if not torch.isfinite(g).all(): raise ValueError('nonfinite prompt gradient')
        return g

    def objective(self,method,S,masks,pixels,query_mask,state):
        if method not in {'D','O'} or not S.requires_grad: raise ValueError('trainable D/O pixels required')
        x,phi,opt=self.initialize(pixels,state)
        masks=masks.to(self.device);query_mask=query_mask.to(self.device)
        logits=self.forward(self.host.model,x,phi)
        if method=='D':
            reference=self.gradient(medical_loss(logits,query_mask,beta_boundary=0).region,phi).detach()
        else:
            scope=self.host.model.resnet if self.task=='polyp' else self.host.model
            losses=[m.bn_loss for m in scope.modules() if isinstance(m,self.host.adabn)]
            if not losses: raise ValueError('empty native loss coverage')
            reference=self.gradient(sum(losses)/len(losses),phi).detach()
        del logits
        syn=self.forward(self.clone,preprocess(S,self.task),phi,True)
        g=self.gradient(medical_loss(syn,masks,beta_boundary=0).region,phi,True)
        if method=='D':
            from .prompt_gradient_matching import cosine_objective
            loss=cosine_objective(g,reference)
        else:
            plus,_=native_adam(phi,reference+.1*g,opt,self.host.optimizer.param_groups[0]['lr'])
            self.counts['differentiable_inner']+=1
            after=self.forward(self.host.model,x,plus)
            loss=medical_loss(after,query_mask,beta_boundary=0).region
        if not torch.isfinite(loss): raise ValueError('nonfinite outer objective')
        return loss

    def clear_graphs(self):
        for model in (self.host.model,self.clone):
            for m in model.modules():
                if hasattr(m,'bn_loss'): m.bn_loss=m.bn_loss.detach()
            for hook in getattr(model,'feature_hooks',[]):
                if isinstance(hook.features,torch.Tensor): hook.features=hook.features.detach()
