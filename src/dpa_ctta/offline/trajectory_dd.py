"""M4 native four-image trajectory. Discrete memory selection stays upstream NumPy."""
import copy
from types import MethodType
import numpy as np
import torch
from torch.func import functional_call
from .adaptation_dd import OfflineEpisode, FiniteSqrt, preprocess, cpu_tree
from .prompt_gradient_matching import cosine_objective
from ..medical_losses import medical_loss


class ForeachFiniteSqrt(FiniteSqrt):
    @staticmethod
    def forward(ctx,x):
        y=torch._foreach_sqrt([x])[0];ctx.save_for_backward(y);return y


def native_adam(phi,gradient,state,lr,foreach=None):
    """Functional native dispatcher: CUDA foreach, CPU single-tensor by default.

    Scalar-list division in CUDA foreach is observably different from the
    single-tensor expression at step 19. Keep the actual native operation order,
    while inheriting the validated zero-second-moment derivative convention.
    The optional flag only permits primitive tests to exercise both backends.
    """
    step=int(state.get('step',0))+1
    old_m=state.get('exp_avg',torch.zeros_like(phi))
    old_v=state.get('exp_avg_sq',torch.zeros_like(phi))
    foreach=phi.device.type=='cuda' if foreach is None else foreach
    if foreach:
        m=torch._foreach_lerp([old_m],[gradient],1-.9)[0]
        v=torch._foreach_addcmul(torch._foreach_mul([old_v],.99),[gradient],[gradient],1-.99)[0]
        denominator=torch._foreach_add(torch._foreach_div([ForeachFiniteSqrt.apply(v)],[(1-.99**step)**.5]),1e-8)
        plus=torch._foreach_addcdiv([phi],[m],denominator,[-lr/(1-.9**step)])[0]
    else:
        m=old_m.lerp(gradient,1-.9)
        v=torch.addcmul(old_v*.99,gradient,gradient,value=1-.99)
        denominator=FiniteSqrt.apply(v)/(1-.99**step)**.5+1e-8
        plus=torch.addcdiv(phi,m,denominator,value=-lr/(1-.9**step))
    return plus,dict(step=step,exp_avg=m,exp_avg_sq=v)


def tensor_batch(self, samples, attention):
    # Exact upstream float32 attention and summation order; only values carry graphs.
    weights=np.array(attention/.2)
    weights=np.exp(weights)/np.sum(np.exp(weights))
    value=samples[0]*float(weights[0])
    for i in range(1,len(samples)):value=value+samples[i]*float(weights[i])
    return value.float()


def detach_state(state):
    return {'prompt':state['prompt'].detach(),
            'adam':{k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in state['adam'].items()},
            'memory':{k:v.detach() for k,v in state['memory'].items()},'count':state['count']}


class TrajectoryEpisode(OfflineEpisode):
    def __init__(self,task,source_state,device='cpu'):
        super().__init__(task,source_state,device)
        self.memory=copy.deepcopy(self.host.memory_bank)
        self.memory._prepare_batch=MethodType(tensor_batch,self.memory)
        self.counts.update(memory_pushes=0,retrievals=0,source_visits=0)

    def initial(self,history=None):
        h=self.host
        if history is None:
            return dict(prompt=torch.ones_like(h.prompt.data_prompt),adam={},memory={},count=0)
        opt=next(iter(history['adam']['state'].values()),{})
        return dict(prompt=history['prompt']['data_prompt'].to(self.device),
            adam={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in opt.items()},
            memory={k:torch.as_tensor(v.copy(),device=self.device) for k,v in history['memory'].items()},
            count=next(iter(history['counters'].values()))[0])

    def step(self,method,S,masks,pixels,label,state,capture=False):
        if method not in ('D4','L4','T4'):raise ValueError('M4 objective')
        if method!='T4':state=detach_state(state)
        h=self.host;x=preprocess(pixels.to(self.device),self.task)
        self.count=state['count']+1
        with torch.no_grad():_,key=functional_call(h.prompt,{'data_prompt':state['prompt']},(x,))
        self.counts['prompt_forwards']+=1
        key=key.cpu().numpy()
        self.memory.memory=dict(state['memory'])
        if self.memory.get_size()>=h.neighbor:
            value=self.memory.get_neighbours(key,h.neighbor)
            phi=value[0] if self.task=='fundus' else value
            self.counts['retrievals']+=1
        else:phi=torch.ones_like(h.prompt.data_prompt)
        if not phi.requires_grad:phi=phi.detach().requires_grad_(True)
        logits=self.forward(h.model,x,phi)
        scope=h.model.resnet if self.task=='polyp' else h.model
        losses=[m.bn_loss for m in scope.modules() if isinstance(m,h.adabn)]
        host_loss=sum(losses)/len(losses)
        if method=='D4':
            reference,=torch.autograd.grad(medical_loss(logits,label.to(self.device),beta_boundary=0).region,phi,retain_graph=True)
            reference=reference.detach();self.counts['gradient_calls']+=1
        syn=self.forward(self.clone,preprocess(S,self.task),phi,True)
        proxy_loss=medical_loss(syn,masks.to(self.device),beta_boundary=0).region
        loss=None
        if method=='D4':
            proxy_gradient=self.gradient(proxy_loss,phi,True)
            loss=cosine_objective(proxy_gradient,reference)
        # Native scales the proxy loss before backward, not its resulting gradient.
        # L4's incoming state was detached, so keeping this graph introduces no
        # cross-image path; T4 retains the host Hessian path through that state.
        update_gradient,=torch.autograd.grad(host_loss+.1*proxy_loss,phi,
            create_graph=method!='D4',retain_graph=True)
        self.counts['gradient_calls']+=1
        plus,adam=native_adam(phi,update_gradient,state['adam'],h.optimizer.param_groups[0]['lr'])
        self.counts['differentiable_inner']+=1
        if capture:self.last_inner=cpu_tree(dict(initial_prompt=phi,gradient=update_gradient,input_adam=state['adam']))
        if method=='D4':
            with torch.no_grad():prediction=self.forward(h.model,x,plus)
        else:
            prediction=self.forward(h.model,x,plus)
            loss=medical_loss(prediction,label.to(self.device),beta_boundary=0).region
        # Upstream push retains byte keys, insertion/overwrite order and >40 eviction.
        self.memory.push(key,plus)
        state=dict(prompt=plus,adam=adam,memory=dict(self.memory.memory),count=self.count)
        self.counts['memory_pushes']+=1;self.counts['source_visits']+=1
        if not torch.isfinite(loss) or not torch.isfinite(plus).all():raise ValueError('nonfinite M4 trajectory')
        return loss,state,prediction

    def window(self,method,S,masks,queries,state,capture=False):
        if len(queries) not in (1,4):raise ValueError('four-image window or unit single-step check required')
        state=detach_state(state);losses=[];trace=[]
        for pixels,label in queries:
            loss,state,pred=self.step(method,S,masks,pixels,label,state,capture=capture);losses.append(loss)
            if capture:trace.append(dict(state=cpu_tree(state),prediction=pred.detach().cpu(),loss=loss.detach().cpu(),inner=self.last_inner))
        return torch.stack(losses).mean(),state,trace
