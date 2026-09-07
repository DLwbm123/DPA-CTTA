"""Isolated source-only H1 diagnostics; native trajectories remain unchanged."""
import copy
from contextlib import contextmanager
import random
import time

import numpy as np
import torch

from .hosts.vptta import model_input_from_pixels
from .medical_losses import medical_loss
from .source_pilot import assemble, evaluate_after_step
from .source_pilot_release import TOLERANCE, source_unchanged

BRANCHES = {'A_cf':(1,0,0,0,1), 'B_cf':(1,.1,0,0,1), 'C_cf':(1,.1,.01,0,1),
            'A_small':(1,0,0,0,.1), 'O_native':(0,0,0,1,1), 'O_small':(0,0,0,1,.1)}
POSITIONS = (1,2,3,4,17,18,19,20)


def rng():
    return (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None,
            copy.deepcopy(np.random.get_state()), random.getstate())


def restore(value):
    torch.set_rng_state(value[0])
    if value[1] is not None: torch.cuda.set_rng_state(value[1])
    np.random.set_state(value[2]); random.setstate(value[3])


def close(a, b, *, exact=False):
    if isinstance(a, torch.Tensor):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('nonfinite state')
        torch.testing.assert_close(a,b,rtol=0 if exact else TOLERANCE['rtol'],atol=0 if exact else TOLERANCE['atol'])
    elif isinstance(a,np.ndarray):
        if exact: np.testing.assert_array_equal(a,b)
        else: np.testing.assert_allclose(a,b,**TOLERANCE)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError('state keys differ')
        for k in a: close(a[k],b[k],exact=exact or k=='step')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise ValueError('state lengths differ')
        for x,y in zip(a,b): close(x,y,exact=exact)
    elif a != b: raise ValueError('state scalar differs')


@contextmanager
def isolated_rng():
    state=rng()
    try: yield
    finally: restore(state)


def snapshot(host):
    return dict(prompt=copy.deepcopy(host.prompt.state_dict()), adam=copy.deepcopy(host.optimizer.state_dict()),
        memory=copy.deepcopy({k:v for k,v in vars(host.memory_bank).items() if not callable(v)}),
        counters=[(n,m.sample_num,m.new_sample) for n,m in host.model.named_modules() if isinstance(m,host.adabn)],
        buffers={n:t.detach().clone() for n,t in host.model.named_buffers()})


def metrics(logits, mask, task):
    values=evaluate_after_step(logits,mask,task)
    hard=logits.detach().cpu().sigmoid()>=.5; gt=mask.bool()
    for c,m in enumerate(values):
        a,b=hard[0,c],gt[0,c]; total=a.numel(); pos=int(b.sum()); pred=int(a.sum())
        fp=int((a & ~b).sum()); fn=int((~a & b).sum())
        m.update(pred_foreground_pixels=pred,gt_foreground_pixels=pos,total_pixels=total,
            false_positive_pixels=fp,false_negative_pixels=fn,false_positive_denominator=total-pos,
            false_negative_denominator=pos,false_positive_rate=fp/(total-pos) if total!=pos else None,
            false_negative_rate=fn/pos if pos else None)
    if task=='fundus':
        outside=int((hard[0,1] & ~hard[0,0]).sum()); denominator=int(hard[0,1].sum())
        for m in values: m.update(pred_oc_outside_od_pixels=outside,pred_oc_pixels_denominator=denominator,
                                  pred_oc_outside_od_ratio=outside/denominator if denominator else None)
    return values


def angle(a,b):
    norm=float(a.norm())*float(b.norm())
    return float(torch.dot(a.flatten(),b.flatten())/norm) if norm else None


class DiagnosticPair:
    """Two native histories plus reusable independent diagnostic models, one task at a time."""
    def __init__(self, task, state, proxy, device='cpu'):
        self.task=task; self.state=state; self.device=torch.device(device); self.visit=0
        self.reference=assemble(task,'A',state,device=device); self.reference_rng=rng()
        self.watched=assemble(task,'A',state,device=device); self.watched_rng=rng()
        close(self.reference_rng,self.watched_rng,exact=True)
        self.diagnostic=assemble(task,'C',state,proxy,device=device)
        self.noadapt=assemble(task,'N',state,device=device)
        d=self.diagnostic
        d.audit_initial_state(); d._initial_hook.remove(); d._started=True
        d.model.requires_grad_(False)  # Only isolated prompt gradients; live native source .grad remains untouched.
        if len(d._proxy_input)!=4: raise ValueError('K must be full batch four')
        self.counts=dict(reference_steps=0,watched_steps=0,counterfactual_steps=0,
                        reference_pushes=0,watched_pushes=0,gradient_calls=0)
        self.handles=[]; self.captured={}; self.retrieval=[]
        self.source_versions={label:{n:(t,t._version) for n,t in h.model.state_dict(keep_vars=True).items()}
                              for label,h in [('reference',self.reference),('watched',self.watched),('diagnostic',d),('N',self.noadapt)]}
        for label,h in [('reference',self.reference),('watched',self.watched),('diagnostic',d)]:
            self.counts[label+'_model_forwards']=0;self.counts[label+'_prompt_forwards']=0
            def count_model(m,args,label=label): self.counts[label+'_model_forwards']+=1
            def count_prompt(m,args,label=label): self.counts[label+'_prompt_forwards']+=1
            self.handles.extend([h.model.register_forward_pre_hook(count_model),h.prompt.register_forward_pre_hook(count_prompt)])
            def count_step(opt,args,kwargs,label=label): self.counts['counterfactual_steps' if label=='diagnostic' else label+'_steps']+=1
            self.handles.append(h.optimizer.register_step_post_hook(count_step))
        self.counts.update(N_model_forwards=0,proxy_model_forwards=0,proxy_images=0)
        def n_count(m,args): self.counts['N_model_forwards']+=1
        def proxy_count(m,args):
            if len(args[0])!=4: raise ValueError('microbatch forbidden')
            self.counts['proxy_model_forwards']+=1;self.counts['proxy_images']+=4
        self.handles.extend([self.noadapt.model.register_forward_pre_hook(n_count),d._proxy_model.register_forward_pre_hook(proxy_count)])
        for label,h in [('reference',self.reference),('watched',self.watched)]:
            original=h.memory_bank.push
            def push(*args,_original=original,_label=label,**kw):
                result=_original(*args,**kw);self.counts[_label+'_pushes']+=1;return result
            h.memory_bank.push=push
        original_get=self.watched.memory_bank.get_neighbours
        def get(*args,**kw):
            result=original_get(*args,**kw);self.retrieval.append(self.visit);return result
        self.watched.memory_bank.get_neighbours=get
        def capture(opt,args,kwargs):
            p=self.watched.prompt.data_prompt
            self.captured=dict(prompt=p.detach().clone(),adam=copy.deepcopy(opt.state_dict()),
                gradient=p.grad.detach().clone(),count=sorted({m.sample_num for m in self.watched.model.modules() if isinstance(m,self.watched.adabn)}))
        self.handles.append(self.watched.optimizer.register_step_pre_hook(capture))

    def reset_diagnostic(self, prompt, lr_factor=1):
        d=self.diagnostic
        d.prompt.update(prompt);d.prompt.train()
        d.optimizer.load_state_dict(copy.deepcopy(self.captured['adam']))
        for group in d.optimizer.param_groups: group['lr']*=lr_factor
        d.optimizer.zero_grad(set_to_none=True)
        for model in (d.model,d._proxy_model):
            for m in model.modules():
                if isinstance(m,d.adabn): m.sample_num=self.visit;m.new_sample=False

    def forward(self,x):
        d=self.diagnostic
        output=d.model(d.prompt(x)[0])
        return output[0] if isinstance(output,tuple) else output

    def gradient(self,loss,retain_graph=False):
        self.counts['gradient_calls']+=1
        value,=torch.autograd.grad(loss,self.diagnostic.prompt.data_prompt,retain_graph=retain_graph)
        if not torch.isfinite(value).all(): raise ValueError('nonfinite diagnostic gradient')
        return value.detach()

    def native(self,image):
        self.visit+=1
        restore(self.reference_rng);reference=self.reference.step(image);self.reference_rng=rng()
        restore(self.watched_rng);watched=self.watched.step(image);self.watched_rng=rng()
        close(self.reference_rng,self.watched_rng,exact=True);close(reference,watched)
        close(snapshot(self.reference),snapshot(self.watched))
        for a,b in zip(self.reference.model.parameters(),self.watched.model.parameters()): close(a.grad,b.grad)
        if self.captured['count']!=[self.visit]: raise ValueError('native counter mismatch')
        for label in ('reference','watched'):
            if self.counts[label+'_steps']!=self.visit or self.counts[label+'_pushes']!=self.visit: raise ValueError('native lifecycle mismatch')
        return reference.detach(),watched.detach(),float((reference-watched).abs().max())

    def diagnose(self,image,watched,mask_reader,counterfactual):
        """All non-oracle updates/predictions finish before mask_reader is called."""
        d=self.diagnostic;p=d.prompt.data_prompt;init=self.captured['prompt'];base_rng=rng()
        before=snapshot(self.watched); before_hooks=[h.features for h in getattr(self.watched.model,'feature_hooks',[])]
        started=time.perf_counter()
        with isolated_rng():
            x=model_input_from_pixels(image,self.task).to(self.device)
            with torch.no_grad():
                n=self.noadapt.step(image)
                self.reset_diagnostic(torch.ones_like(init));g=self.forward(x)
                self.reset_diagnostic(init);i=self.forward(x)
            logits=dict(N=n,G=g,I=i,U=watched);branch_logits={};updates={};gradients={};branch_state={}
            if counterfactual:
                self.reset_diagnostic(init)
                self.forward(x)
                modules=d.model.resnet.modules() if self.task=='polyp' else d.model.modules()
                losses=[m.bn_loss for m in modules if isinstance(m,d.adabn)]
                gradients['H']=self.gradient(sum(losses)/len(losses))
                close(gradients['H'],self.captured['gradient'])
                # Separate backward graphs reduce memory; linear gradient combinations retain one joint Adam update.
                loss=d._proxy_term()
                gradients['R']=self.gradient(d.last_proxy_loss.region,retain_graph=True)
                gradients['D']=self.gradient(d.last_proxy_loss.boundary)
                del loss; d.last_proxy_loss=None
                for name in ('A_cf','B_cf','C_cf','A_small'):
                    restore(base_rng);h,r,b,q,lr=BRANCHES[name];self.reset_diagnostic(init,lr)
                    p.grad=(h*gradients['H']+r*gradients['R']+b*gradients['D']).clone()
                    d.optimizer.step();updates[name]=(p.detach()-init).clone()
                    branch_state[name]=dict(adam_step=int(d.optimizer.state[p]['step']),lr=d.optimizer.param_groups[0]['lr'])
                    with torch.no_grad(): branch_logits[name]=self.forward(x).detach()
                    if name=='A_cf':
                        close(p,self.watched.prompt.data_prompt);close(branch_logits[name],watched)
                        close(d.optimizer.state_dict(),self.watched.optimizer.state_dict())
                # Source query supervision is first obtained here, after all non-oracle branches are fixed.
            mask=mask_reader()
            scores={name:metrics(value,mask,self.task) for name,value in logits.items()}
            branches={};gradient_report={}
            if counterfactual:
                self.reset_diagnostic(init);pre=self.forward(x)
                gradients['Q']=self.gradient(medical_loss(pre,mask.to(self.device),beta_boundary=0).region)
                del pre
                for name in ('O_native','O_small'):
                    restore(base_rng);self.reset_diagnostic(init,BRANCHES[name][-1]);p.grad=gradients['Q'].clone()
                    d.optimizer.step();updates[name]=(p.detach()-init).clone()
                    branch_state[name]=dict(adam_step=int(d.optimizer.state[p]['step']),lr=d.optimizer.param_groups[0]['lr'])
                    with torch.no_grad(): branch_logits[name]=self.forward(x).detach()
                for name,value in branch_logits.items():
                    u=updates[name];cos=angle(gradients['Q'],u)
                    norm=float(gradients['Q'].norm())*float(u.norm())
                    alignment=-float(torch.dot(gradients['Q'].flatten(),u.flatten()))/(norm+1e-12) if norm else None
                    branches[name]=dict(metrics=metrics(value,mask,self.task),update_norm=float(u.norm()),
                        distance_from_A_cf=float((u-updates['A_cf']).norm()),alignment=alignment,
                        **branch_state[name],adam_calls=1)
                gradient_report=dict(raw_norm={k:float(v.norm()) for k,v in gradients.items()},
                    weighted_norm={'H':float(gradients['H'].norm()),'0.1R':float((.1*gradients['R']).norm()),
                                   '0.01D':float((.01*gradients['D']).norm()),'Q':float(gradients['Q'].norm())},
                    cosine={a+'_'+b:angle(gradients[a],gradients[b]) for a,b in [('H','Q'),('R','Q'),('D','Q'),('H','R'),('H','D'),('R','D')]})
            result=dict(predictions=scores,branches=branches,gradients=gradient_report,
                retrieval_observed=self.visit in self.retrieval,adam_step=self.visit,
                memory_size=self.watched.memory_bank.get_size(),counter=self.captured['count'],
                native_update_norm=float((self.watched.prompt.data_prompt.detach()-init).norm()),
                diagnostics_elapsed_seconds=time.perf_counter()-started)
        close(base_rng,rng(),exact=True);close(before,snapshot(self.watched),exact=True)
        for a,b in zip(self.reference.model.parameters(),self.watched.model.parameters()): close(a.grad,b.grad)
        for old,hook in zip(before_hooks,getattr(self.watched.model,'feature_hooks',[])):
            if old is not hook.features: raise ValueError('diagnostic polluted live feature hook')
        for label,h in [('reference',self.reference),('watched',self.watched),('diagnostic',d),('N',self.noadapt)]:
            current=h.model.state_dict(keep_vars=True)
            if any(current[n] is not t or t._version!=version for n,(t,version) in self.source_versions[label].items()):
                raise ValueError('source parameter/buffer mutation')
        if {m.sample_num for m in d.model.modules() if isinstance(m,d.adabn)}!={self.visit}: raise ValueError('diagnostic advanced counters')
        return result

    def identity_check(self,image):
        with isolated_rng(),torch.no_grad():
            x=model_input_from_pixels(image,self.task).to(self.device)
            self.diagnostic.prompt.update(torch.ones_like(self.diagnostic.prompt.data_prompt))
            y=self.diagnostic.prompt(x)[0]
            negligible=torch.allclose(x,y,**TOLERANCE)
            return dict(max_abs_error=float((x-y).abs().max()),within_tolerance=negligible,
                        interpretation='normalization' if negligible else 'normalization_and_identity_prompt_numerical_path')

    def finish(self):
        for h in (self.reference,self.watched,self.diagnostic,self.noadapt): source_unchanged(h,self.state)
        if any(p.grad is not None for p in self.diagnostic.model.parameters()): raise ValueError('isolated source gradient changed')
        for handle in self.handles: handle.remove()
        return dict(self.counts)
