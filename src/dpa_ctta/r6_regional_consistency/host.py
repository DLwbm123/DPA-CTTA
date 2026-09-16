"""Unchanged C chain; narrow loss seam, image-only committed payload boundary."""
import time
import torch
from ..b1_host import Host as Core
from ..hosts.vptta import model_input_from_pixels
from .loss import ARMS,PHYSICAL,TOLERANCES,finite,objective


def physical(core):
    return dict(network_forwards=core.counts['forwards'],loss_backward_calls=core.counts['backwards'],adam_calls=core.counts['base_adam'],jacobian_vjp_calls=0)


class Host:
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ARMS:raise ValueError('R6 arm')
        self.arm=arm;self.core=Core('C',state,device,model)
        self.model=self.core.model;self.params=self.core.params;self.base=self.core.base
        if model is None and (len(self.params),sum(p.numel() for p in self.params))!=(82,19136):raise ValueError('registered BN inventory')
        self.phase='IDLE';self.failed=False;self.pending={};self.payload=None;self.visit=0;self.physical=physical(self.core)
        self.handles=[self.model.register_forward_hook(self._observe),self.base.register_step_pre_hook(self._update)]
        original=self.core.opt.cal_consis_loss
        self.core.opt.cal_consis_loss=lambda data:original(data,criterion=self._criterion)
        finite(list(self.model.parameters()));finite(list(self.model.buffers()))
    def _observe(self,module,args,output):
        i=self.pending.get('forwards',0)
        if i==0 and self.phase=='WEAK':self.pending['pre_logits']=output[0].detach().clone()
        elif i<6 and self.phase=='WEAK':pass
        elif i==6 and self.phase=='WEAK':self.phase='STRONG_LOSS'
        elif i==7 and self.phase=='UPDATE':self.phase='POST'
        else:raise ValueError('R6 forward lifecycle')
        self.pending['forwards']=i+1
    def _criterion(self,z,q):
        if self.phase!='STRONG_LOSS' or q.shape!=(1,2,512,512):raise ValueError('actual criterion contract')
        self.pending['q']=q.detach().clone();loss,rows=objective(z,q,self.arm,self.visit)
        self.pending.update(channels=rows,actual_loss=float(loss.detach()),gradient_hooks=0)
        def observed(g):
            finite(g);norm=g.detach().double().square().sum(dim=(2,3)).sqrt().flatten().cpu()
            want=torch.tensor([r['expected_logit_gradient_l2'] for r in rows],dtype=torch.float64)
            if not torch.allclose(norm,want,**TOLERANCES[z.dtype]):raise ValueError('strong gradient hook mismatch')
            for r,v in zip(rows,norm.tolist()):r['actual_logit_gradient_l2']=v
            self.pending['gradient_hooks']+=1
            return g
        self.handles.append(z.register_hook(observed))
        return loss
    def _update(self,*args):
        if self.phase!='STRONG_LOSS' or self.pending['gradient_hooks']!=1:raise ValueError('one strong backward required')
        self.phase='UPDATE'
    def step(self,pixels):
        if self.failed or self.phase!='IDLE' or self.payload is not None:raise ValueError('host not ready / pending payload')
        start=time.monotonic();before=physical(self.core);prior=[p.detach().clone() for p in self.params]
        try:
            finite(self.base.state);finite(self.base.param_groups)
            x=model_input_from_pixels(pixels,'fundus')
            if x.shape!=(1,3,512,512) or x.requires_grad:raise ValueError('one RGB image only')
            self.visit+=1;self.phase='WEAK';self.pending={}
            post,_=self.core.normalized_step(x)
            finite(self.params);finite(list(self.model.buffers()));finite(self.base.state);finite(self.base.param_groups)
            self.physical=physical(self.core);delta={k:v-before[k] for k,v in self.physical.items()}
            if self.phase!='POST' or delta!=PHYSICAL or self.core.steps!=self.visit:raise ValueError('physical / lifecycle drift')
            self.phase='COMMIT'
            trace=dict(arm=self.arm,visit=self.visit,counts=delta,cumulative=self.physical.copy(),adam_step=self.core.steps,
                channels=self.pending['channels'],actual_loss=self.pending['actual_loss'],
                bn_gradient_l2=float(sum(p.grad.detach().double().square().sum() for p in self.params).sqrt()),
                adam_affine_displacement_l2=float(sum((p.detach().double()-v.double()).square().sum() for p,v in zip(self.params,prior)).sqrt()),
                source_unchanged=True,transaction_complete=True,host_seconds=time.monotonic()-start)
            self.payload=dict(pre_logits=self.pending['pre_logits'],q=self.pending['q'],post_logits=post)
            self.pending.clear();self.core.last.clear()
            # Per-logit hook must not retain an earlier graph or image.
            for handle in self.handles[2:]:handle.remove()
            self.handles=self.handles[:2];self.phase='EVALUATION_RELEASE'
            return post.detach(),trace
        except BaseException:
            self.failed=True;self.phase='FAILED';self.pending.clear();self.payload=None
            for handle in self.handles[2:]:handle.remove()
            self.handles=self.handles[:2]
            raise
    def take_evaluation(self):
        if self.phase!='EVALUATION_RELEASE' or self.payload is None:raise ValueError('no committed payload')
        value=self.payload;self.payload=None;self.phase='IDLE';return value
    def finish(self,state):
        if self.failed or self.phase!='IDLE' or self.payload is not None:raise ValueError('unfinished / failed host')
        self.core.finish(state)
        for handle in self.handles:handle.remove()
