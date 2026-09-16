"""Unchanged B1 C candidate; transaction only changes whether it is retained."""
import copy,time
import torch
from ..b1_host import Host as Core,official
from ..hosts.vptta import model_input_from_pixels
from .rule import Rule,PHYSICAL,measurements,finite


def snapshot(core):
    return dict(params=[p.detach().clone() for p in core.params],
        buffers=[(m,k,None if v is None else v.detach().clone()) for m in core.model.modules() for k,v in m._buffers.items()],
        state={p:copy.deepcopy(v) for p,v in core.base.state.items()},
        groups=[(list(g['params']),copy.deepcopy({k:v for k,v in g.items() if k!='params'})) for g in core.base.param_groups],steps=core.steps)


@torch.no_grad()
def rollback(core,saved):
    for p,v in zip(core.params,saved['params']):p.copy_(v)
    for m,k,v in saved['buffers']:
        now=m._buffers[k]
        if v is None:m._buffers[k]=None
        elif now is not None and now.shape==v.shape and now.dtype==v.dtype:now.copy_(v)
        else:m._buffers[k]=v.clone()
    core.base.state.clear()
    for p,v in saved['state'].items():core.base.state[p]=copy.deepcopy(v)
    if len(core.base.param_groups)!=len(saved['groups']):raise ValueError('optimizer groups changed')
    for g,(params,meta) in zip(core.base.param_groups,saved['groups']):g.clear();g.update(copy.deepcopy(meta));g['params']=list(params)
    core.base.zero_grad(set_to_none=True);core.steps=saved['steps']


class Host:
    def __init__(self,arm,state=None,device='cpu',model=None,p_accept=None):
        self.rule=Rule(arm,p_accept);self.arm=arm;self.core=Core('C',state,device,model)
        self.model=self.core.model;self.params=self.core.params;self.base=self.core.base
        if model is None and (len(self.params),sum(p.numel() for p in self.params))!=(82,19136):raise ValueError('registered BN dimensions')
        if arm=='C_HALF':self.base.param_groups[0]['lr']=5e-5
        self.phase='IDLE';self.failed=False;self.pending={};self.payload=None
        self.totals=dict(n_visits=0,n_candidate_adam=0,n_committed=0,n_rejected=0,n_forced=0,n_eligible=0,parameter_restorations=0)
        self.physical={k:0 for k in PHYSICAL};self.handles=[self.model.register_forward_hook(self._observe)]
        original=self.core.opt.cal_consis_loss
        self.core.opt.cal_consis_loss=lambda data:original(data,criterion=self._criterion)
        finite(list(self.model.parameters()));finite(list(self.model.buffers()))
    def _observe(self,module,args,output):
        if self.phase!='CANDIDATE':raise ValueError('forward outside candidate')
        i=self.pending.get('forwards',0);z=output[0]
        if i==0:self.pending['pre_logits']=z.detach().clone()
        if i<6:
            z=z.detach().cpu()
            if i:z=official().Rotate_and_Flip().inverse(z,i-1)
            self.pending.setdefault('view_logits',[]).append(z.clone())
        self.pending['forwards']=i+1
    def _criterion(self,z,q):
        # Preserve the actual device and BCEWithLogitsLoss reduction of GraTa.
        self.pending['q']=q.detach().clone()
        return torch.nn.BCEWithLogitsLoss()(z,q)
    def step(self,pixels):
        if self.failed or self.phase!='IDLE' or self.payload is not None:raise ValueError('host not ready or previous evaluation not released')
        started=time.monotonic();saved=None
        try:
            x=model_input_from_pixels(pixels,'fundus')
            if len(x)!=1 or x.requires_grad:raise ValueError('one current image')
            self.phase='CANDIDATE';self.pending={}
            if self.arm in ('C_VERIFY','C_RANDOM'):saved=snapshot(self.core)
            # The C candidate path and its Adam-step assertion remain untouched.
            trial,_=self.core.normalized_step(x)
            finite(list(self.model.parameters()));finite(list(self.model.buffers()));finite(self.base.state);finite(self.base.param_groups)
            if self.pending['forwards']!=8 or len(self.pending['view_logits'])!=6:raise ValueError('six weak / strong / trial contract')
            pre=self.pending['pre_logits'];q=self.pending['q'];views=torch.stack(self.pending.pop('view_logits')).sigmoid()
            self.phase='DECIDE';obs=measurements(pre.sigmoid(),q,trial.sigmoid(),views)
            decision=self.rule.decide(obs);accept=decision['accept']
            self.phase='COMMIT' if accept else 'ROLLBACK'
            if not accept:rollback(self.core,saved)
            self.rule.append(obs['r'])
            t=self.totals;t['n_visits']+=1;t['n_candidate_adam']+=1;t['n_committed']+=int(accept);t['n_rejected']+=int(not accept)
            t['n_forced']+=int(decision['forced']);t['n_eligible']+=int(decision['eligible']);t['parameter_restorations']+=int(not accept)
            if t['n_visits']!=t['n_committed']+t['n_rejected'] or self.core.steps!=t['n_committed']:raise ValueError('commit conservation')
            if any(int(self.base.state[p]['step'])!=t['n_committed'] for p in self.params if p in self.base.state):raise ValueError('committed Adam steps')
            for k,v in PHYSICAL.items():self.physical[k]+=v
            if self.core.counts['forwards']!=self.physical['network_forwards'] or self.core.counts['backwards']!=t['n_candidate_adam'] or self.core.counts['base_adam']!=t['n_candidate_adam']:raise ValueError('physical calls')
            emit=trial if accept else pre
            self.payload=dict(pre_logits=pre,trial_logits=trial,q=q.detach(),emit_logits=emit)
            self.pending.clear();self.core.last.clear();self.phase='EVALUATION_RELEASE'
            trace=dict(arm=self.arm,**decision,counts=PHYSICAL.copy(),totals=t.copy(),adam_committed_step=self.core.steps,buffer_count_after=len(self.rule.history),source_unchanged=True,transaction_complete=True,host_seconds=time.monotonic()-started)
            return emit.detach(),trace
        except BaseException:
            self.failed=True;self.phase='FAILED';self.pending.clear();self.payload=None;raise
    def take_evaluation(self):
        if self.phase!='EVALUATION_RELEASE' or self.payload is None:raise ValueError('no committed payload')
        value=self.payload;self.payload=None;self.phase='IDLE';return value
    def finish(self,state):
        if self.failed or self.phase!='IDLE' or self.payload is not None:raise ValueError('unfinished/failed host')
        self.core.finish(state)
        for h in self.handles:h.remove()
