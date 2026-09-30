"""Reuse native B recurrence and existing exact endpoint FiLM scaling."""
import copy,time
import torch
from ..r9_current_first.deployment import Segmenter
from ..r8_ba.host import OnlineHost
from ..r8_ba.context import capture
from ..r7_shared.context import tensor_digest
from ..r7_shared.numerics import finite
from ..r10_use_write_rl.factory import construct as original_construct

CONDITIONS={'B_ZERO':(0.,False),'B_RESET_1':(1.,True),'B_FULL_READOUT_025':(.25,False),'B_RESET_READOUT_025':(.25,True),'B_FULL_1':(1.,False),'C0':(1.,False)}

def state_sha(state):
    return tensor_digest([(k,v) for k,v in sorted(state.items()) if torch.is_tensor(v)])

class ReadoutSegmenter(Segmenter):
    def _film(self,h,v,name):
        out=super()._film(h,v,name)
        rms=float(h.detach().square().mean().sqrt());denom=max(rms,1e-8)
        key=name.removesuffix('_residual_rms')
        self.diagnostics.update({key+'_feature_rms':rms,key+'_near_zero_feature':rms<1e-8,
            key+'_original_increment_ratio':self.diagnostics[name]/denom,
            key+'_readout_increment_ratio':float((out-h).detach().square().mean().sqrt())/denom})
        return out

class CarrierHost(OnlineHost):
    def __init__(self,*args,alpha,**kw):
        self.alpha=alpha
        super().__init__(*args,**kw)
    def check_frozen(self,boundary=False):
        if self.segmenter.alpha!=self.alpha:raise ValueError('carrier output alpha changed')
        super().check_frozen(boundary)
    @torch.no_grad()
    def step(self,image):
        logits,trace=super().step(image)
        if self.method is not None:
            v=self.method.ambient(self.state);finite(v)
            trace.update(code_norm=float(self.method.code(self.state).norm()),ambient_norm=float(v.norm()),
                tanh_saturation=float((v.abs()>3).float().mean()),B_z_norm=float(self.state['z'].norm()),
                B_d_norm=float(self.state['d'].norm()),B_counter=self.state['counter'],B_state_sha256=state_sha(self.state),
                output_alpha=self.alpha,**self.segmenter.diagnostics)
        return logits,trace

def resolved(name):
    if name not in CONDITIONS:raise ValueError('unregistered carrier condition')
    alpha,reset=CONDITIONS[name]
    return dict(arm='C0' if name=='C0' else 'B_CARRIER_FULL_RESET' if reset else 'B_CARRIER_FULL',
        method='C0' if name=='C0' else 'B_PARENT_FULL',seed=20260924,source_job=None,artifact=None,
        diagnostic_condition=name,output_alpha=alpha,history='RESET_HISTORY' if reset else 'FULL',
        readout='h+alpha*(original_film(h,v)-h); amplitude=0.3 unchanged')

def construct(name,c,lock):
    r=resolved(name);old,close=original_construct(r,c,c['output_root'],lock)
    seg=old.segmenter;seg.__class__=ReadoutSegmenter;seg.alpha=r['output_alpha']
    conf=copy.deepcopy(old.context['payload']['config']);source=old.context['payload']['source']
    conf['carrier_readout']=dict(alpha=seg.alpha,history=r['history'],definition=r['readout'])
    host=CarrierHost(seg,old.method,conf,source,capture(seg,old.method,conf,source),old.ablation,alpha=seg.alpha)
    return host,close

def source_item(source,index,visit,seed=20260924):
    from ..r7_shared.source import simulate
    ids,roles,mode=source.schedule('val',index,seed);group=roles[visit][1]
    row=source.data.get(group,'val')
    image=simulate(row.image,source.banks['val'][ids[visit]],f'R10_QUERY|val|{seed}|{index}|{visit}|{group}')
    return image,row.label,mode

def equal_state(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
