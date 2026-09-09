"""B1: pinned GraTa ent/consis functions on the unchanged P2 segmentation model.

Upstream MIT code is imported, not copied. See docs/B1_GRATA_TRANSFER_CONTRACT.md.
"""
import copy
import functools
import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import torch
from .source_pilot import SourceOnlyHost
from .hosts.vptta import model_input_from_pixels
from .host_diagnostic import rng,restore

GRATA_COMMIT='33ae20d664f305af34739ec54a5bec7da53ffa0b'


@functools.lru_cache(maxsize=1)
def official():
    root=Path(os.environ['DPA_GRATA_ROOT']).resolve()
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=GRATA_COMMIT or subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('dirty/wrong GraTa dependency')
    if importlib.metadata.version('batchgenerators')!='0.25.2':raise ValueError('augmentation dependency version')
    prefixes=('dataloaders','custom_optimizers')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('foreign GraTa import cache')
    previous=sys.path[:];bytecode=sys.dont_write_bytecode;sys.dont_write_bytecode=True;sys.path.insert(0,str(root/'GraTa-master'))
    try:
        module=importlib.import_module('custom_optimizers.grata')
        return module
    finally:
        for k in tuple(sys.modules):
            if k.split('.')[0] in prefixes:del sys.modules[k]
        sys.path[:]=previous;sys.dont_write_bytecode=bytecode


def configure(model):
    model.train().requires_grad_(False);names=[];params=[]
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.BatchNorm2d):
            if type(m) is not torch.nn.BatchNorm2d or not m.affine:raise ValueError('standard affine BN required')
            m.requires_grad_(True);m.track_running_stats=False;m.running_mean=None;m.running_var=None
            for key,p in m.named_parameters(recurse=False):names.append(name+'.'+key);params.append(p)
    if not params or [n for n,p in model.named_parameters() if p.requires_grad]!=names:raise ValueError('BN ownership')
    return names,params


def finite(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():raise ValueError('B1 nonfinite tensor')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)


class Host:
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ['C','G']:raise ValueError('B1 arm')
        self.arm=arm;self.device=torch.device(device)
        self.model=SourceOnlyHost('fundus',state,device).model if model is None else model.to(device)
        self.names,self.params=configure(self.model)
        self.base=torch.optim.Adam(self.params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.opt=official().GraTa(self.params,self.base,self.model,device=str(self.device))
        self.counts=dict(forwards=0,backwards=0,base_adam=0,perturb=0,restore=0)
        self.last={};self.steps=0;self.handles=[]
        self.frozen={n:(p,p._version) for n,p in self.model.named_parameters() if not p.requires_grad}
        self.handles.append(self.model.register_forward_hook(self._forward))
        self.handles.append(self.params[0].register_hook(self._backward))
        self.handles.append(self.base.register_step_pre_hook(self._before_adam))
        self.handles.append(self.base.register_step_post_hook(self._after_adam))
        for key in ['cal_ent_loss','cal_consis_loss','perturb_weights_sub','unperturb','get_cosine']:
            original=getattr(self.opt,key)
            def observed(*args,_key=key,_fn=original,**kwargs):
                result=_fn(*args,**kwargs)
                if _key.startswith('cal_'):
                    finite(result);self.last[_key]=float(result.detach())
                elif _key=='get_cosine':finite(result);self.last['cosine']=float(result)
                else:
                    self.counts['perturb' if _key=='perturb_weights_sub' else 'restore']+=1
                    if _key=='unperturb':
                        for p in self.params:
                            if not torch.equal(p,self.opt.state[p]['old_p']):raise ValueError('parameter restoration')
                return result
            setattr(self.opt,key,observed)
        def blocked(*args,**kwargs):raise ValueError('B1 forbids labels and auxiliary heads')
        for key in ['cal_groundtruth_loss','cal_recon_loss','cal_supres_loss','cal_denoise_loss','cal_rotate_loss']:
            setattr(self.opt,key,blocked)
        self.rng=rng()

    def _forward(self,m,args,output):
        self.counts['forwards']+=1;finite(output[0])

    def _backward(self,g):
        self.counts['backwards']+=1;finite(g);return g

    def _before_adam(self,opt,args,kwargs):
        if [id(p) for g in opt.param_groups for p in g['params']]!=[id(p) for p in self.params]:raise ValueError('optimizer ownership drift')
        if any(p.grad is None for p in self.params):raise ValueError('unused registered BN')
        finite([p.grad for p in self.params]);finite(opt.param_groups[0]['lr'])

    def _after_adam(self,opt,args,kwargs):self.counts['base_adam']+=1

    def step(self,pixels):
        if not isinstance(pixels,torch.Tensor) or len(pixels)!=1 or pixels.requires_grad:raise ValueError('one current image only')
        return self.normalized_step(model_input_from_pixels(pixels,'fundus'))

    def normalized_step(self,x):
        """Internal normalized input; production uses step(), toy tests use this seam."""
        if x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=4 or len(x)!=1 or x.requires_grad:raise ValueError('normalized image contract')
        finite(x);restore(self.rng);before=self.counts.copy();self.last={}
        # Strong augmentation mutates its numpy dictionary. Preserve original final x.
        data={'data':x.numpy().copy()};original=x.to(self.device).clone()
        if self.arm=='G':self.opt.step(data,aux='ent',pse='consis')
        else:self.opt.cal_consis_loss(data);self.base.step()
        with torch.no_grad():logits=self.model(original)[0].detach()
        finite(self.base.state);finite(self.params);self.steps+=1;self.rng=rng()
        for p in self.params:
            if int(self.base.state[p]['step'])!=self.steps:raise ValueError('Adam step drift')
        if any(p._version!=v or p.grad is not None for p,v in self.frozen.values()):raise ValueError('non-BN parameter changed')
        counts={k:v-before[k] for k,v in self.counts.items()}
        if counts!=expected_counts(self.arm):raise ValueError('B1 physical counters')
        return logits,dict(counts=counts,adam_step=self.steps,lr=float(self.base.param_groups[0]['lr']),diagnostics=self.last.copy())

    def finish(self,state):
        allowed=set(self.names)
        for n,p in self.model.state_dict().items():
            if n not in allowed and not torch.equal(p.cpu(),state[n]):raise ValueError('non-affine source state changed')
        for handle in self.handles:handle.remove()


def expected_counts(arm):
    return dict(forwards=8 if arm=='C' else 9,backwards=1 if arm=='C' else 2,base_adam=1,perturb=int(arm=='G'),restore=int(arm=='G'))


def reference_step(model,opt,arm,x):
    """Faithful published functions, bypassing Host's wrapper and its instrumentation."""
    data={'data':x.numpy().copy()};original=x.to(opt.device).clone()
    if arm=='G':official().GraTa.step(opt,data,aux='ent',pse='consis')
    else:official().GraTa.cal_consis_loss(opt,data);opt.base_optimizer.step()
    with torch.no_grad():return model(original)[0].detach()
