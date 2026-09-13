"""Composition C engine. Algorithms receive only the current RGB tensor.

C/RP delegate to unchanged R1. New families use the same pinned weak/strong
operations, with explicit intervention points and independently counted VJPs.
"""
import copy,time
import torch
from torch.nn import functional as F
from ..b1_host import configure,official,finite
from ..source_pilot import SourceOnlyHost
from ..hosts.vptta import model_input_from_pixels
from ..host_diagnostic import rng,restore
from ..r1.host import Host as LegacyHost,segmentation
from ..r1.region_memory import grid
from .stats import Memory
from .kernels import unit
from .teachers import density_target,graph_target
from .displacement import flatten,probe_rows,transform
from .contexts import Contexts
from .transport import commit as transport_commit

ARMS=('C','RP','T_LR','T_ISO','T_DIAG','U_PCA','U_RAND','U_SCALE','S_JOINT','S_SHARED','S_NOPCA','M_TRANSPORT','M_IDPOST','M_SHUFFLE','G_PCA','G_ISO','G_ORDER')


class Host:
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ARMS:raise ValueError('unknown R3 arm')
        self.arm=arm;self.visit=0;self.failed=False;self.pending={};self.phase='IDLE';self.reference=None;self.contexts=None
        self.legacy=LegacyHost('C' if arm=='C' else 'C_PCA_REGION',state,device,model) if arm in ('C','RP') else None
        if self.legacy:
            self.model=self.legacy.model;self.params=self.legacy.params;self.names=self.legacy.names;self.base=self.legacy.base
            self.memory=self.legacy.memory;return
        self.device=torch.device(device)
        self.model=SourceOnlyHost('fundus',state,device).model if model is None else model.to(device)
        saved=rng()
        try:
            if arm.startswith(('T_','S_')):
                self.reference=copy.deepcopy(self.model).eval().requires_grad_(False)
                if any(not m.track_running_stats or m.running_mean is None or m.running_var is None for m in self.reference.modules() if isinstance(m,torch.nn.BatchNorm2d)):
                    raise ValueError('measurement clone requires original source BN buffers')
        finally:restore(saved)
        self.names,self.params=configure(self.model)
        if not hasattr(self.model,'seg_head'):raise ValueError('32D segmentation head input required')
        self.base=torch.optim.Adam(self.params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.memory=Memory();self.rng=rng()
        self.frozen={n:(p,p._version) for n,p in self.model.named_parameters() if not p.requires_grad}
        self.ref_versions={} if self.reference is None else {n:(v,v._version) for n,v in self.reference.state_dict(keep_vars=True).items()}
        self.handles=[self.model.seg_head.register_forward_pre_hook(self._tap),self.model.register_forward_hook(self._count_forward),self.base.register_step_pre_hook(self._count_adam)]
        if self.reference is not None:self.handles.extend([self.reference.seg_head.register_forward_pre_hook(self._ref_tap),self.reference.register_forward_hook(self._count_forward)])
        if arm.startswith('S_'):self.contexts=Contexts(self.params,self.names,self.base,arm=='S_SHARED')
        self.total=dict(network_forwards=0,loss_backward_calls=0,jacobian_vjp_calls=0,adam_calls=0,actual_parameter_replacements=0)

    def _tap(self,module,args):
        if self.phase in ('ORIGINAL','STRONG','PREDICT'):
            value=grid(args[0])
            if value.shape!=(1024,32):raise ValueError('seg_head input dimension')
            self.pending[self.phase]=value

    def _ref_tap(self,module,args):
        raw=args[0];self.pending['REFERENCE']=grid(raw).detach().cpu()
        if self.arm.startswith('S_'):
            desc=torch.cat([raw.double().mean((0,2,3)),raw.double().std((0,2,3),unbiased=False)]).cpu()
            self.pending['descriptor']=desc/desc.norm().clamp_min(1e-12)

    def _count_forward(self,module,args,output):
        self.total['network_forwards']+=1;finite(segmentation(output))

    def _count_adam(self,opt,args,kwargs):self.total['adam_calls']+=1

    def _count_vjp(self):self.total['jacobian_vjp_calls']+=1

    def step(self,pixels):
        if self.failed:raise RuntimeError('failed trajectory cannot be resumed')
        if self.legacy:
            z,trace=self.legacy.step(pixels);self.visit=self.legacy.visit
            c=trace['counts'];trace['r3']=dict(arm=self.arm,global_visit=self.visit,counts=dict(network_forwards=c['forwards'],loss_backward_calls=c['backwards'],jacobian_vjp_calls=0,adam_calls=c['base_adam'],actual_parameter_replacements=0))
            return z,trace
        x=model_input_from_pixels(pixels,'fundus')
        return self.normalized_step(x)

    def normalized_step(self,x):
        if self.legacy:raise ValueError('baseline uses original public step')
        if self.failed:raise RuntimeError('failed trajectory cannot be resumed')
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.shape!=(1,3,512,512) or x.requires_grad:
            raise ValueError('one detached current normalized RGB')
        finite(x);restore(self.rng);self.visit+=1;before_counts=self.total.copy();started=time.monotonic()
        trace=dict(arm=self.arm,global_visit=self.visit);route=None
        try:
            data={'data':x.numpy().copy()};original=x.to(self.device).clone()
            if self.reference is not None:
                saved=rng()
                with torch.no_grad():self.reference(original)
                restore(saved)
                if self.contexts:
                    self.memory,route=self.contexts.load(self.pending['descriptor'])
                    self.total['actual_parameter_replacements']+=route['actual_parameter_replacements']
            snapshots=self.memory.snapshots(self.visit);densities=self.memory.density_snapshots(self.visit)
            trace['basis_versions_used']=[None if s is None else s[2] for s in snapshots]
            trace['density_versions_used']=[None if s is None else s[2] for s in densities]
            self.phase='ORIGINAL'
            with torch.set_grad_enabled(self.arm.startswith('U_')):logits=segmentation(self.model(original))
            predictions=[logits.detach().cpu()];del logits
            self.phase='WEAK';weak=official().Rotate_and_Flip()
            for factor in range(5):
                with torch.no_grad():z=segmentation(self.model(weak(original,factor)));predictions.append(weak.inverse(z,factor).detach().cpu())
            views=torch.stack([grid(z).sigmoid() for z in predictions])
            q=torch.stack(predictions).sigmoid().mean(0).to(self.device);del predictions,z
            pre=self.pending['ORIGINAL'];memory_raw=self.pending['REFERENCE'] if self.arm.startswith('T_') else pre
            selected,ids,vectors,audit=self.memory.select(memory_raw,grid(q),views,self.visit)
            del memory_raw
            trace['memory_input']=audit;target=q
            if self.arm.startswith('T_'):target,trace['teacher']=density_target(q,self.pending['REFERENCE'],densities,self.arm[2:])
            if self.arm.startswith('G_'):target,trace['graph']=graph_target(q,pre,views,densities,self.arm[2:])
            rows=None;definitions=[];probes=[]
            if self.arm.startswith('U_'):
                t=time.monotonic();rows,definitions,probes=probe_rows(pre,selected,snapshots,self.params,self.arm=='U_RAND',self._count_vjp)
                trace['jacobian_seconds']=time.monotonic()-t
                # Drop all original autograd references before strong loss backward.
                pre=pre.detach();self.pending['ORIGINAL']=pre
            self.base.zero_grad();self.phase='STRONG'
            aug=official().augmentation_strong_style(data)
            aug=official().normalize_image_to_0_1(torch.from_numpy(aug).float().to(self.device))
            strong=segmentation(self.model(aug));loss=F.binary_cross_entropy_with_logits(strong,target.detach())
            base_loss=float(loss.detach());extra=loss.new_zeros(())
            if self.arm.startswith('M_') or self.arm in ('S_JOINT','S_SHARED'):
                extra=self.memory.loss(self.pending['STRONG'],ids,snapshots);loss=loss+.05*extra
            finite(loss);before=flatten(self.params);self.total['loss_backward_calls']+=1;loss.backward()
            if any(p.grad is None for p in self.params):raise ValueError('unused BN in C loss')
            finite([p.grad for p in self.params]);self.base.step()
            if rows is not None:
                trace['update']=transform(before,self.params,rows,self.arm[2:])
                self.total['actual_parameter_replacements']+=trace['update']['actual_parameter_replacements']
            self.phase='PREDICT'
            with torch.no_grad():result=segmentation(self.model(original)).detach()
            post=self.pending['PREDICT'].detach()
            if definitions:
                postv=unit(post);values=[float(postv[i.to(post.device)].mean(0)@u) for i,u in definitions]
                trace['update']['actual_post_probe_difference_l2']=float(torch.tensor(values,dtype=torch.float64).sub(torch.tensor(probes)).norm())
            self.phase='COMMIT'
            if self.arm.startswith('M_'):trace['transport']=transport_commit(self.memory,pre,post,selected,self.visit,self.arm[2:])
            else:self.memory.merge(vectors,self.visit)
            if self.contexts:
                self.contexts.commit(self.pending['descriptor'],route);trace['context']=route
            elif any(int(self.base.state[p]['step'])!=self.visit for p in self.params):raise ValueError('global Adam step drift')
            finite(self.params);finite(self.base.state)
            if any(p._version!=version or p.grad is not None for p,version in self.frozen.values()):raise ValueError('frozen source parameter changed')
            if any(p._version!=version or p.requires_grad for p,version in self.ref_versions.values()):raise ValueError('measurement clone changed')
            self.rng=rng()
            counts={k:v-before_counts[k] for k,v in self.total.items()}
            if counts['network_forwards']!=(9 if self.reference is not None else 8) or counts['loss_backward_calls']!=1 or counts['adam_calls']!=1 or counts['jacobian_vjp_calls']>8:
                raise ValueError('R3 physical count mismatch')
            trace.update(counts=counts,base_loss=base_loss,extra_loss=float(extra.detach()),banks=self.memory.audit(),
                optimizer_update_l2=float((flatten(self.params)-before).double().norm()),
                final_raw_probability_nested_violations=int((result[:,1].sigmoid()>result[:,0].sigmoid()).sum()),host_seconds=time.monotonic()-started)
            finite(trace)
            return result,dict(r3=trace)
        except BaseException:
            self.failed=True;raise
        finally:self.pending.clear();self.phase='IDLE'

    def finish(self,state):
        if self.legacy:return self.legacy.finish(state)
        for name,value in self.model.state_dict().items():
            if name not in self.names and not torch.equal(value.cpu(),state[name]):raise ValueError('frozen source state changed')
        for handle in self.handles:handle.remove()
