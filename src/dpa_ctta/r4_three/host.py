"""Image-only R4T host. Dense evaluator payloads leave only after state commit."""
import copy,time,types
import torch
from torch.nn import functional as F
from ..b1_host import configure,official,finite
from ..source_pilot import SourceOnlyHost
from ..hosts.vptta import model_input_from_pixels
from ..host_diagnostic import rng,restore
from ..r1.host import Host as OldHost,segmentation
from ..r1.region_memory import Memory,tokens,grid
from ..r1.recovery import local_seed
from .kernel_geometry import KernelGeometry
from .boundary_graph import refine_target

ARMS=('C','RP','MT','MT_RP','FT','FT_RP','KDG','K_ALL','K_MAG','K_FREE','G_BOUND','G_CONST','G_SHUFFLE','G_GLOBAL')
RP_ARMS=('RP','MT_RP','FT_RP')
K_PATHS=(('res.conv1',(64,3,7,7),(2,2),(3,3)),('res.layer1.2.conv2',(64,64,3,3),(1,1),(1,1)))


class ObservedMemory(Memory):
    """Unchanged R1 preparation; retain only exact selected positions for evaluation."""
    def __init__(self):
        super().__init__('REGION');self.selected=None
    def prepare(self,raw,q,views,visit):
        result=super().prepare(raw,q,views,visit)
        regions,reliable=tokens(q,views);norm=raw.detach().cpu().norm(dim=1);selected=[]
        for r,ids in enumerate(regions):
            ids=ids[reliable[ids]];g=torch.Generator().manual_seed(local_seed('sample_'+str(r),visit))
            if len(ids)>32:ids=ids[torch.randperm(len(ids),generator=g)[:32]]
            selected.append(ids[norm[ids%1024]>0]%1024)
        if [len(i) for i in selected]!=result[2]['selected_region_counts']:raise ValueError('selected token audit')
        self.selected=selected
        return result


def effective_forward(module,x):
    return F.conv2d(x,module.r4_adapter(),None,module.stride,module.padding,module.dilation,module.groups)


class Host:
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ARMS:raise ValueError('R4T arm')
        self.arm=arm;self.device=torch.device(device);self.visit=0;self.failed=False;self.phase='IDLE'
        self.pending={};self.evaluation_pending=None;self.teacher=None;self.kernels={};self.handles=[]
        self.legacy=OldHost('C' if arm=='C' else 'C_PCA_REGION',state,device,model) if arm in ('C','RP') else None
        if self.legacy:
            h=self.legacy;self.model=h.model;self.names=h.names;self.params=h.params;self.base=h.base
            if arm=='RP':h.memory=ObservedMemory()
            self.memory=h.memory;self.bn_params=h.params
            if arm=='RP':
                original=h._criterion
                def criterion(z,q):
                    self.pending['q']=q.detach().cpu();return original(z,q)
                h._criterion=criterion
            else:
                def criterion(z,q):
                    self.pending['q']=q.detach().cpu();return F.binary_cross_entropy_with_logits(z,q)
                original=h.opt.cal_consis_loss
                h.opt.cal_consis_loss=lambda data:original(data,criterion=criterion)
            return
        self.model=SourceOnlyHost('fundus',state,device).model if model is None else model.to(device)
        self.names,self.bn_params=configure(self.model)
        if model is None and (len(self.names),sum(p.numel() for p in self.bn_params))!=(82,19136):raise ValueError('BN dimensions')
        self.frozen={n:(p,p._version) for n,p in self.model.named_parameters() if not p.requires_grad}
        if arm.startswith(('MT','FT')):
            saved=rng();self.teacher=copy.deepcopy(self.model).requires_grad_(False);restore(saved)
            self.teacher_params=[dict(self.teacher.named_parameters())[n] for n in self.names]
            self.teacher_frozen={n:(p,p._version) for n,p in self.teacher.named_parameters() if n not in self.names}
        extra=[]
        if arm.startswith('K'):
            for path,shape,stride,padding in K_PATHS:
                m=self.model.get_submodule(path)
                if not isinstance(m,torch.nn.Conv2d) or tuple(m.weight.shape)!=shape or m.stride!=stride or m.padding!=padding or m.groups!=1 or m.bias is not None:raise ValueError('fixed kernel entry mismatch: '+path)
                m.add_module('r4_adapter',KernelGeometry(m.weight,arm))
                m.forward=types.MethodType(effective_forward,m)
                self.kernels[path]=m.r4_adapter;extra.extend(m.r4_adapter.parameters())
            if sum(p.numel() for p in extra)!=dict(KDG=2147,K_ALL=2147,K_MAG=128,K_FREE=4288)[arm]:raise ValueError('kernel coordinates')
        self.params=list(self.bn_params)+extra
        self.base=torch.optim.Adam(self.params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.memory=ObservedMemory() if arm in RP_ARMS else None
        self.rng=rng();self.total=dict(network_forwards=0,loss_backward_calls=0,jacobian_vjp_calls=0,adam_calls=0,actual_parameter_replacements=0)
        self.handles=[self.model.register_forward_hook(self._forward),self.base.register_step_pre_hook(self._adam)]
        if self.memory:self.handles.append(self.model.seg_head.register_forward_pre_hook(self._features))
        if self.teacher is not None:self.handles.append(self.teacher.register_forward_hook(self._forward))

    def _forward(self,module,args,output):
        self.total['network_forwards']+=1;finite(segmentation(output))
    def _adam(self,opt,args,kwargs):
        if [id(p) for g in opt.param_groups for p in g['params']]!=[id(p) for p in self.params]:raise ValueError('optimizer ownership')
        if any(p.grad is None for p in self.params):raise ValueError('unused adaptable coordinate')
        finite([p.grad for p in self.params]);self.total['adam_calls']+=1
    def _features(self,module,args):
        if self.phase in ('PRE','STRONG'):
            value=grid(args[0])
            if value.shape!=(1024,32):raise ValueError('student feature dimension')
            self.pending[self.phase]=value.detach().cpu() if self.phase=='PRE' else value

    def take_evaluation(self):
        if self.phase!='IDLE' or self.evaluation_pending is None:raise ValueError('evaluation only after complete state commit')
        value=self.evaluation_pending;self.evaluation_pending=None
        return value

    def step(self,pixels):
        if self.failed:raise RuntimeError('failed trajectory cannot resume')
        if self.evaluation_pending is not None:raise ValueError('previous dense evaluation not released')
        if not isinstance(pixels,torch.Tensor) or pixels.shape!=(1,3,512,512) or pixels.device.type!='cpu' or pixels.dtype!=torch.float32 or pixels.requires_grad:raise ValueError('one detached current raw RGB')
        finite(pixels)
        if bool(((pixels<0)|(pixels>1)).any()):raise ValueError('raw RGB range')
        self.visit+=1;started=time.monotonic();self.phase='TEACHER'
        try:
            if self.legacy:
                result,old=self.legacy.step(pixels);self.rng=self.legacy.rng
                c=old['counts'];trace=dict(arm=self.arm,global_visit=self.visit,counts=dict(network_forwards=c['forwards'],loss_backward_calls=c['backwards'],adam_calls=c['base_adam'],jacobian_vjp_calls=0,actual_parameter_replacements=0),pca=old['pca'],teacher_update=False,source_unchanged=True)
                q=self.pending['q'];self.evaluation_pending=dict(q=q,selected=None if self.memory is None else self.memory.selected)
                if self.memory:self.memory.selected=None
                return result,dict(r4t=trace)
            restore(self.rng);before=self.total.copy();x=model_input_from_pixels(pixels,'fundus')
            data={'data':x.numpy().copy()};original=x.to(self.device).clone()
            snapshots=self.memory.snapshots(self.visit) if self.memory else None
            teacher=self.teacher if self.teacher is not None else self.model
            with torch.no_grad():z=segmentation(teacher(original))
            logits=[z.detach().cpu()];weak=official().Rotate_and_Flip();self.phase='WEAK'
            for factor in range(5):
                with torch.no_grad():z=segmentation(teacher(weak(original,factor)))
                logits.append(weak.inverse(z,factor).detach().cpu())
            views=torch.stack(logits).sigmoid();q=views.mean(0);views=views[:,0];del logits,z
            audit=None;vectors=ids=None
            if self.memory:
                self.phase='PRE'
                with torch.no_grad():self.model(original)
                vectors,ids,audit=self.memory.prepare(self.pending['PRE'],grid(q),torch.stack([grid(v[None]) for v in views]),self.visit)
            target=q;graph=None;masks={}
            if self.arm.startswith('G_'):
                t=time.monotonic();target,graph,masks=refine_target(pixels,q,views,self.arm,self.visit);graph['graph_seconds']=time.monotonic()-t
            self.phase='STRONG';self.base.zero_grad()
            aug=official().augmentation_strong_style(data)
            aug=official().normalize_image_to_0_1(torch.from_numpy(aug).float().to(self.device))
            strong=segmentation(self.model(aug));loss=F.binary_cross_entropy_with_logits(strong,target.to(self.device).detach());base_loss=float(loss.detach());sub=loss.new_zeros(())
            if self.memory:sub=self.memory.loss(self.pending['STRONG'],ids,snapshots);loss=loss+.05*sub
            finite(loss);loss.backward();self.total['loss_backward_calls']+=1;self.base.step()
            self.phase='PREDICT'
            with torch.no_grad():result=segmentation(self.model(original)).detach()
            self.phase='COMMIT'
            if self.memory:self.memory.merge(vectors,self.visit)
            if self.arm.startswith('MT'):
                with torch.no_grad():
                    for t,p in zip(self.teacher_params,self.bn_params):t.mul_(.99).add_(p,alpha=.01)
            finite(self.params);finite(self.base.state)
            if self.teacher is not None:finite(self.teacher_params)
            if any(int(self.base.state[p]['step'])!=self.visit for p in self.params):raise ValueError('Adam step count')
            if any(p._version!=version or p.grad is not None for p,version in self.frozen.values()):raise ValueError('source weight changed')
            if self.teacher is not None and any(p._version!=v for p,v in self.teacher_frozen.values()):raise ValueError('non-affine teacher changed')
            self.rng=rng();counts={k:v-before[k] for k,v in self.total.items()}
            if counts!=dict(network_forwards=9 if self.arm in ('MT_RP','FT_RP') else 8,loss_backward_calls=1,adam_calls=1,jacobian_vjp_calls=0,actual_parameter_replacements=0):raise ValueError('R4T physical count')
            pca=None if self.memory is None else dict(input=audit,banks=self.memory.audit(),basis_versions_used=[None if s is None else s[2] for s in snapshots],base_loss=base_loss,subloss=float(sub.detach()))
            kernels={path:k.audit() for path,k in self.kernels.items()}
            trace=dict(arm=self.arm,global_visit=self.visit,counts=counts,pca=pca,graph=graph,kernels=kernels,teacher_update=self.arm.startswith('MT'),source_unchanged=True,base_loss=base_loss,host_seconds=time.monotonic()-started,
                       trainable_scalars=sum(p.numel() for p in self.params),teacher_affine_student_l2=None if self.teacher is None else float(sum((t-p).double().square().sum() for t,p in zip(self.teacher_params,self.bn_params)).sqrt()),
                       final_raw_probability_nested_violations=int((result[:,1].sigmoid()>result[:,0].sigmoid()).sum()))
            self.evaluation_pending=dict(q=q,selected=None if self.memory is None else self.memory.selected)
            if graph:self.evaluation_pending.update(qstar=target,**masks)
            if self.memory:self.memory.selected=None
            finite(trace);return result,dict(r4t=trace)
        except BaseException:
            self.failed=True;self.evaluation_pending=None;raise
        finally:self.pending.clear();self.phase='IDLE'

    def finish(self,state):
        if self.legacy:return self.legacy.finish(state)
        for name,value in self.model.state_dict().items():
            if name not in self.names and '.r4_adapter.' not in name and not torch.equal(value.cpu(),state[name]):raise ValueError('source state changed')
        for h in self.handles:h.remove()
