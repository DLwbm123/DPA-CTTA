"""Six frozen diagnostic arms. Six C0 views, one reused strong view, fresh Adam."""
import copy
import torch
from torch.nn import functional as F
from ..b1_host import official,GRATA_COMMIT
from ..hosts.vptta import model_input_from_pixels
from ..r7_shared.context import tensor_digest,stamp
from ..r7_shared.numerics import COUNTS,finite
from ..r8_ba.methods import R8B
from .deployment import Host

STEPS={'B_G1':1,'B_G3':3,'B_RESET_G3':3,'COLD_G1':1,'COLD_G3':3,'BN_RESET_G1':1}
LATENT_LR=(.001,.01,.1)
BN_LR=(1e-5,1e-4,1e-3)


@torch.no_grad()
def teacher_views(segmenter,image,api,observe):
    x=model_input_from_pixels(image,'fundus');original=x.to(segmenter.projection.device)
    result=segmenter(image,observe=observe)
    zero,raw,tokens=result if observe else (result,None,None)
    weak=api.Rotate_and_Flip();ps=[zero.sigmoid()]
    for factor in range(5):
        logits=segmenter.normalized(weak(original,factor).cpu())
        ps.append(weak.inverse(logits.to(original.device),factor).cpu().sigmoid())
    target=torch.stack(ps).mean(0).detach()
    strong=api.augmentation_strong_style({'data':x.numpy().copy()})
    strong=api.normalize_image_to_0_1(torch.from_numpy(strong).float().to(original.device)).cpu()
    return target,strong,raw,tokens


class GradientHost(Host):
    def __init__(self,segmenter,method,config,source,expected_context,arm,lr,scale=None,api=None):
        if arm not in STEPS or lr not in (BN_LR if arm=='BN_RESET_G1' else LATENT_LR):raise ValueError('R9 gradient arm/LR')
        if config.get('gradient_arm')!=arm or config.get('gradient_lr')!=lr or config.get('grata_commit')!=GRATA_COMMIT:
            raise ValueError('R9 gradient binding')
        if segmenter.alpha!=1:raise ValueError('gradient arms do not tune output-alpha')
        self.arm,self.lr,self.api=arm,lr,official() if api is None else api
        self.scale=None;self.bn={};self.bn_source={}
        if arm=='BN_RESET_G1':
            if method is not None:raise ValueError('BN reset has no learned latent host')
            for name,m in segmenter.model.named_modules():
                if isinstance(m,torch.nn.BatchNorm2d):
                    if type(m) is not torch.nn.BatchNorm2d or not m.affine:raise ValueError('registered standard affine BN only')
                    self.bn.update({name+'.'+n:p for n,p in m.named_parameters(recurse=False)})
            if not self.bn or sorted(self.bn)!=config.get('bn_affine_names'):raise ValueError('BN affine registration')
            self.bn_source={n:p.detach().clone() for n,p in self.bn.items()}
        else:
            if not isinstance(method,R8B) or method.static or scale.shape!=(method.rank,) or scale.dtype!=torch.float64:
                raise ValueError('R9 matched B FULL latent space')
            finite(scale)
            if (scale<1e-3).any():raise ValueError('scale floor')
            self.scale=scale.detach().clone()
            if config.get('scale_sha256')!=tensor_digest([('scale',scale)]):raise ValueError('scale binding')
        super().__init__(segmenter,method,config,source,expected_context)
        self.frozen_versions={n:(id(p),p._version) for n,p in segmenter.model.named_parameters() if n not in self.bn}

    def check_frozen(self,boundary=False):
        super().check_frozen(boundary)
        c=self.context['payload']['config']
        if c.get('gradient_arm')!=self.arm or c.get('gradient_lr')!=self.lr:raise ValueError('gradient identity changed')
        if self.scale is not None and c.get('scale_sha256')!=tensor_digest([('scale',self.scale)]):raise ValueError('scale changed')
        if any(not torch.equal(p,self.bn_source[n]) or p.requires_grad for n,p in self.bn.items()):raise ValueError('BN reset boundary')

    def _reset_bn(self):
        with torch.no_grad():
            for name,p in self.bn.items():p.copy_(self.bn_source[name]);p.requires_grad_(False);p.grad=None
        if {n:(id(p),p._version) for n,p in self.segmenter.model.named_parameters() if n not in self.bn}!=self.frozen_versions:
            raise ValueError('non-BN weights changed')
        self.stamp=stamp(self.segmenter,self.method,self.ablation)

    def step(self,image):
        if self.failed:raise RuntimeError('R9 gradient host stopped')
        before=COUNTS.copy()
        try:
            self.check_frozen()
            cold=self.arm.startswith('COLD');bn=self.arm=='BN_RESET_G1'
            target,strong,raw,tokens=teacher_views(self.segmenter,image,self.api,not cold and not bn)
            if bn:
                for p in self.bn.values():p.requires_grad_(True)
                parameters=list(self.bn.values());u0=[p.detach().clone() for p in parameters]
                next_state=None
            else:
                with torch.no_grad():
                    initial_state=self.method.initial() if self.arm=='B_RESET_G3' else self.state
                    if cold:next_state=self.method.initial()
                    else:next_state,_=self.method.update(raw,tokens,initial_state)
                    next_state['counter']=self.visits+1
                u0=next_state['z'].detach().clone()/self.scale
                u=torch.nn.Parameter(u0.clone());parameters=[u]
            optimizer=torch.optim.Adam(parameters,lr=self.lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
            losses=[]
            for _ in range(STEPS[self.arm]):
                optimizer.zero_grad(set_to_none=True)
                logits=self.segmenter.normalized(strong,None if bn else (self.method.basis@(self.scale*u)).float())
                prox=sum((p-v).square().sum() for p,v in zip(parameters,u0))/sum(p.numel() for p in parameters) if bn else (u-u0).square().mean()
                loss=F.binary_cross_entropy_with_logits(logits,target)+.01*prox;finite(loss)
                loss.backward();COUNTS['target_backward_calls']+=1
                finite(*(p.grad for p in parameters));optimizer.step();COUNTS['target_Adam']+=1
                finite(*parameters);losses.append(float(loss.detach()))
            with torch.no_grad():
                if bn:logits=self.segmenter(image)
                else:
                    corrected=(self.scale*u.detach()).clone()
                    logits=self.segmenter(image,(self.method.basis@corrected).float())
                    if self.arm in ('B_G1','B_G3'):next_state['z']=corrected
                    else:next_state=self.method.initial();next_state['counter']=self.visits+1
                    self.method.validate_state(next_state)
            finite(logits)
            if bn:self._reset_bn()
            counts=dict(COUNTS-before);k=STEPS[self.arm]
            if (counts.get('backbone_forwards')!=7+k or counts.get('target_backward_calls')!=k or counts.get('target_Adam')!=k):
                raise ValueError('R9 physical call count mismatch')
            self.state=copy.deepcopy(next_state);self.visits+=1
            return logits.detach(),dict(visit=self.visits,state_committed=True,counts=counts,arm=self.arm,lr=self.lr,loss=losses)
        except BaseException:
            self.failed=True
            if self.bn:self._reset_bn()
            raise
