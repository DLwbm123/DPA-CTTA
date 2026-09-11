"""Current-statistics zero-update baseline and frozen C recipe on PraNet RGB."""
import torch
import torch.nn.functional as F
from .b1_host import Host as FundusC,configure,finite,official,expected_counts
from .source_pilot import SourceOnlyHost
from .hosts.vptta import model_input_from_pixels
from .host_diagnostic import rng,restore,close


def logits(model,x):
    out=model(x);z=out[0] if isinstance(out,(tuple,list)) else out
    if not isinstance(z,torch.Tensor) or z.ndim!=4 or z.shape[0]!=1:raise ValueError('segmentation logits keep batch/channel')
    finite(z);return z


def deterministic_forward(model):
    if any(isinstance(m,(torch.nn.Dropout,torch.nn.Dropout2d,torch.nn.Dropout3d,torch.nn.AlphaDropout)) and m.training and m.p for m in model.modules()):raise ValueError('active random dropout')


class C0:
    def __init__(self,task,state,device='cpu'):
        self.task=task;self.device=torch.device(device);self.model=SourceOnlyHost(task,state,device).model
        self.names,_=configure(self.model);self.model.requires_grad_(False);deterministic_forward(self.model)
        self.counts=dict(forwards=0,backwards=0,base_adam=0,perturb=0,restore=0)
        self.versions={n:(p,p._version) for n,p in self.model.state_dict(keep_vars=True).items()}
    def step(self,pixels):
        before=rng()
        x=model_input_from_pixels(pixels,self.task).to(self.device)
        self.counts['forwards']+=1
        with torch.no_grad():z=logits(self.model,x)
        close(before,rng(),exact=True)
        if any(p._version!=v or p.grad is not None for p,v in self.versions.values()):raise ValueError('C0 state mutation')
        return z.detach(),dict(counts=dict(forwards=1,backwards=0,base_adam=0,perturb=0,restore=0),adam_step=0,lr=None,diagnostics=None,stateless_checked=True)
    def finish(self,state):
        for n,p in self.model.state_dict().items():
            if not torch.equal(p.cpu(),state[n]):raise ValueError('C0 source parameter/buffer drift')


class PolypC(FundusC):
    def __init__(self,state,device='cpu'):
        model=SourceOnlyHost('polyp',state,device).model
        super().__init__('C',device=device,model=model);deterministic_forward(self.model)
        self.seen_bn=set();self.bn_hooks=[m.register_forward_pre_hook(lambda m,x,n=n:self.seen_bn.add(n)) for n,m in self.model.named_modules() if type(m) is torch.nn.BatchNorm2d]
    def _forward(self,m,args,out):
        self.counts['forwards']+=1
        if not isinstance(out,torch.Tensor) or out.shape!=(1,1,352,352):raise ValueError('PraNet tensor output')
        finite(out)
    def step(self,pixels):
        # Frozen RGB reader contract and the original CPU ImageNet normalization.
        original=model_input_from_pixels(pixels,'polyp').to(self.device)
        if len(pixels)!=1 or pixels.requires_grad:raise ValueError('current single RGB image only')
        restore(self.rng);before=self.counts.copy();api=official();weak=api.Rotate_and_Flip();views=[]
        with torch.no_grad():
            views.append(logits(self.model,original).detach().cpu())
            for factor in (0,1,2,3,4):
                rgb=weak(pixels,factor);z=logits(self.model,model_input_from_pixels(rgb,'polyp').to(self.device))
                views.append(weak.inverse(z,factor).detach().cpu())
            q=torch.stack(views).sigmoid().mean(0).to(self.device)
        self.base.zero_grad()
        data={'data':pixels.numpy().copy()};augmented=api.augmentation_strong_style(data)
        # Strong style and min/max are RGB-space CPU operations; channel norm once.
        rgb=api.normalize_image_to_0_1(torch.from_numpy(augmented).float())
        z=logits(self.model,model_input_from_pixels(rgb,'polyp').to(self.device))
        loss=F.binary_cross_entropy_with_logits(z,q);finite(loss);loss.backward();self.base.step()
        with torch.no_grad():prediction=logits(self.model,original).detach()
        finite(self.base.state);finite(self.params);self.steps+=1;self.rng=rng()
        if any(int(self.base.state[p]['step'])!=self.steps for p in self.params):raise ValueError('PraNet Adam step')
        if any(p._version!=v or p.grad is not None for p,v in self.frozen.values()):raise ValueError('PraNet non-BN mutation')
        names={n.rsplit('.',1)[0] for n in self.names}
        if names!=self.seen_bn:raise ValueError('registered BN did not participate')
        counts={k:v-before[k] for k,v in self.counts.items()}
        if counts!=expected_counts('C'):raise ValueError('PraNet physical counts')
        return prediction,dict(counts=counts,adam_step=self.steps,lr=1e-4,diagnostics=dict(point_bce=float(loss.detach()),bn_gradient_l2=float(sum(p.grad.double().square().sum() for p in self.params).sqrt())))
    def finish(self,state):
        super().finish(state)
        for h in self.bn_hooks:h.remove()


def make_C(task,state,device='cpu'):
    return FundusC('C',state,device) if task=='fundus' else PolypC(state,device)


def reference_polyp_step(h,x):
    """Independent literal recomposition for smoke; never calls PolypC.step."""
    api=official();aug=api.Rotate_and_Flip();preds=[]
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    with torch.no_grad():
        for factor in (None,0,1,2,3,4):
            view=x if factor is None else aug(x,factor)
            out=h.model(((view-mean)/std).to(h.device))
            if factor is not None:out=aug.inverse(out,factor)
            preds.append(out.detach().cpu())
        target=torch.stack(preds).sigmoid().mean(0).to(h.device)
    h.base.zero_grad()
    image=api.augmentation_strong_style({'data':x.numpy().copy()})
    image=api.normalize_image_to_0_1(torch.from_numpy(image).float())
    output=h.model(((image-mean)/std).to(h.device))
    loss=F.binary_cross_entropy_with_logits(output,target);loss.backward();h.base.step()
    with torch.no_grad():return h.model(((x-mean)/std).to(h.device)).detach()
