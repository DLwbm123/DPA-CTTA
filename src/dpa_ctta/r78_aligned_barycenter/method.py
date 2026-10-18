"""Keep the native teacher/W/LSO; change only the loss evaluation location."""
import copy
import types
import torch
from ..r37_cw_orientation.method import Host as Retained, candidates
from ..b1_host import finite
from ..host_diagnostic import rng, restore
from ..hosts.vptta import model_input_from_pixels

ARMS=('NATIVE','VERTEX','BARY')
READOUTS=('O','A')


def two_view_step(n,x):
    # Same native C protocol, with an explicit nine-forward contract.
    if x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=4 or len(x)!=1 or x.requires_grad:
        raise ValueError('normalized current-image contract')
    finite(x);restore(n.rng);before=n.counts.copy();n.last={}
    data={'data':x.numpy().copy()};original=x.to(n.device).clone()
    n.opt.cal_consis_loss(data);n.base.step()
    with torch.no_grad():logits=n.model(original)[0].detach()
    finite(n.base.state);finite(n.params);n.steps+=1;n.rng=rng()
    if any(int(n.base.state[p]['step'])!=n.steps for p in n.params):raise ValueError('Adam step drift')
    if any(p._version!=v or p.grad is not None for p,v in n.frozen.values()):raise ValueError('non-BN parameter changed')
    counts={k:v-before[k] for k,v in n.counts.items()}
    if counts!=dict(forwards=9,backwards=1,base_adam=1,perturb=0,restore=0):raise ValueError('two-view counters')
    return logits,dict(counts=counts,adam_step=n.steps,lr=float(n.base.param_groups[0]['lr']),diagnostics=n.last.copy())


class Host(Retained):
    def __init__(self,state,arm,seed,identity,**kwargs):
        if arm not in ARMS:raise ValueError('unregistered training arm')
        self.arm=arm;self.strong=None;self.extra=False
        super().__init__(state,copy.deepcopy(candidates()[3]),seed,identity,**kwargs)
        if self.teacher is not None or self.adapter is not None:raise ValueError('native W/LSO only')
        if sum(p.numel() for p in self.native.params)!=19136:raise ValueError('fixed BN ownership')
        head=self.native.model.seg_head
        if type(head) is not torch.nn.Conv2d or head.kernel_size!=(1,1) or head.in_channels!=32 or head.out_channels!=2:
            raise ValueError('fixed pointwise linear head required')
        if arm!='NATIVE':
            self.native.normalized_step=types.MethodType(two_view_step,self.native)
            self.handles.append(self.native.model.register_forward_pre_hook(self._strong_input))

    def _strong_input(self,m,args):
        if self.collect and not self.extra and len(self.weak)==6 and torch.is_grad_enabled():
            self.strong=args[0]

    def _criterion(self,z,q):
        if self.arm=='NATIVE':return super()._criterion(z,q)
        if self.strong is None or not z.requires_grad:raise ValueError('strong gradient path absent')
        self.extra=True
        try:other=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
        finally:self.extra=False;self.strong=None
        def record(key):
            def hook(g):
                finite(g);self.diag[key]=float(g.norm())
                return g
            return hook
        z.register_hook(record('branch0_gradient_norm'));other.register_hook(record('branch1_gradient_norm'))
        if self.arm=='BARY':loss=super()._criterion((z+other)*.5,q)
        else:
            first=super()._criterion(z,q);first_diag=copy.deepcopy(self.diag)
            second=super()._criterion(other,q)
            self.diag['vertex_first']=first_diag
            loss=(first+second)*.5
        self.diag['joint_loss']=float(loss.detach());return loss

    @torch.no_grad()
    def readouts(self,x,z):
        # Restore counters, gradients, optimizer/native RNG as well as model buffers.
        saved=self.snapshot();diag=copy.deepcopy(self.diag);weak=self.weak;collect=self.collect
        try:
            self.collect=False
            flipped=self.native.model(model_input_from_pixels(torch.flip(x,(-1,)),'fundus').to(self.device))[0]
            avg=(z.sigmoid()+torch.flip(flipped.sigmoid().cpu(),(-1,)))*.5
            finite(avg)
        finally:
            self.restore(saved);self.diag=diag;self.weak=weak;self.collect=collect
        return torch.stack((z.sigmoid(),avg))


def self_check():
    torch.manual_seed(78)
    a=torch.randn(1,2,3,4,requires_grad=True);b=torch.randn_like(a,requires_grad=True);q=torch.rand_like(a)
    loss=torch.nn.functional.binary_cross_entropy_with_logits((a+b)*.5,q);loss.backward()
    expected=((a.detach()+b.detach()).mul(.5).sigmoid()-q)/a.numel()*.5
    assert torch.allclose(a.grad,expected) and torch.allclose(b.grad,expected)
    assert torch.equal(torch.flip(torch.flip(a,(-1,)),(-1,)),a)
    head=torch.nn.Conv2d(32,2,1);u=torch.randn(1,32,3,4);v=torch.randn_like(u)
    assert torch.allclose(head((u+v)*.5),(head(u)+head(v))*.5,atol=2e-6,rtol=2e-5)
    print('BARYCENTER_ALGEBRA_SELF_CHECK_PASS',flush=True)
