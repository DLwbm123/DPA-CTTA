"""C with its existing final classifier unfrozen at a fixed conservative rate."""
import torch
from ..r30_state_history.method import StateHost, norm
from ..r8_ba.native_host import tensor_digest

ARMS = ('C_CONT', 'ANCHOR', 'C_HEAD')


class Host(StateHost):
    def __init__(self, state, arm, seed, identity, device='cuda:0', model=None):
        if arm not in ARMS:
            raise ValueError('unregistered readout arm')
        super().__init__(state, 'C_CONT' if arm == 'C_HEAD' else arm, seed, identity, device, model)
        self.arm = arm
        if arm != 'C_HEAD':
            return
        n = self.native
        head = n.model.seg_head
        if type(head) is not torch.nn.Conv2d or tuple(head.weight.shape) != (2, 32, 1, 1) or head.bias is None:
            raise ValueError('expected existing 66-scalar final classifier')
        names = ['seg_head.weight', 'seg_head.bias']
        parameters = list(head.parameters())
        head.requires_grad_(True)
        n.base.add_param_group(dict(params=parameters, lr=1e-5))
        n.names.extend(names); n.params.extend(parameters)
        n.opt.param_groups = n.base.param_groups
        for name in names:
            del n.frozen[name]
        self.source_affine.update({name: p.detach().clone() for name, p in zip(names, parameters)})
        self.fixed_digest = tensor_digest([(k, v) for k, v in n.model.state_dict().items() if k not in n.names])

    def _lr(self, optimizer, args, kwargs):
        super()._lr(optimizer, args, kwargs)
        if self.arm == 'C_HEAD':
            optimizer.param_groups[1]['lr'] = 1e-5
            self.diag['head_gradient_norm'] = norm(p.grad for p in self.native.model.seg_head.parameters())

    def diagnostics(self):
        result = super().diagnostics()
        if self.arm == 'C_HEAD':
            result['head_drift'] = norm(p-self.source_affine['seg_head.'+k] for k, p in self.native.model.seg_head.named_parameters())
            result['head_gradient_norm'] = self.diag.get('head_gradient_norm', 0.)
        return result
