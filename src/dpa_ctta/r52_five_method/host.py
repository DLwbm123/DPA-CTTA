"""Reuse native VPTTA and the registered MGIPT paper-equation adapter."""
import copy
import os
import sys
from pathlib import Path
import torch
from ..source_pilot import seed_all
from ..hosts.vptta import VPTTAHost, model_input_from_pixels
from ..host_diagnostic import snapshot as vptta_snapshot


class Host:
    def __init__(self, state, candidate, seed, identity):
        seed_all(seed)
        self.kind = candidate['id']
        self.context = {'sha256': identity}
        self.visits = 0
        self.teacher = self.f0 = None
        if self.kind == 'VPTTA':
            self.native = VPTTAHost('fundus', source_state=state, device='cuda:0')
            self.model = self.native.model
        elif self.kind == 'MGIPT':
            sys.path.insert(0, str(Path(os.environ['MGIPT_SUITE_ROOT'])/'src'))
            from ctta_suite import models
            from ctta_suite.mgipt import Adapter
            models.ROOT = Path(os.environ['MGIPT_SUITE_ROOT'])
            self.model = models.build_model('fundus')
            self.model.load_state_dict(state, strict=True)
            self.model.cuda()
            self.adapter = Adapter(self.model, 'fundus', 'paper_equation', seed=seed)
            self.native = self
        else:
            raise ValueError('unregistered method')
        self.initial = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

    def step(self, pixels):
        if self.kind == 'VPTTA':
            z = self.native.step(pixels).detach().float().cpu()
            diagnostic = {'memory_size': self.native.memory_bank.get_size()}
        else:
            x = model_input_from_pixels(pixels, 'fundus').cuda()
            record = self.adapter.adapt(x, {})
            p = self.adapter.predict(x, {}).detach().float().cpu()
            if not torch.isfinite(p).all() or (p<0).any() or (p>1).any():
                raise ValueError('invalid probability')
            # Hard predictions are preserved exactly, including probability 0.5.
            z = torch.logit(p.clamp(1e-7, 1-1e-7))
            diagnostic = record
        if z.shape != (1,2,512,512) or not torch.isfinite(z).all():
            raise ValueError('invalid output')
        self.visits += 1
        return z, dict(visit=self.visits, diagnostics=diagnostic, state_committed=True)

    def check_frozen(self, boundary=False):
        if boundary and any(not torch.equal(v.detach().cpu(), self.initial[k])
                            for k,v in self.model.state_dict().items()):
            raise ValueError('source weights or buffers changed')

    def snapshot(self):
        # No recovery is authorized; retain native adaptive state for audit only.
        state = vptta_snapshot(self.native) if self.kind=='VPTTA' else self.adapter.state_dict()
        return dict(context=self.context, visits=self.visits, adaptive=copy.deepcopy(state), recovery_authorized=False)

    def close(self):
        self.check_frozen(True)
