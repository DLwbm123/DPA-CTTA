"""Reuse R1 grid, selection, all-token loss regions, salts and independent shuffles."""
import torch
from ..r1.region_memory import Memory as R1Memory
from ..r1.streaming_pca import StreamingSubspace
from .weighted_pca import WeightedSubspace, audit
from .feature_losses import paired_loss, projection_residual, energies

SPECS = {'R2_E':('absolute',True,False), 'R2_D':('pair',False,False),
         'R2_DE':('pair',True,False), 'R2_F':('full',True,False), 'R2_DE_S':('pair',True,True)}


class Memory(R1Memory):
    def __init__(self, arm):
        self.loss_kind, weighted, shuffled = SPECS[arm]
        super().__init__('SHUFFLED' if shuffled else 'REGION')
        self.banks = [WeightedSubspace() if weighted else StreamingSubspace() for _ in range(4)]

    def loss(self, fs, f0, ids, snapshots):
        terms, diagnostics = [], []
        for index,(tokens,snap) in enumerate(zip(ids,snapshots)):
            if not len(tokens) or snap is None: continue
            center,U,version = snap
            # The same token always selects the same position on both sides.
            positions = tokens.to(fs.device)%1024
            strong, original = fs[positions], f0[tokens.cpu()%1024]
            terms.append(projection_residual(strong,center,U) if self.loss_kind=='absolute'
                         else paired_loss(strong,original,U,full=self.loss_kind=='full'))
            diagnostics.append(dict(bank=index,tokens=len(tokens),**energies(strong,original,center,U)))
        return (torch.stack(terms).mean() if terms else fs.sum()*0), diagnostics

    def audit(self):
        values = [audit(b) for b in self.banks]
        if sum(v['state_bytes'] for v in values) >= 1024**2: raise ValueError('bounded state exceeded')
        return values
