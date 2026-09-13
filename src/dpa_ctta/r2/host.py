"""R1's complete online sequence, with only the extra loss and memory replaced."""
import torch
from ..r1.host import Host as R1Host, finite
from ..r1.region_memory import grid
from .memory import Memory, SPECS

ARMS = tuple(SPECS)
EXTRA_WEIGHT = .05


class Host(R1Host):
    def __init__(self, arm, state=None, device='cpu', model=None):
        if arm not in (*ARMS,'C','C_PCA_REGION'): raise ValueError('R2 arm or explicit compatibility path required')
        self.r2_arm = arm if arm in ARMS else None
        self.scalar_diagnostics = {}
        super().__init__('C_PCA_REGION' if self.r2_arm else arm,state,device,model)
        if self.r2_arm: self.memory = Memory(arm)

    def _criterion(self,z,q):
        if not self.r2_arm: return super()._criterion(z,q)
        base = torch.nn.functional.binary_cross_entropy_with_logits(z,q)
        vectors,ids,audit = self.memory.prepare(self.pending['f0'],grid(q),torch.stack(self.pending['views']),self.visit)
        extra,diagnostics = self.memory.loss(self.pending['fs'],self.pending['f0'],ids,self.pending['snapshots'])
        finite(extra)
        self.pending.update(vectors=vectors,memory_input=audit,subloss=float(extra.detach()),base_loss=float(base.detach()))
        self.scalar_diagnostics = dict(base_loss=float(base.detach()),extra_loss=float(extra.detach()),
            weighted_extra_loss=float((EXTRA_WEIGHT*extra).detach()),active_regions=len(diagnostics),
            region_energies=diagnostics,shadow_readiness_only=self.r2_arm=='R2_F')
        return base+EXTRA_WEIGHT*extra

    def step(self,pixels):
        try:
            prediction,record = super().step(pixels)
            record['r2'] = dict(self.scalar_diagnostics) if self.r2_arm else None
            return prediction,record
        finally: self.scalar_diagnostics.clear()
