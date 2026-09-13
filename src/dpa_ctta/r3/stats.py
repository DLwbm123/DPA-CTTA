"""R1-compatible sampling plus bounded covariance snapshots; no retained tokens."""
import torch
from ..r1.region_memory import Memory as RegionalMemory, tokens
from ..r1.recovery import local_seed
from .kernels import unit, transport_state


class Memory(RegionalMemory):
    def __init__(self):
        super().__init__('REGION')
        self.covariances = [None]*4
        self.frame_version = 0

    def select(self, raw, q, views, visit):
        vectors, regions, audit = self.prepare(raw, q, views, visit)
        _, reliable = tokens(q, views)
        selected = []
        norms = raw.detach().cpu().norm(dim=1)
        for r, ids in enumerate(regions):
            ids = ids[reliable[ids]]
            g = torch.Generator().manual_seed(local_seed('sample_'+str(r), visit))
            if len(ids)>32: ids = ids[torch.randperm(len(ids), generator=g)[:32]]
            selected.append(ids[norms[ids%1024]>0]%1024)
        if [len(i) for i in selected] != audit['selected_region_counts']:
            raise ValueError('selection quota drift')
        return selected, regions, vectors, audit

    def merge(self, vectors, visit):
        versions = [b.version for b in self.banks]
        super().merge(vectors, visit)
        for i, b in enumerate(self.banks):
            if b.version != versions[i]:
                self.covariances[i] = (b.mean.clone(), ((b.M2+b.M2.T)/2)/(b.n-1), b.version)

    def density_snapshots(self, visit):
        if any(s is not None and s[2]>=visit for s in self.covariances):
            raise ValueError('current image in density state')
        return [None if s is None else (s[0].clone(),s[1].clone(),s[2]) for s in self.covariances]

    @torch.no_grad()
    def transport(self, rotation, visit):
        if visit<=self.frame_version: raise ValueError('frame must advance')
        for i,b in enumerate(self.banks):
            b.mean,b.M2,b.U = transport_state(b.mean,b.M2,b.U,rotation)
            if b.center is not None: b.center = rotation@b.center
            s = self.covariances[i]
            if s is not None: self.covariances[i]=(rotation@s[0], rotation@s[1]@rotation.T, s[2])
        self.frame_version=visit

    def audit(self):
        out=super().audit()
        for b,s in zip(out,self.covariances):
            b['frame_version']=self.frame_version
            b['basis_stat_version']=b['version']
            b['density_stat_version']=None if s is None else s[2]
            b['state_bytes']+=0 if s is None else sum(x.numel()*x.element_size() for x in s[:2])
        if sum(b['state_bytes'] for b in out)>=1024**2:raise ValueError('bounded regional memory')
        return out
