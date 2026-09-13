"""Synthetic prior construction for CPU tests / explicitly authorized smoke only.

Never called by the formal trajectory initializer. No real observations retained.
"""
import torch
from ..r1.region_memory import Memory as OldMemory
from .stats import Memory
from .kernels import unit


def ready(memory):
    if isinstance(memory,Memory):memory.__init__()
    else:OldMemory.__init__(memory,'REGION')
    g=torch.Generator().manual_seed(91)
    if isinstance(memory,Memory):
        for visit in range(1,17):memory.merge([unit(torch.randn(8,32,generator=g)) for _ in range(4)],visit)
        memory.covariances=[(s[0],s[1],0) for s in memory.covariances]
    else:
        for b in memory.banks:
            for visit in range(1,17):b.merge(unit(torch.randn(8,32,generator=g)),visit)
    for b in memory.banks:b.version=b.last_visit=0
