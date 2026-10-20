"""Fixed smoothing-bit / fill-hole composition; no learned state."""
import numpy as np
from scipy.ndimage import binary_fill_holes
from .topology_projection import project,mass_matched,BG
ARMS=('SOURCE','SMOOTH','FILL_SOURCE','TOPO_SMOOTH','FILL_SMOOTH','MASS_MATCH_SMOOTH')
def outputs(source,smooth):
    if source.shape!=smooth.shape or source.ndim!=3 or source.shape[0]!=2 or source.dtype!=bool or smooth.dtype!=bool:
        raise ValueError('two independent native binary heads required')
    out={a:[] for a in ARMS}
    for a,b in zip(source,smooth):
        filled=binary_fill_holes(b,structure=BG)
        values=(a,b,binary_fill_holes(a,structure=BG),project(b),filled,mass_matched(b,filled))
        assert np.all(filled[b]) and not np.any(b&~filled)
        assert np.array_equal(binary_fill_holes(filled,structure=BG),filled)
        for arm,value in zip(ARMS,values):out[arm].append(value)
    return {a:np.stack(v) for a,v in out.items()}
