"""2D cubical critical-pixel adapter; no model or data loading.

GUDHI top-dimensional cells use column-major coface indices. Equal values
use the library tie selection; repeatability is not a Clarke certificate.
"""
import numpy as np

def critical_pairs(probability):
    import gudhi
    p=np.asarray(probability,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('finite two-dimensional probabilities required')
    cc=gudhi.CubicalComplex(top_dimensional_cells=1-p)
    cc.compute_persistence(min_persistence=-1)
    regular,essential=cc.cofaces_of_persistence_pairs()
    h,w=p.shape
    def convert(v):
        a=np.asarray(v,dtype=np.int64)
        return (a%h)*w+a//h
    finite=[convert(regular[d]) if len(regular)>d else np.empty((0,2),dtype=np.int64) for d in (0,1)]
    main=convert(essential[0])
    if main.shape!=(1,):raise RuntimeError('rectangular domain must have one essential H0 feature')
    return finite,main

def correction(probability,pairs=None,include_holes=True):
    """Differentiable probability-unit lifetime sum, beta0=1/beta1=0.

Pairs may be cached for the original curriculum. Their indices, rather
than a stale scalar loss, are reevaluated at the current probability.
"""
    if probability.ndim!=2:raise ValueError('one independent Bernoulli head')
    if pairs is None:pairs=critical_pairs(probability.detach().cpu().double().numpy())
    finite,essential=pairs
    flat=probability.reshape(-1)
    loss=1-flat[int(essential[0])]
    # Sublevel filtration is 1-p, so persistence = p[birth]-p[death].
    for d in range(2 if include_holes else 1):
        a=finite[d]
        if len(a):loss=loss+(flat[a[:,0]]-flat[a[:,1]]).sum()
    return loss
