"""Binary topology projection qualification, not the original TopoSculpt optimizer."""
import numpy as np
from scipy.ndimage import label,binary_fill_holes,generate_binary_structure
ARMS=('SOURCE','SMOOTH','TOPO_SOURCE','TOPO_SMOOTH','LCC_SOURCE','FILL_SOURCE','MASS_MATCH_SOURCE','MASS_MATCH_SMOOTH')
FG=generate_binary_structure(2,2);BG=generate_binary_structure(2,1)
def lcc(m):
    cc,n=label(m,structure=FG)
    if n==0:return m.copy()
    sizes=np.bincount(cc.ravel());sizes[0]=0
    return cc==int(sizes.argmax()) # ties choose raster-first component

def project(m):return binary_fill_holes(lcc(m),structure=BG)
def mass_matched(m,target):
    p=m.copy();flat=m.ravel();t=target.ravel();out=p.ravel()
    for state in (False,True):
        group=np.flatnonzero(flat==state);selected=np.flatnonzero((flat[group]!=t[group]))
        if len(selected):out[group[(selected+len(group)//2+1)%len(group)]]=not state
    assert p.sum()==target.sum() and int((p!=m).sum())==int((target!=m).sum())
    return p

def outputs(source,smooth):
    if source.shape!=smooth.shape or source.ndim!=3 or source.shape[0]!=2 or source.dtype!=bool or smooth.dtype!=bool:raise ValueError('two independent binary native heads')
    out={a:[] for a in ARMS}
    for a,b in zip(source,smooth):
        p=project(a);q=project(b)
        values=(a,b,p,q,lcc(a),binary_fill_holes(a,structure=BG),mass_matched(a,p),mass_matched(b,q))
        for name,v in zip(ARMS,values):out[name].append(v)
    result={a:np.stack(v) for a,v in out.items()}
    assert np.array_equal(project(result['TOPO_SOURCE'][0]),result['TOPO_SOURCE'][0])
    return result
