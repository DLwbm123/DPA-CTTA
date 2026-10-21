"""Fixed MSRCP-derived float-input contract; not the original IPOL binary."""
import numpy as np
from scipy.fft import dctn,idctn

SCALES=(15.,80.,250.)

def balance(x):
    lo,hi=np.quantile(x,[.01,.99],method='linear')
    if hi-lo<=1e-12:return None
    return np.clip((x-lo)/(hi-lo),0.,1.)

def blur(x,sigma):
    h,w=x.shape;omega2=(np.pi*np.arange(h)[:,None]/h)**2+(np.pi*np.arange(w)[None,:]/w)**2
    return idctn(dctn(x,type=2,norm='ortho')*np.exp(-.5*sigma*sigma*omega2),type=2,norm='ortho')

def apply(x,gain):
    maximum=x.max(0);cap=1/np.maximum(maximum,1e-12)
    y=x*np.minimum(gain,cap)[None]
    span=y.max()-y.min()
    if span<=1e-12:return x.astype(np.float32).copy()
    return ((y-y.min())/span).astype(np.float32)

def variants(x):
    x=x.astype(np.float64);intensity=x.mean(0);positive=np.maximum(intensity,1/255.)
    surround=sum(np.log(np.maximum(blur(positive,s),1/255.)) for s in SCALES)/3
    mapped=balance(np.log(positive)-surround)
    gain=np.ones_like(intensity) if mapped is None else mapped/positive
    global_mapped=balance(intensity)
    global_gain=np.ones_like(intensity) if global_mapped is None else global_mapped/positive
    # Spatial null has the same un-capped gain multiset, with positions destroyed.
    # Per-pixel gamut clipping can change final dose; no matched-dose claim is made.
    return np.stack((x.astype(np.float32),apply(x,gain),apply(x,global_gain),apply(x,np.roll(gain,257,axis=1))))

def check():
    x=np.ones((16,16),np.float64)*.4
    assert np.max(np.abs(blur(x,15)-x))<1e-12
    grid=np.linspace(0,1,256).reshape(16,16);rgb=np.stack((grid,grid*.7,grid*.2));out=variants(rgb)
    assert out.shape==(4,3,16,16) and np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    # A common multiplier preserves RGB proportions before final global affine normalization.
    raw=rgb*np.minimum(np.ones_like(grid)*1.3,1/np.maximum(rgb.max(0),1e-12))[None]
    assert np.max(np.abs(raw[1]-.7*raw[0]))<1e-12
    assert balance(x) is None
