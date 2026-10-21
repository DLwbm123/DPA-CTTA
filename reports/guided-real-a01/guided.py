"""Fixed scalar guided filtering of raw logits, no learned or label-fitted values."""
import numpy as np
from scipy.ndimage import uniform_filter
RADIUS=8
EPSILON=1e-3

def box(x):
    return uniform_filter(np.asarray(x,dtype=np.float64),size=(1,)*(x.ndim-2)+(2*RADIUS+1,)*2,mode='reflect')

def guided(i,p):
    i=np.asarray(i,np.float64);p=np.asarray(p,np.float64)
    mu=box(i);mp=box(p);var=np.maximum(box(i*i)-mu*mu,0)
    a=(box(i*p)-mu*mp)/(var+EPSILON);b=mp-a*mu
    q=box(a)*i+box(b)
    assert np.isfinite(q).all()
    return q

def variants(rgb,z):
    i=np.einsum('c,chw->hw',np.array([.299,.587,.114]),rgb.astype(np.float64))
    return np.stack((z,guided(i,z),box(box(z)),guided(np.roll(i,257,axis=1),z))).astype(np.float32)

def check():
    rng=np.random.default_rng(41);i=rng.random((27,29));p=rng.normal(size=(2,27,29))
    assert np.allclose(guided(i,np.ones_like(p)),1,atol=1e-12)
    assert np.allclose(guided(np.zeros_like(i),p),box(box(p)),atol=1e-12)
    assert np.allclose(guided(i,2*p+3),2*guided(i,p)+3,atol=1e-10)
    assert not np.allclose(guided(i,p),guided(np.roll(i,7,axis=1),p))
