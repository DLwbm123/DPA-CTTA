"""Image-only phase/amplitude/gradient fields and frozen contour mean readout."""
import numpy as np
from scipy import ndimage as ndi
import phase
FULL=np.ones((3,3),bool)
CROSS=ndi.generate_binary_structure(2,1)
EPS=1e-12
FIELD_NAMES=('PHASE','AMPLITUDE','ENERGY','GRADIENT','SHIFT')

def fields(grey):
    grey=np.asarray(grey,np.float64)
    ev=phase.measure(grey)
    gx=ndi.sobel(grey,axis=1,mode='reflect')/8
    gy=ndi.sobel(grey,axis=0,mode='reflect')/8
    out=dict(PHASE=ev['pc'],AMPLITUDE=ev['amplitude'],ENERGY=ev['energy'],GRADIENT=np.hypot(gx,gy),SHIFT=np.roll(ev['pc'],grey.shape[1]//2+1,axis=1))
    for v in out.values():
        assert v.shape==grey.shape and np.isfinite(v).all() and (v>=0).all()
        v.setflags(write=False)
    return out

def outer(mask):
    lab,n=ndi.label(np.asarray(mask,bool),FULL)
    if not n:return np.zeros_like(mask,dtype=bool)
    counts=np.bincount(lab.ravel());counts[0]=0
    main=lab==int(np.argmax(counts));main=ndi.binary_fill_holes(main,structure=CROSS)
    return main&~ndi.binary_erosion(main,FULL,border_value=0)

def score(fields,base,candidate,base_boundary=None):
    b=outer(base) if base_boundary is None else base_boundary
    c=outer(candidate);n0=int(b.sum());n1=int(c.sum())
    if not n0 or not n1:return np.zeros(6),False
    support=bool(np.any((b^c)&(fields['PHASE']>0)))
    if not support:return np.zeros(6),False
    q=[float(fields[name][c].mean()-fields[name][b].mean()) for name in FIELD_NAMES]
    q.append(-abs(n1-n0)/n0)
    assert np.isfinite(q).all()
    return np.asarray(q),True

def checks():
    y,x=np.indices((33,33));r=np.hypot(y-16,x-16);b=r<=8;c=r<=10
    f={name:np.exp(-((r-10)/.7)**2) for name in FIELD_NAMES}
    q,u=score(f,b,c);rev,ur=score(f,c,b)
    assert u and ur and q[0]>0 and rev[0]==-q[0]
    zero,_=score(f,b,b);assert np.array_equal(zero,np.zeros(6))
    constant={name:np.ones((33,33)) for name in FIELD_NAMES};q,u=score(constant,b,c);assert u and (q[:5]==0).all()
    none={name:np.zeros((33,33)) for name in FIELD_NAMES};q,u=score(none,b,c);assert not u and (q==0).all()
    q,u=score(f,b,np.zeros_like(b));assert not u and (q==0).all()
    h=c.copy();h[16,16]=False;assert np.array_equal(outer(h),outer(c))
    island=c.copy();island[0,0]=True;assert np.array_equal(outer(island),outer(c))
    cached,_=score(f,b,c,outer(b));uncached,_=score(f,b,c);assert np.array_equal(cached,uncached)
    return 9
