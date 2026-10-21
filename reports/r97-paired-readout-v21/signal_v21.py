"""Single v2.1 signal; no labels, learned thresholds or trajectory writes."""
import numpy as np
EPS=1e-7

def margin(p,kind):
    if kind=='sigmoid':return p-.5
    if kind!='softmax':raise ValueError('unknown decoder')
    assert p.shape[0]>=2
    return np.stack([p[c]-np.max(np.delete(p,c,axis=0),axis=0) for c in range(len(p))])

def decode(p,kind):
    if kind=='sigmoid':return p>=.5
    arg=np.argmax(p,axis=0)
    return np.stack([arg==c for c in range(len(p))])

def aggregate(v,mask,classes):
    vals=[float(np.mean(v[c][mask[c]])) for c in classes if mask[c].any()]
    return float(np.mean(vals)) if vals else 0.

def entropy(p,kind):
    p=np.clip(p,1e-7,1-1e-7)
    if kind=='sigmoid':return -(p*np.log(p)+(1-p)*np.log1p(-p))/np.log(2)
    return -np.sum(p*np.log(p),axis=0)/np.log(len(p))

def compute(p0,pt,pk,pq,views0,viewst,kind='sigmoid',valid=None):
    assert p0.shape==pt.shape==pk.shape==pq.shape
    assert views0.shape==viewst.shape and views0.shape[1:]==p0.shape
    assert len(views0)>1 and all(np.isfinite(a).all() for a in (p0,pt,pk,pq,views0,viewst))
    b0,bt,bk,bq=[decode(a,kind) for a in (p0,pt,pk,pq)]
    e=bt.astype(np.int8)-b0.astype(np.int8)
    E=e!=0;D=bk!=bq
    valid=np.ones(p0.shape[1:],dtype=bool) if valid is None else valid
    assert valid.shape==p0.shape[1:] and valid.any(),'invalid common support'
    allowed=valid.copy()
    if kind=='softmax':
        for p in (p0,pt):allowed &= (p==p.max(0)).sum(0)==1
    E=E & allowed;D=D & valid;R=E&D
    classes=list(range(len(p0))) if kind=='sigmoid' else list(range(1,len(p0)))
    dm=np.stack([margin(b,kind)-margin(a,kind) for a,b in zip(views0,viewst)])
    miss=1-np.mean(e[None]*dm>EPS,axis=0)
    U=aggregate(miss,R,classes)
    current=np.stack([decode(v,kind) for v in viewst])
    one=np.mean(current!=bt[None],axis=0)
    signed=np.where(dm>EPS,1,np.where(dm < -EPS,-1,0))
    nopos=1-np.abs(signed.mean(0))
    hd=entropy(pk,kind)-entropy(pq,kind)
    if kind=='softmax':hd=np.broadcast_to(hd,p0.shape)
    a=e*(bk.astype(np.int8)-bq.astype(np.int8))
    nR=[int(R[c].sum()) for c in classes];nD=[int(D[c].sum()) for c in classes]
    out=dict(U=U,z=int(U>.5 and any(nR)),eligible=int(any(nR)),
             entropyD=aggregate(hd,D,classes),entropyFull=float(hd[classes][:,valid].mean()),
             singleR=aggregate(one,R,classes),noAction=aggregate(miss,E,classes),
             noHistory=aggregate(nopos,D,classes),Earea=float(E[classes].mean()),
             Darea=float(np.any(D[classes],axis=0)[valid].mean()),
             R_count=nR,D_count=nD,reverse_count=[int(((a[c]<0)&R[c]).sum()) for c in classes],
             U_class=[float(miss[c][R[c]].mean()) if R[c].any() else None for c in classes],
             excluded_tie_pixels=int((valid & ~allowed).sum()))
    return out

def self_check():
    def shaped(x):return np.array(x,dtype=np.float64).reshape(-1,1,1)
    a=shaped([.45,.40,.15]);b=shaped([.33,.36,.31]);v0=np.stack([a]*6);vt=np.stack([b]*6)
    assert b[1]<a[1] and decode(b,'softmax')[1] and not decode(a,'softmax')[1]
    o=compute(a,b,b,a,v0,vt,'softmax');assert o['U']==0 and o['z']==0
    assert abs(float(margin(b,'softmax')[1]-margin(a,'softmax')[1])-.08)<1e-12
    a=shaped([.4]);b=shaped([.6]);assert np.array_equal(margin(b,'sigmoid')-margin(a,'sigmoid'),b-a)
    v0=np.stack([a]*6);vt=np.stack([b]+[shaped([.3])]*5)
    o=compute(a,b,b,a,v0,vt);assert np.isclose(o['U'],5/6) and o['z']==1
    rev=compute(a,b,a,b,v0,vt);assert rev['reverse_count']==[1]
    deletion=compute(1-a,1-b,1-b,1-a,1-v0,1-vt);assert deletion['U']==o['U']
    assert compute(a,a,b,a,v0,v0)['eligible']==0
    assert compute(a,b,b,b,v0,vt)['eligible']==0
    aa=np.concatenate([a,a]);bb=np.concatenate([a,b]);vv0=np.stack([aa]*6);vvt=np.stack([bb]+[np.concatenate([a,shaped([.3])])]*5)
    assert np.isclose(compute(aa,bb,bb,aa,vv0,vvt)['U'],5/6)
    tie=shaped([.5,.5]);assert compute(tie,shaped([.4,.6]),shaped([.4,.6]),tie,np.stack([tie]*6),np.stack([shaped([.4,.6])]*6),'softmax')['eligible']==0
    try:compute(a,b,b,a,v0,vt,valid=np.zeros((1,1),bool))
    except AssertionError:pass
    else:raise AssertionError('invalid support accepted')
    return dict(status='PASS',softmax_counterexample_fixed=True,sigmoid_equivalent=True,
                deletion_symmetric=True,single_active_class=True,empty_cases=True,
                reverse_action_diagnosed=True,tie_excluded=True,invalid_support_rejected=True,
                synthetic_only=True,model_calls=0)
