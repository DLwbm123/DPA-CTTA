"""Independent NumPy references for R7 equations, not production model code.
No GPU, network, patient data, source checkpoint, or training is used.
"""
from __future__ import annotations
import numpy as np


def array(x):
    value = np.asarray(x, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('non-finite input')
    return value


def spd_solve(a, b):
    a, b = array(a), array(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        raise ValueError('symmetric square matrix required')
    # No jitter, eigenvalue clipping, or fallback to a different model.
    l = np.linalg.cholesky(a)
    return np.linalg.solve(l.T, np.linalg.solve(l, b))


def film(h, gamma, beta):
    h, gamma, beta = array(h), array(gamma), array(beta)
    return h + np.expm1(.1*np.tanh(gamma))*h + .1*np.tanh(beta)


def stable_matrix(w, rho=.95):
    w = array(w)
    if not 0 < rho < 1:
        raise ValueError('rho must lie in (0,1)')
    return rho*w/max(1., float(np.linalg.norm(w, 2)))


def gaussian_filter(m, p, f, q, observation, r):
    m, p, f, q, observation, r = map(array, (m,p,f,q,observation,r))
    n = m.size
    if m.shape != (n,) or observation.shape != (n,) or any(x.shape != (n,n) for x in (p,f,q,r)):
        raise ValueError('dimension mismatch')
    spd_solve(p, np.eye(n)); spd_solve(q, np.eye(n)); spd_solve(r, np.eye(n))
    pred_m, pred_p = f@m, f@p@f.T+q
    pred_p = (pred_p+pred_p.T)/2
    precision = spd_solve(pred_p,np.eye(n))+spd_solve(r,np.eye(n))
    new_p = spd_solve(precision,np.eye(n))
    new_p = (new_p+new_p.T)/2
    new_m = spd_solve(precision,spd_solve(pred_p,pred_m)+spd_solve(r,observation))
    return new_m, new_p


def predictive_subspace(jcov, jprobe, curvature, rank, ridge=.01):
    jc,jp,w = map(array,(jcov,jprobe,curvature))
    if jc.ndim!=2 or jp.ndim!=2 or jc.shape[1]!=jp.shape[1] or w.shape!=(jc.shape[0],):
        raise ValueError('Jacobian shapes')
    if rank<1 or rank>min(jp.shape) or ridge<=0 or np.any(w<0):
        raise ValueError('rank/ridge/curvature')
    n=jc.shape[1]
    precision=ridge*np.eye(n)+(jc.T*w)@jc/len(jc)
    cov=spd_solve(precision,np.eye(n));cov=(cov+cov.T)/2
    vals,vecs=np.linalg.eigh(cov)
    root=(vecs*np.sqrt(vals))@vecs.T
    gram=root@jp.T@jp@root;gram=(gram+gram.T)/2
    eig,vec=np.linalg.eigh(gram)
    selected=vec[:,np.argsort(eig)[::-1][:rank]]
    # Sign convention preserves subspace/covariance and reproducible artifacts.
    for col in range(rank):
        k=np.argmax(np.abs(selected[:,col]))
        if selected[k,col]<0:selected[:,col]*=-1
    b=root@selected
    return b,cov


def soft_threshold(x, threshold):
    x=array(x)
    if not np.isfinite(threshold) or threshold<0:raise ValueError('threshold')
    return np.sign(x)*np.maximum(np.abs(x)-threshold,0.)


def sparse_energy(delta,h,o,prior,lam=.01,mu=.1,kappa=1.):
    delta,h,o,prior=map(array,(delta,h,o,prior))
    if lam<0 or mu<=0 or kappa<=0:raise ValueError('sparse parameters')
    residual=o-h@(prior+delta)
    return .5*float(residual@residual)/kappa**2+lam*float(np.abs(delta).sum())+.5*mu*float(delta@delta)


def sparse_correct(h,o,prior,steps=5,lam=.01,mu=.1,kappa=1.):
    h,o,prior=map(array,(h,o,prior))
    if h.shape!=(o.size,prior.size) or o.ndim!=1 or prior.ndim!=1 or type(steps)is not int or steps<0:
        raise ValueError('sparse dimensions/steps')
    if not all(np.isfinite(v) for v in (lam,mu,kappa)) or lam<0 or mu<=0 or kappa<=0:
        raise ValueError('sparse parameters')
    eta=1/(float(np.linalg.norm(h,2))**2/kappa**2+mu)
    delta=np.zeros_like(prior);history=[sparse_energy(delta,h,o,prior,lam,mu,kappa)]
    for _ in range(steps):
        grad=h.T@(h@(prior+delta)-o)/kappa**2+mu*delta
        delta=soft_threshold(delta-eta*grad,eta*lam)
        history.append(sparse_energy(delta,h,o,prior,lam,mu,kappa))
    return prior+delta,delta,np.asarray(history)


def huber(x, cutoff=1.):
    x=array(x)
    if cutoff<=0:raise ValueError('positive cutoff')
    a=np.abs(x)
    return np.where(a<=cutoff,.5*x*x,cutoff*a-.5*cutoff**2)


def robust_energy(z,h,o,variance,prior,lam=.1,cutoff=1.):
    z,h,o,variance,prior=map(array,(z,h,o,variance,prior))
    if np.any(variance<=0) or lam<=0:raise ValueError('positive variance/ridge')
    res=(o-h@z)/np.sqrt(variance)
    return float(huber(res,cutoff).mean()+.5*lam*np.sum((z-prior)**2))


def robust_correct(h,o,variance,prior,steps=3,lam=.1,cutoff=1.):
    h,o,variance,prior=map(array,(h,o,variance,prior))
    if h.shape!=(o.size,prior.size) or o.ndim!=1 or variance.shape!=o.shape or prior.ndim!=1:
        raise ValueError('robust dimensions')
    if o.size<1 or np.any(variance<=0) or lam<=0 or cutoff<=0 or type(steps)is not int or steps<0:
        raise ValueError('robust parameters')
    z=prior.copy();history=[robust_energy(z,h,o,variance,prior,lam,cutoff)];weight_history=[]
    for _ in range(steps):
        res=(o-h@z)/np.sqrt(variance)
        nu=np.minimum(1.,cutoff/np.maximum(np.abs(res),cutoff))
        weights=nu/variance
        lhs=(h.T*weights)@h/o.size+lam*np.eye(prior.size)
        rhs=h.T@(weights*o)/o.size+lam*prior
        z=spd_solve(lhs,rhs)
        history.append(robust_energy(z,h,o,variance,prior,lam,cutoff));weight_history.append(weights)
    return z,np.asarray(history),weight_history
