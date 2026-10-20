"""Image-only, fixed-scale vessel path geometry; no file, model, or label access."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
EPS=1e-12
CROSS=ndi.generate_binary_structure(2,1)
FULL=np.ones((3,3),bool)

def enhance(green,od):
    g=np.asarray(green,np.float64);od=np.asarray(od,bool)
    if g.ndim!=2 or g.shape!=od.shape or not np.isfinite(g).all() or g.min()<0 or g.max()>1:raise ValueError('finite 2D green [0,1] and matching OD required')
    hyy=4*ndi.gaussian_filter(g,2,order=(2,0),mode='reflect',truncate=4)
    hxx=4*ndi.gaussian_filter(g,2,order=(0,2),mode='reflect',truncate=4)
    hxy=4*ndi.gaussian_filter(g,2,order=(1,1),mode='reflect',truncate=4)
    disc=np.sqrt((hxx-hyy)**2+4*hxy*hxy);lo=(hxx+hyy-disc)/2;hi=(hxx+hyy+disc)/2
    swap=np.abs(lo)>np.abs(hi);l1=np.where(swap,hi,lo);l2=np.where(swap,lo,hi)
    rb=np.divide(l1,l2,out=np.zeros_like(g),where=np.abs(l2)>EPS)
    v=np.exp(-rb*rb/(2*.5**2))*(-np.expm1(-(l1*l1+l2*l2)/(2*.03**2)))
    v[(l2<=EPS)|~od]=0
    low=(v>=.03)&od;labels,n=ndi.label(low,FULL);seed=np.unique(labels[(v>=.10)&od]);seed=seed[seed!=0]
    mask=np.isin(labels,seed)&low
    return v,skeletonize(mask,method='zhang')

def graph(s):
    nodes={tuple(int(x) for x in p) for p in np.argwhere(s)};adj={p:[] for p in nodes}
    for y,x in sorted(nodes):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                q=(y+dy,x+dx)
                if (dy or dx) and q in nodes:
                    if dy and dx and ((y,x+dx) in nodes or (y+dy,x) in nodes):continue
                    adj[y,x].append(q)
        adj[y,x].sort()
    return adj

def measure(green,od):
    v,s=enhance(green,od);adj=graph(s);critical={p for p,a in adj.items() if len(a)!=2};bad=set(critical);front=set(critical)
    for _ in range(4):
        front={q for p in front for q in adj[p]}-bad;bad|=front
    seen=set();paths=[]
    for begin in sorted(critical):
        for nxt in adj[begin]:
            edge=tuple(sorted((begin,nxt)))
            if edge in seen:continue
            seen.add(edge);path=[begin,nxt];prev,cur=begin,nxt
            while cur not in critical:
                choices=[p for p in adj[cur] if p!=prev]
                if len(choices)!=1:raise AssertionError('path degree')
                nxt=choices[0];edge=tuple(sorted((cur,nxt)))
                if edge in seen:break
                seen.add(edge);path.append(nxt);prev,cur=cur,nxt
            paths.append(path)
    points=[];reliability=[];curvature=[];pid=[]
    for pathid,path in enumerate(paths):
        for i in range(4,len(path)-4):
            p=path[i]
            if p in bad:continue
            u=np.asarray(path[i-4],float)-p;w=np.asarray(path[i+4],float)-p
            den=np.linalg.norm(u)*np.linalg.norm(w)
            if den<=EPS:continue
            k=np.pi-np.arccos(np.clip(np.dot(u,w)/den,-1,1));k=k if k>=np.pi/18 else 0.
            points.append(p);reliability.append(v[p]);curvature.append(k);pid.append(pathid)
    pts=np.asarray(points,int).reshape(-1,2);a=np.asarray(reliability);k=np.asarray(curvature);pid=np.asarray(pid,int)
    weights=a*k;shifted=weights.copy();moved=changed=0
    for j in np.unique(pid):
        ix=np.flatnonzero(pid==j);rolled=np.roll(weights[ix],len(ix)//2);shifted[ix]=rolled
        moved+=len(ix) if len(ix)>1 else 0;changed+=int(np.count_nonzero(rolled!=weights[ix]))
    result=dict(points=pts,reliability=a,curvature=k,path_id=pid,bend=weights,vessel=a.copy(),shift=shifted,
        critical=np.asarray(sorted(critical),int).reshape(-1,2),forbidden=np.asarray(sorted(bad),int).reshape(-1,2),
        skeleton_points=len(adj),junction_points=sum(len(a)>2 for a in adj.values()),permuted_positions=moved,permuted_weights_changed=changed)
    for value in result.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    return result

def outer(mask):
    m=np.asarray(mask,bool);lab,n=ndi.label(m,FULL)
    if n==0:return np.zeros_like(m)
    size=np.bincount(lab.ravel());size[0]=0;main=lab==int(np.argmax(size));filled=ndi.binary_fill_holes(main,structure=CROSS)
    return filled&~ndi.binary_erosion(filled,FULL,border_value=0)

def distances(mask,points):
    boundary=outer(mask)
    if not boundary.any():return None
    d=ndi.distance_transform_edt(~boundary)
    return np.minimum((d[points[:,0],points[:,1]]/3)**2,1)

def score(evidence,base,candidate,mode='bend'):
    if mode not in ('bend','vessel','shift'):raise ValueError('unknown mode')
    if evidence['bend'].sum()<=EPS:return 0.,False,'NO_BEND_EVIDENCE'
    points=evidence['points'];d0=distances(base,points);d1=distances(candidate,points)
    if d0 is None or d1 is None:return 0.,False,'NO_OUTER_CONTOUR'
    w=evidence[mode];den=w.sum()
    if den<=EPS:raise AssertionError('undefined control denominator')
    supported=bool(np.any((np.abs(d0-d1)>EPS)&(evidence['bend']>0)))
    q=float(np.dot(w,d0-d1)/den)
    if not np.isfinite(q):raise ValueError('nonfinite energy')
    return (q if supported else 0.),supported,'SUPPORTED' if supported else 'NO_INFLUENCE'
