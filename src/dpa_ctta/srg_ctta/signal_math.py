"""Label-free, fixed E6-B block scores. No model, labels or oracle imports."""
import numpy as np
from scipy.ndimage import label,binary_erosion,distance_transform_edt
from scipy.special import expit

NAMES=('FEATURE','LOGIT_MATCH','MARGIN_DIRECT','GEOMETRY_ONLY','FEATURE_SHIFT','STRATIFIED_ROLL')

def components(s,b):
    g,n=label((s!=b).any(0),structure=np.ones((3,3),int))
    count=lambda m:np.bincount(g.ravel(),weights=m.ravel(),minlength=n+1)[1:].astype(int)
    od,oc=count(s[0]!=b[0]),count(s[1]!=b[1]);sizes=np.bincount(g.ravel(),minlength=n+1)[1:]
    return g,sizes,od,oc

def block_anchors(g,j,eligible,prob,agree):
    yy,xx=np.nonzero(g==j);h,w=g.shape
    y0,y1=max(0,int(yy.min())-16),min(h,int(yy.max())+17)
    x0,x1=max(0,int(xx.min())-16),min(w,int(xx.max())+17)
    d=distance_transform_edt(g[y0:y1,x0:x1]!=j)
    local=(d<=16)&eligible[y0:y1,x0:x1]&agree[y0:y1,x0:x1]
    anchors=[]
    for good in (prob[y0:y1,x0:x1]>=.9,prob[y0:y1,x0:x1]<=.1):
        ay,ax=np.nonzero(local&good)
        order=np.lexsort((ax+x0,ay+y0,d[ay,ax]))[:8]
        anchors.append(np.column_stack((ay[order]+y0,ax[order]+x0)))
    return np.column_stack((yy,xx)),anchors

def evidence(q,fg,bg):
    arrays=(q,fg,bg)
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError('nonfinite representation')
    norms=[np.sqrt((a*a).sum(1)) for a in arrays]
    if any((n==0).any() for n in norms):return None
    q,fg,bg=[a/n[:,None] for a,n in zip(arrays,norms)]
    return (q@fg.T).mean(1)-(q@bg.T).mean(1)

def stratify(records):
    groups={}
    for r in records:
        if r['common']:
            key=(r['direction'],r['size_bucket'],r['distance_bucket'])
            groups.setdefault(key,[]).append(r)
    moved=changed=0
    for rows in groups.values():
        rows.sort(key=lambda r:r['block_id'])
        for i,r in enumerate(rows):
            other=rows[(i-1)%len(rows)];r['scores']['STRATIFIED_ROLL']=other['scores']['FEATURE']
            r['stratified_from_block']=other['block_id'];moved+=r['block_id']!=other['block_id'];changed+=r['scores']['FEATURE']!=other['scores']['FEATURE']
    return dict(common=sum(r['common'] for r in records),permutation_positions_moved=moved,permutation_values_changed=changed)

def score_frame(f,z,s,b):
    h,w=s.shape[1:]
    assert f.shape==(h,w,32) and z.shape==(h,w,2)
    if not np.isfinite(f).all() or not np.isfinite(z).all():raise ValueError('nonfinite frozen output')
    g,sizes,od,oc=components(s,b);D=g>0;eligible=distance_transform_edt(~D)>3
    prob=expit(z[:,:,1]);agree=s[1]==b[1]
    surf=(s[1]^binary_erosion(s[1],border_value=0))|(b[1]^binary_erosion(b[1],border_value=0))
    distance=distance_transform_edt(~surf) if surf.any() else np.zeros((h,w))
    shift=w//2+1;records=[]
    for index in np.flatnonzero((od==0)&(oc>0)):
        j=int(index+1);q,anchors=block_anchors(g,j,eligible,prob,agree)
        delta=s[1,q[:,0],q[:,1]].astype(int)-b[1,q[:,0],q[:,1]].astype(int)
        assert (delta!=0).all() # OC-only union is precisely the OC edit set
        add=bool((delta<0).any());remove=bool((delta>0).any())
        dist=float(distance[q[:,0],q[:,1]].mean());size=int(sizes[index])
        r=dict(block_id=j,size=size,oc_pixels=int(oc[index]),direction='MIXED' if add and remove else 'EXPAND' if add else 'SHRINK',size_bucket='1' if size==1 else '2_4' if size<=4 else 'GE5',pred_boundary_distance=dist,distance_bucket='LE1' if dist<=1 else 'LE3' if dist<=3 else 'GT3',query=q.tolist(),anchors=[a.tolist() for a in anchors],common=False,intrinsic_valid={},scores={name:None for name in NAMES})
        if min(map(len,anchors))<8:r['reason']='ANCHORS_SHORT';records.append(r);continue
        fg,bg=anchors
        take=lambda a,pts:a[pts[:,0],pts[:,1]]
        shifted=lambda pts:f[pts[:,0],(pts[:,1]-shift)%w]
        reps={'FEATURE':[take(f,pts) for pts in (q,fg,bg)],'LOGIT_MATCH':[take(z,pts) for pts in (q,fg,bg)],'FEATURE_SHIFT':[shifted(pts) for pts in (q,fg,bg)]}
        for name,v in reps.items():
            e=evidence(*v);r['intrinsic_valid'][name]=e is not None
            if e is not None:r['scores'][name]=float(np.mean(delta*e))
        r['scores']['MARGIN_DIRECT']=float(np.abs(take(z,q)[:,1]).mean())
        dfg=np.sqrt(((q[:,None,:]-fg[None,:,:])**2).sum(2)).mean(1)
        dbg=np.sqrt(((q[:,None,:]-bg[None,:,:])**2).sum(2)).mean(1)
        r['scores']['GEOMETRY_ONLY']=float(np.mean(delta*(dbg-dfg)/16))
        r['common']=all(r['intrinsic_valid'].values());r['reason']='COMMON' if r['common'] else 'ZERO_NORM'
        assert all(v is None or np.isfinite(v) for v in r['scores'].values())
        records.append(r)
    return records,stratify(records)

def auc(scores,utility,weights=None):
    s=np.asarray(scores,float);u=np.asarray(utility,float);w=np.ones(len(u)) if weights is None else np.asarray(weights,float)
    pos=u>0;neg=u<0
    if not pos.any() or not neg.any():return None
    p,n=s[pos,None],s[None,neg];pairs=w[pos,None]*w[None,neg]
    credit=(p>n).astype(float)+.5*(p==n)
    return float((credit*pairs).sum()/pairs.sum())
