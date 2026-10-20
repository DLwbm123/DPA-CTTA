"""Candidate-relative angular partition score on a common, finite direction domain."""
import numpy as np
NAMES=('CROP_Q','GLOBAL_Q','RESAMPLE_Q','LOGIT_Q','CROP_LOGIT_Q','E6_FEATURE','RAW_MARGIN','GEOMETRY_ONLY','AREA_DELTA','CROP_Q_SHIFT','STRATIFIED_ROLL')

def window(query):
    q=np.asarray(query);start=np.floor((q.min(0)+q.max(0)+1)/2).astype(int)-128;start=np.clip(start,0,256)
    fits=bool(((q>=start)&(q<start+256)).all())
    return start.tolist(),fits

def norm(v):return np.sqrt((v*v).sum(-1))
def contrast(a,b):
    na,nb=norm(a),norm(b)
    if na==0 or nb==0:return None
    return float(1-np.dot(a/na,b/nb))

def partitions(representations,b,s,edit):
    valid=np.ones(b.shape,bool);norms={}
    for name,v in representations.items():
        if not np.isfinite(v).all():raise ValueError('nonfinite representation')
        norms[name]=norm(v);valid&=norms[name]>0
    nvalid=int(valid.sum())
    if not valid[edit].all():return None,'ZERO_QUERY',nvalid
    new=b.copy();new[edit]=s[edit]
    if min(int((valid&m).sum()) for m in (b,~b,new,~new))<3:return None,'SHORT_PARTITION',nvalid
    delta=s[edit].astype(int)-b[edit].astype(int);answer={}
    for name,v in representations.items():
        u=v[valid]/norms[name][valid,None]
        inside=b[valid];a=u[inside].sum(0);bb=u[~inside].sum(0)
        change=(v[edit]/norms[name][edit,None]*delta[:,None]).sum(0)
        old=contrast(a,bb);after=contrast(a+change,bb-change)
        if old is None or after is None:return None,'ZERO_PARTITION_SUM',nvalid
        answer[name]=after-old
    if not all(np.isfinite(v) for v in answer.values()):raise ValueError('nonfinite partition score')
    return answer,'COMMON',nvalid

def stratify(rows):
    groups={}
    for r in rows:
        if r['common']:groups.setdefault((r['direction'],r['size_bucket'],r['distance_bucket']),[]).append(r)
    moved=changed=0
    for group in groups.values():
        group.sort(key=lambda r:r['block_id'])
        for i,r in enumerate(group):
            other=group[(i-1)%len(group)];r['scores']['STRATIFIED_ROLL']=other['scores']['CROP_Q'];r['stratified_from_block']=other['block_id']
            moved+=r['block_id']!=other['block_id'];changed+=r['scores']['CROP_Q']!=other['scores']['CROP_Q']
    return dict(common=sum(r['common'] for r in rows),permutation_positions_moved=moved,permutation_values_changed=changed)
