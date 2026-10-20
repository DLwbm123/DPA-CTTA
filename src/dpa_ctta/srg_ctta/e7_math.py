"""Frozen masking-response scores; no labels, oracle, selector, or learned parameters."""
import numpy as np
from scipy.special import expit
from .signal_math import score_frame as common_frame

NAMES=('MARS_FD','RAW_MARGIN','NULL_FUSION','CONSISTENCY_ONLY','MASKED_MEAN_DIRECT','GEOMETRY_ONLY','RESPONSE_SHIFT','STRATIFIED_ROLL')

def masks(case,condition):
    rng=np.random.default_rng(20261018+case*4+condition)
    for _ in range(8):
        rho=float(rng.uniform(.2,.4));ids=rng.choice(128*128,int(np.floor(128*128*rho)),replace=False)
        coarse=np.zeros((128,128),bool);coarse.ravel()[ids]=True
        yield coarse.repeat(4,0).repeat(4,1),dict(rho=rho,patch_ids=ids.tolist())

def entropy_from_logit(z):
    # H=sigmoid(z)*softplus(-z)+sigmoid(-z)*softplus(z), no cancellation/epsilon.
    if not np.isfinite(z).all():raise ValueError('nonfinite source logit')
    return expit(z)*np.logaddexp(0,-z)+expit(-z)*np.logaddexp(0,z)

def fields(z,q,s):
    if not np.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('invalid masked probability')
    p=expit(z);h=entropy_from_logit(z);c=p*q+(1-p)*(1-q)
    shift=q.shape[1]//2+1;qs=np.roll(q,shift,axis=1);cs=p*qs+(1-p)*(1-qs)
    return dict(MARS_FD=-h*(1-c),RAW_MARGIN=np.abs(z),NULL_FUSION=-h*2*p*(1-p),CONSISTENCY_ONLY=c,MASKED_MEAN_DIRECT=(2*s.astype(int)-1)*(q-.5),RESPONSE_SHIFT=-h*(1-cs))

def stratify(rows):
    groups={}
    for r in rows:
        if r['common']:groups.setdefault((r['direction'],r['size_bucket'],r['distance_bucket']),[]).append(r)
    moved=changed=0
    for group in groups.values():
        group.sort(key=lambda r:r['block_id'])
        for i,r in enumerate(group):
            other=group[(i-1)%len(group)];r['scores']['STRATIFIED_ROLL']=other['scores']['MARS_FD'];r['stratified_from_block']=other['block_id']
            moved+=r['block_id']!=other['block_id'];changed+=r['scores']['MARS_FD']!=other['scores']['MARS_FD']
    return dict(common=sum(r['common'] for r in rows),permutation_positions_moved=moved,permutation_values_changed=changed)

def score_frame(f,z,s,b,q,maskcount):
    rows,_=common_frame(f,z,s,b);ff=fields(z[:,:,1],q,s[1])
    for r in rows:
        xy=np.asarray(r['query']);y,x=xy[:,0],xy[:,1];geometry=r['scores']['GEOMETRY_ONLY']
        r['query_mask_rate']=float(maskcount[y,x].mean()/8)
        r['scores']={n:None for n in NAMES}
        if r['common']:
            for n,a in ff.items():r['scores'][n]=float(a[y,x].mean())
            r['scores']['GEOMETRY_ONLY']=geometry
    return rows,stratify(rows)
