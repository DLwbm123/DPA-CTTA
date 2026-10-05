"""Metadata-only split and predeclared, transitive outer-search decisions."""
import copy,hashlib,math
from collections import defaultdict,Counter

SEED=43127

def key(content):return hashlib.sha256(f'{SEED}:{content}'.encode()).hexdigest()

def split(rows,screen_size=384):
    primary=[r for r in rows if r['subset']=='remaining_dev']
    if any(r.get('patient_linkage')!='UNKNOWN' for r in rows):raise ValueError('real patient linkage requires group-aware registration before execution')
    domains=defaultdict(list)
    identities={}
    for r in primary:
        identity=r['image_sha256']
        if identity in identities and identities[identity]!=r['domain']:raise ValueError('cross-domain duplicate requires grouped stratification')
        identities[identity]=r['domain']
    for i,d in identities.items():domains[d].append(i)
    quota={d:int(.6*len(v)) for d,v in domains.items()};left=round(.6*len(identities))-sum(quota.values())
    for d in sorted(domains,key=lambda d:(-(.6*len(domains[d])-quota[d]),d))[:left]:quota[d]+=1
    assignment={};ordered={}
    for d,xs in domains.items():
        xs=sorted(xs,key=key);ordered[d]=xs[:quota[d]]
        assignment.update({x:'SEARCH' if i<quota[d] else 'SEALED_REVIEW' for i,x in enumerate(xs)})
    counts=dict.fromkeys(domains,0)
    while sum(counts.values())<min(screen_size,sum(quota.values())):
        progressed=False
        for d in sorted(domains):
            if counts[d]<len(ordered[d]) and sum(counts.values())<screen_size:counts[d]+=1;progressed=True
        if not progressed:break
    pool={x for d in domains for x in ordered[d][:counts[d]]}
    audit=dict(seed=SEED,patient_dependence='UNKNOWN',grouping='image content hash; duplicates bound',primary_observations=len(primary),primary_identities=len(identities),SEARCH=sum(quota.values()),SEALED_REVIEW=len(identities)-sum(quota.values()),screen_identities=len(pool),domains={d:dict(primary=len(domains[d]),SEARCH=quota[d],SEALED_REVIEW=len(domains[d])-quota[d],screen=counts[d]) for d in sorted(domains)})
    return assignment,pool,audit

def ranking(results):
    return sorted(results,key=lambda x:(not x['nonrisk'],-math.floor(x['delta_pp']/.05+1e-9),-min(x['order_delta_pp']),x['seconds_per_image'],x['id']))

def promote(results,configs,n=4):
    methods=[r for r in results if configs[r['id']]['family']!='control'];rank=ranking(methods);chosen=[]
    for family in sorted({configs[r['id']]['family'] for r in rank}):
        xs=[r for r in rank if configs[r['id']]['family']==family]
        if xs and not all(d < -2 for d in xs[0]['order_delta_pp']):chosen.append(xs[0]['id'])
    for r in rank:
        if len(chosen)>=n:break
        if r['id'] not in chosen:chosen.append(r['id'])
    chosen=sorted(chosen,key=lambda x:next(i for i,r in enumerate(rank) if r['id']==x))[:n]
    control=ranking([r for r in results if configs[r['id']]['family']=='control'])[0]['id']
    return list(dict.fromkeys(['G',control]+chosen)),control

def signature(c):
    import json
    return json.dumps({k:v for k,v in c.items() if k!='id'},sort_keys=True)

def extensions(results,configs):
    rank=ranking([r for r in results if configs[r['id']]['family']!='control']);out=[];seen={signature(c) for c in configs.values()}
    params={'teacher':('teacher_mix',.05,.75),'pixel_weight':('variance_temperature',.005,.2),'online_adapter':('adapter_lr_multiplier',.1,10),'target_prototype':('prototype_mix',.05,.75)}
    for r in rank[:2]:
        base=configs[r['id']];p,lo,hi=params[base['family']]
        for suffix,factor in [('HALF',.5),('DOUBLE',2),('LR15',1)]:
            c=copy.deepcopy(base);c['id']=base['id']+'_'+suffix
            if suffix=='LR15':c['final_lr_multiplier']=1.5
            else:c['params'][p]=max(lo,min(hi,c['params'][p]*factor))
            if signature(c) not in seen:seen.add(signature(c));out.append(c)
    families={'teacher':'T','pixel_weight':'W','online_adapter':'A','target_prototype':'P'};best={}
    for r in rank:
        k=families[configs[r['id']]['family']]
        if k not in best and r['delta_pp']>=0 and r['worst_cell_pp']>=-2:best[k]=r
    pairs=[]
    for a,b in [('T','W'),('P','W'),('A','T'),('A','W')]:
        if a in best and b in best:pairs.append((best[a]['delta_pp']+best[b]['delta_pp'],a+'+'+b,a,b))
    for _,_,a,b in sorted(pairs,key=lambda x:(-x[0],x[1]))[:2]:
        ca,cb=configs[best[a]['id']],configs[best[b]['id']]
        out.append(dict(id=ca['id']+'_'+cb['id'],family='combo',host='G',params={},components={a:copy.deepcopy(ca['params']),b:copy.deepcopy(cb['params'])}))
    return out

def ablations(primary,configs):
    from .method import components
    modules=components(primary);a=copy.deepcopy(primary);a.update(id='ABL_C_'+primary['id'],host='C',final_lr_multiplier=1);a['params'].pop('final_lr_multiplier',None)
    out=[a];reused=[]
    if len(modules)==2:
        out=[]
        for removed in sorted(modules):
            keep={k:v for k,v in modules.items() if k!=removed};out.append(dict(id='ABL_DROP_'+removed+'_'+primary['id'],family='combo',host='G',params={},components=keep))
    else:
        k=next(iter(modules));b=copy.deepcopy(primary);b['id']='ABL_KEY_'+primary['id'];p=b['params']
        if k=='T':p['current_student']=True
        elif k=='W':p['variance_disabled']=True # Predeclared choice: remove variance; retain boundary if present.
        elif k=='A':
            factor=float(primary.get('final_lr_multiplier',1))
            if factor==1:reused.append('G');b=None
            else:b=dict(id=b['id'],family='control',host='native_grata',params=dict(final_lr_multiplier=factor))
        else:p['current_only']=True
        if b:out.append(b)
    return out,reused
