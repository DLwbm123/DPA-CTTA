"""Descriptive paired scientific review. No automatic gate or next-run nomination."""
import math
from collections import defaultdict
import numpy as np

def review(rows):
    """Scalar rows: arm/order/content/domain/subset/dice[OD,OC]/assd[OD,OC].

    ASSD None is retained, never changed to zero. Content identifiers are used
    privately for pairing and excluded from the aggregate returned here.
    """
    arms=['C_BASE','C0']+[g+'_'+v for g in 'ABC' for v in ('FULL','STATIC')]
    lookup={};domains=set()
    for row in rows:
        if row['arm'] not in arms or row['order'] not in (0,1,4):raise ValueError('screen scope')
        if row['subset']!='remaining_dev':continue
        key=(row['arm'],row['order'],row['content'])
        if key in lookup:raise ValueError('duplicate scalar')
        if len(row['dice'])!=2 or len(row['assd'])!=2:raise ValueError('OD/OC')
        if any(not math.isfinite(x) for x in row['dice']):raise ValueError('finite Dice')
        if any(x is not None and (not math.isfinite(x) or x<0) for x in row['assd']):raise ValueError('ASSD missing/finite')
        lookup[key]=row;domains.add(row['domain'])
    if not domains:raise ValueError('no scored rows')
    keys={o:{c for a,k,c in lookup if a=='C_BASE' and k==o} for o in (0,1,4)}
    if not keys[0] or any(keys[o]!=keys[0] for o in keys):raise ValueError('same-content stream coverage')
    for arm in arms:
        for o in keys:
            if {c for a,k,c in lookup if a==arm and k==o}!=keys[o]:raise ValueError('paired arm coverage')
    absolute=[];paired=[]
    for arm in arms:
        for o in keys:
            for domain in sorted(domains):
                rr=[lookup[arm,o,c] for c in keys[o] if lookup['C_BASE',o,c]['domain']==domain]
                if not rr or any(r['domain']!=domain for r in rr):raise ValueError('domain pairing')
                values=np.asarray([r['dice'] for r in rr],dtype=float)
                absolute.append(dict(arm=arm,order=o,domain=domain,n=len(rr),OD=float(values[:,0].mean()),OC=float(values[:,1].mean()),macro=float(values.mean())))
    for g in 'ABC':
        for ref in ('C_BASE',g+'_STATIC'):
            for o in keys:
                for domain in sorted(domains):
                    pairs=[(lookup[g+'_FULL',o,c],lookup[ref,o,c]) for c in keys[o] if lookup[ref,o,c]['domain']==domain]
                    delta=np.asarray([np.asarray(a['dice'])-b['dice'] for a,b in pairs]);macro=delta.mean(1)
                    assd=[]
                    for ch in range(2):
                        valid=[a['assd'][ch]-b['assd'][ch] for a,b in pairs if a['assd'][ch] is not None and b['assd'][ch] is not None]
                        assd.append(dict(channel=('OD','OC')[ch],common_valid=len(valid),undefined=len(pairs)-len(valid),mean_delta=float(np.mean(valid)) if valid else None))
                    paired.append(dict(group=g,reference=ref,order=o,domain=domain,n=len(pairs),OD=float(delta[:,0].mean()),OC=float(delta[:,1].mean()),macro=float(macro.mean()),negative=int((macro<0).sum()),positive=int((macro>0).sum()),zero=int((macro==0).sum()),paired_tail_quantiles={str(q):float(np.quantile(macro,q)) for q in (.05,.25,.5,.75,.95)},ASSD=assd))
    summary=[]
    for g in 'ABC':
        for ref in ('C_BASE',g+'_STATIC'):
            selected=[r for r in paired if r['group']==g and r['reference']==ref]
            summary.append(dict(group=g,reference=ref,primary=float(np.mean([r['macro'] for r in selected if r['order'] in (0,1)])),recurrence=float(np.mean([r['macro'] for r in selected if r['order']==4])),worst_domain_order=min(r['macro'] for r in selected)))
    return dict(status='DESCRIPTIVE_RESEARCH_REVIEW_NOT_A_GATE',absolute=absolute,paired=paired,summary=summary,automatic_nomination=False,next_execution_authorized=False,source_and_online_costs='MUST_ATTACH_SEPARATE_MEASURED_COST_LEDGER',limitations='development contents; different orders are not new patients; no significance, equivalence or clinical safety claim')
