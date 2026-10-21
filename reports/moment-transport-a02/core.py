"""Finite synthetic mechanism falsification; no model/data dependencies."""
import json
import math
import os
from pathlib import Path

B1, B2, LR, EPS, T, H = .9, .99, .05, 1e-8, 48, 32
HC = [[1., .2], [.2, 2.]]

def mv(a, x):
    return [sum(a[i][j]*x[j] for j in range(2)) for i in range(2)]

def add(a, b):
    return [x+y for x,y in zip(a,b)]

def state(gs, direct=False):
    if direct:
        return dict(m=[sum((1-B1)*B1**(len(gs)-1-k)*g[j] for k,g in enumerate(gs)) for j in range(2)],
                    r=[sum((1-B2)*B2**(len(gs)-1-k)*g[j] for k,g in enumerate(gs)) for j in range(2)],
                    v=[sum((1-B2)*B2**(len(gs)-1-k)*g[j]**2 for k,g in enumerate(gs)) for j in range(2)],t=len(gs))
    s=dict(m=[0.,0.],r=[0.,0.],v=[0.,0.],t=0)
    for g in gs:
        s['m']=[B1*a+(1-B1)*b for a,b in zip(s['m'],g)]
        s['r']=[B2*a+(1-B2)*b for a,b in zip(s['r'],g)]
        s['v']=[B2*a+(1-B2)*b*b for a,b in zip(s['v'],g)]
        s['t']+=1
    return s

def transport(s, d, first_only=False):
    out=json.loads(json.dumps(s));s1=1-B1**s['t'];s2=1-B2**s['t']
    out['m']=[x+s1*y for x,y in zip(s['m'],d)]
    if not first_only:
        out['r']=[x+s2*y for x,y in zip(s['r'],d)]
        out['v']=[v+2*dj*r+s2*dj*dj for v,r,dj in zip(s['v'],s['r'],d)]
    assert min(out['v'])>=-1e-10
    out['v']=[max(0.,v) for v in out['v']]
    return out

def error(a,b):
    return max(abs(x-y) for name in ('m','r','v') for x,y in zip(a[name],b[name]))

def loss(x):
    return .5*sum(a*b for a,b in zip(x,mv(HC,x)))

def future(s):
    s=json.loads(json.dumps(s));x=[1.,0.];losses=[]
    for _ in range(H):
        losses.append(loss(x));g=mv(HC,x);s['t']+=1
        for j in range(2):
            s['m'][j]=B1*s['m'][j]+(1-B1)*g[j]
            s['v'][j]=B2*s['v'][j]+(1-B2)*g[j]**2
            x[j]-=LR*(s['m'][j]/(1-B1**s['t']))/(math.sqrt(s['v'][j]/(1-B2**s['t']))+EPS)
    return dict(mean_pre_loss=sum(losses)/H,first_post_loss=losses[1],final_loss=loss(x))

def gate(a,b):
    na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b))
    if not all(math.isfinite(x) for x in a+b):return False
    if na==nb==0:return True
    if min(na,nb)==0:return False
    return sum(x*y for x,y in zip(a,b))>0 and math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))/(na+nb+1e-12)<=.5

def run(c):
    direct=os.environ['AUDIT_MODE']=='synthetic2';rows=[]
    for name in ('shared','alternating','adversarial'):
        gs=[];translated=[];delta=[1.,0.]
        for k in range(T):
            # Each g is realizable by a convex quadratic at an arbitrary historical point.
            # Its center is point - inverse(H_k) g; no real training trace is claimed.
            hk=HC if name=='shared' or (name=='alternating' and k%2==0) else [[4.,0.],[0.,.5]]
            g=[-2.,.1] if name=='adversarial' else [math.sin(k*.4),math.cos(k*.3)]
            gs.append(g);translated.append(add(g,mv(hk,delta)))
        s=state(gs,direct);oracle=state(translated,direct);d=mv(HC,delta)
        trans=transport(s,d)
        controls=dict(keep=s,fresh=state([],direct),transport=trans,first_only=transport(s,d,True),historical_oracle=oracle)
        rows.append(dict(case=name,gate_pass=gate(d,d),transport_oracle_max_abs_error=error(trans,oracle),
                         d=d,nonrestored_coordinate_shift=d[1],arms={k:future(v) for k,v in controls.items()}))
    checks=dict(shared_exact=rows[0]['transport_oracle_max_abs_error']<1e-10,
                zero_jump_identity=error(transport(state([[1.,2.]]),[0.,0.]),state([[1.,2.]]))==0,
                zero_gate=gate([0.,0.],[0.,0.]),one_zero_rejected=not gate([0.,0.],[1.,0.]),
                opposite_rejected=not gate([1.,0.],[-1.,0.]),nonfinite_rejected=not gate([float('nan'),0.],[1.,0.]),
                cross_group_shift_nonzero=rows[0]['nonrestored_coordinate_shift']!=0)
    assert all(checks.values()),checks
    counter=rows[2];arms=counter['arms']
    result=dict(schema=1,checks=checks,cases=rows,model_calls=0,
                counterexample_detected=counter['gate_pass'] and counter['transport_oracle_max_abs_error']>1e-6 and arms['transport']['mean_pre_loss']>arms['fresh']['mean_pre_loss'])
    root=Path(c['root']);out=root/'public'/('direct.json' if direct else 'recurrence.json')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
