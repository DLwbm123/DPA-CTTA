"""Frozen finite opportunity test on optimizer-generated quadratic histories."""
import json,math,os
from pathlib import Path
from core import B1,B2,LR,EPS,T,H,HC,mv,add,state,transport,error,gate,loss

def step(s,x,g):
    s['t']+=1
    for j in range(2):
        s['m'][j]=B1*s['m'][j]+(1-B1)*g[j]
        s['r'][j]=B2*s['r'][j]+(1-B2)*g[j]
        s['v'][j]=B2*s['v'][j]+(1-B2)*g[j]*g[j]
        x[j]-=LR*s['m'][j]/(1-B1**s['t'])/(math.sqrt(s['v'][j]/(1-B2**s['t']))+EPS)

def future(s,x):
    s=json.loads(json.dumps(s));x=x[:];ls=[]
    for _ in range(H):
        ls.append(loss(x));step(s,x,mv(HC,x))
    return dict(mean_pre_loss=sum(ls)/H,final_loss=loss(x))

def run(c):
    direct=os.environ['AUDIT_MODE']=='synthetic2';rows=[]
    for name in ('stationary','abrupt','returning','mixed','null'):
        x=[1.,0.];s=state([]);gs=[];hs=[]
        for k in range(T):
            shifted=name=='abrupt' or (name=='returning' and k>=24) or (name=='mixed' and k%2==1)
            hk=[[4.,0.],[0.,.5]] if shifted else HC
            center=[2.,-1.] if shifted else [0.,0.]
            g=mv(hk,[a-b for a,b in zip(x,center)]);gs.append(g);hs.append(hk);step(s,x,g)
        old=x[:];new=old[:] if name=='null' else [1.,0.];delta=[a-b for a,b in zip(new,old)]
        s=state(gs,direct);d=mv(HC,delta);oracle=state([add(g,mv(hk,delta)) for g,hk in zip(gs,hs)],direct)
        tr=transport(s,d);controls=dict(keep=s,fresh=state([]),transport=tr,first_only=transport(s,d,True),historical_oracle=oracle)
        arms={k:future(v,new) for k,v in controls.items()};arms['native_no_restore']=future(s,old)
        arms['fixed_source']=dict(mean_pre_loss=loss([1.,0.]),final_loss=loss([1.,0.]))
        rows.append(dict(case=name,gate_pass=gate(d,d),old_point=old,restored_point=new,delta=delta,
                         transport_oracle_max_abs_error=error(tr,oracle),arms=arms,
                         improvement_over_fresh=arms['fresh']['mean_pre_loss']-arms['transport']['mean_pre_loss']))
    nonnull=rows[:4]
    result=dict(schema=1,cases=rows,model_calls=0,
                qualified_finite_opportunity=all(r['improvement_over_fresh']>=-1e-10 for r in nonnull) and sum(r['improvement_over_fresh']>1e-6 for r in nonnull)>=2,
                checks=dict(stationary_exact=rows[0]['transport_oracle_max_abs_error']<1e-10,
                            null_identity=abs(rows[4]['arms']['keep']['mean_pre_loss']-rows[4]['arms']['transport']['mean_pre_loss'])<1e-10))
    assert all(result['checks'].values())
    p=Path(c['root'])/'public'/('direct.json' if direct else 'recurrence.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
