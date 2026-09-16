from pathlib import Path
from collections import deque
import copy,json,math,random,runpy
ROOT=Path(__file__).resolve().parent
ns=runpy.run_path(str(ROOT/'probe_review.py'))
rule=ns['rule']
def bound(row,identity):
    if row['binding']!=identity:raise ValueError('binding')
# The test uses exact identical row/expected bindings; this stricter fixture bound
# is not a substitute for reviewing the full repository authorization machinery.
g=dict(deque=deque,random=random,json=json,math=math,bound=bound,PHYSICAL=rule.PHYSICAL,
       validate_metric=ns['audit_ns']['validate_metric'])
exec((ROOT/'sources/replay_join_extract.py').read_text(),g)
traces=[];labels=[];ordered=[];r=rule.Rule('C');identity={'fixture':'CPU_AUDIT_PROBE'}
def metric(ch,pred=100,k=90):
    return dict(channel=ch,pred_pixels=pred,gt_pixels=100,intersection=k,total_pixels=262144,
                dice=2*k/(pred+100),gt_empty=False,gt_full=False,pred_empty=pred==0,pred_full=pred==262144,assd=1.)
for i in range(1,34):
    ep=.1;et=.2
    obs=dict(e_pre=dict(count=524288,sse=ep*524288,mean=ep),
             e_trial=dict(count=524288,sse=et*524288,mean=et),
             regions=[dict(region=name,count=100,sse=1.,mean=.01) for name in ('OD_fg','OD_bg','OC_fg','OC_bg')],r=.01)
    z=r.decide(obs);r.append(obs['r'])
    z.update(arm='C',counts=rule.PHYSICAL.copy(),source_unchanged=True,transaction_complete=True,
             totals=dict(n_visits=i,n_candidate_adam=i,n_committed=i,n_rejected=0,n_forced=min(i,32),n_eligible=max(0,i-32),parameter_restorations=0),
             adam_committed_step=i,buffer_count_after=len(r.history))
    e=dict(group_id=f'g{i}',sample_id=f's{i}',domain=f'D{i%4}',subset='remaining_dev');ordered.append(e)
    row=dict(binding=identity,visit=i,**e);traces.append(dict(**row,trace=z))
    pre=[metric(ch) for ch in ('OD','OC')]
    trial=[metric(ch,262144,0) for ch in ('OD','OC')] if i==33 else copy.deepcopy(pre)
    labels.append(dict(**row,evaluation=dict(transaction_before_GT=True,metrics=dict(pre=pre,q=copy.deepcopy(pre),trial=trial,emit=copy.deepcopy(trial)))))
rs=g['join'](traces,labels,ordered,dict(arm='C',records=33),identity)
assert rs[-1]['L']==[-90.,-90.] and rs[-1]['trace']['eligible'] and not rs[-1]['trace']['shadow_accept']
report=dict(status='IMPOSSIBLE_METRICS_ACCEPTED_BY_REPLAY_AND_JOIN',records=len(rs),
    impossible_last_L_pp=rs[-1]['L'],eligible=rs[-1]['trace']['eligible'],shadow_accept=rs[-1]['trace']['shadow_accept'],
    full_recompute_run=False,scope='exact replay/join excerpts + exact inherited metric validator; fixture identity comparator; no registered data or model')
(ROOT/'join_probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
