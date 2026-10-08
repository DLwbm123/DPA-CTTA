"""Frozen three-arm SEARCH-informed development follow-up; no learned policy."""
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from .method import Host, ARMS
from ..r25_parameter_policy import run as prior
from ..r26_residual_action.run import Campaign as ResidualCampaign
from ..r20_model_only_search import runtime as base
from ..r20_model_only_search.report import public,csvout,aggregate,bootstrap_pair,diagnostics
from ..r10_12h_core.run import read,save,sha

ID='R27_NESTING_SELECTION'
SEEDS=[20260907,17011,29009]
CONTRASTS=[('GREEDY','ANCHOR'),('GREEDY','RANDOM4')]


def report(c,state):
    root=Path(c['output_root']);out=root/'public';out.mkdir(exist_ok=True)
    resource=base.ledger(c,state)
    for name,value in [('RUN_STATE.json',state),('RESOURCE_LEDGER.json',resource),('CANDIDATES.json',c['candidates'])]:
        save(out/name,public(value))
    for name in ['FROZEN.json','MECHANICAL_TESTS.json','PROFILE_ADMISSION.json','DATA_SPLIT_AUDIT.json']:
        if (root/name).exists():save(out/name,public(read(root/name)))
    jobs=read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done=[j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows=prior.scoring.load_rows(root,'FINAL',done,final=True) if done else []
    main,domains=aggregate(rows);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    csvout(out/'MECHANISM_DIAGNOSTICS.csv',diagnostics(rows))
    csvout(out/'COST.csv',[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],wall_seconds=a['wall_seconds'],**a['cost']) for a in resource['attempts']])
    pairs=[];cis=[];negative=[]
    for a,b in CONTRASTS:
        for role in ('SEARCH','SEALED_REVIEW'):
            result,ci,neg=bootstrap_pair(rows,a,b,SEEDS,role=role)
            pairs.append(result);cis+=ci;negative+=neg
    csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',cis);csvout(out/'ALL_NEGATIVE_CELLS.csv',negative)
    rv={r['baseline']:r for r in pairs if r['role']=='SEALED_REVIEW'}
    qualified=state['status']=='COMPLETE' and len(done)==18 and all(r.get('status')=='COMPLETE' for r in rv.values())
    signal=None
    if qualified:
        a=rv['ANCHOR'];b=rv['RANDOM4']
        signal=(a['delta_pp']>=.3 and b['delta_pp']>0 and min(a['order_delta_pp'])>0 and min(b['order_delta_pp'])>0
            and a['positive_trajectories']>=5 and b['positive_trajectories']>=5
            and a['imageweighted_delta_pp']>=0 and a['worst_seed_averaged_cell_pp']>=-2)
    save(out/'FINAL_DECISION.json',dict(status=state['status'],primary='GREEDY',qualified_complete_matrix=qualified,
        development_signal=signal,development_selection=True,independent_generalization=False,novelty_established=False,patient_dependence='UNKNOWN'))
    lines=['# R27: nesting-only residual candidate selection','',f"Status: **{state['status']}**. Development signal: **{signal}**.",'',
        '| Arm | Legacy REVIEW macro Dice (%) | Trajectories |','|---|---:|---:|']
    for arm in ARMS:
        rr=[r for r in main if r['condition']==arm and r['role']=='SEALED_REVIEW']
        if rr:lines.append(f"| {arm} | {statistics.mean(r['macro_Dice_percent'] for r in rr):.4f} | {len(rr)} |")
    lines+=['','All content is development-exposed. The nesting reward was selected after the R26 SEARCH audit; legacy SEALED_REVIEW is not independent validation. Three seeds and two orders reuse the same content. Patient linkage and ROI provenance remain UNKNOWN.',
        '', 'Primary:512px hard Dice, equal domain/channel then order/seed; empty/empty=1. Image-weighted metrics, all fixed paired contrasts and negative cells are retained. Content bootstrap does not resolve patient dependence or multiplicity.64px proposal masks are diagnostic only and remain private.',
        '', f"Wall:{resource['wall_seconds']/3600:.3f}h; GPU-worker:{resource['gpu_worker_seconds']/3600:.3f}h; CPU-worker:{resource['cpu_worker_seconds']/3600:.3f}h. No cumulative GPU cap. Code:`{c['code_sha']}`.",
        '', 'No policy learning, covariance or DPO is added. RANDOM4 and GREEDY differ only in candidate choice, not candidate generation or committed update count. Nesting alone cannot guarantee anatomical correctness or avoid empty predictions. Final GitHub delivery is separate from execution completion.']
    if state.get('reason'):lines+=['','Incomplete reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


class Campaign(ResidualCampaign):
    def execute(self):
        self.qualify();planned=self.matrix(list(ARMS),SEEDS)
        freeze=dict(primary='GREEDY',arms=list(ARMS),seeds=SEEDS,contrasts=CONTRASTS,code_sha=self.c['code_sha'],
            frozen_at=time.time(),registered_jobs=18,development_exposed=True,reward='negative soft nesting',
            reward_selected_from='R26 SEARCH-only eight-component audit',gpu_worker_cap_seconds=None)
        save(self.root/'FROZEN.json',freeze);self.state['frozen']=freeze;self.save()
        actual=self.run_pairs('FORMAL',planned,required=True)
        if len(actual)!=18:raise RuntimeError('incomplete registered18-trajectory matrix')
        if any(read(p).get('active') for p in (self.root/'processes').glob('online_*.json')):raise ValueError('online workers remain')
        save(self.root/'stages/FINAL.barrier.json',dict(status='ALL_WORKERS_RETIRED',at=time.time()))
        save(self.root/'FINAL_JOBS.json',self.jobs)
        release=self.root/'REVIEW_RELEASE.json'
        if release.exists():raise ValueError('duplicate label release')
        save(release,dict(frozen_sha256=sha(freeze),all_formal_workers_retired=True,at=time.time(),development_exposed=True))
        self.state.update(status='FINAL_SCORING',review_opened=True);self.save();self.score('FINAL',self.jobs,final=True)
        self.state.update(status='COMPLETE',execution_finished=True,ended=time.time(),delivery='PENDING_GITHUB')
        self.save();report(self.c,self.state)


def main():
    prior.ID=ID;prior.ARMS=ARMS;prior.Campaign=Campaign;prior.report=report
    base.ID=ID;base.Host=Host;prior.scoring.metrics=prior.hard_metrics;torch.set_num_threads(2)
    mode=os.environ['RUN_MODE']
    if mode=='worker':
        if os.environ['RUN_PHASE'].startswith('score_'):prior.scoring.metrics=prior.diagnostic_metrics(base.config())
        result=base.worker();sys.exit(0 if result['status']=='COMPLETE' else 1)
    elif mode=='watch':base.watch()
    elif mode=='supervise':prior.supervise()
    else:raise ValueError('unregistered mode')


if __name__=='__main__':main()
