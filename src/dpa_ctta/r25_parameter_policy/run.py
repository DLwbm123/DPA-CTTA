"""Blind, replicated mechanism matrix; existing journal and hard-cap watchdog."""
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from .method import Host, ARMS
from ..r20_model_only_search import runtime as base, score as scoring
from ..r20_model_only_search.schedule import Campaign as ExistingCampaign
from ..r20_model_only_search.report import public, csvout, aggregate, bootstrap_pair, diagnostics
from ..r24_c_context.run import hard_metrics
from ..r10_12h_core.run import read, save, sha
from ..r9_current_first.storage import lease

ID = 'R25_PARAMETER_POLICY'
CORE = list(ARMS[:8])
SEEDS = [20260907, 17011, 29009]
CONTRASTS = [('PG_RETAIN', 'C'), ('PG_RETAIN', 'ANCHOR'), ('PG_RETAIN', 'REPEAT5'),
             ('GREEDY', 'RANDOM4'), ('PG_RETAIN', 'GREEDY'), ('PG_STRUCT', 'PG_CONS'),
             ('PG_RETAIN', 'PG_STRUCT'), ('PG_COV', 'PG_RETAIN'), ('RN_DPO', 'GREEDY')]


class Campaign(ExistingCampaign):
    def qualify(self):
        if read(self.root/'MECHANICAL_TESTS.json')['status'] != 'PASS':
            raise ValueError('mechanical tests missing')
        self.state['status'] = 'PROFILING'; self.save()
        for offset in range(0, len(ARMS), 4):
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                futures = {pool.submit(base.run_task, self.c, 'profile_'+cid, gpu, 1800,
                    dict(candidate=self.candidates[cid])): cid
                    for cid, gpu in zip(ARMS[offset:offset+4], self.c['gpu_assignments'])}
                for future in concurrent.futures.as_completed(futures):
                    cid = futures[future]; result = future.result()
                    self.state['engineering'][cid] = result['status']
                    if result['status'] == 'COMPLETE':
                        self.profiles[cid] = result['result']
                    self.save()
        save(self.root/'PROFILE_ADMISSION.json', dict(profiles=self.profiles,
             charged_GPU_seconds=base.amounts(self.c)[0], label_reads=0))
        if len(self.profiles) != len(ARMS):
            raise RuntimeError('real engineering/profile failure; no formal launch')

    def matrix(self, arms, seeds):
        return [job for seed in seeds for job in self.make_jobs('FORMAL',
                [self.candidates[a] for a in arms], seed=seed)]

    def execute(self):
        self.qualify()
        full = self.matrix(list(ARMS), SEEDS)
        if not self.fits(full):
            raise RuntimeError('registered three-seed full matrix does not fit 46h online cap; no reduced scientific substitute')
        # Timing-only expansion is frozen before any formal predictions or labels.
        seeds = SEEDS.copy()
        extended = self.matrix(list(ARMS), seeds + [41017, 53003])
        estimated_wall = sum(self.estimate(j['candidate']) for j in full)/4
        if estimated_wall < 16*3600 and self.fits(extended):
            seeds += [41017, 53003]
        freeze = dict(primary='PG_RETAIN', controls=['C', 'ANCHOR', 'REPEAT5', 'RANDOM4', 'GREEDY'],
            arms=list(ARMS), seeds=seeds, contrasts=CONTRASTS, code_sha=self.c['code_sha'],
            frozen_at=time.time(), allocation='3 seeds; 5 iff three-seed estimated wall <16h and full five-seed matrix fits',
            label_based_selection=False, review_historically_exposed=True,
            three_seed_estimated_wall_hours=estimated_wall/3600,
            registered_jobs=len(ARMS)*len(seeds)*2)
        save(self.root/'FROZEN.json', freeze)
        self.state['frozen'] = freeze; self.save()
        # Each priority block includes every registered seed and both orders.
        for name, arms in [('CORE', CORE), ('EXTENSION', list(ARMS[8:]))]:
            planned = self.matrix(arms, seeds)
            actual = self.run_pairs(name, planned, required=True)
            if len(actual) != len(planned):
                raise RuntimeError('incomplete '+name+' matrix; preserve every attempt, no scientific retry')
        active = [p for p in (self.root/'processes').glob('online_*.json') if read(p).get('active')]
        if active:
            raise ValueError('online workers remain at label barrier')
        save(self.root/'stages/FINAL.barrier.json', dict(status='ALL_WORKERS_RETIRED', at=time.time()))
        save(self.root/'FINAL_JOBS.json', self.jobs)
        release = self.root/'REVIEW_RELEASE.json'
        if release.exists():
            raise ValueError('duplicate label release')
        save(release, dict(frozen_sha256=sha(freeze), all_formal_workers_retired=True, at=time.time(), selection_closed=True))
        self.state.update(status='FINAL_SCORING', review_opened=True); self.save()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = {pool.submit(base.run_task, self.c, 'score_final_'+j['id'], None, 3600,
                       dict(job=j, stage='FINAL', final=True)): j for j in self.jobs}
            for future in concurrent.futures.as_completed(futures):
                if future.result()['status'] != 'COMPLETE':
                    raise RuntimeError('independent scorer incomplete: '+futures[future]['id'])
        self.state.update(status='COMPLETE', execution_finished=True, ended=time.time(), delivery='PENDING_GITHUB')
        self.save(); report(self.c, self.state)


def diagnostic_metrics(c):
    request = json.loads(os.environ['RUN_JOB'])
    path = Path(c['output_root'])/'target'/request['job']['id']/'visits.jsonl'
    traces = path.open()
    def metrics(mask, label):
        trace = json.loads(next(traces))
        result = hard_metrics(mask, label)
        encoded = trace.get('proposal_masks_b64')
        if encoded:
            gt = torch.nn.functional.interpolate(label.float(), (64, 64), mode='nearest').numpy()[0].astype(bool)
            proposals = [np.unpackbits(np.frombuffer(base64.b64decode(a), dtype=np.uint8)).reshape(2,64,64).astype(bool) for a in encoded]
            for channel, item in enumerate(result):
                item['proposal_Dice64'] = [float(2*(a[channel]&gt[channel]).sum()/(a[channel].sum()+gt[channel].sum()))
                    if a[channel].sum()+gt[channel].sum() else 1. for a in proposals]
        return result
    return metrics


def report(c, state):
    root = Path(c['output_root']); out = root/'public'; out.mkdir(exist_ok=True)
    resource = base.ledger(c, state)
    for name, value in [('RUN_STATE.json', state), ('RESOURCE_LEDGER.json', resource), ('CANDIDATES.json', c['candidates'])]:
        save(out/name, public(value))
    for name in ('FROZEN.json', 'MECHANICAL_TESTS.json', 'PROFILE_ADMISSION.json', 'DATA_SPLIT_AUDIT.json'):
        if (root/name).exists():
            save(out/name, public(read(root/name)))
    rows = []
    if (root/'FINAL_JOBS.json').exists():
        done = [j for j in read(root/'FINAL_JOBS.json') if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
        rows = scoring.load_rows(root, 'FINAL', done, final=True)
    main, domains = aggregate(rows)
    csvout(out/'FULL_RESULTS.csv', main); csvout(out/'DOMAIN_CHANNEL.csv', domains)
    csvout(out/'MECHANISM_DIAGNOSTICS.csv', diagnostics(rows))
    csvout(out/'COST.csv', [dict(phase=a['phase'], attempt=a['attempt'], status=a['status'],
           wall_seconds=a['wall_seconds'], **a['cost']) for a in resource['attempts']])
    freeze = read(root/'FROZEN.json') if (root/'FROZEN.json').exists() else None
    pairs, cis, negative = [], [], []
    if rows and freeze:
        for a, b in CONTRASTS:
            for role in ('SEARCH', 'SEALED_REVIEW'):
                result, ci, neg = bootstrap_pair(rows, a, b, freeze['seeds'], role=role)
                pairs.append(result); cis += ci; negative += neg
    csvout(out/'PAIRED_SUMMARY.csv', pairs); csvout(out/'CONTENT_BOOTSTRAP_CI.csv', cis)
    csvout(out/'ALL_NEGATIVE_CELLS.csv', negative)
    from scipy.stats import spearmanr
    calibration = []
    for arm in ARMS:
        for role in ('SEARCH', 'SEALED_REVIEW'):
            selected = [r for r in rows if r['condition']==arm and r['role']==role and 'proposal_Dice64' in r['metrics'][0]]
            gaps, regrets, correlations = [], [], []
            for r in selected:
                dice = np.mean([m['proposal_Dice64'] for m in r['metrics']], axis=0)
                rewards = r['diagnostics']['rewards']; index = r['diagnostics']['selected']
                gaps.append(100*(dice.max()-dice.min())); regrets.append(100*(dice.max()-dice[index]))
                if np.ptp(dice)>0 and np.ptp(rewards)>0:
                    correlations.append(float(spearmanr(dice,rewards).statistic))
            if selected:
                calibration.append(dict(condition=arm, role=role, observations=len(selected),
                    rank_estimable=len(correlations), mean_within_case_Spearman=statistics.mean(correlations) if correlations else None,
                    mean_proposal_range_pp=statistics.mean(gaps), mean_selection_regret_pp=statistics.mean(regrets),
                    diagnostic_resolution=64, primary_metric_resolution=512,
                    note='Post-hoc only; repeated content/orders/seeds; no online label feedback'))
    csvout(out/'REWARD_CALIBRATION.csv', calibration)
    rv={(r['candidate'], r['baseline']):r for r in pairs if r['role']=='SEALED_REVIEW'}
    required=[('PG_RETAIN', b) for b in ('C','ANCHOR','REPEAT5','GREEDY')]
    qualified=state['status']=='COMPLETE' and all(rv.get(k,{}).get('status')=='COMPLETE' for k in required)
    signal=None
    if qualified:
        x=rv['PG_RETAIN','C']
        signal=(x['delta_pp']>=.3 and min(x['order_delta_pp'])>0 and
                x['positive_trajectories']>=math_ceil(.8*x['trajectory_count']) and
                x['imageweighted_delta_pp']>=0 and x['worst_seed_averaged_cell_pp']>=-2 and
                all(rv[k]['delta_pp']>0 for k in required))
    save(out/'FINAL_DECISION.json', dict(status=state['status'], primary='PG_RETAIN',
        qualified_complete_matrix=qualified, development_signal=signal, independent_generalization=False,
        novelty_established=False, patient_dependence='UNKNOWN', label_based_selection=False))
    lines=['# R25: conditional adapter parameter exploration', '',
        f"Status: **{state['status']}**. Qualified complete matrix: **{qualified}**. Development signal: **{signal}**.", '',
        'All configurations and seeds are frozen before formal execution. No performance-based pruning or tuning. '
        'PG_RETAIN is the primary; a better extension does not replace it retrospectively. '
        'The policy is a contextual bandit, not long-horizon RL. Methods transfer selected mechanisms, not complete paper reproductions.', '',
        '| Arm | REVIEW macro Dice (%) | Trajectories |', '|---|---:|---:|']
    for arm in ARMS:
        rr=[r for r in main if r['condition']==arm and r['role']=='SEALED_REVIEW']
        if rr:lines.append(f"| {arm} | {statistics.mean(r['macro_Dice_percent'] for r in rr):.4f} | {len(rr)} |")
    lines += ['', 'Each trajectory contains 1,951 arrivals, with 1,017 SEARCH and 678 campaign-held REVIEW identities plus 256 context identities. '
        'Both orders and all seeds share historically exposed content. Patient linkage and ROI crop provenance remain UNKNOWN. '
        'The 512px hard-Dice metric weights domain/channel equally; empty/empty Dice is 1. ASSD and soft metrics are not computed. '
        'The 64px proposal calibration is only a post-hoc mechanism diagnostic and is not primary Dice.', '',
        'PAIRED_SUMMARY.csv and ALL_NEGATIVE_CELLS.csv include all registered contrasts and adverse cells. '
        'Content bootstrap couples orders/seeds and cannot establish patient independence. '
        'Costs include initialization, failed work, all temporary proposals, policy operations and profiling. '
        'REPEAT5 is an additional-update control, not exact FLOP matching. Public aggregate files exclude images, masks, checkpoints and identities.', '',
        f"Runtime wall: {resource['wall_seconds']/3600:.3f} h; GPU-worker: {resource['gpu_worker_seconds']/3600:.3f} h; CPU-worker: {resource['cpu_worker_seconds']/3600:.3f} h. Hard cap: 48h from registered T0; online cap 46h. Code: `{c['code_sha']}`.", '',
        'GitHub final delivery is separate from execution completion.']
    if state.get('reason'):lines += ['', 'Incomplete reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def math_ceil(x):
    return int(np.ceil(x))


def supervise():
    c=base.config(); root=Path(c['output_root']); campaign=Campaign(c)
    with lease(root/'supervisor', dict(experiment_id=ID, config_sha256=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('campaign already started')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)))
        try:campaign.execute()
        except BaseException as e:
            traceback.print_exc()
            campaign.state.update(status='INCOMPLETE', reason=str(e), ended=time.time(), delivery='PENDING_GITHUB')
            campaign.save(); report(c,campaign.state)


def main():
    base.ID=ID; base.Host=Host; scoring.metrics=hard_metrics
    torch.set_num_threads(2)
    mode=os.environ['RUN_MODE']
    if mode=='worker':
        if os.environ['RUN_PHASE'].startswith('score_'):scoring.metrics=diagnostic_metrics(base.config())
        result=base.worker();sys.exit(0 if result['status']=='COMPLETE' else 1)
    elif mode=='watch':base.watch()
    elif mode=='supervise':supervise()
    else:raise ValueError('unregistered mode')


if __name__=='__main__':main()
