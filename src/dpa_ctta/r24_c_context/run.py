"""Bounded C-based campaign using the existing guarded workers and journal."""
import concurrent.futures
from pathlib import Path
import json
import os
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from .method import Host
from ..r20_model_only_search import runtime as base, score as scoring
from ..r20_model_only_search.schedule import Campaign as ExistingCampaign
from ..r20_model_only_search.report import public, csvout, aggregate, bootstrap_pair, diagnostics
from ..r10_12h_core.run import read, save, sha
from ..r9_current_first.storage import lease

ID = 'R24_C_ANCHORED_CONTEXT'


def hard_metrics(mask, label):
    gt = label.numpy()[0].astype(bool)
    result = []
    for k, name in enumerate(('OD', 'OC')):
        intersection = int((mask[k] & gt[k]).sum())
        predicted, actual = int(mask[k].sum()), int(gt[k].sum())
        result.append(dict(channel=name, dice=2 * intersection / (predicted + actual) if predicted + actual else 1.,
                           intersection=intersection, pred_pixels=predicted, gt_pixels=actual,
                           total_pixels=int(mask[k].size), pred_empty=not predicted, gt_empty=not actual,
                           TP=intersection, FP=predicted-intersection, FN=actual-intersection,
                           assd=None, soft_dice=None, Brier=None, ASSD_status='NOT_COMPUTED'))
    return result


def stage_results(rows, seed):
    results = {}
    for arm in ('C', 'C_LR15', 'STATIC', 'VIEW', 'ANCHOR'):
        rr = [r for r in rows if r['condition'] == arm]
        if not rr:
            continue
        cc = {f'{o}/{d}/{k}': statistics.mean(m['dice'] * 100 for r in rr if r['order'] == o and r['domain'] == d
                                            for m in r['metrics'] if m['channel'] == k)
              for o in (0, 1) for d in sorted({r['domain'] for r in rr}) for k in ('OD', 'OC')}
        results[arm] = dict(macro_Dice_percent=statistics.mean(cc.values()), cells=cc,
                            imageweighted_Dice_percent=statistics.mean(statistics.mean(m['dice'] * 100 for m in r['metrics']) for r in rr),
                            seconds_per_image=statistics.mean(r['seconds'] for r in rr), observations=len(rr))
    pairs = {}
    for control in ('C', 'C_LR15', 'STATIC', 'VIEW'):
        pairs[control] = bootstrap_pair(rows, 'ANCHOR', control, [seed], role='SEARCH', repeats=500)[0]
    thresholds = {'C': .2, 'C_LR15': 0., 'STATIC': .1, 'VIEW': .1}
    advance = all(v['status'] == 'COMPLETE' and v['delta_pp'] >= thresholds[k]
                  and v['delta_pp'] > 0 and min(v['order_delta_pp']) > 0
                  and v['imageweighted_delta_pp'] >= 0 and v['worst_single_seed_cell_pp'] >= -2
                  for k, v in pairs.items())
    return dict(summaries=results, ANCHOR_comparisons=pairs, advance=advance, primary='ANCHOR', review_labels_read=False)


class Campaign(ExistingCampaign):
    def qualify(self):
        if read(self.root/'MECHANICAL_TESTS.json')['status'] != 'PASS':
            raise ValueError('mechanical tests missing')
        ids = ('C', 'STATIC', 'VIEW', 'ANCHOR')
        self.state['status'] = 'PROFILING'; self.save()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            ff = {pool.submit(base.run_task, self.c, 'profile_'+cid, gpu, 600,
                              dict(candidate=self.candidates[cid])): cid
                  for cid, gpu in zip(ids, self.c['gpu_assignments'])}
            for f in concurrent.futures.as_completed(ff):
                cid = ff[f]; result = f.result()
                self.state['engineering'][cid] = result['status']
                if result['status'] == 'COMPLETE':
                    self.profiles[cid] = result['result']
                self.save()
        save(self.root/'PROFILE_ADMISSION.json', dict(profiles=self.profiles, charged_GPU_seconds=base.amounts(self.c)[0]))
        if len(self.profiles) != len(ids):
            raise RuntimeError('real engineering/profile failure; preserve receipts for repair')

    def score_stage(self, stage, jobs, final=False):
        self.state['status'] = 'FINAL_SCORING' if final else 'SEARCH_SCORING'; self.save()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            ff = {pool.submit(base.run_task, self.c, ('score_final_' if final else 'score_'+stage+'_')+j['id'],
                              None, 3600, dict(job=j, stage=stage, final=final)): j for j in jobs}
            for f in concurrent.futures.as_completed(ff):
                if f.result()['status'] != 'COMPLETE':
                    raise RuntimeError('CPU scorer incomplete: '+ff[f]['id'])
        if final:
            return
        rows = scoring.load_rows(self.root, stage, jobs)
        result = stage_results(rows, self.c['native_seed'])
        save(self.root/'stages'/f'{stage}.selection.json', result)
        self.state['stages'][stage]['status'] = 'SCORED'; self.save()
        return result

    def execute(self):
        self.qualify()
        initial = self.make_jobs('SEARCH', list(self.candidates.values()))
        if not self.fits(initial):
            raise RuntimeError('entire registered five-condition SEARCH matrix exceeds budget')
        jobs = self.run_pairs('SEARCH', initial, required=True)
        if len(jobs) != len(initial):
            raise RuntimeError('incomplete SEARCH matrix; no scientific substitution')
        result = self.score_stage('SEARCH', jobs)
        self.state['SEARCH_gate'] = result['advance']
        seeds = [self.c['native_seed']]
        confirmation = []
        budget_reason = None
        if result['advance']:
            first = self.make_jobs('CONFIRM_17011', list(self.candidates.values()), seed=17011)
            second = self.make_jobs('CONFIRM_29009', list(self.candidates.values()), seed=29009)
            if self.fits(first + second):
                seeds += [17011, 29009]; confirmation = [('CONFIRM_17011', first), ('CONFIRM_29009', second)]
            elif self.fits(first):
                seeds += [17011]; confirmation = [('CONFIRM_17011', first)]
                budget_reason = 'third seed NOT_RUN_BUDGET, determined before new-seed labels and review'
            else:
                budget_reason = 'NOT_CONFIRMED_BUDGET'
        freeze = dict(primary='ANCHOR', controls=['C', 'C_LR15', 'STATIC', 'VIEW'], seeds=seeds,
                      SEARCH_gate=result['advance'], budget_reason=budget_reason,
                      code_sha=self.c['code_sha'], frozen_at=time.time(), settings_unchanged=True)
        save(self.root/'FROZEN.json', freeze)
        self.state['status'] = 'CONFIGURATION_FROZEN'; self.save()
        for stage, planned in confirmation:
            actual = self.run_pairs(stage, planned, required=True)
            if len(actual) != len(planned):
                raise RuntimeError('registered confirmation incomplete; preserve all attempts')
        if any(read(p).get('active') for p in (self.root/'processes').glob('online_*.json')):
            raise ValueError('online workers remain before review')
        save(self.root/'stages/FINAL.barrier.json', dict(status='ALL_WORKERS_RETIRED', at=time.time()))
        save(self.root/'FINAL_JOBS.json', self.jobs)
        release = self.root/'REVIEW_RELEASE.json'
        if release.exists():
            raise ValueError('review release already exists')
        save(release, dict(frozen_sha256=sha(freeze), all_formal_workers_retired=True, at=time.time()))
        self.state['review_opened'] = True; self.save()
        self.score_stage('FINAL', self.jobs, final=True)
        self.state.update(status='COMPLETE', ended=time.time(), execution_finished=True, delivery='PENDING_GITHUB')
        self.save()
        report(self.c, self.state)


def report(c, state):
    root = Path(c['output_root']); out = root/'public'; out.mkdir(exist_ok=True)
    resource = base.ledger(c, state)
    for name, value in [('RUN_STATE.json', state), ('RESOURCE_LEDGER.json', resource),
                        ('CANDIDATES.json', c['candidates'])]:
        save(out/name, public(value))
    for name in ('FROZEN.json', 'MECHANICAL_TESTS.json', 'PROFILE_ADMISSION.json', 'DATA_SPLIT_AUDIT.json'):
        if (root/name).exists():
            save(out/name, public(read(root/name)))
    search = read(root/'stages/SEARCH.selection.json') if (root/'stages/SEARCH.selection.json').exists() else None
    if search is not None:
        save(out/'SEARCH_RESULTS.json', search)
    rows = []
    if (root/'FINAL_JOBS.json').exists():
        done = [j for j in read(root/'FINAL_JOBS.json') if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
        rows = scoring.load_rows(root, 'FINAL', done, final=True)
    main, domain = aggregate(rows)
    for r in domain:
        r['ASSD_status'] = 'NOT_COMPUTED'
        for k in ('ASSD_defined', 'ASSD_undefined', 'ASSD_conditional_mean_pixels'):
            r.pop(k, None)
    csvout(out/'FULL_RESULTS.csv', main)
    csvout(out/'DOMAIN_CHANNEL.csv', domain)
    csvout(out/'MECHANISM_DIAGNOSTICS.csv', diagnostics(rows))
    costs = [dict(phase=a['phase'], attempt=a['attempt'], status=a['status'], wall_seconds=a['wall_seconds'], **a['cost'])
             for a in resource['attempts']]
    csvout(out/'COST.csv', costs)
    pairs = []; cis = []; negative = []; signal = False
    freeze = read(root/'FROZEN.json') if (root/'FROZEN.json').exists() else None
    if rows and freeze:
        for control in freeze['controls']:
            for role in ('SEARCH', 'SEALED_REVIEW'):
                a, b, d = bootstrap_pair(rows, 'ANCHOR', control, freeze['seeds'], role=role)
                pairs.append(a); cis += b; negative += d
        review = {x['baseline']: x for x in pairs if x['role'] == 'SEALED_REVIEW'}
        if len(freeze['seeds']) >= 2 and all(x['status'] == 'COMPLETE' for x in review.values()):
            b = review['C']
            signal = (b['delta_pp'] >= .3 and min(b['order_delta_pp']) > 0 and
                      b['positive_trajectories'] >= (5 if len(freeze['seeds']) == 3 else 3) and
                      b['imageweighted_delta_pp'] >= 0 and b['worst_seed_averaged_cell_pp'] >= -2 and
                      all(review[k]['delta_pp'] > 0 for k in ('C_LR15', 'STATIC', 'VIEW')))
    csvout(out/'PAIRED_SUMMARY.csv', pairs)
    csvout(out/'CONTENT_BOOTSTRAP_CI.csv', cis)
    csvout(out/'ALL_NEGATIVE_CELLS.csv', negative)
    save(out/'FINAL_DECISION.json', dict(status=state['status'], primary='ANCHOR', strong_development_signal=signal,
                                      SEARCH_gate=None if search is None else search['advance'],
                                      review_historically_exposed=True, patient_dependence='UNKNOWN',
                                      clinical_validation=False, novelty_established=False))
    lines = ['# R24: C strong baseline and anchored conditional adaptation', '',
             f"Status: **{state['status']}**. Strong development signal: **{signal}**.", '',
             'The frozen primary is ANCHOR; C is the main baseline. STATIC and VIEW are mechanism controls, and C_LR15 checks a simple learning-rate explanation. No W weighting, reliability gate, source retraining, source images, additional pretrained encoder or online labels are used.', '',
             f"Full-stream arrivals per trajectory: 1,951; primary SEARCH/REVIEW identities: 1,017/678. Frozen seeds: {None if freeze is None else freeze['seeds']}. All content was historically exposed; this is campaign-held review, not independent generalization. Patient linkage and ROI crop-center provenance remain UNKNOWN.", '',
             '| Condition | REVIEW macro Dice (%) | Imageweighted (%) | Seed-order trajectories |',
             '|---|---:|---:|---:|']
    for arm in ('C', 'C_LR15', 'STATIC', 'VIEW', 'ANCHOR'):
        rr = [r for r in main if r['condition'] == arm and r['role'] == 'SEALED_REVIEW']
        if rr:
            lines.append(f"| {arm} | {statistics.mean(r['macro_Dice_percent'] for r in rr):.4f} | {statistics.mean(r['imageweighted_Dice_percent'] for r in rr):.4f} | {len(rr)} |")
    lines += ['', '| ANCHOR vs control | REVIEW delta (pp) | Positive trajectories | Worst seed-mean cell (pp) |',
              '|---|---:|---:|---:|']
    for x in pairs:
        if x['role'] == 'SEALED_REVIEW' and x['status'] == 'COMPLETE':
            lines.append(f"| {x['baseline']} | {x['delta_pp']:+.4f} | {x['positive_trajectories']}/{x['trajectory_count']} | {x['worst_seed_averaged_cell_pp']:+.4f} |")
    lines += ['', 'A negative gate is retained. A winning static adapter, higher learning rate or per-view condition does not retrospectively become the registered primary. Conditional adaptation is established prior art; this experiment does not establish novelty. The descriptor may capture content as well as appearance. Zero-U initialization does not make later pseudo-labels correct.', '',
              'Dice is 2TP/(predicted+GT), or 1 when both masks are empty. Domain and OD/OC receive equal weight; orders and seeds are then averaged. ASSD and soft-probability metrics are NOT_COMPUTED in this bounded round. FULL_RESULTS.csv, DOMAIN_CHANNEL.csv and ALL_NEGATIVE_CELLS.csv retain denominators and adverse results. Bootstrap resamples content within domains and couples orders/seeds; it does not resolve patient dependence or selection bias.', '',
              f"Elapsed runtime: {resource['wall_seconds']/3600:.3f} h; GPU workers: {resource['gpu_worker_seconds']/3600:.3f}/12 h; CPU scorers: {resource['cpu_worker_seconds']/3600:.3f} worker h. Code: `{c['code_sha']}`. Timing includes initialization, I/O and actual concurrency; no exact equal-cost superiority claim.", '',
              'All attempts, including profile and failed/recovered work, are charged in COST.csv. Code/protocol/aggregate metrics are public; images, labels, patient/content identifiers, predictions, checkpoints, private paths and credentials remain private. GitHub delivery is recorded separately after proxy push and anonymous verification.']
    if state.get('reason'):
        lines += ['', 'Incomplete reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def supervise():
    c = base.config(); root = Path(c['output_root']); campaign = Campaign(c)
    with lease(root/'supervisor', dict(experiment_id=ID, config_sha256=sha(c))):
        marker = root/'execution_started.json'
        if marker.exists():
            raise ValueError('campaign already launched')
        save(marker, dict(at=time.time(), config_sha256=sha(c)))
        try:
            campaign.execute()
        except BaseException as e:
            traceback.print_exc()
            campaign.state.update(status='INCOMPLETE', reason=str(e), ended=time.time(), delivery='PENDING_GITHUB')
            campaign.save(); report(c, campaign.state)


def main():
    base.ID = ID; base.Host = Host; scoring.metrics = hard_metrics
    torch.set_num_threads(2)
    mode = os.environ['RUN_MODE']
    if mode == 'worker':
        result = base.worker(); sys.exit(0 if result['status'] == 'COMPLETE' else 1)
    elif mode == 'watch':
        base.watch()
    elif mode == 'supervise':
        supervise()
    else:
        raise ValueError('unregistered entry mode')


if __name__ == '__main__':
    main()
