"""One frozen full-flow matrix, existing guarded workers and independent CPU scoring."""
import concurrent.futures
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from .method import Host
from ..r20_model_only_search import runtime as base, score as scoring
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout, diagnostics
from ..r24_c_context.run import hard_metrics
from ..r10_12h_core.run import read, save, sha
from ..r9_current_first.storage import lease

ID = 'R36_INCREMENTAL_MODULES'


def filesystem_guard(c):
    root=Path(c['output_root']).resolve(); checkpoint=Path(c['checkpoint_path']).resolve()
    mount=root.parent.parent
    libraries=[Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    code=[Path(os.environ[k]).resolve() for k in ('DPA_CTTA_BASE_ROOT','DPA_GRATA_ROOT','PYTHONPATH')]
    current={'image':None}; opened=set(); rejected=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(args[0])).resolve()
        protected=any(p.is_relative_to(root/d) for d in ('scorer','scores','stages','reports'))
        allowed=p in (checkpoint,current['image']) or p.is_relative_to(root) or any(p.is_relative_to(x) for x in libraries+code)
        asset=p.suffix.lower() in ('.png','.jpg','.jpeg','.pt','.pth','.safetensors','.npy','.npz')
        if protected or (p.is_relative_to(mount) or asset) and not allowed:
            rejected.append(str(p)); raise PermissionError('registered current-image/model-only boundary')
        if p.is_relative_to(mount): opened.add(str(p))
    sys.addaudithook(audit)
    for p in (root/'scorer/denial_probe.json',root.parent/'data/source_denial_probe'):
        try:p.read_bytes()
        except PermissionError:continue
        raise AssertionError('access boundary failed closed')
    return current,opened,rejected


def public(value):
    # Receipts may contain internal paths inside exception text.
    import re
    from ..r20_model_only_search.report import public as existing
    value = existing(value)
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'/remote-home/[^\s\"\']+', '[PRIVATE_PATH]', value)
    return value


def report(c, state):
    root = Path(c['output_root']); out = root/'public'
    resources = base.ledger(c, state)
    jobs = read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done = [j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows = scoring.load_rows(root, 'FINAL', done, final=True)
    main, domain = aggregate(rows)
    csvout(out/'FULL_RESULTS.csv', main); csvout(out/'DOMAIN_CHANNEL.csv', domain)
    csvout(out/'MECHANISM_DIAGNOSTICS.csv', diagnostics(rows))
    costs = [dict(phase=a['phase'], attempt=a['attempt'], status=a['status'],
                  wall_seconds=a['wall_seconds'], **a['cost']) for a in resources['attempts']]
    csvout(out/'COST.csv', costs)
    pairs, intervals, negative = [], [], []
    for candidate in c['candidates'][2:]:
        for baseline in ('W', 'C'):
            for role in ('SEARCH', 'SEALED_REVIEW'):
                a, b, d = bootstrap_pair(rows, candidate['id'], baseline, c['seeds'], role=role)
                pairs.append(a); intervals.extend(b); negative.extend(d)
    csvout(out/'PAIRED_SUMMARY.csv', pairs)
    csvout(out/'CONTENT_BOOTSTRAP_CI.csv', intervals)
    csvout(out/'ALL_NEGATIVE_CELLS.csv', negative)
    decisions = []
    for candidate in c['candidates'][2:]:
        tests = [x for x in pairs if x['candidate'] == candidate['id'] and x['role']=='SEARCH']
        signal = state['status']=='COMPLETE' and len(tests)==2 and all(
            x['status']=='COMPLETE' and x['delta_pp']>=.3 and min(x['order_delta_pp'])>0
            and x['positive_trajectories']>=5 and x['imageweighted_delta_pp']>=0
            and x['worst_seed_averaged_cell_pp']>=-2 for x in tests)
        decisions.append(dict(candidate=candidate['id'], strong_development_signal=signal,
                              scope='fixed configuration; no independent generalization claim'))
    save(out/'FINAL_DECISION.json', dict(primary='W_TP', decisions=decisions,
                                       review_descriptive_only=True, automatic_retuning=False))
    for name, value in [('RUN_STATE.json', state), ('RESOURCE_LEDGER.json', resources),
                        ('CANDIDATES.json', c['candidates'])]:
        save(out/name, public(value))
    for name in ('FROZEN.json', 'MECHANICAL_TESTS.json', 'PROFILE_ADMISSION.json'):
        if (root/name).exists(): save(out/name, public(read(root/name)))
    audit = dict(formal_planned=len(c['jobs']), formal_complete=sum(state['jobs'].get(j['id'])=='COMPLETE' for j in c['jobs']),
                 scored_jobs=len(done), scored_rows=len(rows), expected_scored_rows=1951*len(c['jobs']),
                 all_formal_workers_retired=not any(read(p).get('active') for p in (root/'processes').glob('online_*.json')),
                 online_target_label_reads=0, source_image_reads=0, all_attempts_charged=True)
    save(out/'COMPLETION_AUDIT.json', audit)
    lines = ['# R36: incremental modules on the retained W baseline', '',
             f"Status: **{state['status']}**. Fixed primary: W_TP. No controller training.", '',
             'W retains R20 six-view pseudo supervision, variance/boundary weighting, GraTa and LR multiplier 1.5. C is the strong plain-consistency control. TP is a simplified cross-threshold connected-component overlap weight, not a PH/OT TopoOT reproduction. LSO/LSM are independently implemented sigmoid structure-tensor losses using current detached soft pseudo-targets. BAL borrows frequency weighting for BCE; it is not original entropy-based DSBR.', '',
             'All seven conditions, three seeds and two orders were frozen before labels. Each full stream has 1,951 arrivals. No source retraining, external models, online labels, output ensemble or scientific retries. SEARCH and legacy REVIEW have historical exposure; REVIEW is descriptive, not independent confirmation. Patient linkage is UNKNOWN.', '',
             '| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |',
             '|---|---:|---:|']
    for candidate in c['candidates']:
        values = []
        for role in ('SEARCH', 'SEALED_REVIEW'):
            rs = [x['macro_Dice_percent'] for x in main if x['condition']==candidate['id'] and x['role']==role]
            values.append('NA' if not rs else f'{statistics.mean(rs):.4f}')
        lines.append(f"| {candidate['id']} | {values[0]} | {values[1]} |")
    lines += ['', 'PAIRED_SUMMARY includes both W and C comparisons. ALL_NEGATIVE_CELLS retains all adverse domain/channel/order/seed cells. A development signal requires SEARCH gain >=0.3pp against both W and C, both orders positive, >=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell >=-2pp. These are investment thresholds, not clinical or significance thresholds. Secondary candidates cannot retrospectively replace the primary.', '',
              f"GPU-worker {resources['gpu_worker_seconds']/3600:.3f}/48h; CPU-worker {resources['cpu_worker_seconds']/3600:.3f}h. Wall limit 24h. Failed preparation/profile/formal attempts remain recorded; incomplete cells are not success or zero.", '',
              'Dice uses 2TP/(prediction+GT), or 1 when both empty; equal domains and OD/OC, then equal seeds/orders. ASSD, soft Dice and Brier were not computed. Bootstrap couples repeated content across seeds/orders and does not resolve patient dependence. Novelty and medical benefit are unestablished.', '',
              'Only anonymous aggregate results, own implementation, settings, cost and audits are public. Data, masks, predictions, model states, private logs and third-party PDFs remain private. Execution completion and verified GitHub delivery are distinct.']
    if state.get('reason'): lines += ['', 'Stopped reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def supervise():
    c = base.config(); root = Path(c['output_root'])
    state = dict(status='PROFILING', T0=c['origin']['T0'], code_sha=c['code_sha'],
                 jobs={j['id']:'NOT_RUN' for j in c['jobs']}, delivery='PENDING_GITHUB')
    def persist(): save(root/'RUN_STATE.json', state); base.ledger(c, state)
    def batch(jobs, prefix, seconds, cpu=False):
        results = []
        width = 2 if cpu else len(c['gpu_assignments'])
        for offset in range(0, len(jobs), width):
            current = jobs[offset:offset+width]
            with concurrent.futures.ThreadPoolExecutor(width) as pool:
                fs = {}
                for i, j in enumerate(current):
                    if prefix=='online_': state['jobs'][j['id']]='RUNNING'
                    gpu = None if cpu else c['gpu_assignments'][i]
                    fs[pool.submit(base.run_task, c, prefix+j['id'], gpu, seconds, j)]=j
                persist()
                for f in concurrent.futures.as_completed(fs):
                    receipt = f.result(); results.append(receipt)
                    if prefix=='online_': state['jobs'][fs[f]['id']]=receipt['status']
                    persist()
            if any(x['status']!='COMPLETE' for x in results):
                raise RuntimeError('registered task failed; no automatic scientific retry')
        return results
    with lease(root/'supervisor', dict(experiment_id=ID, config_sha256=sha(c))):
        if (root/'execution_started.json').exists(): raise ValueError('already launched')
        save(root/'execution_started.json', dict(at=time.time()))
        persist()
        try:
            if read(root/'MECHANICAL_TESTS.json')['status']!='PASS': raise ValueError('mechanical tests absent')
            profiles = batch([dict(id=x['id'], candidate=x) for x in c['candidates']], 'profile_', 1200)
            estimates = {x['result']['candidate']: 1.2*(1951*(max(x['result']['seconds'])+.08)+x['result']['initialization_seconds']+30) for x in profiles}
            seconds = sum(estimates[j['candidate']['id']] for j in c['jobs'])
            workers = len(c['gpu_assignments'])
            admitted = base.amounts(c)[0]+seconds < c['origin']['gpu_worker_cap_seconds']-120 and time.time()+seconds/workers+max(estimates.values()) < c['origin']['online_deadline_epoch']-600
            save(root/'PROFILE_ADMISSION.json', dict(admitted=admitted, profiles=[x['result'] for x in profiles],
                                                    projected_GPU_seconds=seconds, estimates=estimates,
                                                    formal_jobs=len(c['jobs']), max_concurrent=workers))
            if not admitted: raise RuntimeError('full frozen matrix exceeds budget; no pruning or retuning')
            state['status']='RUNNING_FIXED_MATRIX'; persist()
            batch(c['jobs'], 'online_', max(1800, 3*max(estimates.values())))
            save(root/'stages/FINAL.barrier.json', dict(status='ALL_WORKERS_RETIRED', at=time.time()))
            save(root/'FINAL_JOBS.json', c['jobs'])
            if (root/'REVIEW_RELEASE.json').exists(): raise ValueError('review already released')
            save(root/'REVIEW_RELEASE.json', dict(frozen_sha256=sha(read(root/'FROZEN.json')), all_formal_workers_retired=True, at=time.time()))
            state['status']='SCORING'; persist()
            batch([dict(id=j['id'], job=j, stage='FINAL', final=True) for j in c['jobs']], 'score_final_', 7200, cpu=True)
            state.update(status='COMPLETE', ended=time.time())
        except BaseException as e:
            traceback.print_exc(); state.update(status='INCOMPLETE', reason=str(e), ended=time.time())
        finally:
            persist(); report(c, state)


def main():
    base.ID=ID; base.Host=Host; base.filesystem_guard=filesystem_guard; scoring.metrics=hard_metrics
    mode=os.environ['RUN_MODE']
    if mode=='worker':
        receipt=base.worker(); sys.exit(0 if receipt['status']=='COMPLETE' else 1)
    elif mode=='watch': base.watch()
    elif mode=='supervise': supervise()
    else: raise ValueError('unknown mode')


if __name__=='__main__': main()
