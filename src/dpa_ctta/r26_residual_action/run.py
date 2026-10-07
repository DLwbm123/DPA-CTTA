"""A fixed six-arm follow-up; reuse the sealed journal, scorer and paired reports."""
import concurrent.futures
import os
from pathlib import Path
import sys
import time
import torch
from .method import Host, ARMS
from ..r25_parameter_policy import run as prior
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import read, save, sha

ID = 'R26_RESIDUAL_ACTION'
SEEDS = [20260907, 17011, 29009]
CONTRASTS = [('PG_RETAIN', 'C'), ('PG_RETAIN', 'ANCHOR'), ('PG_RETAIN', 'REPEAT5'),
             ('GREEDY', 'RANDOM4'), ('PG_RETAIN', 'GREEDY')]


original_report = prior.report


def report(c, state):
    original_report(c, state)
    path = Path(c['output_root'])/'public/REPORT.md'
    text = path.read_text().replace('# R25: conditional adapter parameter exploration',
                                    '# R26: bounded residual-space action exploration')
    text = text.replace('Hard cap: 48h from registered T0; online cap 46h.',
                        'No cumulative GPU-worker cap. Per-task deadlines, a seven-day supervisory safety timeout and 64 GiB storage guard remain.')
    text += ('\nR26 is a development follow-up selected after inspecting R25. Both SEARCH and '
             'the legacy SEALED_REVIEW field are development-exposed, not independent validation. '
             'The unchanged arm labels refer to residual-output actions in this round, not R25 gate-logit actions. '
             'The fixed six-arm, three-seed, two-order matrix has 36 trajectories; no extra seeds or performance pruning. '
             'PG_RETAIN remains primary even if another arm performs better.\n')
    path.write_text(text)


class Campaign(prior.Campaign):
    def fits(self, jobs, reserve=0):
        # GPU cost is recorded without a cumulative cap; the finite matrix is fixed.
        estimates = [self.estimate(j['candidate']) for j in jobs]
        wall = max(estimates, default=0) + sum(estimates)/len(self.c['gpu_assignments'])
        return time.time()+wall < self.c['origin']['online_deadline_epoch']-600

    def execute(self):
        self.qualify()
        planned = self.matrix(list(ARMS), SEEDS)
        freeze = dict(primary='PG_RETAIN', arms=list(ARMS), seeds=SEEDS,
            contrasts=CONTRASTS, code_sha=self.c['code_sha'], frozen_at=time.time(),
            registered_jobs=36, label_based_selection=False, development_exposed=True,
            review_historically_exposed=True, action_space='bounded adapter residual output',
            gpu_worker_cap_seconds=None, residual_multiplier='exp(0.75*tanh(action))',
            allocation='fixed three seeds and two orders; no expansion or performance pruning')
        save(self.root/'FROZEN.json', freeze)
        self.state['frozen'] = freeze; self.save()
        actual = self.run_pairs('FORMAL', planned, required=True)
        if len(actual) != len(planned):
            raise RuntimeError('incomplete registered matrix; preserve all attempts')
        if any(read(p).get('active') for p in (self.root/'processes').glob('online_*.json')):
            raise ValueError('online workers remain at label barrier')
        save(self.root/'stages/FINAL.barrier.json', dict(status='ALL_WORKERS_RETIRED', at=time.time()))
        save(self.root/'FINAL_JOBS.json', self.jobs)
        release = self.root/'REVIEW_RELEASE.json'
        if release.exists():
            raise ValueError('duplicate label release')
        save(release, dict(frozen_sha256=sha(freeze), all_formal_workers_retired=True,
                          at=time.time(), selection_closed=True, development_exposed=True))
        self.state.update(status='FINAL_SCORING', review_opened=True); self.save()
        self.score('FINAL', self.jobs, final=True)
        self.state.update(status='COMPLETE', execution_finished=True, ended=time.time(), delivery='PENDING_GITHUB')
        self.save(); report(self.c, self.state)


def main():
    # Rebind the existing runner within this new process only; R25 artifacts stay intact.
    prior.ID = ID; prior.ARMS = ARMS; prior.CONTRASTS = CONTRASTS
    prior.Campaign = Campaign
    prior.report = report
    base.ID = ID; base.Host = Host
    prior.scoring.metrics = prior.hard_metrics
    torch.set_num_threads(2)
    mode = os.environ['RUN_MODE']
    if mode == 'worker':
        if os.environ['RUN_PHASE'].startswith('score_'):
            prior.scoring.metrics = prior.diagnostic_metrics(base.config())
        result = base.worker(); sys.exit(0 if result['status']=='COMPLETE' else 1)
    elif mode == 'watch': base.watch()
    elif mode == 'supervise': prior.supervise()
    else: raise ValueError('unregistered mode')


if __name__ == '__main__': main()
