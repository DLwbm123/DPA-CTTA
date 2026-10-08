"""Apply the user's later GPU-budget amendment only to newly started workers."""
import copy
import json
import os
from pathlib import Path
import runpy
import time
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import save

AMENDMENT = '20261009_NO_CUMULATIVE_GPU_CAP'


class AmendedGuard(base.Guard):
    def __init__(self, c, phase):
        if c['experiment_id'] != 'R30_DELAY_AND_STATE_HISTORY':
            raise ValueError('budget amendment is R30-specific')
        effective = dict(c, origin=dict(c['origin'], gpu_worker_cap_seconds=None))
        super().__init__(effective, phase)
        save(self.root/'private'/f'{phase}-budget.json', dict(amendment=AMENDMENT, pid=os.getpid(), at=time.time(),
             original_GPU_cap_seconds=c['origin']['gpu_worker_cap_seconds'], effective_GPU_cap_seconds=None,
             unchanged_T0=c['origin']['T0_epoch'], task_deadline_retained=True))


def main():
    if os.environ['ENTRY_MODULE'] != 'dpa_ctta.r30_state_history.run':
        raise ValueError('unexpected entry module')
    if os.environ.get('RUN_MODE') == 'budget_check':
        c = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
        original = copy.deepcopy(c)
        guard = AmendedGuard(c, 'amendment_check')
        assert guard.c['origin']['gpu_worker_cap_seconds'] is None and c == original
        assert guard.c['origin']['T0_epoch'] == c['origin']['T0_epoch']
        assert guard.c['origin']['absolute_deadline_epoch'] == c['origin']['absolute_deadline_epoch']
        print('PASS: effective GPU cap removed; original config, T0 and safety deadlines preserved')
        return
    if os.environ['RUN_MODE'] == 'worker':
        base.Guard = AmendedGuard
    runpy.run_module(os.environ['ENTRY_MODULE'], run_name='__main__')


if __name__ == '__main__':
    main()
