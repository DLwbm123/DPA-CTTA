"""Neutral CPU stages plus an independent bound on this audit's identities only."""
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path


def identity(pid):
    try:
        return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def used(root):
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += os.lstat(Path(base) / name).st_size
            except FileNotFoundError:
                pass
    return total


def terminate(pid, start):
    if identity(pid) != start:
        return
    fd = os.pidfd_open(pid)
    try:
        if identity(pid) == start:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            time.sleep(1)
            if identity(pid) == start:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)


def guard(c, root):
    rec = dict(pid=os.getpid(), start_ticks=identity(os.getpid()), active=True, exit_code=None)
    write(root / 'processes/guard.json', rec)
    while not (root / 'FINAL.json').exists():
        elapsed = time.clock_gettime(time.CLOCK_BOOTTIME) - c['origin_boottime']
        if elapsed >= 3540 or used(root) >= 1073741824:
            write(root / 'HARD_STOP.json', dict(elapsed=elapsed, bytes=used(root)))
            for p in sorted((root / 'processes').glob('*.json'), key=lambda p: p.name == 'controller.json'):
                if p.name == 'guard.json':
                    continue
                item = json.loads(p.read_text())
                if item['active']:
                    terminate(item['pid'], item['start_ticks'])
            if not (root / 'FINAL.json').exists():
                write(root / 'FINAL.json', dict(status='HARD_STOP', exit_code=None))
            break
        time.sleep(1)
    rec['active'] = False
    write(root / 'processes/guard.json', rec)


def controller(c, root):
    me = dict(pid=os.getpid(), start_ticks=identity(os.getpid()), active=True, exit_code=None)
    write(root / 'processes/controller.json', me)
    guard_env = dict(os.environ, AUDIT_MODE='guard')
    with (root / 'logs/guard.log').open('x') as log:
        g = subprocess.Popen([c['python'], c['entry']], env=guard_env, stdout=log, stderr=log, start_new_session=True)
    stages = []
    modes = ('score',) if c.get('completed_generation_preserved') else ('worker1', 'worker2', 'score') if c.get('completed_check_preserved') else ('check', 'worker1', 'worker2', 'score')
    for mode in modes:
        begin = time.time()
        remaining = 3530 - (time.clock_gettime(time.CLOCK_BOOTTIME) - c['origin_boottime'])
        if remaining <= 0:
            break
        env = dict(os.environ, AUDIT_MODE=mode, CUDA_VISIBLE_DEVICES=c['gpu_uuid'] if mode in ('worker1','worker2') else '', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        with (root / 'logs' / (mode + '.log')).open('x') as log:
            p = subprocess.Popen([c['python'], c['entry']], env=env, stdout=log, stderr=log)
            rec = dict(pid=p.pid, start_ticks=identity(p.pid), active=True, exit_code=None)
            write(root / 'processes' / (mode + '.json'), rec)
            args = subprocess.check_output(['ps', '-p', str(p.pid), '-o', 'args='], text=True).strip()
            assert args == c['python'] + ' ' + c['entry']
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                terminate(p.pid, rec['start_ticks'])
                rc = p.wait()
        rec.update(active=False, exit_code=rc, seconds=time.time() - begin)
        write(root / 'processes' / (mode + '.json'), rec)
        stages.append(dict(stage=mode, exit_code=rc, seconds=rec['seconds']))
        write(root / 'STATE.json', dict(stages=stages, completed_stages=len(stages)))
        if rc:
            break
    complete = len(stages) == len(modes) and all(v['exit_code'] == 0 for v in stages)
    write(root / 'public/RESOURCE_LEDGER.json', dict(stages=stages, all_stage_workers_retired=True,
          source_sha=c['source_sha'], previous_attempts=c.get('previous_attempts', []), planned_model_calls=128, planned_decoder_only_calls=0, planned_path_convolution_calls=256, planned_head_only_calls=384, elapsed_seconds=time.clock_gettime(time.CLOCK_BOOTTIME) - c['origin_boottime']))
    write(root / 'FINAL.json', dict(status='COMPLETE' if complete else 'FAILED', application_exit_code=0 if complete else 1))
    grc = g.wait(timeout=5)
    grec = json.loads((root / 'processes/guard.json').read_text())
    grec.update(active=False, exit_code=grc)
    write(root / 'processes/guard.json', grec)
    me['active'] = False
    write(root / 'processes/controller.json', me)


if __name__ == '__main__':
    root = Path(os.environ['AUDIT_ROOT'])
    c = json.loads((root / 'private/config.json').read_text())
    sys.path.insert(0, str(root / 'code'))
    mode = os.environ['AUDIT_MODE']
    if mode == 'guard':
        guard(c, root)
    elif mode == 'controller':
        controller(c, root)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == (c['gpu_uuid'] if mode in ('worker1','worker2') else '')
        import importlib
        importlib.import_module('pipeline').run(c)
