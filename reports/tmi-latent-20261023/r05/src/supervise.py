"""Bound both seeds, checks and evaluation with the same durable clock."""
import os, json, time, signal, subprocess
from pathlib import Path

R = Path(os.environ['WORK_ROOT'])
c = json.loads((R / 'config.private.json').read_text())


def write(n, x):
    p = R / n; t = p.with_suffix('.tmp')
    t.write_text(json.dumps(x, indent=2)); t.replace(p)


def size():
    total = 0
    for p in R.rglob('*'):
        try:
            if p.is_file(): total += p.stat().st_blocks * 512
        except FileNotFoundError:
            pass
    return total


receipt = R / 'RUNTIME_RECEIPT.json'
old = json.loads(receipt.read_text()) if receipt.exists() else None
origin = old['origin_unix'] if old else time.time()
deadline = origin + c['hard_seconds']
if (R / 'TRAINING_FINISHED.json').exists(): raise RuntimeError('Refuse completed run')
if (R / 'PROCESS.json').exists():
    previous = json.loads((R / 'PROCESS.json').read_text())
    # PID alone is not sufficient identity evidence.
    proc = Path('/proc') / str(previous['controller_pid'])
    if proc.exists() and proc.joinpath('stat').read_text().split()[21] == previous.get('controller_start_ticks'):
        raise RuntimeError('Original controller still alive')
remaining = int(deadline - c['stop_margin'] - time.time())
if remaining < 60: raise RuntimeError('Original budget exhausted')
write('RUNTIME_RECEIPT.json', {'origin_unix': origin, 'deadline_unix': deadline, 'hard_seconds': c['hard_seconds'],
      'soft_stop_bytes': c['storage_soft'], 'hard_storage_bytes': c['storage_hard'], 'save_margin_seconds': c['stop_margin'],
      'baseline_bytes': old['baseline_bytes'] if old else size(), 'expected_peak_bytes': 10000000, 'retries_reset_clock': False})
write('STATE.json', {'status': 'RUNNING', 'origin_unix': origin, 'deadline_unix': deadline})
with (R / 'train.log').open('a') as out:
    p = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=60s', str(remaining) + 's', c['python'], c['neutral_train']],
                         env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    write('PROCESS.json', {'worker_pid': p.pid, 'controller_pid': os.getpid(),
          'controller_start_ticks': Path('/proc/self/stat').read_text().split()[21],
          'worker_start_ticks': (Path('/proc') / str(p.pid) / 'stat').read_text().split()[21],
          'controller_argv': [c['python'], c['neutral_supervisor']], 'worker_entry': c['neutral_train']})
    reason = None
    while p.poll() is None:
        if time.time() >= deadline - c['stop_margin']: reason = 'TIME_SOFT_STOP'
        if size() >= c['storage_soft']: reason = 'STORAGE_SOFT_STOP'
        if reason:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            break
        time.sleep(5)
    write('EXIT.json', {'code': p.returncode, 'reason': reason, 'seconds': time.time() - origin,
                       'allocated_bytes': size(), 'ended_unix': time.time()})
    if p.returncode == 0 and (R / 'TRAINING_FINISHED.json').exists():
        state = json.loads((R / 'TRAINING_FINISHED.json').read_text())['status']
    else:
        state = 'FAILED' if reason is None else 'STOPPED'
    write('STATE.json', {'status': state, 'origin_unix': origin, 'deadline_unix': deadline,
                        'exit_code': p.returncode, 'stop_reason': reason, 'analysis_pending': True})
