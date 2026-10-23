"""Local-only deadline guardian plus independent native timeout fallback.

Stage this file, job.json and the probe on local /tmp before invocation.
The scientific worker and its descendants must stay in the timeout process group.
"""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

R = Path(os.environ['CONTROL_ROOT'])
# A flat, private, non-symlink local spool prevents accidental NFS critical-path I/O.
if R.parent != Path('/tmp') or not re.fullmatch(r'[A-Za-z0-9_-]+', R.name) or R.is_symlink():
    raise RuntimeError('Control spool must be a direct non-symlink /tmp child')
mounts = []
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    a, b = line.split(' - ', 1); mounts.append((a.split()[4], b.split()[0]))
kind = max((m for m in mounts if str(R) == m[0] or str(R).startswith(m[0].rstrip('/') + '/')), key=lambda m: len(m[0]))[1]
if kind not in {'tmpfs', 'ext4', 'xfs', 'overlay', 'btrfs'}:
    raise RuntimeError('Control spool is not on a supported local filesystem')
if R.stat().st_uid != os.getuid() or R.stat().st_mode & 0o077:
    raise RuntimeError('Control spool must be private to the current uid')
C = json.loads((R/'job.json').read_text())


def write(name, value):
    p = R/name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(p)


def identity(pid):
    try:
        s = Path('/proc', str(pid), 'stat').read_text()
        fields = s[s.rfind(')')+2:].split()
        return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_active(pgid):
    # procfs only; never inspect work files or wait synchronously for a child.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            s = (p/'stat').read_text(); fields = s[s.rfind(')')+2:].split()
            if int(fields[2]) == pgid and fields[0] not in {'Z','X'}: return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def send(pgid, sig):
    try: os.killpg(pgid, sig)
    except ProcessLookupError: pass


def term_members(pgid):
    # Do not signal GNU timeout itself: it owns the time-based TERM window.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == pgid: continue
        try:
            text = (p/'stat').read_text(); fields = text[text.rfind(')')+2:].split()
            if int(fields[2]) == pgid and fields[0] not in {'Z','X'}:
                os.kill(int(p.name),signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError): pass


def spool_bytes():
    return sum(p.lstat().st_blocks*512 for p in R.iterdir() if p.is_file())


def neutral(argv):
    return (len(argv) == 4 and argv[:3] == ['/usr/bin/python3','-I','-S'] and
            re.fullmatch(r'/tmp/e[0-9]+\.py', argv[-1]))


def main():
    if (R/'LAUNCH.json').exists() or (R/'EXIT.json').exists():
        write('REFUSED.json', {'reason':'CONTROL_SPOOL_ALREADY_USED', 'worker_started':False})
        return 2
    assert C['origin_unix'] <= time.time()+.001, 'Origin cannot be in the future'
    assert neutral(C['worker_argv']) and neutral(C['probe_argv']), 'Neutral entries required'
    assert 0 < C['kill_reserve'] < C['stop_margin'] < C['hard_seconds']
    assert 0 < C['probe_timeout'] < C['stop_margin']
    assert C['storage_hard'] - C['storage_soft'] > C['stdout_cap'] + 65536
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (R/'CLOCK.json').exists():
        clock = json.loads((R/'CLOCK.json').read_text())
        assert clock['boot_id'] == boot and clock['origin_unix'] == C['origin_unix'], 'Clock cannot reset'
        deadline = clock['deadline_monotonic']
    else:
        deadline = time.monotonic() + C['origin_unix'] + C['hard_seconds'] - time.time()
        write('CLOCK.json', {'origin_unix':C['origin_unix'], 'deadline_unix':C['origin_unix']+C['hard_seconds'],
                            'deadline_monotonic':deadline, 'boot_id':boot})
    soft = deadline - C['stop_margin']; kill_at = deadline - C['kill_reserve']
    worker = None; probe = None; probe_started = 0.; next_probe = 0.
    error = None
    reason = None; term_sent = False; kill_sent = False; captured = 0
    stop_signal = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_signal.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop_signal.__setitem__(0, True))
    events = []
    log = (R/'stdout.log').open('wb')
    try:
        while True:
            now = time.monotonic()
            if worker is not None: worker.poll()
            if probe is not None: probe.poll()
            if now >= deadline:
                if reason is None: reason = 'ABSOLUTE_DEADLINE'
                if worker: send(worker.pid, signal.SIGKILL)
                if probe: send(probe.pid, signal.SIGKILL)
                break
            if now >= soft and reason is None: reason = 'TIME_SOFT_STOP'
            if stop_signal[0] and reason is None: reason = 'GUARDIAN_STOP_SIGNAL'
            if reason and worker is None:
                if probe: send(probe.pid, signal.SIGKILL)
                break
            if probe is not None:
                if probe.returncode is not None:
                    if not reason:
                        if probe.returncode != 0:
                            reason = ('STORAGE_PROBE_TIMEOUT' if probe.returncode == 124 or
                                      now-probe_started >= C['probe_timeout'] else 'STORAGE_PROBE_FAILED')
                        else:
                            try:
                                data = json.loads(os.read(probe.stdout.fileno(), 4096))
                                remote_used = int(data['allocated_bytes'])
                                if remote_used < 0: raise ValueError('Negative allocation')
                                used = remote_used + spool_bytes() + 65536
                                if used >= C['storage_soft']: reason = 'STORAGE_SOFT_STOP'
                                elif worker is None and used+C['expected_peak_bytes'] >= C['storage_hard']: reason = 'INSUFFICIENT_STORAGE_RESERVE'
                                elif worker is None:
                                    # Recompute the remaining timeout AFTER all preflight I/O.
                                    left = soft - time.monotonic()
                                    if left <= 0: reason = 'PREFLIGHT_BUDGET_EXHAUSTED'
                                    else:
                                        env = os.environ.copy(); env.update(C.get('worker_env', {}))
                                        cmd = ['/usr/bin/timeout','--signal=TERM',
                                               '--kill-after=%.6fs' % (kill_at-soft), '%.6fs' % left, *C['worker_argv']]
                                        worker = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, start_new_session=True, cwd='/tmp')
                                        os.set_blocking(worker.stdout.fileno(), False)
                                        write('LAUNCH.json', {'wrapper':identity(worker.pid), 'launched_monotonic':time.monotonic(),
                                             'soft_monotonic':soft, 'kill_monotonic':kill_at, 'deadline_monotonic':deadline,
                                             'native_timeout_argv':cmd})
                            except (ValueError, KeyError, TypeError, BlockingIOError): reason = 'INVALID_STORAGE_RECEIPT'
                    probe.stdout.close(); probe = None; next_probe = now + C['probe_interval']
                elif now-probe_started >= C['probe_timeout']:
                    send(probe.pid, signal.SIGKILL)
                    if reason is None: reason = 'STORAGE_PROBE_TIMEOUT'
            if worker is not None:
                try: chunk = os.read(worker.stdout.fileno(), 65536)
                except BlockingIOError: chunk = b''
                if chunk and captured < C['stdout_cap']:
                    chunk = chunk[:C['stdout_cap']-captured]; log.write(chunk); log.flush(); captured += len(chunk)
                if worker.returncode is not None:
                    if not group_active(worker.pid): break
                    if reason is None: reason = 'CHILDREN_REMAIN'
                if reason and not term_sent:
                    events.append({'signal':'NATIVE_TERM_WINDOW' if reason == 'TIME_SOFT_STOP' else 'TERM_MEMBERS', 'monotonic':now,'reason':reason})
                    write('SIGNALS.json',events)
                    if reason != 'TIME_SOFT_STOP': term_members(worker.pid)
                    term_sent=True
                    if reason != 'TIME_SOFT_STOP': kill_at = min(kill_at, now+C['failure_grace'])
                if (now >= kill_at or (term_sent and now >= kill_at)) and not kill_sent:
                    events.append({'signal':'KILL','monotonic':now,'reason':reason or 'ABSOLUTE_KILL_DEADLINE'})
                    write('SIGNALS.json',events); send(worker.pid,signal.SIGKILL); kill_sent=True
            if probe is None and not reason and now >= next_probe:
                probe_cmd = ['/usr/bin/timeout','--signal=TERM','--kill-after=.1s',
                             '%.6fs' % C['probe_timeout'], *C['probe_argv']]
                probe = subprocess.Popen(probe_cmd, env={**os.environ,'WORK_ROOT':C['work_root']},
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, cwd='/tmp')
                os.set_blocking(probe.stdout.fileno(), False)
                probe_started = now
            time.sleep(.02)
    except BaseException as exc:
        reason = reason or 'GUARDIAN_ERROR'
        error = {'type':type(exc).__name__,'message':str(exc)}
    finally:
        if probe:
            send(probe.pid,signal.SIGKILL)
            try: probe.wait(timeout=.1)
            except subprocess.TimeoutExpired: pass
        if worker:
            worker.poll()
            if group_active(worker.pid): send(worker.pid,signal.SIGKILL)
        log.close()
    now = time.monotonic()
    alive = worker is not None and group_active(worker.pid)
    probe_retired = probe is None or probe.poll() is not None
    write('EXIT.json', {'error':error, 'reason':reason or ('NORMAL_EXIT' if worker and worker.returncode == 0 else 'WORKER_EXIT'),
          'wrapper_exit_code':None if worker is None else worker.poll(), 'worker_started':worker is not None,
          'termination_confirmed':not alive, 'probe_termination_confirmed':probe_retired, 'observed_monotonic':now, 'deadline_monotonic':deadline,
          'within_deadline':not alive and probe_retired and now<=deadline, 'stdout_saved_bytes':captured,
          'stdout_may_be_truncated':captured>=C['stdout_cap'], 'scientific_evidence':False})
    return 0 if not alive and worker is not None and worker.returncode == 0 and not reason else 1


if __name__ == '__main__':
    sys.exit(main())
