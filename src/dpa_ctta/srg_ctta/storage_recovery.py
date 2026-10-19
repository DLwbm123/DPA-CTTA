"""Adopt monitoring of the registered live worker; never launch another GPU worker."""
import errno,os,sys,time,stat,select,threading,signal,subprocess,resource,json,ctypes,platform
from pathlib import Path
from .run import read,save

def _pidfd_syscall(number,*args):
    # The deployed interpreter lacks wrappers; kernel UAPI numbers are x86_64.
    if platform.machine()!='x86_64':raise OSError(errno.ENOTSUP,'pidfd syscall ABI not qualified')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(number),*args)
    if result<0:
        code=ctypes.get_errno();raise OSError(code,os.strerror(code))
    return result

def pidfd_open(pid):
    return os.pidfd_open(pid,0) if hasattr(os,'pidfd_open') else _pidfd_syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))

def pidfd_signal(fd,sig):
    if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(fd,sig)
    return _pidfd_syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(None),ctypes.c_uint(0))

def bytes_used(root):
    total=0
    def onerror(e):
        if e.errno!=errno.ENOENT:raise e
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
        for name in files:
            try:s=os.lstat(os.path.join(directory,name))
            except FileNotFoundError:continue
            if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total

def proc_record(pid,proc_root=Path('/proc')):
    try:text=(proc_root/str(pid)/'stat').read_text()
    except FileNotFoundError:return None
    fields=text.rsplit(')',1)[1].split()
    return dict(state=fields[0],start_ticks=int(fields[19]),wait_status=int(fields[49]),pt_fields_readable=bool(int(fields[23])))

def exit_code_from_record(value,expected_ticks):
    if value is None or value['start_ticks']!=expected_ticks or value['state'] not in ('Z','X','x'):return None
    return os.waitstatus_to_exitcode(value['wait_status'])

def completion_certificate(root):
    try:
        s=read(root/'RUN_STATE.json');q=read(root/'QUALIFICATION.json');f=read(root/'FULL_REPEATABILITY.json');m=read(root/'MECHANISM_AUDIT.json');a=read(root/'ACCESS_AUDIT.json');rows=read(root/'private/predictions.json')
        assert s['status']=='GPU_COMPLETE' and s['predictions']==384 and s['unique_inputs']==32 and s['optimizer_steps']==11880
        assert q['status']=='PASS' and f['all_arms_exact'] and f['final_model_exact'] and f['final_adam_exact']
        assert m['status']=='PASS' and m['total_optimizer_steps']==11880 and m['source_parameters_and_buffers_immutable'] and not m['bn_buffers_changed']
        assert a['evaluation_label_reads']==0 and a['prior_outcome_score_reads']==0
        assert len(rows)==384 and len({(v['case'],v['condition'],v['arm']) for v in rows})==384
        assert len(list((root/'predictions').glob('*.bits')))==384
        n=0
        with (root/'private/optimization_steps.jsonl').open() as inp:
            for n,line in enumerate(inp,1):assert json.loads(line)['total_step']==n
        assert n==11880
    except (AssertionError,KeyError,FileNotFoundError,ValueError):return False
    return True

def controller(c,repair):
    root=Path(c['output_root']);old=read(root/'CONTROLLER_IDENTITY.private.json');prior=proc_record(old['pid'])
    assert prior is None or prior['start_ticks']!=old['start_ticks']
    v=read(root/'processes/worker.json');pid=v['pid'];ticks=v['start_ticks'];initial=proc_record(pid)
    assert v['active'] and initial is not None and initial['start_ticks']==ticks and initial['state'] not in ('Z','X','x')
    assert initial['pt_fields_readable'] and not (root/'RESOURCE_LEDGER.json').exists() and not (root/'processes/score.json').exists()
    with (root/'RECOVERY_START_ONCE.json').open('x') as f:f.write('{"gpu_relaunch":false}')
    save(root/'ORIGINAL_WORKER_PROCESS.private.json',v)
    fd=pidfd_open(pid);assert proc_record(pid)['start_ticks']==ticks
    done=threading.Event();observed={};space_stopped=False
    hz=os.sysconf('SC_CLK_TCK');start_boot=ticks/hz;guardian_start=time.time()
    def watch_exit():
        try:
            poll=select.poll();poll.register(fd,select.POLLIN);poll.poll()
            observed.update(exit_code=None,pidfd_exit_observed=True,end_boot=time.clock_gettime(time.CLOCK_BOOTTIME))
            try:
                record=proc_record(pid);observed.update(exit_code=exit_code_from_record(record,ticks),proc_record_at_exit=record)
            except (OSError,ValueError,IndexError) as e:observed['exit_record_error_type']=type(e).__name__
        finally:done.set()
    threading.Thread(target=watch_exit,daemon=True).start()
    save(root/'RECOVERY_STATE.json',dict(status='MONITORING_EXISTING_WORKER',gpu_relaunch=False,pidfd_open=True,original_controller_exception='FileNotFoundError during recursive scan of a disappearing temporary directory',registered_storage_cap_bytes=c['output_cap_bytes']))
    while not done.wait(2):
        if bytes_used(root)>c['output_cap_bytes']:
            pidfd_signal(fd,signal.SIGTERM);space_stopped=True
            if not done.wait(10):pidfd_signal(fd,signal.SIGKILL)
            done.wait();break
    os.close(fd)
    if not observed.get('pidfd_exit_observed'):raise RuntimeError('pidfd watcher failed; no success inference')
    code=observed['exit_code'];lifetime=observed['end_boot']-start_boot
    # A readable pidfd proves exit, not the exit code after PID1 has reaped it.
    cert=completion_certificate(root) if code in (0,None) and not space_stopped else False
    v.update(active=False,exit_code=code,ended=time.time(),exit_code_observed=code is not None,pidfd_exit_observed=True,completion_certificate=cert)
    save(root/'processes/worker.json',v);save(root/'WORKER_EXIT_CAPTURE.private.json',observed)
    records=[dict(mode='worker',status='COMPLETE' if code==0 and cert else 'EXITED_CODE_UNOBSERVED' if code is None and cert else 'FAILED',exit_code=code,exit_code_observed=code is not None,gpu_worker_seconds=lifetime,cpu_worker_seconds=0.,storage_stop=space_stopped,lifetime_method='CLOCK_BOOTTIME minus original kernel start_ticks; one clock-tick start resolution',completion_certificate=cert)]
    save(root/'attempts/worker.json',records[0])
    if cert and code in (0,None):
        mode='score' if code==0 else 'score_reconstruction'
        if mode=='score_reconstruction':
            save(root/'RECONSTRUCTION_ADMISSION.json',dict(status='PASS',worker_exited=True,pidfd_exit_observed=True,os_exit_code=None,full_inference_repeatability_and_access_certificates=True,original_exit0_gate_restored=False,labels_previously_scored=False))
        env=dict(os.environ,DIAG_MODE='score' if mode=='score' else 'score_reconstruction',CUDA_VISIBLE_DEVICES='')
        entry=repair['original_entry'] if mode=='score' else repair['entry'];start=time.time()
        def limit_file():resource.setrlimit(resource.RLIMIT_FSIZE,(c['output_cap_bytes'],c['output_cap_bytes']))
        with (root/'logs'/f'{mode}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,entry],env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit_file)
            child_stat=proc_record(child.pid)
            process=dict(pid=child.pid,start_ticks=child_stat['start_ticks'] if child_stat else None,active=True,mode=mode)
            save(root/'processes'/f'{mode}.json',process);cpu_space=False
            while child.poll() is None:
                time.sleep(2)
                if bytes_used(root)>c['output_cap_bytes']:
                    child.terminate();cpu_space=True
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill()
                    break
            result=child.wait()
        process.update(active=False,exit_code=result,ended=time.time());save(root/'processes'/f'{mode}.json',process)
        record=dict(mode=mode,status='COMPLETE' if result==0 else 'FAILED',exit_code=result,cpu_worker_seconds=time.time()-start,gpu_worker_seconds=0.,storage_stop=cpu_space);records.append(record);save(root/'attempts'/f'{mode}.json',record)
    ledger=read(Path(c['storage_window_root'])/'STORAGE_LEDGER.json')
    for item in ledger['registered_remote_roots']:
        if item['root']==str(root):item.update(reserved_bytes=0,active=False,all_files_bytes=bytes_used(root))
    ledger['actual_registered_bytes']=sum(item.get('all_files_bytes',0) for item in ledger['registered_remote_roots'])
    assert ledger['actual_registered_bytes']+ledger['ancillary_reserve_bytes']<ledger['workingBytes']
    save(Path(c['storage_window_root'])/'STORAGE_LEDGER.json',ledger)
    complete=len(records)==2 and records[1]['exit_code']==0
    save(root/'RESOURCE_LEDGER.json',dict(attempts=records,all_workers_retired=True,no_gpu_retry=True,original_controller_failed=True,original_controller_exit_code_unobserved=True,recovery_source_sha=repair['source_sha'],gpu_worker_seconds=lifetime,cpu_worker_seconds=sum(x['cpu_worker_seconds'] for x in records),guardian_elapsed_seconds=time.time()-guardian_start,time_gpu_quotas_removed_by_user=True,storage_cap_bytes=c['output_cap_bytes'],original_protocol_exit0_gate_met=code==0,reconstruction_only=code is None and complete))
    save(root/'RECOVERY_STATE.json',dict(status='COMPLETE' if complete else 'FAILED',worker_exit_code=code,worker_exit_code_observed=code is not None,scorer_mode=records[-1]['mode'] if len(records)>1 else None,gpu_relaunch=False))
    return 0 if complete else 1

if __name__=='__main__':
    repair=read(os.environ['RECOVERY_CONFIG']);c=read(os.environ['DIAG_CONFIG'])
    if os.environ.get('DIAG_MODE')=='score_reconstruction':
        from .topology_reconstruction_score import score
        score(c)
    else:sys.exit(controller(c,repair))
