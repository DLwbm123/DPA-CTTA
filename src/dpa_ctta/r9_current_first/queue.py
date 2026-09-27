"""Finite serial dispatcher: separate GPU online and CPU score processes, no monitor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
from .protocol import graph,digest,RECOVERY_POLICY
from .admission import require_authorized
from .identity import verify_runtime
from .ledger import Ledger
from .storage import write_json,lease
from .runtime import read,disk


def trajectory_identity(resolved,config,lock):
    return dict(code_inventory=digest(config['source_inventory']),source_lock=lock['sha256'],
                backbone=config['bindings']['checkpoint_sha256'],preprocessing='PINNED_R8_FP32',
                source_job=resolved['source_job'],source_origin=resolved.get('source_origin'),artifact=resolved.get('artifact'),
                arm=resolved['arm'] if resolved['source_job'] is None else None,
                diagnostic=resolved['diagnostic'],alpha=resolved['alpha'],gradient=resolved['gradient'],
                seed_rng=resolved['seed'],order=resolved['order'],registration=config['bindings']['refs']['target']['sha256'],
                output_policy='FP32_SIGMOID_OD_OC',scoring='R9_HARD_SOFT_ASSD_V1')


def verify_attempt(record,node,attempt,identity,ledger_root):
    if (record.get('schema')!='R9_ATTEMPT_V1' or record.get('identity')!=identity or record.get('node')!=node or record.get('attempt')!=attempt):raise ValueError('worker receipt identity')
    if record['status'] not in ('COMPLETE','FAILED') or (record['failure'] is None)!=(record['status']=='COMPLETE'):raise ValueError('worker receipt status')
    state=read(Path(ledger_root)/'state.json');entry=state['attempts'][attempt]
    ledger=Ledger(ledger_root,identity,state['gpus'])
    # Worker writes its immutable receipt first. Reconcile that durable receipt
    # exactly once if either process died before ledger/queue settlement.
    ledger.observe(attempt,entry['token'],record['actual'],settle=True,failed=record['status']=='FAILED',allow_finished=True)
    if read(Path(ledger_root)/'state.json')['attempts'][attempt]['status']!=record['status']:raise ValueError('receipt/ledger status conflict')
    ledger.sync_disk()
    return record


def accept_receipt(root,state,node,record,config,ledger):
    """Single durable transition used by both normal wait and queue restart."""
    name=node['id'];attempt=record['attempt']
    if record['status']=='COMPLETE':
        state['nodes'][name]=dict(status='COMPLETE',attempt=attempt)
        if node['kind']=='score':
            from .resolution import resolve
            from .runtime import source_receipts
            r=resolve(node['job'],read(root/'recipe_selection.json'),source_receipts(root))
            key=digest(trajectory_identity(r,config,read(root/'source_lock.json')))
            state['reused'][key]=node['job']['id']
    else:
        f=record['failure'];retry=False
        if f['class']=='INFRASTRUCTURE':
            from .recovery import bound_failure
            f=bound_failure(record)
            chain=[a for a in ledger._read()['attempts'] if a.rsplit('.',1)[0]==name]
            if len(chain)==1:retry=ledger.claim_recovery(node.get('job',{}).get('id',name),attempt,digest(record))
        state['nodes'][name]=dict(status='RETRY' if retry else 'FAILED',attempt=attempt,failure=f)
        if f['class'] not in ('INFRASTRUCTURE','NUMERICAL'):
            write_json(root/'queue.json',state);raise RuntimeError('R9 stopped: '+f['reason'])
    write_json(root/'queue.json',state)


def await_worker(process,ledger,attempt,token,gpu,started):
    """Only the aggregate worker-time cap can stop a hung owned GPU process."""
    while True:
        try:return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if not gpu:continue
            state=read(ledger.root/'state.json');entry=state['attempts'][attempt]
            if entry['status']!='RUNNING':continue
            from .protocol import CAPS
            actual=dict(entry.get('observed',dict.fromkeys(CAPS,0)))
            actual['gpu_seconds']=time.monotonic()-started
            try:ledger.observe(attempt,token,actual,allow_finished=True)
            except RuntimeError:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise


def run(config,worker_entry,python):
    root=require_authorized(config);identity=verify_runtime(config)
    if any(x in str(python).lower()+' '+str(worker_entry).lower() for x in ('wangbomin','ctta','current_first','r9')):
        raise ValueError('neutral interpreter and worker entry required')
    root.mkdir(parents=True,exist_ok=True);(root/'queue').mkdir(exist_ok=True)
    nodes=graph()['nodes'];ledger=Ledger(root/'ledger',identity,[g['physical_id'] for g in config['gpu_assignments']])
    if not (root/'ledger'/'state.json').exists():ledger.create()
    statefile=root/'queue.json'
    state=read(statefile) if statefile.exists() else dict(identity=identity,config_sha256=digest(config),nodes={},reused={},recovery_policy=RECOVERY_POLICY)
    if state.get('recovery_policy')!=RECOVERY_POLICY or state['identity']!=identity or state['config_sha256']!=digest(config):raise ValueError('queue binding changed')
    with lease(root/'queue',identity):
        launchpath=root/'queue'/'launch.json'
        if launchpath.exists() and read(launchpath)!=config:raise ValueError('saved launch config changed')
        write_json(launchpath,config)
        # ponytail: serial jobs bound transient FP32 outputs to one LONG10 stream;
        # parallelize only after profiling concurrent disk/CPU pressure.
        while True:
            done={k for k,v in state['nodes'].items() if v['status']=='COMPLETE'}
            terminal=done|{k for k,v in state['nodes'].items() if v['status']=='FAILED'}
            eligible=[n for n in nodes if n['id'] not in terminal and set(n['needs'])<=done]
            if not eligible:break
            # Score and retire each output before issuing any next GPU target job.
            node=min(eligible,key=lambda n:(0 if n['kind']=='score' else 1,nodes.index(n)))
            name=node['id'];prior=state['nodes'].get(name)
            if prior and prior['status']=='RUNNING':
                receipt=root/'attempts'/f"{prior['attempt']}.json"
                if not receipt.exists():raise RuntimeError('previous worker has no final receipt; inspect process/failure evidence before resume')
                record=verify_attempt(read(receipt),name,prior['attempt'],identity,root/'ledger')
                accept_receipt(root,state,node,record,config,ledger);continue
            if node['kind']=='online':
                from .resolution import resolve
                from .runtime import source_receipts
                resolved=resolve(node['job'],read(root/'recipe_selection.json'),source_receipts(root))
                key=digest(trajectory_identity(resolved,config,read(root/'source_lock.json')))
                if key in state['reused']:
                    original=state['reused'][key];p=root/'target'/original
                    from .target import _digest
                    seal=read(p/'score_complete.json')
                    if seal['scalar_sha256']!=_digest(p/'scalars.private.jsonl'):raise ValueError('reuse score seal')
                    target=root/'target'/node['job']['id'];target.mkdir(parents=True,exist_ok=False)
                    write_json(target/'reuse.json',dict(schema='R9_EXACT_SLOT_REUSE_V1',original_slot=original,identity_sha256=key,original_score_receipt=digest(seal),new_model_forwards=0,new_backward_calls=0,new_optimizer_steps=0))
                    for suffix in ('online','score'):state['nodes'][node['job']['id']+'__'+suffix]=dict(status='COMPLETE',reuse=original)
                    write_json(statefile,state);continue
            assignment=config['gpu_assignments'][0] if node['resource']=='gpu' else None
            chain=[a for a in ledger._read()['attempts'] if a.rsplit('.',1)[0]==name]
            attempt=name+'.'+str(len(chain))
            if chain:
                claim=ledger._read()['recovery_claims'].get(node.get('job',{}).get('id',name))
                if len(chain)!=1 or prior['status']!='RETRY' or not claim or claim['attempt']!=prior['attempt']:raise ValueError('unclaimed or exhausted recovery')
            ledger.sync_disk()
            budget=config['profile']['node_budgets'][name].copy()
            if chain and node['kind']=='online':
                from .recovery import verify_failure
                from .target import PREDICTION_BYTES
                jobroot=root/'target'/node['job']['id']
                verify_failure(root,jobroot,name,identity,prior['failure'])
                prefix=(jobroot/'predictions.bits').stat().st_size
                maximum=(19510 if node['job']['order']=='LONG10' else 1951)*PREDICTION_BYTES
                if not 0<=prefix<=maximum or prefix>budget['disk_bytes']:raise ValueError('recovery probability occupancy')
                budget['disk_bytes']-=prefix
            from .protocol import CAPS
            if disk(root)+budget['disk_bytes']+8*1024**2>CAPS['disk_bytes']:
                ledger.stop('aggregate resource cap: projected occupied output bytes');raise RuntimeError('aggregate resource cap: output bytes')
            if assignment:
                from .assets import available_memory
                available_memory(assignment,config['profile']['peak_vram_bytes'][name])
            token=ledger.reserve(attempt,assignment['physical_id'] if assignment else None,budget)
            packet=dict(config_path=str(launchpath),node=node,assignment=assignment,attempt=attempt,token=token,failure=prior.get('failure') if prior else None)
            packetpath=root/'queue'/f'{attempt}.json';write_json(packetpath,packet)
            state['nodes'][name]=dict(status='RUNNING',attempt=attempt);write_json(statefile,state)
            env=os.environ.copy();env.update(R9_PACKET=str(packetpath),CUDA_VISIBLE_DEVICES=str(assignment['physical_id']) if assignment else '',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/'queue'/f'{attempt}.log').open('ab') as log:
                started=time.monotonic()
                process=subprocess.Popen([str(python),str(worker_entry)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['nodes'][name]['pid']=process.pid;write_json(statefile,state)
                code=await_worker(process,ledger,attempt,token,assignment is not None,started)
            receiptpath=root/'attempts'/f'{attempt}.json'
            if not receiptpath.exists():raise RuntimeError(f'worker exit {code} without receipt; retain reservation and inspect')
            receipt=verify_attempt(read(receiptpath),name,attempt,identity,root/'ledger')
            accept_receipt(root,state,node,receipt,config,ledger)
        state['status']='COMPLETE' if all(v['status']=='COMPLETE' for v in state['nodes'].values()) and len(state['nodes'])==len(nodes) else 'INCOMPLETE'
        state['blocked_nodes']=[n['id'] for n in nodes if n['id'] not in state['nodes']]
        write_json(statefile,state);return state


def worker_main():
    from .runtime import worker
    packet=read(os.environ['R9_PACKET']);packet['config']=read(packet.pop('config_path'));receipt=worker(**packet)
    return 0 if receipt['status']=='COMPLETE' else 1
