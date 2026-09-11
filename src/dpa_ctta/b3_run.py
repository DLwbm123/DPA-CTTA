"""Frozen algorithm dispatch, one paired smoke, ten finite trajectories."""
import copy,gc,json,os,subprocess,time
from pathlib import Path
import torch
from .b3_orders import ARMS,ORDERS,BUDGET,build_stream,check_balance
from .b3_entry import Entry,direct,capture
from .b3_analysis import old_records,evaluate,recompute
from .b3_runtime import process_audit
from .b1_host import official,GRATA_COMMIT
from .m1_run import Observed,new_log,Budget as OldBudget
from .source_pilot import seed_all
from .source_pilot_release import digest,check_registered_files,environment,source_unchanged
from .source_io import read_pixels,read_mask
from .host_diagnostic_run import private_json,append
from .host_diagnostic import rng,close
from .m2_run import deterministic_smoke_pair
from .p1_run import sync
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT
from .b1_run import metadata
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/b3_frozen_order_balanced_v1.json'
BASELINE='3a6418b9f83485aa2f304484c17bfa42dfb6e231'


class Budget(OldBudget):
    def check(self):
        if self.seconds()>=28800:raise RuntimeError('8 hour active cap')
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=1024**3:raise RuntimeError('1 GiB output cap')


def register(b2,out):
    b2=Path(b2);r=json.loads((b2/'receipt.run.json').read_text())
    if r['commit']!='35d9d26c5554a23d81713f610d6ae53d93f54774' or digest(b2/'registration.json')!=r['registration_sha256']:raise ValueError('B2 binding')
    if json.loads((b2/'verification.json').read_text())['status']!='B2_INTERVAL_CONSISTENCY_COMPLETE':raise ValueError('B2 incomplete')
    reg=json.loads((b2/'registration.json').read_text());reg.update(b2_directory=str(b2),formal_budget=BUDGET)
    reg['identities'] += [metadata(b2/n) for n in ('registration.json','receipt.run.json','verification.json','execution_audit.json','run.environment.json')]
    reg['identities'] += [metadata(p) for p in sorted(b2.glob('fundus_*.jsonl'))]
    reg['execution_provenance']={'old_order0_1':{'A':'d3ee6901379be293f47abd1687808caaf5b04266','C':'029a79340057b58b8074ccf86c7b5671b5fdbd1c','U_S_I':r['commit']},'new_order2_3':'bound by B3 execution receipt','N':'old P2 records rearranged by identity; zero new forwards'}
    streams={str(o):build_stream(reg,o) for o in ORDERS}
    private_json(out/'streams.json',{o:[dict(visit=i+1,**{k:v[k] for k in ('group_id','domain','subset','sample_id')}) for i,v in enumerate(rs)] for o,rs in streams.items()})
    reg['identities'].append(metadata(out/'streams.json'));reg['order_balance']=check_balance()
    check_registered_files(reg)
    for o in (0,1):old_records(reg,o)
    private_json(out/'registration.json',reg);private_json(out/'registration.completion.json',dict(status='B3_READY',groups=1951,order_balance=reg['order_balance'],formal_budget=BUDGET,old_records_validated=True,exit_code=0))
    return reg


def normalized_counts(h,arm,watch=None):
    if arm!='A':return h.counts.copy()
    c=watch.counts;return dict(forwards=c['model_forwards'],backwards=c['backward_calls'],base_adam=c['online_adam'],perturb=0,restore=0)


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    budget=Budget(out);state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    physical={k:0 for k in BUDGET if k!='records'};completed=physical.copy();calls=0;prompt_calls=0;h=None;watch=None;arm=None;evidence={};audit_done=False
    try:
        with deterministic_smoke_pair(),new_log(out/'smoke.calls.jsonl') as log:
            for arm in ARMS:
                saved=[];ev=[]
                for path in ('direct','entry'):
                    budget.check();seed_all(20260907)
                    if path=='direct':h=direct(arm,state,'cuda:0');watch=Observed(h) if arm=='A' else None
                    else:entry=Entry(arm,state,'cuda:0');h=entry.host;watch=entry.watch;entry.trainable()
                    if not audit_done:process_audit(out,'smoke');audit_done=True
                    opt=h.optimizer if arm=='A' else h.base
                    def counted(opt,args,kwargs):
                        nonlocal calls
                        calls+=1;append(log,dict(arm=arm,path=path,visit=i+1,base_adam=calls,current_counts=normalized_counts(h,arm,watch)))
                    handle=opt.register_step_post_hook(counted);torch.cuda.reset_peak_memory_stats()
                    for i in range(2):
                        x=pixels('fundus',i)
                        if path=='entry':z,m=entry.step(x)
                        elif arm=='A':z=h.step(x);watch.verify()
                        else:z,m=h.step(x)
                        snap=dict(logits=z.detach().cpu(),state=capture(h,arm),rng=rng(),counts=normalized_counts(h,arm,watch),native_counts=watch.counts.copy() if arm=='A' else None)
                        if path=='direct':saved.append(snap)
                        else:
                            close(snap['rng'],saved[i]['rng'],exact=True);close(snap['counts'],saved[i]['counts'],exact=True);close(snap['native_counts'],saved[i]['native_counts'],exact=True)
                            close(snap['logits'],saved[i]['logits']);close(snap['state'],saved[i]['state'])
                            ev.append(dict(visit=i+1,max_logits_difference=float((snap['logits']-saved[i]['logits']).abs().max()),counts=snap['counts'],state_RNG_match=True))
                        del z,snap
                    c=normalized_counts(h,arm,watch)
                    if arm=='A':prompt_calls+=watch.counts['prompt_forwards']
                    for k in physical:physical[k]+=c[k]
                    if arm=='A':source_unchanged(h,state);watch.release()
                    else:h.finish(state)
                    handle.remove();completed=physical.copy();h=None;watch=None
                    if path=='entry':del entry
                    del opt;gc.collect()
                evidence[arm]=dict(visits=ev,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                print(json.dumps(dict(stage='smoke',arm=arm,status='PASS',base_adam=calls)),flush=True)
        if physical!=dict(forwards=136,backwards=20,base_adam=20,perturb=0,restore=0) or calls!=20 or prompt_calls!=8:raise ValueError('smoke budget')
        private_json(out/'smoke.completion.json',dict(status='B3_SMOKE_PASS',base_adam=calls,physical=physical,evidence=evidence,A_prompt_forwards=prompt_calls,A_fft2_calls=prompt_calls,A_ifft2_calls=prompt_calls,FFT_accounting="one fft2 and ifft2 per observed pinned Prompt.forward",gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')}))
    except Exception as e:
        current=normalized_counts(h,arm,watch) if h is not None else {k:0 for k in completed}
        private_json(out/'smoke.failure.json',dict(status='INCOMPLETE',reason=str(e),base_adam=calls,physical={k:completed[k]+current[k] for k in completed},gpu_seconds=budget.seconds(),exit_code=1));raise


def run(receipt,out,reg):
    budget=Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress={k:0 for k in BUDGET};where={};entry=None;trainable={};audited=False
    state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    private_json(out/'run.start.json',dict(status='B3_RUNNING',commit=receipt['commit']))
    try:
        for order in (2,3):
            stream=build_stream(reg,order)
            for arm in ARMS:
                seed_all(20260907);entry=Entry(arm,state,'cuda:0');torch.cuda.reset_peak_memory_stats();scope=entry.trainable()
                if arm in trainable and scope!=trainable[arm]:raise ValueError('parameter registration drift')
                trainable[arm]=scope
                if not audited:process_audit(out,'run');audited=True
                with new_log(out/f'fundus_{order}_{arm}.jsonl') as f:
                    for i,row in enumerate(stream):
                        where=dict(order=order,arm=arm,visit=i+1);budget.check();start=time.monotonic();x=read_pixels(row['image_path'],'fundus',row['image_size'])
                        sync();hoststart=time.monotonic();logits,m=entry.step(x);sync();hostseconds=time.monotonic()-hoststart
                        for k,v in m['counts'].items():progress[k]+=v
                        p=logits.sigmoid();mask=read_mask(row['mask_path'],'fundus',row['image_size']);metrics=evaluate(p,mask,'fundus')
                        record=dict(task='fundus',**where,**{k:row[k] for k in ('domain','subset','sample_id','group_id')},**m,metrics=metrics,prediction_fixed_before_label=True,source_unchanged=arm=='A',frozen_parameters_checked=arm!='A',host_seconds=hostseconds,pipeline_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        append(f,record);progress['records']+=1;del logits,p,mask,x
                        if (i+1)%64==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
                entry.finish(state);entry=None;gc.collect()
        if progress!=BUDGET:raise ValueError('B3 counts')
        check_registered_files(reg)
        private_json(out/'run.completion.json',dict(status='B3_RUN_COMPLETE',progress=progress,trainable=trainable,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='INCOMPLETE',reason=str(e),where=where,progress=progress,current_counts=entry.counts if entry else None,gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(r,stage):
    if r['authorization']!='direct_user_B3_FROZEN_ORDER_BALANCED':raise ValueError('new B3 authorization required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('B3 execution checkout')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old scientific code changed')
    if digest(CONFIG)!=r['config_sha256'] or r['reference_commit']!=REFERENCE_COMMIT or r['grata_commit']!=GRATA_COMMIT:raise ValueError('B3 code bindings')
    checkout_root();out=Path(r['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_mode&0o077 or out.stat().st_uid!=os.getuid():raise ValueError('private output')
    if digest(out/'registration.json')!=r['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if reg['formal_budget']!=BUDGET or json.loads(CONFIG.read_text())['formal_budget']!=BUDGET:raise ValueError('B3 frozen budget')
    cpu=json.loads((out/'CPU_validation.json').read_text())
    if cpu['exit_code']!=0 or cpu['tests']!=5 or cpu['commit']!=r['commit']:raise ValueError('B3 CPU acceptance gate')
    if stage!='recompute':
        official();devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if r['physical_gpu'] not in [4,5,6,7] or mapping[r['physical_gpu']]!=r['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:raise ValueError('B3 GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='B3_SMOKE_PASS' or s['base_adam']!=20 or digest(out/'smoke.completion.json')!=r['smoke_sha256'] or any(s[k]!=r[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('B3 smoke gate')
    return out,reg


def main():
    from types import SimpleNamespace
    a=SimpleNamespace(stage=os.environ['RUN_STAGE'],receipt=Path(os.environ['RUN_RECEIPT']));os.umask(0o077)
    if a.stage=='smoke':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=json.loads(a.receipt.read_text());out=Path(r['output_directory'])
    try:
        out,reg=validate(r,a.stage)
        if a.stage!='recompute':
            env=environment();private_json(out/(a.stage+'.environment.json'),env)
            if env!=json.loads((Path(reg['b1_directory'])/'run.environment.json').read_text()):raise ValueError('B1 environment drift')
        if a.stage=='recompute':recompute(out,reg,r)
        else:{'smoke':smoke,'run':run}[a.stage](r,out,reg)
    except Exception as e:
        f=out/(a.stage+'.failure.json')
        if not f.exists():private_json(f,dict(status='INCOMPLETE',stage=a.stage,reason=str(e),exit_code=1))
        raise
