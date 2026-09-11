"""One frozen transfer attempt, canonical zero-update inference, CPU closeout."""
import copy,gc,json,os,subprocess,time
from pathlib import Path
import torch
from .b4_host import PolypC,logits
from .b5_host import Host,H0,make_host,check_policy,SCORE_HEAD_BN
from .b5_data import BUDGET,SMOKE,stream,register
from .b5_analysis import recompute
from .b1_host import official,GRATA_COMMIT
from .source_pilot import seed_all,SourceOnlyHost
from .source_pilot_release import digest,check_registered_files,environment
from .source_io import read_pixels,read_mask
from .p1_analysis import evaluate
from .m1_run import new_log,Budget as OldBudget
from .host_diagnostic_run import private_json,append
from .host_diagnostic import rng,close
from .offline.adaptation_dd import cpu_tree
from .hosts.vptta import model_input_from_pixels
from .m2_run import deterministic_smoke_pair
from .p1_run import sync
from .b3_runtime import process_audit
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/b5_score_head_preservation_v1.json';BASELINE='8bbbc1f914223d5fdc7c9dbbf055323148b563f5'


class Budget(OldBudget):
    def check(self):
        if self.seconds()>=21600:raise RuntimeError('6 hour active cap')
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=512*1024**2:raise RuntimeError('512 MiB cap')


class Calls:
    def __init__(self):self.counts={k:0 for k in BUDGET if k!='records'};self.handles=[]
    def attach(self,h,update=False):
        def forward(m,x):self.counts['forwards']+=1
        self.handles.append(h.model.register_forward_pre_hook(forward))
        if update:
            def backward(g):self.counts['backwards']+=1;return g
            def adam(o,a,k):self.counts['base_adam']+=1
            self.handles.extend([h.params[0].register_hook(backward),h.base.register_step_post_hook(adam)])
    def release(self):
        for h in self.handles:h.remove()
        self.handles=[]


def capture(h):return dict(affine=[p.detach().cpu().clone() for p in h.params],adam=cpu_tree(copy.deepcopy(h.base.state_dict())),rng=rng())


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'));from test_vptta_host import pixels
    budget=Budget(out);calls=Calls();ev={}
    try:
        state=torch.load(reg['tasks']['polyp']['checkpoint']['path'],map_location='cpu',weights_only=True)
        with deterministic_smoke_pair(),new_log(out/'smoke.calls.jsonl') as log:
            saved=[]
            for path in ('old','new'):
                seed_all(20260907);h=PolypC(state,'cuda:0') if path=='old' else make_host('all-adapt',state,'cuda:0')
                if path=='old':process_audit(out,'smoke')
                calls.attach(h,True)
                for i in range(4):
                    budget.check();p,m=h.step(pixels('polyp',i));snap=dict(logits=p.cpu(),state=capture(h))
                    if path=='old':saved.append(snap)
                    else:close(snap,saved[i]);close(snap['state']['rng'],saved[i]['state']['rng'],exact=True)
                    append(log,dict(stage=path,visit=i+1,physical=calls.counts.copy()))
                calls.release();h.finish(state);del h,p,snap;gc.collect()
            ev['all_adapt']=dict(four_image_B4_match=True)
            inventories={}
            for arm in ('F','H'):
                seed_all(20260907);h=Host(arm,state,'cuda:0');calls.attach(h,True)
                for i in range(2):
                    budget.check();p,m=h.step(pixels('polyp',i))
                    if set(h.head_input_gradient)!=set(SCORE_HEAD_BN) or any(v<=0 for v in h.head_input_gradient.values()):raise ValueError('programmatic upstream gradient disconnected')
                    check_policy(h);append(log,dict(stage=arm,visit=i+1,physical=calls.counts.copy()))
                inventories[arm]=dict(names=h.names,BN_layers=len(h.names)//2,affine_scalars=sum(p.numel() for p in h.params),parameter_tensors=len(h.params))
                ev[arm]=dict(layer_policy=h.policy,frozen_heads_unchanged=True,upstream_gradient_connected=True)
                calls.release();h.finish(state);del h,p;gc.collect()
            if inventories['F']!=inventories['H']:raise ValueError('F/H optimizer inventory differs')
            ev['inventory']=inventories
            for i in range(2):
                seed_all(20260907);h=H0(state,'cuda:0');calls.attach(h);p,_=h.step(pixels('polyp',i));p=p.cpu();calls.release();h.finish(state);del h;gc.collect()
                seed_all(20260907);h=Host('H',state,'cuda:0');calls.attach(h)
                with torch.no_grad():q=logits(h.model,model_input_from_pixels(pixels('polyp',i),'polyp').to('cuda:0'))
                close(p,q.cpu());calls.release();h.finish(state);del h,p,q;gc.collect()
            ev['H0']=dict(matches_fresh_H_first_original=True,images=2)
        if calls.counts!=SMOKE:raise ValueError('smoke physical budget')
        private_json(out/'smoke.completion.json',dict(status='B5_SMOKE_PASS',base_adam=12,physical=calls.counts,evidence=ev,gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')}))
    except Exception as e:
        private_json(out/'smoke.failure.json',dict(status='INCOMPLETE',reason=str(e),physical=calls.counts,gpu_seconds=budget.seconds(),exit_code=1));raise


def run(receipt,out,reg):
    budget=Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress={k:0 for k in BUDGET};where={};h=None;trainable={};audited=False
    private_json(out/'run.start.json',dict(status='B5_RUNNING',commit=receipt['commit']))
    try:
        for task,arm,order in (('polyp','H0',None),('polyp','F',0),('polyp','H',0),('polyp','F',1),('polyp','H',1)):
            seed_all(20260907);state=torch.load(reg['tasks'][task]['checkpoint']['path'],map_location='cpu',weights_only=True)
            h=make_host(arm,state,'cuda:0');torch.cuda.reset_peak_memory_stats()
            if not audited:process_audit(out,'run');audited=True
            if arm!='H0':
                entry=dict(names=h.names,layers=len(h.names)//2,parameters=sum(p.numel() for p in h.params),lr=1e-4,betas=[.9,.999])
                if arm in trainable and entry!=trainable[arm]:raise ValueError('trainable drift')
                trainable[arm]=entry
            name=f'{task}_canonical_H0.jsonl' if arm=='H0' else f'{task}_{order}_{arm}.jsonl'
            with new_log(out/name) as f:
                for i,row in enumerate(stream(reg,order or 0)):
                    where=dict(task=task,arm=arm,order=order,visit=i+1);budget.check();start=time.monotonic();x=read_pixels(row['image_path'],task,row['image_size'])
                    sync();t=time.monotonic();p,m=h.step(x);sync();seconds=time.monotonic()-t
                    for k,v in m['counts'].items():progress[k]+=v
                    probability=p.sigmoid();mask=read_mask(row['mask_path'],task,row['image_size']);metrics=evaluate(probability,mask,task)
                    record=dict(**where,**{k:row[k] for k in ('domain','subset','sample_id','group_id')},**m,metrics=metrics,prediction_origin='canonical_stateless' if arm=='H0' else 'continuous_update',prediction_fixed_before_label=True,frozen_parameters_checked=True,host_seconds=seconds,pipeline_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    append(f,record);progress['records']+=1;del p,probability,mask,x
                    if (i+1)%64==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
            h.finish(state);h=None;del state;gc.collect()
        if progress!=BUDGET:raise ValueError('B5 totals')
        check_registered_files(reg);private_json(out/'run.completion.json',dict(status='B5_RUN_COMPLETE',progress=progress,trainable=trainable,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='INCOMPLETE',reason=str(e),where=where,progress=progress,current_counts=h.counts if h else None,gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(r,stage):
    if r['authorization']!='direct_user_B5_SCORE_HEAD_PRESERVATION':raise ValueError('new B5 authorization required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('B5 execution checkout')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old scientific code changed')
    if digest(CONFIG)!=r['config_sha256'] or r['reference_commit']!=REFERENCE_COMMIT or r['grata_commit']!=GRATA_COMMIT:raise ValueError('B5 code bindings')
    checkout_root();out=Path(r['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_mode&0o077 or out.stat().st_uid!=os.getuid():raise ValueError('private output')
    if digest(out/'registration.json')!=r['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if reg['formal_budget']!=BUDGET or json.loads(CONFIG.read_text())['formal_budget']!=BUDGET:raise ValueError('B5 frozen budget')
    cpu=json.loads((out/'CPU_validation.json').read_text())
    if cpu['exit_code']!=0 or cpu['tests']!=8 or cpu['commit']!=r['commit']:raise ValueError('B5 CPU acceptance gate')
    if stage!='recompute':
        official();devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if r['physical_gpu'] not in [3,4,5,6,7] or mapping[r['physical_gpu']]!=r['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:raise ValueError('B5 GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='B5_SMOKE_PASS' or s['base_adam']!=12 or digest(out/'smoke.completion.json')!=r['smoke_sha256'] or any(s[k]!=r[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('B5 smoke gate')
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
