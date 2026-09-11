"""One frozen transfer attempt, canonical zero-update inference, CPU closeout."""
import copy,gc,json,os,subprocess,time
from pathlib import Path
import torch
from .b4_host import C0,PolypC,FundusC,make_C,logits,reference_polyp_step
from .b4_data import BUDGET,stream,register
from .b4_analysis import recompute
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
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/b4_frozen_c_transfer_v1.json';BASELINE='de492dc16dba58884b119fb1945e67c011939303'


class Budget(OldBudget):
    def check(self):
        if self.seconds()>=21600:raise RuntimeError('6 hour active cap')
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=256*1024**2:raise RuntimeError('256 MiB cap')


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
    budget=Budget(out);calls=Calls();ev={};first={};audited=False
    try:
        with deterministic_smoke_pair(),new_log(out/'smoke.calls.jsonl') as log:
            for task in ('fundus','polyp'):
                budget.check();state=torch.load(reg['tasks'][task]['checkpoint']['path'],map_location='cpu',weights_only=True);saved=[]
                for path in ('reference','new'):
                    seed_all(20260907);h=(FundusC('C',state,'cuda:0') if task=='fundus' else PolypC(state,'cuda:0')) if path=='reference' else make_C(task,state,'cuda:0')
                    if not audited:process_audit(out,'smoke');audited=True
                    calls.attach(h,True)
                    def logged(o,a,k):append(log,dict(task=task,path=path,completed_counts=calls.counts.copy()))
                    handle=h.base.register_step_post_hook(logged)
                    def original(m,x,output):
                        if task not in first:first[task]=(output[0] if isinstance(output,(tuple,list)) else output).detach().cpu().clone()
                    first_handle=h.model.register_forward_hook(original)
                    for i in range(2):
                        x=pixels(task,i)
                        if task=='polyp' and path=='reference':p=reference_polyp_step(h,x)
                        else:p,_=h.step(x)
                        snap=dict(logits=p.cpu(),state=capture(h))
                        if path=='reference':saved.append(snap)
                        else:
                            close(snap['state']['rng'],saved[i]['state']['rng'],exact=True);close(snap['logits'],saved[i]['logits']);close(snap['state'],saved[i]['state'])
                        del p,snap
                    ev[task+'_C']=dict(pair_match=True,BN_layers=len(h.names)//2,affine_scalars=sum(p.numel() for p in h.params),names=h.names)
                    first_handle.remove();handle.remove();calls.release();h.finish(state);del h;gc.collect()
                seed_all(20260907);h=C0(task,state,'cuda:0');calls.attach(h);outputs={}
                for i in (0,1,1,0):
                    p,m=h.step(pixels(task,i))
                    if i in outputs:close(p.cpu(),outputs[i])
                    else:outputs[i]=p.cpu()
                close(outputs[0],first[task]);h.finish(state);calls.release();ev[task+'_C0']=dict(order_independent=True,RNG_unchanged=True,matches_first_C_original=True);del h,p,outputs;gc.collect()
                seed_all(20260907);h=SourceOnlyHost(task,state,'cuda:0');calls.attach(h);x=pixels(task,0)
                p=h.step(x)
                with torch.no_grad():q=logits(h.model,model_input_from_pixels(x,task).to('cuda:0'))
                close(p,q);calls.release();ev[task+'_source_adapter']=dict(source_eval_match=True,shape=list(q.shape));del h,p,q,x,state,saved;gc.collect()
                print(json.dumps(dict(stage='smoke',task=task,status='PASS',physical=calls.counts)),flush=True)
        if calls.counts!=dict(forwards=76,backwards=8,base_adam=8,perturb=0,restore=0):raise ValueError('smoke physical budget')
        private_json(out/'smoke.completion.json',dict(status='B4_SMOKE_PASS',base_adam=8,physical=calls.counts,evidence=ev,gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')}))
    except Exception as e:
        private_json(out/'smoke.failure.json',dict(status='INCOMPLETE',reason=str(e),physical=calls.counts,gpu_seconds=budget.seconds(),exit_code=1));raise


def run(receipt,out,reg):
    budget=Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress={k:0 for k in BUDGET};where={};h=None;trainable={};audited=False
    private_json(out/'run.start.json',dict(status='B4_RUNNING',commit=receipt['commit']))
    try:
        for task,arm,order in (('fundus','C0',None),('polyp','C0',None),('polyp','C',0),('polyp','C',1)):
            seed_all(20260907);state=torch.load(reg['tasks'][task]['checkpoint']['path'],map_location='cpu',weights_only=True)
            h=C0(task,state,'cuda:0') if arm=='C0' else PolypC(state,'cuda:0');torch.cuda.reset_peak_memory_stats()
            if not audited:process_audit(out,'run');audited=True
            if arm=='C':
                entry=dict(names=h.names,layers=len(h.names)//2,parameters=sum(p.numel() for p in h.params),lr=1e-4,betas=[.9,.999])
                if task in trainable and entry!=trainable[task]:raise ValueError('trainable drift')
                trainable[task]=entry
            name=f'{task}_canonical_C0.jsonl' if arm=='C0' else f'{task}_{order}_C.jsonl'
            with new_log(out/name) as f:
                for i,row in enumerate(stream(reg,task,order or 0)):
                    where=dict(task=task,arm=arm,order=order,visit=i+1);budget.check();start=time.monotonic();x=read_pixels(row['image_path'],task,row['image_size'])
                    sync();t=time.monotonic();p,m=h.step(x);sync();seconds=time.monotonic()-t
                    for k,v in m['counts'].items():progress[k]+=v
                    probability=p.sigmoid();mask=read_mask(row['mask_path'],task,row['image_size']);metrics=evaluate(probability,mask,task)
                    record=dict(**where,**{k:row[k] for k in ('domain','subset','sample_id','group_id')},**m,metrics=metrics,prediction_origin='canonical_stateless' if arm=='C0' else 'continuous_update',prediction_fixed_before_label=True,frozen_parameters_checked=True,host_seconds=seconds,pipeline_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    append(f,record);progress['records']+=1;del p,probability,mask,x
                    if (i+1)%64==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
            h.finish(state);h=None;del state;gc.collect()
        if progress!=BUDGET:raise ValueError('B4 totals')
        check_registered_files(reg);private_json(out/'run.completion.json',dict(status='B4_RUN_COMPLETE',progress=progress,trainable=trainable,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='INCOMPLETE',reason=str(e),where=where,progress=progress,current_counts=h.counts if h else None,gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(r,stage):
    if r['authorization']!='direct_user_B4_FROZEN_C_TRANSFER':raise ValueError('new B4 authorization required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('B4 execution checkout')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old scientific code changed')
    if digest(CONFIG)!=r['config_sha256'] or r['reference_commit']!=REFERENCE_COMMIT or r['grata_commit']!=GRATA_COMMIT:raise ValueError('B4 code bindings')
    checkout_root();out=Path(r['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_mode&0o077 or out.stat().st_uid!=os.getuid():raise ValueError('private output')
    if digest(out/'registration.json')!=r['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if reg['formal_budget']!=BUDGET or json.loads(CONFIG.read_text())['formal_budget']!=BUDGET:raise ValueError('B4 frozen budget')
    cpu=json.loads((out/'CPU_validation.json').read_text())
    if cpu['exit_code']!=0 or cpu['tests']!=5 or cpu['commit']!=r['commit']:raise ValueError('B4 CPU acceptance gate')
    if stage!='recompute':
        official();devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if r['physical_gpu'] not in [3,4,5,6,7] or mapping[r['physical_gpu']]!=r['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:raise ValueError('B4 GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='B4_SMOKE_PASS' or s['base_adam']!=8 or digest(out/'smoke.completion.json')!=r['smoke_sha256'] or any(s[k]!=r[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('B4 smoke gate')
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
