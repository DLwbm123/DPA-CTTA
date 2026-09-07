"""Finite H1 receipt-bound execution. Errors preserve prefixes; no resume or retry."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .host_diagnostic import DiagnosticPair, POSITIONS, BRANCHES, metrics
from .host_diagnostic_analysis import distribution, paired, channels
from .source_io import read_pixels, read_mask, source_proxy
from .source_pilot import DEFAULT_CONFIG, expected_visits, validate_arm
from .source_pilot_release import CONFIG_SHA, digest, check_registered_files, environment
from .integrations.ctta_suite import REFERENCE_COMMIT, checkout_root

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/host_mechanism_diagnostic_v1.json'
AUTHORIZATION='direct_user_H1_HOST_MECHANISM_DIAGNOSTIC_gpu7_only'
PILOT_COMMIT='1f8f1fab8d7858b89ed2f238d4c48660f48a0265'


def private_json(path,value):
    data=json.dumps(value,allow_nan=False,indent=2)+'\n'
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def append(f,value):
    data=json.dumps(value,allow_nan=False)+'\n'
    if f.tell()+len(data.encode())>32*1024**2: raise ValueError('JSONL budget exceeded')
    f.write(data);f.flush();os.fsync(f.fileno())


def validate_receipt(r,stage):
    if r.get('authorization')!=AUTHORIZATION or r.get('stage')!=stage:
        raise ValueError('UNAUTHORIZED_H1')
    if r.get('physical_gpu')!=7 or r.get('reference_commit')!=REFERENCE_COMMIT or r.get('tasks')!=['fundus','polyp']:
        raise ValueError('RECEIPT_SCOPE_MISMATCH')
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual!=r.get('commit') or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('EXECUTION_COMMIT_DIRTY_OR_MISMATCH')
    if digest(CONFIG)!=r.get('diagnostic_config_sha256') or digest(DEFAULT_CONFIG)!=CONFIG_SHA:
        raise ValueError('CONFIG_MISMATCH')
    checkout_root()
    old=Path(r['pilot_directory']);out=Path(r['output_directory'])
    if not out.is_absolute() or out.resolve().is_relative_to(ROOT) or out.resolve()==old.resolve(): raise ValueError('private output location mismatch')
    if out.stat().st_uid!=os.getuid() or out.stat().st_mode & 0o077: raise ValueError('output must be owned private 0700')
    old_receipt=json.loads((old/'receipt.pilot.private.json').read_text())
    if old_receipt['commit']!=PILOT_COMMIT or old_receipt['config_sha256']!=CONFIG_SHA: raise ValueError('wrong prior run')
    if digest(old/'registration.json')!=r.get('registration_sha256') or r['registration_sha256']!=old_receipt['registration_sha256']:
        raise ValueError('REGISTRATION_MISMATCH')
    if digest(old_receipt['source_spec_path'])!=old_receipt['source_spec_sha256']: raise ValueError('prior source specification drift')
    if json.loads((old/'pilot.completion.json').read_text())['status']!='SOURCE_PILOT_COMPLETE': raise ValueError('prior pilot incomplete')
    if stage in ('smoke','formal'):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']: raise ValueError('GPU_UUID_ENV_MISMATCH')
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        devices={int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.splitlines()}
        if devices.get(7)!=r['gpu_uuid']: raise ValueError('PHYSICAL_GPU_UUID_MISMATCH')
    if stage=='formal':
        if digest(out/'smoke.completion.json')!=r.get('smoke_evidence_sha256'): raise ValueError('SMOKE_EVIDENCE_MISMATCH')
        smoke=json.loads((out/'smoke.completion.json').read_text())
        if smoke['status']!='H1_SMOKE_PASS' or smoke['commit']!=actual or smoke['registration_sha256']!=r['registration_sha256'] or smoke['gpu_uuid']!=r['gpu_uuid'] or smoke['diagnostic_config_sha256']!=r['diagnostic_config_sha256']:
            raise ValueError('SMOKE_BINDING_MISMATCH')
        if smoke['counts']['reference_steps']+smoke['counts']['watched_steps']!=68 or smoke['counts']['counterfactual_steps']!=12:
            raise ValueError('SMOKE_BUDGET_MISMATCH')
    return old,out


def validate_records(rows,expected,task,positions):
    if len(rows)!=len(expected) or not rows: raise ValueError('diagnostic coverage mismatch')
    for row,wanted in zip(rows,expected):
        if any(row.get(k)!=v for k,v in wanted.items()): raise ValueError('diagnostic order/identity mismatch')
        v=row['visit'];cf=v in positions
        if type(row['retrieval_observed']) is not bool or not 1<=row['memory_size']<=min(41,v):
            raise ValueError('invalid observed retrieval/memory')
        for key in ('reference_steps','watched_steps','reference_pushes','watched_pushes'):
            if row['counts'][key]!=v: raise ValueError('observed native count mismatch')
        if row['counts']['counterfactual_steps']!=6*sum(p<=v for p in positions):
            raise ValueError('observed branch count mismatch')
        if row['adam_step']!=v or row['counter']!=[v] or row['equivalence_pass'] is not True:
            raise ValueError('diagnostic lifecycle/equivalence mismatch')
        if set(row['predictions'])!={'N','G','I','U'} or set(row['branches'])!=(set(BRANCHES) if cf else set()):
            raise ValueError('diagnostic prediction/branch coverage mismatch')
        for name,ms in list(row['predictions'].items())+[(b,x['metrics']) for b,x in row['branches'].items()]+[('reference',row['reference_metrics'])]:
            # Reuse metric range/channel/empty/ASSD validation, with synthetic N lifecycle fields only.
            adapted=dict(wanted,metrics=ms,adam_step=0,native_counts=[],memory_size=0,optimizer_steps_this_visit=0,
                optimizer_update_norm=0,prompt_state_delta_norm=0,pipeline_elapsed_seconds=0,host_step_elapsed_seconds=0,
                source_versions_unchanged=True,optimizer_state_finite=True)
            validate_arm([adapted],[wanted],task,'N')
            for m in ms:
                for k in ('pred_foreground_pixels','gt_foreground_pixels','false_positive_pixels','false_negative_pixels'):
                    if type(m[k]) is not int or not 0<=m[k]<=m['total_pixels']: raise ValueError('invalid pixel count')
                if m['false_positive_denominator']!=m['total_pixels']-m['gt_foreground_pixels'] or m['false_negative_denominator']!=m['gt_foreground_pixels']:
                    raise ValueError('pixel denominator mismatch')
                if m['pred_foreground_pixels']-m['false_positive_pixels']!=m['gt_foreground_pixels']-m['false_negative_pixels']:
                    raise ValueError('confusion count mismatch')
                for value,num,den in [('false_positive_rate','false_positive_pixels','false_positive_denominator'),('false_negative_rate','false_negative_pixels','false_negative_denominator')]:
                    expected_rate=m[num]/m[den] if m[den] else None
                    if m[value]!=expected_rate: raise ValueError('rate mismatch')
        for b,value in row['branches'].items():
            if value['adam_calls']!=1 or value['adam_step']!=v: raise ValueError('counterfactual Adam not restored')
        # JSON round trip disallows NaN/Inf everywhere, including gradients and timings.
        json.dumps(row,allow_nan=False)
    return True


def aggregate(rows,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp']
    result={}
    for section,lo,hi in [('all',1,len(rows)),('1_5',1,5),('6_16',6,16),('17_plus',17,len(rows))]:
        chosen=[r for r in rows if lo<=r['visit']<=hi]
        if not chosen: continue
        def select(key,c): return channels([{'metrics':r['predictions'][key]} for r in chosen],c)
        value=result[section]=dict(visits=len(chosen),predictions={},deltas={},counterfactual={})
        for key in ('N','G','I','U'):
            value['predictions'][key]={}
            for c in names:
                m=select(key,c)
                value['predictions'][key][c]=dict(dice_percent=distribution([100*x['dice'] for x in m]),
                    assd_conditional_px=distribution([x['assd'] for x in m if x['assd'] is not None]),
                    assd_undefined=sum(x['assd'] is None for x in m))
                if c!='macro':
                    value['predictions'][key][c]['observations']={k:distribution([x[k] for x in m if x[k] is not None])
                        for k in m[0] if k not in ('channel','dice','assd')}
        for label,left,right in [('Delta_norm','G','N'),('Delta_init','I','G'),('Delta_update','U','I'),('U-N','U','N')]:
            value['deltas'][label]={c:paired(select(left,c),select(right,c)) for c in names}
        cf=[r for r in chosen if r['branches']]
        for b in BRANCHES:
            if not cf: continue
            def branch(c): return channels([{'metrics':r['branches'][b]['metrics']} for r in cf],c)
            value['counterfactual'][b]=dict(positions=len(cf),before_after={},versus_A_cf={},observations={})
            entry=value['counterfactual'][b]
            for c in names:
                pre=channels([{'metrics':r['predictions']['I']} for r in cf],c)
                a=channels([{'metrics':r['branches']['A_cf']['metrics']} for r in cf],c)
                entry['before_after'][c]=paired(branch(c),pre)
                entry['versus_A_cf'][c]=paired(branch(c),a)
                if c!='macro':
                    entry['observations'][c]={k:distribution([x[k]-y[k] for x,y in zip(branch(c),pre)])
                                              for k in ('region','pred_foreground_pixels')}
            entry['update_norm']=distribution([r['branches'][b]['update_norm'] for r in cf])
            entry['distance_from_A_cf']=distribution([r['branches'][b]['distance_from_A_cf'] for r in cf])
            entry['alignment']=distribution([r['branches'][b]['alignment'] for r in cf if r['branches'][b]['alignment'] is not None])
        value['retrieval_observed_count']=sum(r['retrieval_observed'] for r in chosen)
        if cf:
            value['gradients']={category:{key:distribution([r['gradients'][category][key] for r in cf if r['gradients'][category][key] is not None])
                              for key in cf[0]['gradients'][category]} for category in ('raw_norm','weighted_norm','cosine')}
    return result


def execute(receipt,stage,old,out):
    registration=json.loads((old/'registration.json').read_text());check_registered_files(registration)
    env=environment();private_json(out/(stage+'.environment.json'),env)
    smoke=stage=='smoke'; total={};evidence={};start=time.perf_counter();pair=None;rows=[];task=None
    try:
        for task,reg in registration['tasks'].items():
            state=torch.load(reg['checkpoint']['path'],weights_only=True,map_location='cpu')
            proxy=source_proxy(reg['selected']['proxy'],task)
            pair=DiagnosticPair(task,state,proxy,'cuda:0');torch.cuda.reset_peak_memory_stats(0)
            sys.path.insert(0,str(ROOT/'tests'))
            from test_vptta_host import pixels,proxy as procedural_proxy
            if smoke:
                selected=[dict(visit=i+1,segment='clean',sample_id=f'PROCEDURAL_{i}',group_id=f'PROCEDURAL_{i}') for i in range(17)]
                positions=(17,);fixture_mask=procedural_proxy(task).mask
            else:
                selected=expected_visits(reg['selected']['query'])[:len(reg['selected']['query'])];positions=POSITIONS
            identity=pair.identity_check(pixels(task))
            rows=[];task_start=time.perf_counter()
            with os.fdopen(os.open(out/f'{stage}_{task}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as handle:
                for i,wanted in enumerate(selected):
                    visit_start=time.perf_counter();r=reg['selected']['query'][i] if not smoke else None
                    image=pixels(task,i) if smoke else read_pixels(r['image_path'],task,r['image_size'])
                    reference,u,error=pair.native(image)
                    mask_cache=[]
                    def label():
                        mask=fixture_mask if smoke else read_mask(r['mask_path'],task,r['image_size'])
                        mask_cache.append(mask);return mask
                    record=pair.diagnose(image,u,label,wanted['visit'] in positions)
                    record.update(wanted,reference_metrics=metrics(reference,mask_cache[0],task),
                        equivalence_pass=True,max_logit_difference=error,counts=dict(pair.counts),
                        elapsed_seconds=time.perf_counter()-visit_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                    validate_records([record],[wanted],task,positions);append(handle,record);rows.append(record)
                    print(json.dumps(dict(stage=stage,task=task,completed=len(rows),adam_calls=sum(pair.counts[k] for k in ('reference_steps','watched_steps','counterfactual_steps')))),flush=True)
                    del reference,u,image,record
            validate_records(rows,selected,task,positions);counts=pair.finish()
            for k,v in counts.items(): total[k]=total.get(k,0)+v
            evidence[task]=dict(visits=len(rows),identity_prompt=identity,counts=counts,source_unchanged=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),elapsed_seconds=time.perf_counter()-task_start,
                retrieval_observed_positions=pair.retrieval,max_logit_difference=max(r['max_logit_difference'] for r in rows))
            private_json(out/f'{stage}_{task}.completion.json',evidence[task])
            del pair,state,proxy,rows;pair=None;rows=[];gc.collect()
        expected=(68,12) if smoke else (104,96)
        if (total['reference_steps']+total['watched_steps'],total['counterfactual_steps'])!=expected: raise ValueError('observed step budget mismatch')
        private_json(out/f'{stage}.completion.json',dict(status='H1_SMOKE_PASS' if smoke else 'H1_FORMAL_COMPLETE',
            commit=receipt['commit'],diagnostic_config_sha256=receipt['diagnostic_config_sha256'],
            registration_sha256=receipt['registration_sha256'],gpu_uuid=receipt['gpu_uuid'],counts=total,evidence=evidence,
            elapsed_seconds=time.perf_counter()-start,exit_code=0))
    except Exception as error:
        private_json(out/f'{stage}.failure.json',dict(status='INCOMPLETE',task=task,completed_current_task=len(rows),
            prior_completed_tasks=list(evidence),counts_prior_tasks=total,counts_current_task=pair.counts if pair else {},
            exception_type=type(error).__name__,private_reason=str(error),exit_code=1))
        raise


def recompute(old,out):
    registration=json.loads((old/'registration.json').read_text());public={}
    for task,reg in registration['tasks'].items():
        rows=[json.loads(line) for line in (out/f'formal_{task}.jsonl').read_text().splitlines()]
        expected=expected_visits(reg['selected']['query'])[:len(reg['selected']['query'])]
        validate_records(rows,expected,task,POSITIONS)
        done=json.loads((out/f'formal_{task}.completion.json').read_text())
        if done['source_unchanged'] is not True or done['visits']!=len(rows): raise ValueError('task completion mismatch')
        if rows[-1]['counts']!=done['counts']: raise ValueError('observed counters mismatch')
        public[task]=aggregate(rows,task)
        previous=[json.loads(line) for line in (old/f'{task}_A.jsonl').read_text().splitlines()][:len(rows)]
        if any(a['group_id']!=b['group_id'] for a,b in zip(rows,previous)): raise ValueError('prior clean identity mismatch')
        names=['OD','OC','macro'] if task=='fundus' else ['polyp']
        public[task]['reference_minus_old_clean']={c:paired(channels([{'metrics':r['reference_metrics']} for r in rows],c),channels(previous,c)) for c in names}
    return public


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','formal','recompute'),required=True);args=p.parse_args(argv)
    os.umask(0o077);r=json.loads(args.receipt.read_text())
    try: old,out=validate_receipt(r,args.stage)
    except Exception as error:
        print(json.dumps(dict(status='BLOCKED_RECEIPT',reason=str(error))));return 2
    try:
        if args.stage=='recompute':
            if json.loads((out/'formal.completion.json').read_text())['status']!='H1_FORMAL_COMPLETE':
                raise ValueError('formal completion missing')
            data=recompute(old,out)
            if torch.cuda.is_initialized(): raise ValueError('recompute initialized CUDA')
            private_json(out/'diagnostic.public.json',data)
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>256*1024**2: raise ValueError('private disk budget exceeded')
            private_json(out/'verification.json',dict(status='H1_DIAGNOSTIC_COMPLETE',independent_jsonl_recompute=True,cuda_initialized=False,exit_code=0))
        else: execute(r,args.stage,old,out)
        return 0
    except Exception as error:
        print(json.dumps(dict(status='INCOMPLETE',stage=args.stage,exception_type=type(error).__name__)));return 1


if __name__=='__main__': raise SystemExit(main())
