"""Export completed R2 aggregates and scalar-only mechanism/cost supplements.

Use the frozen runtime src via PYTHONPATH; set RESULT_DIR, CONTROL_DIR, REPORT_DIR.
The destination must be new and outside the execution checkout. No model runs.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from dpa_ctta.r2.plan import matrix,binding,bound
from dpa_ctta.r2.continuation import read,source_for
from dpa_ctta.host_diagnostic_analysis import distribution


def public(value):
    forbidden={'sample_id','group_id','image_path','mask_path','pid','pgid','device_uuid','source_directory','private_storage_root'}
    if isinstance(value,dict):
        if set(value)&forbidden:raise ValueError('private field in public export')
        for v in value.values():public(v)
    elif isinstance(value,list):
        for v in value:public(v)
    elif isinstance(value,str) and any(s in value for s in ('/data_nas/','/home/','/Users/','GPU-')):
        raise ValueError('private path/device in public export')


def export():
    os.umask(0o077);torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    out,control,dest=(Path(os.environ[k]) for k in ('RESULT_DIR','CONTROL_DIR','REPORT_DIR'))
    aggregate=read(out/'public_aggregate.json');pointer=read(out/'current_result.json');receipt=read(out/'receipt.json')
    exited=read(control/'launcher.exit.json');process=read(out/'matrix.processes.json')
    assert pointer['valid'] and pointer['status']==aggregate['status']=='R2_EXPERIMENT_COMPLETE'
    assert pointer['binding']==aggregate['binding']==receipt['binding']
    assert exited['exit_code']==0 and process['status']=='COMPUTE_COMPLETE' and not any(process['exit_codes'])
    slots={a['job_id']:a['worker'] for a in receipt['schedule']['assignments']}
    jobs=[];mechanisms={};total=0
    for job in matrix()['jobs']:
        folder,origin=source_for(receipt,out,job);done=read(folder/'completion.json');identity=binding(origin,slots[job['job_id']],job)
        bound(done,identity);assert done['status']=='TRAJECTORY_COMPLETE' and not list(folder.glob('*failure.json'))
        rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
        assert len(rows)==done['records']==job['records'];total+=len(rows)
        for row in rows:bound(row,identity)
        counts={k:sum(row['counts'][k] for row in rows) for k in done['physical']}
        assert counts==done['physical']==dict(forwards=job['forwards'],backwards=job['backwards'],base_adam=job['adam'],perturb=0,restore=0)
        banks=[]
        for i in range(4):
            bs=[r['pca']['banks'][i] for r in rows]
            banks.append(dict(bank=i,final=bs[-1],ready_basis_visits=sum(r['pca']['basis_versions_used'][i] is not None for r in rows),
                contributing_visits=sum(r['pca']['input']['assigned_bank_counts'][i]>0 for r in rows),
                distributions={k:distribution([b[k] for b in bs]) for k in ('W','Q','raw_n','images','rank','version')}))
        energies={}
        for i in range(4):
            es=[e for r in rows for e in r['r2']['region_energies'] if e['bank']==i]
            energies[str(i)]={k:distribution([e[k] for e in es]) for k in ('original_absolute_energy','strong_absolute_energy','delta_energy','complement_delta_energy')}
        mechanisms[job['job_id']]=dict(aggregate['mechanism'][job['job_id']],banks=banks,region_energy_distributions=energies,
            loss_distributions={k:distribution([r['r2'][k] for r in rows]) for k in ('base_loss','extra_loss','weighted_extra_loss')},
            optimizer_update_l2=distribution([r['optimizer_update_l2'] for r in rows]),
            no_selected_tokens_per_region=[sum(r['pca']['input']['selected_region_counts'][i]==0 for r in rows) for i in range(4)],
            missing_foreground_visits=[sum(r['pca']['input']['missing_foreground'][i] for r in rows) for i in range(2)])
        jobs.append(dict(job_id=job['job_id'],arm=job['arm'],order=job['order'],runtime_binding=origin['binding'],
            physical_gpu=origin['devices'][slots[job['job_id']]]['index'],device_model=origin['devices'][slots[job['job_id']]]['model'],
            records=len(rows),physical=counts,trajectory_seconds=done['seconds'],
            latency={k:distribution([r[k] for r in rows]) for k in ('host_seconds','pipeline_seconds')},
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows)))
    assert total==aggregate['physical']['new_records']==39020 and len(jobs)==20
    cont=receipt['continuation'];tz=timezone(timedelta(hours=8))
    audit=dict(status=aggregate['status'],binding=receipt['binding'],jobs=jobs,launcher_exit_code=0,
        completed_at=datetime.fromtimestamp(exited['ended_unix'],tz).isoformat(),
        final_attempt_process_exit_codes=process['exit_codes'],selected_formal_physical=aggregate['physical'],
        cumulative_active_worker_seconds=cont['prior_active_seconds']+process['active_seconds'],
        cumulative_attempt_wall_seconds=cont['prior_wall_seconds']+process['wall_seconds'],
        final_attempt_cpu_closeout_seconds=exited['ended_unix']-exited['started_unix']-process['wall_seconds'],
        continuation_accounting=aggregate['continuation'],cuda_initialized=False,real_images_masks_or_weights_read=False,
        validation='Completed frozen CPU recompute plus export-time selected-row bindings, counts and completion consistency. No second model or metric evaluation.')
    values={'public_aggregate.json':aggregate,'MECHANISM_AND_COST.json':mechanisms,'EXECUTION_AUDIT.json':audit}
    for value in values.values():public(value)
    report=(out/'R2_EXPERIMENT_REPORT.md').read_text();public(report)
    dest.mkdir(mode=0o700,exist_ok=False)
    for name,value in values.items():(dest/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (dest/'RUNTIME_REPORT.md').write_text(report)
    print(json.dumps(dict(status=aggregate['status'],selected_records=total,selected_jobs=len(jobs),cuda_initialized=False,exported_files=sorted(p.name for p in dest.iterdir()))))


if __name__=='__main__':export()
