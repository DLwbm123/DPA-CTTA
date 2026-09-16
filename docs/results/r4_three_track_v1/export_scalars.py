"""Export validated scalar evidence, never images, masks or model weights.

Set RESULT_INPUT, RESULT_CONTROL, RESULT_EXPORT; execute using RUN_FILE.
The runtime analyzer is authoritative. Assertions check export coverage/counts.
"""
import collections,csv,datetime,gzip,json,os
from pathlib import Path


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in ('run_id','device_uuid','pid','pgid','result_directory')}
    if isinstance(v,list):return [clean(x) for x in v]
    return v


def encoded(v):
    s=json.dumps(clean(v),indent=2,allow_nan=False)+'\n'
    for forbidden in ('/data_nas/','/home/','/Users/','GPU-','sample_id','group_id','.png','.jpg','.jpeg','wangbomin'):
        assert forbidden not in s, forbidden
    return s


def table(root,name,rows):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    source=Path(os.environ['RESULT_INPUT']);control=Path(os.environ['RESULT_CONTROL']);dest=Path(os.environ['RESULT_EXPORT']);dest.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((source/name).read_text())
    current=read('current_result.json');a=read('public_aggregate.json');m=read('matrix.processes.json');launcher=json.loads((control/'launcher.exit.json').read_text())
    assert current['valid'] and current['status']==a['status']=='R4T_EXPERIMENT_COMPLETE'
    assert current['binding']==a['binding']==m['binding']
    assert m['status']=='COMPUTE_COMPLETE' and not m['unstarted_jobs'] and m['exit_codes']==[0]*73
    assert launcher['exit_code']==0 and not list(source.glob('*/*failure.json'))
    costs=[];completions=[];counts=collections.Counter()
    for process in m['processes']:
        if process['phase']!='formal':continue
        job=process['key'];done=read(job+'/completion.json');assert done['binding']==process['binding'] and done['records']==1951
        cost=dict(job=job,arm=done['binding']['arm'],order=done['binding']['order'],records=0,worker_seconds=done['seconds'],host_seconds=0.,pipeline_seconds=0.,peak_allocated_bytes=0)
        calls=collections.Counter()
        with (source/job/'records.jsonl').open() as f:
            for line in f:
                row=json.loads(line);cost['records']+=1;calls.update(row['r4t']['counts'])
                for k in ('host_seconds','pipeline_seconds'):cost[k]+=row[k]
                cost['peak_allocated_bytes']=max(cost['peak_allocated_bytes'],row['peak_allocated_bytes'])
        assert cost['records']==1951 and dict(calls)==done['physical']
        counts.update(calls);cost.update(calls);costs.append(cost);completions.append(done)
        print('PASS',job,cost['arm'],'records=1951 physical counters matched',flush=True)
    assert len(costs)==len({(v['arm'],v['order']) for v in costs})==70
    assert sum(v['records'] for v in costs)==a['physical']['records']==136570
    assert dict(counts)=={k:v for k,v in a['physical'].items() if k!='records'}
    smokes=[read('device%d/smoke.completion.json'%i) for i in range(3)]
    assert all(s['status']=='MECHANICAL_SMOKE_COMPLETE' for s in smokes)
    for k,v in a['smoke_physical'].items():assert sum(s['physical'][k] for s in smokes)==v
    execution=dict(status=a['status'],binding=a['binding'],review='EXPLICIT_USER_WAIVER_NOT_EXTERNAL_PASS',launcher=launcher,wall_seconds=m['wall_seconds'],active_worker_seconds=m['active_seconds'],process_exit_codes=m['exit_codes'],unstarted_jobs=[],failures=[],completions=completions,smokes=smokes)
    (dest/'execution.json').write_text(encoded(execution))
    with gzip.open(dest/'aggregate.json.gz','wb') as f:f.write(encoded(a).encode())
    table(dest,'costs_by_trajectory.csv',costs)
    scores=[];pairs=[];domains=[]
    for order,subsets in a['target'].items():
        for subset,v in subsets.items():
            for arm,channels in v['domain_equal_dice_percent'].items():scores.append(dict(order=order,subset=subset,arm=arm,**channels))
            if subset!='remaining_dev':continue
            for scope,block in [('pooled',v['pooled_content_paired'])]+[(d,x['paired']) for d,x in v['domains'].items()]:
                for pair,channels in block.items():
                    for ch,x in channels.items():
                        r=dict(order=order,scope=scope,pair=pair,channel=ch,**x['dice_delta_pp'])
                        for k in ('assd_common_valid','assd_not_jointly_defined','assd_left_undefined','assd_right_undefined','assd_delta_mean_px','assd_delta_median_px','assd_adverse_upper_decile_mean_px'):r[k]=x.get(k)
                        pairs.append(r)
            for domain,x in v['domains'].items():
                for arm,channels in x['arms'].items():
                    for ch,y in channels.items():domains.append(dict(order=order,domain=domain,arm=arm,channel=ch,dice_percent=y['dice_percent']['mean']))
    table(dest,'all_subset_domain_equal.csv',scores);table(dest,'remaining_pairs.csv',pairs);table(dest,'remaining_domain_scores.csv',domains)
    mechanism=[dict(job=job,key=k,**v) for job,x in a['mechanism'].items() for k,v in x.items()]
    table(dest,'mechanism_distributions.csv',mechanism)
    print('PASS all 70 trajectories, 136570 records, 73 zero exits; public scalar export, no raw assets',flush=True)
