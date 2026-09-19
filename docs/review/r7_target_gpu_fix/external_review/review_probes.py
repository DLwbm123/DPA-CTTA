"""Zero-model control-flow probes. This is not a project test-suite rerun.

Runs a manually materialized GitHub execute_job function excerpt unchanged.
All data, model, IO validation, scoring, resource-check dependencies are substitutes.
No torch import, network, GPU queries, checkpoint load or model forward occurs.
"""
import json, sys, time, hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'evidence/target_execute_job.excerpt.py'

class Output:
    def __init__(self): self.files={}
    def write(self,name,value): self.files[name]=value
    def evidence(self,name,value): self.files[name]=value

class Reader:
    def __init__(self,root,limit,kind): self.counts=Counter(); self.kind=kind
    def after_check(self): pass

rows=[{'subset':'remaining_dev'} for _ in range(1951)]
job={'arm':'C_BASE','order':0,'arrivals':1951,'scored_contents':1951,
     'network_forwards':15608,'backwards':1951,'Adam':1951}
# Synthetic rows intentionally omit real image paths/content and use a reduced
# semantic fixture, not the actual 1695-of-1951 registered scoring partition.
observed_counts={'forwards':15608,'backwards':1951,'Adam':1951}
approved={'config':{'job_wall_seconds':30,'job_output_bytes':1000000,'max_asset_bytes':1000},
          'binding':{'target_root':'SYNTHETIC_ONLY','artifact_root':'SYNTHETIC_ONLY',
                     **{k:{'path':'SYNTHETIC_ONLY','sha256':'0'*64} for k in ('registration','inventory','checkpoint')}},
          'registration':{},'inventory':{}}

results=[]
for mode in ('success_control','posthoc_failure','online_partial_failure'):
    out=Output(); events=[]
    host=SimpleNamespace(finish=lambda state:events.append('host_closed'),handles=[],counts={})
    def online(host,*args):
        if mode=='online_partial_failure':
            host.counts={'forwards':8,'backwards':1,'Adam':1}
            events.append('synthetic_partial_host_counters_available')
            raise RuntimeError('SYNTHETIC_ONLINE_FAILURE')
        events.append('synthetic_online_counts_returned')
        return dict(observed_counts)
    def posthoc(*args):
        events.append('posthoc_entered')
        if mode=='posthoc_failure':raise ValueError('SYNTHETIC_MASK_DECODE_FAILURE')
    ns={'time':time,'sys':sys,'json':json,'Path':Path,'TargetReader':Reader,
        'resource_check':lambda *a:None,'stream':lambda *a:rows,'make_host':lambda *a:(host,{}),
        'image_records':lambda *a:[],'online':online,'posthoc':posthoc,'verified':lambda *a:b''}
    exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), ns)
    err=None
    try: ns['execute_job'](approved,job,out)
    except Exception as e:err={'type':type(e).__name__,'message':str(e)}
    row={'case':mode,'exception':err,'events':events,'persisted_files':out.files,
         'counts_file_present':'counts.json' in out.files,
         'worker_has_model_counts':any(k in out.files.get('worker.json',{}) for k in ('counts','physical_counts','online_counts'))}
    if mode=='success_control':assert err is None and out.files['counts.json']==observed_counts
    else:
        assert err and out.files['worker.json']['status']=='FAILED'
        assert 'counts.json' not in out.files and not row['worker_has_model_counts']
        assert out.files['first_error.json']==err
    results.append(row)

result={'scope':'ISOLATED_EXECUTE_JOB_CONTROL_FLOW','cases':3,'assertions_passed':True,
        'interpretation':'Controls match the current implementation; both failure cases reproduce missing persisted model-cost counts, not a successful fix.',
        'project_test_suite_rerun':False,'model_calls':0,'optimizer_calls':0,'torch_imported':False,
        'real_asset_reads':0,'GPU_queries':0,'subprocesses_started':0,
        'excerpt_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_ref':'9de62090fb921695aacd7d68027cf7ec853bcc47',
        'source_path':'src/dpa_ctta/r7_target_screen/runner.py',
        'full_source_Git_blob_SHA_from_connector':'5905fddc2c3b128bd3b5cd096edf284c74869b53',
        'full_source_materialized_or_rehashed':False,'results':results}
(HERE/'output/control_flow_probe_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
