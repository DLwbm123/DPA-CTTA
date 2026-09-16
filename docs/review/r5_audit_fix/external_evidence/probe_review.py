"""Independent bounded CPU probes, not a rerun of the project's full suite.

host.py and rule.py are verbatim connector-fetched files and Git-blob verified.
Only snapshot/rollback are AST-loaded from host.py; the original model/augmentation
pipeline is NOT loaded. Two audit functions are verbatim excerpts. Smoke runs
with a procedural failing Old-host fixture, so it tests failure accounting only.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import ast, copy, gc, hashlib, importlib.util, json, math, random, sys, tempfile
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'sources'
torch.set_num_threads(2)
results=[]

def check(name,fn):
    try:
        detail=fn()
        results.append(dict(name=name,passed=True,detail=detail))
    except Exception as exc:
        results.append(dict(name=name,passed=False,error=type(exc).__name__+': '+str(exc)))
        raise

def blob_hash(path):
    b=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

expected={'host.py':'a5384abfabded8749df450e13711fa6faddbe3ca','rule.py':'f4ed32ee9e3cbae728a65f9e962532221db14c15'}
assert all(blob_hash(SOURCE/k)==v for k,v in expected.items())
spec=importlib.util.spec_from_file_location('r5_rule_under_review',SOURCE/'rule.py')
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
mod=ast.parse((SOURCE/'host.py').read_text())
selected=[n for n in mod.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('snapshot','rollback')]
namespace=dict(copy=copy,torch=torch)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE/'host.py'),'exec'),namespace)
snapshot,rollback=namespace['snapshot'],namespace['rollback']


def observation(r,pre=.2,trial=.1):
    return {'r':r,'e_pre':{'mean':pre},'e_trial':{'mean':trial}}


def rule_oracle():
    generated=random.Random(314159)
    series=[(None if t%47==0 else generated.random(), .2, .3 if t%5==0 else .1) for t in range(400)]
    rejected={}
    for arm in rule.ARMS:
        p=.63 if arm=='C_RANDOM' else None
        tested=rule.Rule(arm,p);history=[];draws=random.Random(20260908);nreject=0
        for t,(r,ep,et) in enumerate(series,1):
            old=history[-128:]
            reason='warmup' if t<=32 else 'empty_reliable_regions' if r is None else 'insufficient_history' if len(old)<32 else 'eligible'
            eligible=reason=='eligible'
            quant=float(np.quantile(old,.9,method='linear')) if old else None
            should_keep=not eligible or (et<=ep+1e-12 and r<=quant+1e-12)
            draw=draws.random() if arm=='C_RANDOM' else None
            accept=True if arm in ('C','C_HALF') else should_keep if arm=='C_VERIFY' else (not eligible or draw<p)
            z=tested.decide(observation(r,ep,et))
            assert z['reason']==reason and z['eligible']==eligible
            assert z['accept']==accept and z['shadow_accept']==should_keep and z['random_draw']==draw
            assert z['past_count']==len(old)
            assert z['q90_past'] is None if quant is None else abs(z['q90_past']-quant)<1e-15
            tested.append(r)
            if r is not None:history.append(r)
            assert list(tested.history)==history[-128:]
            nreject+=not z['accept']
        rejected[arm]=nreject
    return dict(rule_decisions=1600,rejected=rejected,reference='separate history + NumPy quantile + separate seeded RNG')


def measured_regions():
    pre=torch.full((1,2,512,512),.2)
    q=torch.full_like(pre,.5);trial=pre.clone();views=q.unsqueeze(0).repeat(6,1,1,1,1)
    obs=rule.measurements(pre,q,trial,views)
    assert obs['r'] is None and all(x['count']==0 and x['mean'] is None for x in obs['regions'])
    q[:,0,:256]=.95;views[:,:,0,:256]=.95;trial[:,0,:256]=.3
    obs=rule.measurements(pre,q,trial,views)
    assert obs['regions'][0]['count']==256*512 and sum(x['count'] for x in obs['regions'][1:])==0
    wanted=float((trial[:,0,:256].double()-pre[:,0,:256].double()).square().mean())
    assert obs['r']==wanted
    return dict(empty_all_is_null=True,partial_region_correct=True)


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def mkcore():
    torch.manual_seed(42)
    m=torch.nn.Linear(3,2);m.register_buffer('probe_buffer',torch.tensor([7.]))
    return SimpleNamespace(model=m,params=list(m.parameters()),base=torch.optim.Adam(m.parameters(),lr=1e-4),steps=0)

def update(c,offset):
    c.base.zero_grad(set_to_none=True)
    loss=c.model(torch.tensor([[.2+offset,.3,.5]])).square().sum()
    loss.backward();c.base.step();c.steps+=1

def rollback_reference(initialized):
    tested=mkcore();reference=mkcore()
    if initialized:
        update(tested,0.);update(reference,0.)
    before_model=copy.deepcopy(reference.model.state_dict());before_adam=copy.deepcopy(reference.base.state_dict())
    param_ids=[id(p) for p in tested.params]
    saved=snapshot(tested)
    update(tested,.1);tested.model.probe_buffer.add_(3)
    tested.base.param_groups[0]['probe']={'nested':[1,2]}
    for value in tested.base.state.values():value['probe_extra']={'v':torch.tensor([10.])}
    rollback(tested,saved)
    assert equal(tested.model.state_dict(),before_model) and equal(tested.base.state_dict(),before_adam)
    assert [id(p) for p in tested.params]==param_ids and all(p.grad is None for p in tested.params)
    # Reference never ran the rejected candidate and never calls snapshot/rollback.
    update(tested,.2);update(reference,.2)
    assert equal(tested.model.state_dict(),reference.model.state_dict())
    assert equal(tested.base.state_dict(),reference.base.state_dict())
    return dict(initialized=initialized,exact_next_update_parity=True,reference_did_not_call_rollback=True)

check('rule_vs_independent_400_step_oracle_each_arm',rule_oracle)
check('full_grid_empty_partial_regions',measured_regions)
check('rollback_uninitialized_next_step_independent_reference',lambda:rollback_reference(False))
check('rollback_initialized_next_step_independent_reference',lambda:rollback_reference(True))

# Load the two verbatim audit excerpts. Package name allows controlled relative-import stubs.
audit_ns={'__name__':'dpa_ctta.r5_update_acceptance.execution','__package__':'dpa_ctta.r5_update_acceptance','sys':sys,'ROOT':ROOT,'gc':gc,'PHYSICAL':rule.PHYSICAL}
exec(compile((SOURCE/'audit_extracts.py').read_text(),str(SOURCE/'audit_extracts.py'),'exec'),audit_ns)


def impossible_count_probe():
    m=dict(channel='OD',pred_pixels=262144,gt_pixels=100,intersection=0,total_pixels=262144,
           dice=0.,gt_empty=False,gt_full=False,pred_empty=False,pred_full=True,assd=1.)
    audit_ns['validate_metric'](m)
    assert m['pred_pixels']+m['gt_pixels']-m['intersection']>m['total_pixels']
    return dict(finding='IMPOSSIBLE_PIXEL_COUNTS_ACCEPTED',accepted=m,
                union=m['pred_pixels']+m['gt_pixels']-m['intersection'],grid=m['total_pixels'])

check('audit_gap_impossible_counts',impossible_count_probe)


def module(name,**items):
    m=ModuleType(name);m.__dict__.update(items);return m


def smoke_failure_probe():
    events=[];saved_json={}
    class FailingHost:
        def __init__(self,*a,**kw):
            self.model=torch.nn.Linear(3,2)
            self.counts={'forwards':0,'backwards':0,'base_adam':0}
        def step(self,pixels):
            for _ in range(3):
                with torch.no_grad():self.model(pixels)
                self.counts['forwards']+=1;events.append('completed_forward')
            raise RuntimeError('INJECTED_AFTER_THREE_COMPLETED_CPU_FORWARDS')
    stubs={
      'dpa_ctta.r1.host':module('dpa_ctta.r1.host',Host=FailingHost),
      'dpa_ctta.source_pilot':module('dpa_ctta.source_pilot',seed_all=lambda seed:torch.manual_seed(seed)),
      'dpa_ctta.b4_run':module('dpa_ctta.b4_run',capture=lambda core:{}),
      'dpa_ctta.host_diagnostic':module('dpa_ctta.host_diagnostic',close=lambda *a,**k:None),
      'dpa_ctta.m2_run':module('dpa_ctta.m2_run',deterministic_smoke_pair=nullcontext),
      'dpa_ctta.b3_runtime':module('dpa_ctta.b3_runtime',process_audit=lambda *a,**k:None),
      'dpa_ctta.r5_update_acceptance.host':module('dpa_ctta.r5_update_acceptance.host',Host=FailingHost),
      'test_vptta_host':module('test_vptta_host',pixels=lambda *a:torch.ones((1,3))),
    }
    audit_ns.update(write=lambda p,v:saved_json.update({p.name:copy.deepcopy(v)}),failure_scope=lambda exc:'probe_failure',backend_policy=lambda:{})
    recipe=dict(recipe='PROCEDURAL_OLD_C4_DIAG_C4_V1',pixel_indices=[0,1,2,3],network_forwards=64,loss_backward_calls=8,adam_calls=8,jacobian_vjp_calls=0)
    previous_path=sys.path.copy()
    try:
        with patch.dict(sys.modules,stubs), tempfile.TemporaryDirectory() as tmp:
            try:audit_ns['smoke']({},'cpu',Path(tmp),{'probe':True},recipe)
            except RuntimeError as exc:assert str(exc)=='INJECTED_AFTER_THREE_COMPLETED_CPU_FORWARDS'
            else:raise AssertionError('injected exception swallowed')
    finally:sys.path[:]=previous_path
    actual=len(events);recorded=saved_json['smoke.failure.json']['physical']['network_forwards']
    assert actual==3 and recorded==0
    return dict(finding='SMOKE_FAILURE_UNDERCOUNTS_COMPLETED_FORWARDS',actual_completed_forwards=actual,recorded_forwards=recorded,
                scope='isolated original smoke orchestration; real procedural CPU linear forwards; mock host/dependencies; no full C model')

check('audit_gap_partial_smoke_cost',smoke_failure_probe)

report=dict(implementation_sha='0c08cece6a91bdf5e3d06c9862dcd192df7787d8',
            source_blob_matches=expected,python=sys.version.split()[0],torch=torch.__version__,
            cuda_initialized=torch.cuda.is_initialized(),registered_checkpoint_reads=0,real_images_or_masks_reads=0,
            full_project_suite_rerun=False,results=results,
            positive_invariants=4,reproduced_audit_gaps=2,
            note='passed=true for a gap probe means the described defect was reproduced, not that production passes review')
(ROOT/'probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
