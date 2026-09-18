#!/usr/bin/env python3
"""Independent boundary probes for f719c703. Standard library only.

Verifies saved full source hashes, then compiles ONLY named boundary definitions
from their ASTs. Does not import the repository, torch, models, or image decoders.
All files are synthetic TemporaryDirectory files. Clock/process/signal/cleanup
are explicit substitutes; preflight metadata audit/identity use test fixtures.
This is not the submitted 32+59 suite or a real SOURCE_PREP run.
"""
from __future__ import annotations
import ast, copy, hashlib, io, json, os, stat, sys, tempfile, time, uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from subprocess import TimeoutExpired

HERE=Path(__file__).resolve().parent
EXPECTED={
    'io.py':'dfbbdb9a9c2a6f99af449451f9dc690502af900f23a9e832c03594a934300daa',
    'runner.py':'f160e2ab99a899ff2994a32cdc6db2322a73df0ca4df896d41f1a0f05f848a5c',
}
TREES={}
for name,wanted in EXPECTED.items():
    raw=(HERE/'evidence'/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=wanted:
        raise RuntimeError('reviewed source hash mismatch: '+name)
    TREES[name]=ast.parse(raw,filename=name)

def select(tree,names):
    return ast.Module(body=[node for node in tree.body if
        isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names
        or isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)],type_ignores=[])

class BrokenStderr:
    def write(self,_):raise OSError('SYNTHETIC_STDERR_FAILURE')
    def flush(self):raise OSError('SYNTHETIC_STDERR_FAILURE')

class Fixture:
    def __init__(self,root):
        self.root=root;self.clock=[0.0];self.reads=[];self.audit_calls=[];self.cleanups=[]
        for name in ('source','storage','code'):(root/name).mkdir()
        self.ns=dict(copy=copy,io=io,json=json,os=os,stat=stat,uuid=uuid,Path=Path,Counter=Counter,
            time=NS(monotonic=lambda:self.clock[0],sleep=lambda _:None),
            signal=NS(SIGTERM=15,SIGINT=2,SIG_IGN=1,signal=lambda *_:None),
            sys=NS(platform='linux',stderr=io.StringIO()),
            subprocess=NS(TimeoutExpired=TimeoutExpired),ROOT=root/'code',SCIENCE={})
        for file,names in [('io.py',{'checked_path','Output'}),('runner.py',{
            'device_policy','preflight','expected_counts','TERMINAL_RESERVE','tree_bytes',
            'resource_check','BudgetOutput','cleanup_owned','supervise_one','publish_completion','main'})]:
            exec(compile(select(TREES[file],names),file,'exec'),self.ns)
        self.real_cleanup=self.ns['cleanup_owned']
        def cleanup(record):self.cleanups.append(record);record.update(cleaned=True)
        self.ns['cleanup_owned']=cleanup
        def verified(path,expected,max_bytes):
            p=self.ns['checked_path'](path)
            assert p.is_relative_to(root) and p.suffix=='.json', 'only synthetic JSON metadata allowed'
            self.reads.append(p.name);raw=p.read_bytes()
            assert len(raw)<=max_bytes and hashlib.sha256(raw).hexdigest()==expected
            return raw
        self.ns.update(verified=verified,code_identity=lambda:'SYNTHETIC_CODE_NOT_AUTH',
            audit=lambda *args:(self.audit_calls.append(1) or {'groups':48}))
        (root/'checkpoint.txt').write_text('synthetic bytes, not a model checkpoint')
        self.config=dict(enabled=True,scope='SOURCE_PREP',device='cpu',physical_GPU_ids=None,
            dtype_policy='R7_CPU_FP32_MODEL_FP64_LATENT_V1',workers=1,threads=2,retry=False,
            wall_seconds=30,output_bytes=500000,max_asset_bytes=100000,max_decoded_bytes=10**10,
            storage_root=str(root/'storage'),resource_authorization={'scope':'SOURCE_PREP','receipt_id':'SYNTHETIC_ONLY_NOT_REAL_AUTH'})
        self.receipt=dict(schema='R7_SOURCE_PREP_AUTH_V1',scope='SOURCE_PREP',enabled=True,
            user_authorization=dict(granted=True,scope='SOURCE_PREP',receipt_id='SYNTHETIC_ONLY_NOT_REAL_AUTH'),
            execution_layer_review=dict(status='PASS',scope='SOURCE_PREP',code_sha='SYNTHETIC_CODE_NOT_AUTH',checkpoint_sha256='4'*64),
            code_sha='SYNTHETIC_CODE_NOT_AUTH',science_sha256={},source_binding_status='BOUND',
            output_dir=str(root/'storage'/'run'),source_root=str(root/'source'),checkpoint_path=str(root/'checkpoint.txt'))
        for key,value in dict(config=self.config,manifest={'checkpoint':{'sha256':'4'*64}},split={},target={}).items():self.meta(key,value)
    def meta(self,key,value):
        p=self.root/(key+'.json');p.write_text(json.dumps(value))
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        self.receipt[key]={'path':str(p),'sha256':digest};self.receipt['execution_layer_review'][key+'_sha256']=digest
    def output(self,cap=500000):return self.ns['BudgetOutput'](self.root/'storage'/'run',{'synthetic':True},cap)
    def process(self,code=0):return NS(pid=987654321,returncode=code,poll=lambda:code)
    def supervise(self,out,code=0,cap=500000,start=None,wall=30):
        return self.ns['supervise_one'](start or (lambda:self.process(code)),dict(wall_seconds=wall,output_bytes=cap),out)
    def completion_fixture(self,out):
        worker=out.path/'worker';worker.mkdir()
        (worker/'execution.json').write_text(json.dumps(dict(status='SOURCE_PREP_COMPLETE_PENDING_REVIEW',source_after_check='UNCHANGED')))
    def read(self,out,name):return json.loads((out.path/name).read_text())

def expect(kind,fn,contains=None,identity=None):
    try:fn()
    except kind as exc:
        if contains is not None:assert contains in str(exc),(contains,str(exc))
        if identity is not None:assert exc is identity
        return exc
    raise AssertionError('expected '+str(kind))

RESULTS=[]
def case(name,fn):
    t=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='r7-review-') as temp:fn(Fixture(Path(temp).resolve()))
        RESULTS.append(dict(name=name,status='PASS',wall_seconds=time.perf_counter()-t))
    except Exception as exc:
        RESULTS.append(dict(name=name,status='FAIL',type=type(exc).__name__,message=str(exc),wall_seconds=time.perf_counter()-t))

def normal(f):
    a=f.ns['preflight'](f.receipt);assert a['audit']['groups']==48
    assert not Path(f.receipt['output_dir']).exists()
case('SP1_normal_preflight_no_output_creation',normal)

for field,relative in [('output_dir','storage/../source/new'),('output_dir','storage/../outside'),
    ('source_root','storage/../source'),('checkpoint_path','storage/../checkpoint.txt')]:
    def probe(f,field=field,relative=relative):
        f.receipt[field]=str(f.root/relative)
        expect(ValueError,lambda:f.ns['preflight'](f.receipt),'parent traversal')
        assert f.audit_calls==[] and not (f.root/'source'/'new').exists()
    case('SP1_'+field+'_'+relative,probe)

def bad_storage(f):
    f.config['storage_root']=str(f.root/'storage/../storage');f.meta('config',f.config)
    expect(ValueError,lambda:f.ns['preflight'](f.receipt),'parent traversal');assert f.audit_calls==[]
case('SP1_noncanonical_storage_with_valid_config_digest',bad_storage)
for key in ('manifest','split','target','config'):
    def probe(f,key=key):
        f.receipt[key]['path']=str(f.root/'storage'/'..'/(key+'.json'))
        expect(ValueError,lambda:f.ns['preflight'](f.receipt),'parent traversal');assert f.audit_calls==[]
    case('SP1_noncanonical_'+key+'_independent_valid_fixture',probe)

def links(f):
    link=f.root/'storage'/'link';link.symlink_to(f.root/'source',target_is_directory=True)
    f.receipt['output_dir']=str(link/'new')
    expect(ValueError,lambda:f.ns['preflight'](f.receipt),'symlink');assert not (f.root/'source'/'new').exists()
case('SP1_symlink_parent_not_hidden',links)
def missing(f):
    f.receipt['output_dir']=str(f.root/'storage'/'missing'/'new')
    expect(FileNotFoundError,lambda:f.ns['preflight'](f.receipt));assert f.audit_calls==[]
case('SP1_missing_parent_rejected',missing)
def constructor(f):
    expect(ValueError,lambda:f.ns['BudgetOutput'](f.root/'storage'/'..'/'source'/'new',{},500000),'parent traversal')
    assert not (f.root/'source'/'new').exists()
case('SP1_output_constructor_guard',constructor)

def evidence_combo(f,kind):
    out=f.output();attempts=[]
    def fail(name,value):attempts.append(name);raise OSError('SYNTHETIC_DISK_FAILURE')
    out.evidence=fail;f.ns['sys'].stderr=BrokenStderr()
    def raise_it(exc):raise exc
    if kind=='start':
        original=LookupError('SYNTHETIC_START');start=lambda:raise_it(original);expected=LookupError
    elif kind=='cleanup':
        original=ArithmeticError('SYNTHETIC_CLEANUP');f.ns['cleanup_owned']=lambda _:raise_it(original)
        start=lambda:f.process();expected=ArithmeticError
    elif kind=='timeout':
        original=None
        def start():f.clock[0]=2;return f.process()
        expected=TimeoutError
    else:original=None;start=lambda:f.process(3);expected=RuntimeError
    expect(expected,lambda:f.supervise(out,start=start,wall=1),identity=original)
    assert len(attempts)==len(set(attempts)) and 'supervisor.json' in attempts
    assert 'supervisor.first_error.json' in attempts and not (out.path/'completion.json').exists()
for kind in ('start','nonzero','timeout','cleanup'):case('SP2_'+kind+'_disk_and_stderr_failures',lambda f,k=kind:evidence_combo(f,k))

def summary_only(f):
    out=f.output();original=out.evidence;attempts=[];fault=OSError('SYNTHETIC_SUMMARY_FAILURE')
    def evidence(name,value):
        attempts.append(name)
        if name=='supervisor.json':raise fault
        original(name,value)
    out.evidence=evidence
    expect(OSError,lambda:f.supervise(out),identity=fault)
    assert f.read(out,'supervisor.first_error.json')['message']=='SYNTHETIC_SUMMARY_FAILURE'
    assert (out.path/'supervisor.evidence_errors.json').exists() and len(attempts)==len(set(attempts))
case('SP2_summary_only_failure_is_not_success',summary_only)
def normal_supervisor(f):
    out=f.output();assert f.supervise(out)==0
    s=f.read(out,'supervisor.json');assert s['complete'] and s['cleaned'] and not s['retry']
    assert len(f.cleanups)==1
case('SP2_normal_supervisor',normal_supervisor)

def oversized(f,code=0):
    out=f.output(200000);(out.path/'worker.log').write_bytes(b'x'*220000)
    expect(ValueError if code==0 else RuntimeError,lambda:f.supervise(out,cap=200000,code=code),
        'output cap' if code==0 else 'nonzero exit 3')
    assert (out.path/'worker.log').stat().st_size==220000 and not (out.path/'completion.json').exists()
    assert 'SOURCE_PREP_EVIDENCE_WRITE_ERROR' in f.ns['sys'].stderr.getvalue()
case('SP3_already_exited_overcap_same_writer_and_supervisor_limit',oversized)
case('SP3_nonzero_overcap_and_evidence_failure_preserves_first',lambda f:oversized(f,3))

def growth(f):
    out=f.output(200000);p=f.process();polls=[]
    def poll():
        polls.append(1)
        if len(polls)==1:return None
        (out.path/'worker').mkdir();(out.path/'worker'/'payload.bin').write_bytes(b'x'*220000);return 0
    p.poll=poll
    expect(ValueError,lambda:f.supervise(out,cap=200000,start=lambda:p),'output cap');assert len(polls)==2
case('SP3_growth_at_final_poll_nested_payload',growth)
for elapsed in (1,2):
    def probe(f,elapsed=elapsed):
        out=f.output()
        def start():f.clock[0]=elapsed;return f.process()
        expect(TimeoutError,lambda:f.supervise(out,start=start,wall=1))
        assert not f.read(out,'supervisor.json')['complete']
    case('SP3_terminal_time_'+str(elapsed),probe)

def reserve(f):
    out=f.output(200000);(out.path/'worker.log').write_bytes(b'x'*110000)
    expect(ValueError,lambda:f.supervise(out,cap=200000),'terminal reserve')
    assert not f.read(out,'supervisor.json')['complete'] and f.ns['tree_bytes'](out.path)<200000
case('SP3_global_terminal_reserve',reserve)
def writer_tree(f):
    out=f.output(200000);(out.path/'worker').mkdir();(out.path/'worker'/'payload').write_bytes(b'x'*150000)
    out.used=0
    expect(ValueError,lambda:out.bytes('new.bin',b'abc'),'cap');assert not (out.path/'new.bin').exists()
case('SP3_writer_uses_tree_not_local_used_counter',writer_tree)
def tree_link(f):
    out=f.output();(out.path/'link').symlink_to(f.root/'checkpoint.txt')
    expect(ValueError,lambda:f.ns['tree_bytes'](out.path),'nonordinary')
case('SP3_nonordinary_tree_rejected',tree_link)
def tree_hardlink(f):
    out=f.output();os.link(f.root/'checkpoint.txt',out.path/'linked')
    expect(ValueError,lambda:f.ns['tree_bytes'](out.path),'nonordinary')
case('SP3_hardlinked_output_rejected',tree_hardlink)
def audit_io(f):
    out=f.output();base=f.ns['tree_bytes']
    def fail(path):
        if (out.path/'process.json').exists():raise OSError('SYNTHETIC_AUDIT_IO')
        return base(path)
    f.ns['tree_bytes']=fail
    expect(RuntimeError,lambda:f.supervise(out,code=3),'nonzero exit 3')
    assert 'SYNTHETIC_AUDIT_IO' in f.ns['sys'].stderr.getvalue()
case('SP3_audit_IO_and_evidence_failure_keeps_worker_first',audit_io)

def completion(f,kind):
    out=f.output();f.completion_fixture(out);original=out.evidence
    def evidence(name,value):
        original(name,value)
        if name=='completion.pending.json':
            if kind=='time':f.clock[0]=1
            elif kind=='bytes':(out.path/'worker.log').write_bytes(b'x'*500000)
    out.evidence=evidence
    call=lambda:f.ns['publish_completion'](out,dict(wall_seconds=1,output_bytes=500000),0)
    if kind=='success':
        call();assert f.read(out,'completion.json')['other_scopes_authorized'] is False
        assert not (out.path/'completion.pending.json').exists()
    else:
        expect(TimeoutError if kind=='time' else ValueError,call)
        assert not (out.path/'completion.json').exists() and (out.path/'completion.pending.json').exists()
for kind in ('time','bytes','success'):case('SP3_completion_'+kind,lambda f,k=kind:completion(f,k))

def completion_initial_overcap(f):
    out=f.output(200000);f.completion_fixture(out);(out.path/'worker.log').write_bytes(b'x'*220000)
    expect(ValueError,lambda:f.ns['publish_completion'](out,dict(wall_seconds=30,output_bytes=200000),0),'output cap')
    assert not (out.path/'completion.json').exists()
case('SP3_completion_initial_tree_audit',completion_initial_overcap)
def invalid_worker(f):
    out=f.output();f.completion_fixture(out)
    (out.path/'worker'/'execution.json').write_text(json.dumps(dict(status='FAILED',source_after_check='UNCHANGED')))
    expect(ValueError,lambda:f.ns['publish_completion'](out,dict(wall_seconds=30,output_bytes=500000),0),'completion absent or invalid')
    assert not (out.path/'completion.json').exists()
case('SP3_exit0_not_enough_without_worker_completion',invalid_worker)

def cleanup(f,state):
    first=PermissionError('SYNTHETIC_EPERM');p=f.process();waits=[]
    def wait(timeout):
        waits.append(timeout)
        if state=='alive':raise TimeoutExpired('synthetic',timeout)
        return 0
    def stop(_):raise first
    p.wait=wait;f.ns['sys'].platform='darwin';f.ns['stop_owned']=stop
    f.ns['subprocess'].check_output=lambda *args,**kw:str(p.pid) if state=='group_present' else '1\n2\n'
    record={'process':p}
    if state=='reaped':f.real_cleanup(record);assert record['cleaned']
    else:expect(PermissionError,lambda:f.real_cleanup(record),identity=first);assert not record.get('cleaned')
    assert waits==[.5]
for state in ('reaped','alive','group_present'):case('cleanup_EPERM_'+state,lambda f,s=state:cleanup(f,s))

def denied(f):
    f.receipt['scope']='TARGET_SCREEN'
    expect(PermissionError,lambda:f.ns['preflight'](f.receipt));assert f.reads==[]
case('scope_TARGET_rejected_before_metadata',denied)
def default_entry(f):
    f.ns['os']=NS(environ={})
    expect(PermissionError,f.ns['main'],'receipt absent');assert f.reads==[]
case('scope_default_entry_disabled',default_entry)
def budget(f):
    counts=Counter()
    for row in f.ns['expected_counts']({'fit':list(range(111)),'cal':list(range(23)),'val':list(range(25))}).values():counts.update(row)
    assert counts==Counter(backbone_forwards=135328,source_backward_calls=9072,source_Adam=3072,source_VJP=1024,
        source_AdamW=6000,calibration_backward_calls=1536,calibration_Adam=1536)
case('unchanged_budget_arithmetic_no_model_calls',budget)

report=dict(schema='R7_EXTERNAL_BOUNDARY_PROBES_V1',implementation_sha='f719c703087b38c07bdfbe7ce9dcfa62d88a12d9',
    source_sha256=EXPECTED,tests=len(RESULTS),passed=sum(r['status']=='PASS' for r in RESULTS),
    failures=sum(r['status']!='PASS' for r in RESULTS),results=RESULTS,
    python=sys.version,platform=sys.platform,models_imported='torch' in sys.modules,
    actual_model_forwards=0,actual_backward_calls=0,checkpoint_deserializations=0,image_decodes=0,
    real_source_reads=0,real_target_reads=0,GPU_queries=0,real_children_spawned=0,
    interpretation='Independent AST-selected boundary cases. NOT the submitted 32+59 tests; NOT a full SOURCE_PREP execution.',
    substitutes=['clock','process','signal','normal cleanup','preflight code identity','preflight metadata audit','synthetic metadata verified reader'],
    actual_files='Temporary synthetic JSON/text/binary output only')
print(json.dumps(report,indent=2,ensure_ascii=False))
raise SystemExit(0 if report['failures']==0 else 1)
