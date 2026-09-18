"""Fresh output ownership. Legacy aggregates use the reviewed pinned reader."""
import json,os,time,uuid
from pathlib import Path

class Output:
    def __init__(self,path,binding):
        self.path=Path(path)
        if self.path.exists() or self.path.is_symlink():raise FileExistsError('new output only')
        if any(p.is_symlink() for p in [self.path.parent,*self.path.parents]):raise ValueError('symlink output ancestor')
        self.path.mkdir(parents=False)
        self.owner=uuid.uuid4().hex;self.binding=dict(binding);self.started=time.monotonic()
        self.write('owner.json',dict(owner=self.owner,binding=self.binding))
    def write(self,name,value):
        if Path(name).name!=name or name in ('.','..'):raise ValueError('output path escape')
        if name!='owner.json':
            owner=json.loads((self.path/'owner.json').read_text())
            if owner!=dict(owner=self.owner,binding=self.binding):raise ValueError('output owner/binding changed')
        with (self.path/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    def fail(self,exc,counts):
        self.write('first_error.json',dict(type=type(exc).__name__,message=str(exc)))
        self.write('cost.json',dict(seconds=time.monotonic()-self.started,counts=counts))

def legacy_reader():
    # Reuse only the separately reviewed fixed-version ordinary-file reader.
    # No historical recompute/invalidate/publish path is called.
    import importlib.util,sys
    path=Path(__file__).resolve().parents[3]/'analysis/r6d_posthoc_v1/pinned_io.py'
    spec=importlib.util.spec_from_file_location('r7_pinned_reader',path)
    module=importlib.util.module_from_spec(spec)
    old=sys.path[:]
    cached=sys.modules.get('core')
    if cached is not None and Path(cached.__file__).resolve()!=path.parent/'core.py':raise ValueError('foreign legacy core import')
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(module)
    finally:sys.path[:]=old
    return module

def snapshot_published(source,destination,expected_addendum):
    """Optional scalar input only: ordinary pinned payload, no source writes.

    Validate source/output disjointness before creating any file. The old reader's
    no-follow, nlink, in-read identity and pointer checks remain unchanged.
    """
    source=Path(source).absolute();destination=Path(destination).absolute()
    a,b=source.resolve(),destination.resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('source/output overlap')
    reader_module=legacy_reader();reader=reader_module.Reader(source)
    before,raw=reader_module.layout(reader,expected_addendum)
    out=Output(destination,expected_addendum['historical_binding'])
    try:
        relative=expected_addendum['aggregate_input']['source_relative_path']
        first=reader.record(relative)
        snap=reader.record(relative,out.path/'public_aggregate.json')
        after=reader.record(relative)
        if first!=snap or first!=after:raise ValueError('source payload changed')
        reader_module.check_layout(reader,expected_addendum,before)
        out.write('input_audit.json',dict(binding=expected_addendum['historical_binding'],payload=first,pointer=before['pointer_record'],links=before['links']))
        return out
    except Exception as exc:
        out.fail(exc,{});raise
