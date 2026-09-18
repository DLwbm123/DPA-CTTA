"""Fresh output ownership. Legacy aggregates use the reviewed pinned reader."""
import json,os,time,uuid
from pathlib import Path

class Output:
    def __init__(self,path,binding):
        self.path=Path(path)
        if self.path.exists() or self.path.is_symlink():raise FileExistsError('new output only')
        if self.path.parent.is_symlink():raise ValueError('symlink output parent')
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
    import importlib.util
    path=Path(__file__).resolve().parents[3]/'analysis/r6d_posthoc_v1/pinned_io.py'
    spec=importlib.util.spec_from_file_location('r7_pinned_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
