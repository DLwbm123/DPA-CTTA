"""One fully allocated NFS extent; bounded payloads and two committed metadata banks.
No model calls. Existing experiment files are read only. No filesystem growth
is permitted after create; all runtime logs/state/results must use this API.
"""
import fcntl,io,json,os,stat,struct,time
from contextlib import contextmanager
from pathlib import Path
SIZE=2*1024**2
BANK=16384
START=2*BANK
NORMAL=9961472
HARD=10485760
MAGIC=b'R107A001'
PREFIX=struct.Struct('<8sQI')
FOOTER=struct.Struct('<8sQ')

class CapacityError(RuntimeError):pass

def usage(root):
    total=0
    for p in (Path(root),*Path(root).rglob('*')):
        try:
            s=p.lstat()
            if not stat.S_ISLNK(s.st_mode):total+=max(s.st_size,s.st_blocks*512)
        except FileNotFoundError:pass
    return total

def name_ok(name):
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts:raise ValueError('relative object name required')

class Arena:
    def __init__(self,path,root):
        self.path=Path(path);self.root=Path(root)
        if self.path.stat().st_size!=SIZE:raise ValueError('fixed arena length')
    @contextmanager
    def locked(self):
        with self.path.open('r+b',buffering=0) as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            try:yield f
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    @staticmethod
    def _bank(raw):
        try:
            magic,generation,n=PREFIX.unpack_from(raw)
            end_magic,end_generation=FOOTER.unpack_from(raw,BANK-FOOTER.size)
            if magic!=MAGIC or end_magic!=MAGIC or generation!=end_generation or n>BANK-PREFIX.size-FOOTER.size:return None
            h=json.loads(raw[PREFIX.size:PREFIX.size+n]);assert h['generation']==generation
            assert START<=h['cursor']<=SIZE
            for row in h['objects'].values():assert START<=row['offset']<=row['offset']+row['length']<=h['cursor']
            return h
        except (ValueError,KeyError,AssertionError,struct.error):return None
    def _load(self,f):
        headers=[self._bank(os.pread(f.fileno(),BANK,i*BANK)) for i in (0,1)]
        valid=[h for h in headers if h is not None]
        if not valid:raise ValueError('no committed metadata bank')
        return max(valid,key=lambda h:h['generation'])
    def _commit(self,f,h):
        h=dict(h,generation=h['generation']+1)
        body=json.dumps(h,separators=(',',':'),allow_nan=False).encode()
        if len(body)>BANK-PREFIX.size-FOOTER.size:raise CapacityError('metadata capacity')
        bank=h['generation']%2;offset=bank*BANK
        record=PREFIX.pack(MAGIC,h['generation'],len(body))+body
        record=record.ljust(BANK-FOOTER.size,b'\0')
        self._write(f,offset,record);os.fsync(f.fileno())
        self._write(f,offset+BANK-FOOTER.size,FOOTER.pack(MAGIC,h['generation']));os.fsync(f.fileno())
        return h
    @staticmethod
    def _write(f,offset,data):
        if offset<0 or offset+len(data)>SIZE:raise CapacityError('fixed extent boundary')
        n=0
        while n<len(data):
            got=os.pwrite(f.fileno(),memoryview(data)[n:],offset+n)
            if got<=0:raise OSError('short arena write')
            n+=got
    def inspect(self):
        with self.locked() as f:
            h=self._load(f);s=os.fstat(f.fileno())
            return dict(generation=h['generation'],cursor=h['cursor'],payload_free=SIZE-h['cursor'],objects=h['objects'],state=h['state'],logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,root_usage=usage(self.root))
    def put(self,name,data):
        name_ok(name);data=bytes(data)
        with self.locked() as f:
            h=self._load(f)
            if name in h['objects']:raise FileExistsError(name)
            offset=(h['cursor']+7)//8*8;end=offset+len(data)
            if end>SIZE:raise CapacityError('payload capacity; no bytes written')
            proposed=dict(h,objects=dict(h['objects'],**{name:dict(offset=offset,length=len(data))}),cursor=end)
            # Validate metadata before committing payload, too.
            if len(json.dumps(proposed,separators=(',',':')).encode())>BANK-PREFIX.size-FOOTER.size:raise CapacityError('metadata capacity')
            self._write(f,offset,data);os.fsync(f.fileno());self._commit(f,proposed)
    def get(self,name):
        with self.locked() as f:
            h=self._load(f);row=h['objects'][name];data=os.pread(f.fileno(),row['length'],row['offset'])
            if len(data)!=row['length']:raise OSError('short arena read')
            return data
    def open_read(self,name):return io.BytesIO(self.get(name))
    def set_state(self,key,value):
        name_ok(key)
        with self.locked() as f:
            h=self._load(f);self._commit(f,dict(h,state=dict(h['state'],**{key:value})))
    @classmethod
    def create(cls,path,root):
        path=Path(path);root=Path(root)
        before=usage(root)
        if before+2*SIZE+65536>=NORMAL:raise CapacityError('preallocation headroom')
        readings=[]
        with path.open('x+b',buffering=0) as f:
            os.posix_fallocate(f.fileno(),0,SIZE)
            # Materialize all pages with noncompressible test data before admission.
            for offset in range(0,SIZE,262144):
                cls._write(f,offset,os.urandom(262144));os.fsync(f.fileno())
                s=os.fstat(f.fileno());u=usage(root)
                readings.append(dict(offset=offset,allocated=s.st_blocks*512,root_usage=u))
                if u>=NORMAL:raise CapacityError('preallocation did not fit normal cap')
            os.pwrite(f.fileno(),bytes(START),0);os.fsync(f.fileno())
        a=cls(path,root)
        with a.locked() as f:a._commit(f,dict(generation=0,cursor=START,objects={},state={'phase':'PREALLOCATED'}))
        return a,dict(before=before,readings=readings,after=a.inspect())
