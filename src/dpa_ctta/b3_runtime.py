"""Neutral process arguments; preserve frozen scientific verification and call paths."""
import os
import subprocess
import sys
from .host_diagnostic_run import private_json

ENTRY='import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
FORBIDDEN=('wangbomin','ctta','grata','vptta','b3_run','b2_host','b1_host')


def neutral_subprocesses():
    import io,tarfile
    original_run=subprocess.run;original_popen=subprocess.Popen
    def run(args,*a,**kw):
        values=list(args) if isinstance(args,(list,tuple)) else []
        if values[:1]==['git']:
            values=values[1:];directory=kw.get('cwd')
            if values[:1]==['-C']:directory=values[1];values=values[2:]
            if values[:1]==['archive']:
                if len(values)<4 or values[2]!='--' or a:raise ValueError('unexpected legacy archive contract')
                revision=values[1];paths=values[3:]
                # Selected immutable blobs, same tar input to the unchanged legacy
                # validator. Pathspecs and object IDs never enter process argv.
                tree=original_run(['git','ls-tree','-r','-z',revision],cwd=directory,capture_output=True,check=True).stdout
                entries=[]
                for row in tree.split(b'\0'):
                    if not row:continue
                    info,name=row.split(b'\t',1);mode,kind,oid=info.decode().split();name=name.decode()
                    if kind=='blob' and any(name==p or name.startswith(p.rstrip('/')+'/') for p in paths):entries.append((name,mode,oid))
                data=original_run(['git','cat-file','--batch'],cwd=directory,input=''.join(oid+'\n' for _,_,oid in entries).encode(),capture_output=True,check=True).stdout
                cursor=0;buffer=io.BytesIO()
                with tarfile.open(fileobj=buffer,mode='w') as archive:
                    for name,mode,oid in entries:
                        end=data.index(b'\n',cursor);actual,kind,size=data[cursor:end].decode().split();size=int(size);cursor=end+1
                        if actual!=oid or kind!='blob':raise ValueError('Git blob response')
                        item=tarfile.TarInfo(name);item.mode=int(mode,8);item.size=size
                        archive.addfile(item,io.BytesIO(data[cursor:cursor+size]));cursor+=size+1
                return subprocess.CompletedProcess(args,0,buffer.getvalue(),b'')
            kw['cwd']=directory;args=['git',*values]
        return original_run(args,*a,**kw)
    class Popen(original_popen):
        def __init__(self,args,*a,**kw):
            if any(s in ' '.join(map(str,args)).lower() for s in FORBIDDEN):raise ValueError('non-neutral subprocess arguments')
            super().__init__(args,*a,**kw)
    subprocess.run=run;subprocess.Popen=Popen


def process_audit(out,stage):
    ids=[os.getpid(),os.getppid()]
    ps=subprocess.check_output(['ps','-ww','-p',','.join(map(str,ids)),'-o','pid=,ppid=,args='],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True)
    own=[s for s in gpu.splitlines() if s.split(',')[0].strip()==str(os.getpid())]
    if not own or any(s in (ps+'\n'+'\n'.join(own)).lower() for s in FORBIDDEN):raise ValueError('visible process command audit')
    private_json(out/(stage+'.process_audit.json'),dict(ps=ps,gpu=own,neutral=True))
