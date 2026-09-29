"""Finite source-only execution of the same kernels used by the R10 queue."""
import copy,json,math,time
from pathlib import Path
import torch
from .protocol import CAPS,PROFILE_CAPS,SPEC,graph,digest
from .identity import require
from .profile import units,node_units,projection
from .assets import open_source,validate_metadata
from .source import Source
from .controller import Actor,Host
from .learning import Trainer
from .evaluation import episode,d0,choose_families
from .factory import construct
from .runtime import disk
from ..r9_current_first.storage import write_json,write_torch,load_torch,PhaseJournal
from ..r9_current_first.physical import Meter
from ..r9_current_first.metrics import evaluate_probability
from ..r8_ba.journal import TargetJournal,_digest
from .target import retire_probabilities


class Profile:
    def __init__(self,config):
        self.config=config;self.identity=dict(**require(config,True),gpu_assignments=config['gpu_assignments'],output_root=config['output_root']);self.root=Path(config['output_root'])
        self.private=self.root/'private';self.private.mkdir(parents=True,exist_ok=True)
        self.scratch=self.root/'profile-scratch';self.scratch.mkdir(exist_ok=False)
        self.report=dict(schema='R10_REAL_PROFILE_V1',status='RUNNING',identity=self.identity,execution_authorized=False,measurements=[],snapshots={},trace_bytes={},source_log_bytes=0,active=None)
        self.meter=None;self.last_disk_check=0
    def save(self):write_json(self.private/'profile-progress.json',self.report)
    def guard(self,cost):
        if any(cost[k]>PROFILE_CAPS[k] for k in CAPS if k!='disk_bytes'):raise RuntimeError('aggregate resource cap: R10 profile')
        if time.monotonic()-self.last_disk_check>=1:
            if disk(self.root)+8*1024**2>PROFILE_CAPS['disk_bytes']:raise RuntimeError('aggregate resource cap: R10 profile disk')
            self.last_disk_check=time.monotonic()
    def check(self):self.meter.check()
    def measure(self,unit,call,*,repeats=3,divisor=1,cpu=False,variant=None):
        self.report['active']=dict(unit=unit,variant=variant);self.save();samples=[]
        for _ in range(repeats):
            before=self.meter.cost.copy()
            if not cpu:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.monotonic();call()
            if not cpu:torch.cuda.synchronize()
            elapsed=time.monotonic()-start;self.check()
            row={k:(self.meter.cost[k]-before[k])/divisor for k in CAPS}
            row['gpu_seconds']=0 if cpu else elapsed/divisor;row['disk_bytes']=0
            row.update(cpu_wall_seconds=elapsed/divisor if cpu else 0,max_reserved_bytes=torch.cuda.max_memory_reserved() if not cpu else 0)
            samples.append(row)
        row={k:1.3*max(s[k] for s in samples) for k in CAPS};row.update(measured=True,evidence=digest(samples),binding=self.identity,samples=samples,variant=variant,max_reserved_bytes=max(s['max_reserved_bytes'] for s in samples),cpu_wall_seconds=1.3*max(s['cpu_wall_seconds'] for s in samples))
        self.report['measurements'].append(dict(unit=unit,row=row));self.report['cost']=self.meter.cost.copy();self.save();return row
    def run(self):
        b=self.config['bindings'];gpu=self.config['gpu_assignments'][0]
        try:
            with Meter(None,self.guard) as self.meter:
                try:
                    self.source_profile(b,gpu)
                    self.online_profile(b,gpu)
                    self.report['status']='COMPLETE'
                finally:self.report['cost']=self.meter.cost.copy();self.report['cost']['disk_bytes']=disk(self.root);self.save()
        except BaseException as exc:
            import traceback
            self.report.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());self.save();raise
        return self.assemble()
    def source_profile(self,b,gpu):
        def setup():
            with open_source(b,gpu,self.check):pass
        self.measure('setup/source',setup)
        with open_source(b,gpu,self.check) as (data,seg,oracles,controller):
            handle=self.meter.attach(seg.model);source=Source(data,oracles,seg,controller,digest(b))
            try:
                a=Actor(20260924);t=Trainer(a,controller,source,20260924,'WARM',dict(profile_only=True),guard=self.check)
                for _ in range(4):t.step()
                self.measure('train/WARM',lambda:[t.step() for _ in range(4)],divisor=4,variant='complete eight-visit chunks')
                ref=copy.deepcopy(a)
                for method in SPEC['training']['methods']:
                    actor=copy.deepcopy(ref);tr=Trainer(actor,controller,source,20260924,method,dict(profile_only=True,method=method),ref,self.check)
                    # One full 32-round schedule block covers four modes and all eight windows.
                    # Timed blocks clear only the authorized source observation cache at entry.
                    def block():
                        source.cache.clear()
                        for _ in range(32):tr.step()
                    self.measure('train/'+method,block,divisor=32,variant='three complete curriculum/window blocks; actual two-epoch replay')
                    path=self.scratch/(method+'.pt')
                    def checkpoint():
                        sha=write_torch(path,tr.snapshot());load_torch(path,sha)
                    self.measure('source_checkpoint',checkpoint)
                    self.report['snapshots']['source']=max(self.report['snapshots'].get('source',0),path.stat().st_size)
                    row=tr.step();self.report['source_log_bytes']=max(self.report['source_log_bytes'],len(json.dumps(row).encode())+256)
                    journal=PhaseJournal(self.scratch/method,'io',dict(profile_only=True));journal.begin(tr)
                    self.measure('source_log',lambda:journal.append(row))
                for mode_index in (0,16,32,48):
                    source.cache.clear()
                    self.measure('validation_episode',lambda:episode(source,ref,20260924,mode_index,guard=self.check),variant=dict(index=mode_index))
                for index in range(4):
                    source.cache.clear()
                    self.measure('gap_episode',lambda:episode(source,ref,20260924,index,'cal',False,True,self.check),variant=dict(index=index))
                # D0 production function is evaluated once over its exact 64 contexts.
                self.measure('d0_context',lambda:d0(source,ref,20260924,self.check),repeats=1,divisor=64)
                def finalize():write_json(self.scratch/'source_complete_fixture.json',dict(profile_only=True,points={str(k):{'rows':[{'S':.5}]*64} for k in (500,1000,2000,4000)}))
                self.measure('source_finalize',finalize)
                self.measure('control/bind',lambda:validate_metadata(b),cpu=True)
                self.measure('control/freeze',lambda:write_json(self.scratch/'freeze_fixture.json',dict(spec=SPEC,profile_only=True)),cpu=True)
                fixture={f'FIT_{m}_{s}':{'selection':{'S':.5}} for m in SPEC['training']['methods'] for s in (20260924,20260925)}
                self.measure('control/select',lambda:choose_families(fixture),cpu=True)
                from .target import lock_sources
                sourcefixture={j['id']:{'schema':'R10_SOURCE_COMPLETE_V1','profile_only':True,'steps':2000 if j['kind']=='warmup' else 4000} for j in SPEC['source_jobs']}
                self.measure('control/lock',lambda:lock_sources(sourcefixture,choose_families(fixture)),cpu=True)
                p=self.scratch/'source'/'PROFILE';p.mkdir(parents=True)
                sha=write_torch(p/'actor.pt',dict(actor=ref.state_dict(),profile_only=True))
                self.report['profile_actor']=dict(file='actor.pt',sha256=sha)
                self.io_source=data
            finally:handle.remove()
    def online_profile(self,b,gpu):
        slots=[u.split('/',1)[1] for u in units() if u.startswith('online/')]
        lock={'sha256':digest(dict(profile_only=True,identity=self.identity))}
        names=self.io_source.folds['val'];images=[self.io_source.get(names[i%len(names)],'val').image for i in range(69)]
        for arm in slots:
            resolved=dict(arm='SUP_SEQ' if arm=='POLICY' else arm,method='SUP_SEQ' if arm=='POLICY' else arm,seed=20260924 if arm=='POLICY' else 20260907 if arm in ('VPTTA_NATIVE','C_CTTA','G_CTTA') else None,source_job='PROFILE' if arm=='POLICY' else None,artifact=self.report['profile_actor'] if arm=='POLICY' else None)
            def setup():
                host,close=construct(resolved,self.config,self.scratch,lock);close()
            self.measure('setup/'+arm,setup)
            host,close=construct(resolved,self.config,self.scratch,lock);model=host.native.model if hasattr(host,'native') else host.segmenter.model;handle=self.meter.attach(model)
            position=[0]
            def visit():
                logits,trace=host.step(images[position[0]%len(images)]);position[0]+=1
                self.report['trace_bytes'][arm]=max(self.report['trace_bytes'].get(arm,0),len(json.dumps(trace).encode())+1024)
                return logits,trace
            try:
                for _ in range(2):visit();self.check()
                self.measure('online/'+arm,visit,variant='cold')
                while host.visits<64:visit();self.check()
                row=self.measure('online/'+arm,visit,variant='after64')
                expected={'POLICY':(2,0,0),'B_CARRIER_FULL':(2,0,0),'B_CARRIER_RESET':(2,0,0),'B_CARRIER_STATIC':(2,0,0),'C0':(1,0,0),'N_SOURCE_EVAL':(1,0,0),'VPTTA_NATIVE':(2,1,1),'C_CTTA':(8,1,1),'G_CTTA':(9,2,1)}[arm]
                for sample in row['samples']:
                    if tuple(sample[k] for k in ('model_forwards','backward_calls','optimizer_steps'))!=expected:raise ValueError('actual native/policy operation contract: '+arm)
                path=self.scratch/(arm+'.pt')
                def cp():sha=write_torch(path,host.snapshot());load_torch(path,sha)
                self.measure('target_checkpoint/'+arm,cp)
                self.report['snapshots'][arm]=path.stat().st_size
                if arm=='POLICY':self.io_profile(resolved,lock,images)
            finally:handle.remove();close()
    def io_profile(self,resolved,lock,images):
        host,close=construct(resolved,self.config,self.scratch,lock);handle=self.meter.attach(host.segmenter.model)
        p=self.scratch/'trajectory';j=TargetJournal(p,host,'profile-source-only','0'*64,2*512*512*4);j.create();fixtures=[]
        try:
            for i in range(16):
                logits,trace=host.step(images[i]);prob=logits.sigmoid().detach().cpu().float().contiguous();raw=prob.numpy().astype('<f4',copy=False).tobytes()
                self.measure('target_append',lambda:j.append(raw,trace),repeats=1)
                fixtures.append((prob,self.io_source.get(self.io_source.folds['val'][i%len(self.io_source.folds['val'])],'val').label))
            seal=j.complete(16)
            self.measure('score_visit',lambda:[evaluate_probability(p,y) for p,y in fixtures],divisor=16,cpu=True)
            def score_seal():
                out=p/'scalars.private.jsonl'
                out.write_text(''.join(json.dumps(dict(visit=i+1,metrics=evaluate_probability(x,y)))+'\n' for i,(x,y) in enumerate(fixtures)))
                write_json(p/'score_complete.json',dict(schema='R10_SCORE_COMPLETE_V1',online_sha256=digest(seal),scalar_sha256=_digest(out)))
            self.measure('score_seal',score_seal,cpu=True)
            self.report['scalar_bytes_per_visit']=(p/'scalars.private.jsonl').stat().st_size/16+512
            retire_probabilities(p)
            if (p/'predictions.bits').exists():raise ValueError('profile probability retirement failed')
            self.report['io_retirement']='PASS'
        finally:handle.remove();close()
    def assemble(self):
        rows={}
        for item in self.report['measurements']:
            u=item['unit'];r=item['row']
            if u not in rows:rows[u]=copy.deepcopy(r)
            else:
                for k in CAPS:rows[u][k]=max(rows[u][k],r[k])
                rows[u]['max_reserved_bytes']=max(rows[u]['max_reserved_bytes'],r['max_reserved_bytes'])
                rows[u]['evidence']=digest([rows[u]['evidence'],r['evidence']])
        for family,methods in (('SUP',SPEC['training']['supervised_family']),('GR',SPEC['training']['rl_family'])):
            vs=[rows['train/'+m] for m in methods]
            rows['train/SELECTED_'+family]={k:max(v[k] for v in vs) for k in CAPS}
            rows['train/SELECTED_'+family].update(measured=True,evidence=digest(vs),binding=self.identity,max_reserved_bytes=max(v['max_reserved_bytes'] for v in vs),derived_max_of=methods)
        if set(rows)!=set(units()):raise ValueError('missing measured production units: '+str(set(units())-set(rows)))
        budgets={};peaks={}
        for n in graph():
            u=node_units(n);b={k:sum(math.ceil(rows[x][k]*count) for x,count in u.items()) for k in CAPS};keep=2*1024**2
            if n['kind']=='train':keep+=int(1.3*(6*self.report['snapshots']['source']+4000*self.report['source_log_bytes']+2*1024**2))
            if n['kind']=='online':
                arm=next(x.split('/',1)[1] for x in u if x.startswith('online/'));arr=n['job']['arrivals']
                keep+=int(1.3*(2*self.report['snapshots'][arm]+2*arr*self.report['trace_bytes'][arm]))+arr*2*512*512*4
            if n['kind']=='score':keep+=int(1.3*n['job']['arrivals']*self.report['scalar_bytes_per_visit'])
            b['disk_bytes']=keep;budgets[n['id']]=b
            peaks[n['id']]=max(rows[x]['max_reserved_bytes'] for x in u)
        prior=self.report['cost'].copy();prior['disk_bytes']=disk(self.root)
        profile=dict(identity=self.identity,measurements=rows,node_budgets=budgets,peak_vram_bytes=peaks,prior_cost=prior,raw_digest=digest(self.report))
        proof=projection(profile);write_json(self.private/'profile.json',profile);write_json(self.private/'admission.json',proof)
        return profile,proof
