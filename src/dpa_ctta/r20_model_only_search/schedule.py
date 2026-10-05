"""One bounded outer-search sequence; no review metric enters selection."""
import concurrent.futures,copy,json,math,time,traceback
from pathlib import Path
import numpy as np
from .runtime import config,read,save,sha,run_task,ledger,ID,amounts
from .selection import ranking,promote,extensions,ablations
from .score import load_rows,rank_results
from ..r9_current_first.storage import lease

class Campaign:
    def __init__(self,c):
        self.c=c;self.root=Path(c['output_root']);self.candidates={x['id']:x for x in c['candidates']};self.profiles={};self.jobs=[];self.state=dict(status='IMPLEMENTED',stages={},jobs={},engineering={},T0=c['origin']['T0'],code_sha=c['code_sha'],source_reads=0,online_target_label_reads=0,review_opened=False)
    def save(self):save(self.root/'RUN_STATE.json',self.state)
    def estimate(self,candidate,n=1951):
        family=candidate['family'];xs=[p for p in self.profiles.values() if p['family']==family]
        if candidate['id'] in self.profiles:xs=[self.profiles[candidate['id']]]
        if family=='combo':
            xs=[p for p in self.profiles.values() if p['family']!='control'];return 1.2*(n*sum(max(p['seconds']) for p in sorted(xs,key=lambda p:max(p['seconds']),reverse=True)[:2])+90)
        if not xs:raise ValueError('unprofiled family')
        p=max(xs,key=lambda x:max(x['seconds']));return 1.2*(n*(float(np.quantile(p['seconds'],.95))+.08)+p['initialization_seconds']+30)
    def reserve(self,full_pairs=8):
        # Reserve primary/G/control base and first new seed plus key ablations before optional work.
        values=[self.estimate(x) for x in self.candidates.values() if x['family'] in {p['family'] for p in self.profiles.values()}]
        return full_pairs*2*max(values)
    def fits(self,jobs,reserve=0):
        estimates=[self.estimate(j['candidate'],384 if j['stream']=='screen' else 1951) for j in jobs];cost=sum(estimates);charged=amounts(self.c)[0];workers=len(self.c['gpu_assignments']);wall=max(estimates,default=0)+max(0,cost-max(estimates,default=0))/workers+reserve/workers
        return charged+cost+reserve<self.c['origin']['gpu_worker_cap_seconds']-120 and time.time()+wall<self.c['origin']['online_deadline_epoch']-600
    def make_jobs(self,stage,candidates,stream='full',seed=None,soft=False):
        return [dict(id=f'{stage}_{x["id"]}_s{seed or self.c["native_seed"]}_o{o}',stage=stage,candidate=copy.deepcopy(x),order=o,seed=seed or self.c['native_seed'],stream=stream,soft=soft) for x in candidates for o in (0,1)]
    def record_skip(self,stage,jobs,reason):
        for j in jobs:self.state['jobs'][j['id']]=reason
        self.state['stages'].setdefault(stage,{}).setdefault('not_run',[]).extend(dict(id=j['id'],reason=reason) for j in jobs);self.save()
    def run_pairs(self,stage,jobs,reserve=0,required=False):
        self.state['status']='RUNNING';self.state['stages'].setdefault(stage,{})['status']='RUNNING';accepted=[]
        # Each entire two-order condition is admitted together. No scientific retries.
        for i in range(0,len(jobs),2):
            pair=jobs[i:i+2]
            if len(pair)!=2:raise ValueError('paired orders required')
            if self.fits(accepted+pair,reserve):accepted+=pair
            else:self.record_skip(stage,pair,'NOT_RUN_BUDGET')
        for j in accepted:
            matching=[p for p in self.profiles.values() if p['family']==j['candidate']['family']]
            j['peak_bytes']=max([p['peak_reserved_bytes'] for p in matching] or [p['peak_reserved_bytes'] for p in self.profiles.values()])
            self.state['jobs'][j['id']]='NOT_RUN';self.jobs.append(j)
        print(json.dumps(dict(stage=stage,status='RUNNING',admitted_trajectories=len(accepted),gpu_worker_seconds=amounts(self.c)[0],review_labels_open=False)),flush=True)
        save(self.root/'stages'/f'{stage}.jobs.json',accepted);save(self.root/'stages'/f'{stage}.admission.json',dict(at=time.time(),admitted=[j['id'] for j in accepted],reserved_GPU_seconds=reserve,estimated_GPU_seconds=sum(self.estimate(j['candidate'],384 if j['stream']=='screen' else 1951) for j in accepted),already_charged_GPU_seconds=amounts(self.c)[0],method='family p95 plus initialization/IO and 20 percent margin',cpu_final_reserve_hours=2));self.save()
        for offset in range(0,len(accepted),len(self.c['gpu_assignments'])):
            batch=accepted[offset:offset+len(self.c['gpu_assignments'])]
            if not self.fits(batch,reserve):
                self.record_skip(stage,accepted[offset:],'NOT_RUN_BUDGET');break
            for j in batch:self.state['jobs'][j['id']]='RUNNING'
            self.save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
                fs={ex.submit(run_task,self.c,'online_'+j['id'],gpu,max(900,3*self.estimate(j['candidate'],384 if j['stream']=='screen' else 1951)),j):j for j,gpu in zip(batch,self.c['gpu_assignments'])}
                for f in concurrent.futures.as_completed(fs):
                    j=fs[f];r=f.result();failure=r.get('failure') or {}
                    # Only a recorded transient I/O/network failure can resume once; timeout/budget/numerics cannot.
                    if failure.get('error_type') in ('OSError','ConnectionError','InterruptedError') and failure.get('errno') in (4,5,104,110,116) and self.fits([j],reserve):
                        resumed=dict(j,recovery=dict(class_='INFRASTRUCTURE'))
                        resumed['recovery']={'class':'INFRASTRUCTURE','reason':failure['reason'],'evidence':dict(attempt=0,receipt=sha(r),errno=failure['errno'])}
                        r=run_task(self.c,'online_'+j['id'],self.c['gpu_assignments'][batch.index(j)],max(900,3*self.estimate(j['candidate'])),resumed,attempt=1)
                    self.state['jobs'][j['id']]=r['status'];self.save();ledger(self.c,self.state)
        done=[j for j in accepted if self.state['jobs'][j['id']]=='COMPLETE'];save(self.root/'stages'/f'{stage}.barrier.json',dict(status='ALL_WORKERS_RETIRED',at=time.time(),jobs={j['id']:self.state['jobs'][j['id']] for j in accepted},complete=[j['id'] for j in done]));self.state['stages'][stage]['status']='PREDICTIONS_SEALED';self.save();return done
    def score(self,stage,jobs,final=False):
        # CPU scoring starts after a whole stage has retired, never inside an online worker.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fs={ex.submit(run_task,self.c,('score_final_' if final else 'score_'+stage+'_')+j['id'],None,7200,dict(job=j,stage=stage,final=final)):j for j in jobs}
            for f in concurrent.futures.as_completed(fs):
                r=f.result()
                if r['status']!='COMPLETE':raise RuntimeError('independent scorer failed: '+fs[f]['id'])
        if final:return
        rows=load_rows(self.root,stage,jobs);attempts=[read(p) for p in (self.root/'attempts').glob('online_*.json')];results=rank_results(rows,jobs,attempts);save(self.root/'stages'/f'{stage}.selection.json',dict(selection_mode=self.c['selection_mode'],results=ranking(results),review_labels_read=False));self.state['stages'][stage]['status']='SCORED';self.save();print(json.dumps(dict(stage=stage,status='SCORED',complete_conditions=len(results),gpu_worker_seconds=amounts(self.c)[0],review_labels_open=False)),flush=True);return results
    def qualify(self):
        proof=read(self.root/'MECHANICAL_TESTS.json')
        if proof['status']!='PASS' or proof['tests']<7:raise ValueError('mechanical qualification absent')
        ids=['G','T03','W04','A05','P04','C','G_K2','G_H025']
        self.state['status']='TESTED';self.save()
        for offset in range(0,len(ids),4):
            batch=ids[offset:offset+4]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                fs={ex.submit(run_task,self.c,'profile_'+cid,gpu,1800,dict(candidate=self.candidates[cid])):cid for cid,gpu in zip(batch,self.c['gpu_assignments'])}
                for f in concurrent.futures.as_completed(fs):
                    cid=fs[f];r=f.result()
                    if r['status']=='COMPLETE':self.profiles[cid]=r['result'];self.state['engineering'][cid]='TESTED'
                    else:self.state['engineering'][cid]='NOT_RUN_ENGINEERING'
                    self.save();ledger(self.c,self.state)
        if 'G' not in self.profiles:raise RuntimeError('native G engineering qualification failed')
        qualified=[]
        for c in self.candidates.values():
            rep={'teacher':'T03','pixel_weight':'W04','online_adapter':'A05','target_prototype':'P04'}.get(c['family'],c['id'] if c['id'] in ids else 'G')
            if rep in self.profiles:qualified.append(c)
            else:self.state['engineering'][c['id']]='NOT_RUN_ENGINEERING'
        save(self.root/'PROFILE_ADMISSION.json',dict(profiles=self.profiles,qualified_candidates=[x['id'] for x in qualified],estimated_seconds_per_full_trajectory={x['id']:self.estimate(x) for x in qualified},GPU_worker_cap_hours=48,max_concurrent=4,source_reads=0,target_label_reads=0));self.save();return qualified
    def execute(self):
        qualified=self.qualify();families=['control','teacher','pixel_weight','online_adapter','target_prototype'];queues={f:[c for c in qualified if c['family']==f] for f in families};ordered=[]
        while any(queues.values()):
            for f in families:
                if queues[f]:ordered.append(queues[f].pop(0))
        stage='S1';jobs=self.run_pairs(stage,self.make_jobs(stage,ordered,'screen'),reserve=self.reserve(8));results=self.score(stage,jobs)
        if not any(r['id']=='G' for r in results):raise RuntimeError('paired screen G unavailable')
        selected,control=promote(results,self.candidates);save(self.root/'stages/S2.selection_registration.json',dict(selected=selected,screen_control=control,rule='family diverse then fixed lexicographic ranking',at=time.time()))
        full_jobs=self.run_pairs('S2',self.make_jobs('S2',[self.candidates[i] for i in selected]),reserve=self.reserve(5));full_results=self.score('S2',full_jobs)
        methods=ranking([r for r in full_results if self.candidates[r['id']]['family']!='control'])
        if not methods:raise RuntimeError('no paired full-flow new candidate')
        # One optional registered extension only; never regenerate from its outcome.
        extra=extensions(full_results,self.candidates);extra_jobs=self.make_jobs('S3_SCREEN',extra,'screen');reserve=self.reserve(7)
        if extra and self.fits(extra_jobs,reserve):
            for x in extra:self.candidates[x['id']]=x
            save(self.root/'stages/S3.candidates.json',extra)
            # Stage-matched G is required; exact S1 G reused since stream, seed and configuration are identical.
            ej=self.run_pairs('S3_SCREEN',extra_jobs,reserve=reserve);g_screen=[j for j in jobs if j['candidate']['id']=='G'];self.score_extensions(ej,g_screen)
            er=read(self.root/'stages/S3_SCREEN.selection.json')['results'];top=[r['id'] for r in er if r['id']!='G'][:2]
            more=self.run_pairs('S3_FULL',self.make_jobs('S3_FULL',[self.candidates[i] for i in top]),reserve=self.reserve(5));self.score_extensions(more,[j for j in full_jobs if j['candidate']['id']=='G'],stage='S3_FULL',base_stage='S2')
            full_jobs+=more;full_results+=read(self.root/'stages/S3_FULL.selection.json')['results'];full_results=list({r['id']:r for r in full_results}.values())
        else:self.state['stages']['S3_SCREEN']=dict(status='NOT_RUN_BUDGET',reason='confirmation and key-ablation reserve takes priority');self.save()
        methods=ranking([r for r in full_results if self.candidates[r['id']]['family']!='control']);primary=methods[0]['id'];backup=methods[1]['id'] if len(methods)>1 else None;bestcontrol=ranking([r for r in full_results if self.candidates[r['id']]['family']=='control'])[0]['id'];main=list(dict.fromkeys([primary,'G',bestcontrol]));abs_,reuse=ablations(self.candidates[primary],self.candidates)
        # Freeze all choices before any new-seed or review scoring.
        first=self.make_jobs('CONFIRM_17011',[self.candidates[i] for i in main],seed=17011,soft=True);second=self.make_jobs('CONFIRM_29009',[self.candidates[i] for i in main],seed=29009,soft=True);abjobs=self.make_jobs('ABL',abs_);seeds=[self.c['native_seed'],17011]
        if not self.fits(first):raise RuntimeError('first paired confirmation cannot finish; preserve incomplete evidence')
        if self.fits(first+abjobs+second):seeds.append(29009)
        additional=[]
        gresult=next(r for r in full_results if r['id']=='G')
        if methods[0]['seconds_per_image']>1.5*gresult['seconds_per_image'] and 'G_K2' in self.profiles and not any(j['candidate']['id']=='G_K2' for j in full_jobs):additional=self.make_jobs('COST_CONTROL',[self.candidates['G_K2']])
        freeze=dict(primary=primary,backup=backup,best_simple_control=bestcontrol,comparison_ids=main,seeds=seeds,code_sha=self.c['code_sha'],config_sha256=sha(self.c),candidates={i:self.candidates[i] for i in main+([backup] if backup else [])},ablations=abs_,reused_ablations=reuse,extra_compute_control='G_K2' if additional else None,SEARCH_results=ranking(full_results),initialization='original checkpoint; empty moments/history; independent module RNG; zero-U adapter',primary_metric='equal domain x OD/OC macro hard Dice',frozen_at=time.time())
        save(self.root/'FROZEN.json',freeze);self.state['status']='CONFIGURATION_FROZEN';self.save()
        self.run_pairs('CONFIRM_17011',first,required=True)
        self.run_pairs('ABL',abjobs)
        if additional:self.run_pairs('COST_CONTROL',additional,reserve=sum(self.estimate(j['candidate']) for j in second) if 29009 in seeds else 0)
        if 29009 in seeds:self.run_pairs('CONFIRM_29009',second)
        # Every attempted/registered formal task has now terminated. Release is immutable and occurs once.
        active=[p for p in (self.root/'processes').glob('online_*.json') if read(p).get('active')]
        if active:raise ValueError('formal online workers still active')
        final_jobs=[j for j in self.jobs if j['stream']=='full' and self.state['jobs'].get(j['id'])=='COMPLETE'];save(self.root/'stages/FINAL.barrier.json',dict(status='ALL_WORKERS_RETIRED',at=time.time(),jobs={j['id']:self.state['jobs'][j['id']] for j in self.jobs}));save(self.root/'FINAL_JOBS.json',final_jobs)
        release=self.root/'REVIEW_RELEASE.json'
        if release.exists():raise ValueError('sealed review already released')
        save(release,dict(frozen_sha256=sha(freeze),at=time.time(),all_formal_workers_retired=True,selection_closed=True));self.state['review_opened']=True;self.state['status']='FINAL_SCORING';self.save();self.score('FINAL',final_jobs,final=True)
        required=first+abjobs+(second if 29009 in seeds else [])
        self.state['status']='COMPLETE' if all(self.state['jobs'].get(j['id'])=='COMPLETE' for j in required) else 'INCOMPLETE'
        self.state['execution_finished']=True;self.state['ended']=time.time();self.state['delivery']='PENDING_GITHUB';self.save();ledger(self.c,self.state)
        from .report import report
        report(self.c,self.state)
    def score_extensions(self,jobs,baseline,stage='S3_SCREEN',base_stage='S1'):
        # Copy already-scored scalar files only into the offline scorer area, no new label access.
        import shutil
        target=self.root/'scores'/stage;target.mkdir(parents=True,exist_ok=True)
        for j in baseline:
            for suffix in ('.private.jsonl','.complete.json'):shutil.copyfile(self.root/'scores'/base_stage/(j['id']+suffix),target/(j['id']+suffix))
        if jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                futures=[ex.submit(run_task,self.c,'score_'+stage+'_'+j['id'],None,7200,dict(job=j,stage=stage,final=False)) for j in jobs]
                for f in futures:
                    if f.result()['status']!='COMPLETE':raise RuntimeError('extension scorer failed')
        rows=load_rows(self.root,stage,jobs+baseline);attempts=[read(p) for p in (self.root/'attempts').glob('online_*.json')];results=rank_results(rows,jobs+baseline,attempts);save(self.root/'stages'/f'{stage}.selection.json',dict(results=ranking(results),review_labels_read=False));return results

def supervise():
    c=config();root=Path(c['output_root']);campaign=Campaign(c)
    with lease(root/'supervisor',dict(experiment_id=ID,config_sha256=sha(c))):
        marker=root/'execution_started.json'
        if marker.exists():raise ValueError('campaign already started')
        save(marker,dict(at=time.time(),config_sha256=sha(c)))
        try:campaign.execute()
        except BaseException as e:
            traceback.print_exc();campaign.state.update(status='INCOMPLETE',reason=str(e),ended=time.time(),delivery='PENDING_GITHUB');campaign.save();ledger(c,campaign.state)
            from .report import report
            report(c,campaign.state)
