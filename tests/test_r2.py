"""Stage-I procedural checks. No registered weights, RGB, masks or GPU are read."""
import copy,json,math,tempfile,unittest
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r2 import analyze,plan,run
from dpa_ctta.r2.weighted_pca import WeightedSubspace,RHO,audit
from dpa_ctta.r2.feature_losses import paired_loss,unit,projection_residual
from dpa_ctta.r2.memory import Memory,SPECS
from dpa_ctta.r2.host import Host
from dpa_ctta.r1.streaming_pca import StreamingSubspace
from dpa_ctta.r1.host import Host as R1Host
from dpa_ctta.r1.region_memory import Memory as R1Memory
from dpa_ctta.r1.evidence import write
from dpa_ctta.b1_host import Host as OldC
from dpa_ctta.b4_run import capture
from dpa_ctta.host_diagnostic import close,rng
from dpa_ctta.source_pilot import seed_all
from test_r1 import Toy,ready
from test_r1_fixes import complete_fixture,scalar_row
from test_vptta_host import pixels


def host(arm,cls=Host):
    seed_all(20260907);return cls(arm,model=Toy())


class MathChecks(unittest.TestCase):
    def test_explicit_weighted_reference_multitoken_empty_and_first(self):
        g=torch.Generator().manual_seed(53);b=WeightedSubspace();xs=[];weights=[]
        for t,k in enumerate([0,3,17,0,8]*8,1):
            x=torch.randn(k,32,generator=g,dtype=torch.float64);xs.append(x);weights=[w*RHO for w in weights]+[torch.ones(k,dtype=torch.float64)]
            b.merge(x,t)
            if not b.n:continue
            X=torch.cat(xs);w=torch.cat(weights);W=w.sum();mu=(X*w[:,None]).sum(0)/W;M=(X-mu).T@((X-mu)*w[:,None])
            torch.testing.assert_close(b.mean,mu,rtol=1e-12,atol=1e-12);torch.testing.assert_close(b.M2,M,rtol=1e-12,atol=1e-12)
            self.assertAlmostEqual(b.W,float(W),11);self.assertAlmostEqual(b.Q,float(w.square().sum()),11)
            torch.testing.assert_close(b.covariance(),M/(W-w.square().sum()/W),rtol=1e-12,atol=1e-12)
        self.assertEqual(b.images,24);self.assertEqual(b.eigh_calls,1)
    def test_half_life_empty_visits_snapshot_and_no_expiry(self):
        b=WeightedSubspace();g=torch.Generator().manual_seed(3)
        for v in range(1,17):b.merge(torch.randn(8,32,generator=g),v)
        old=b.snapshot();W,Q,M,mu=b.W,b.Q,b.M2.clone(),b.mean.clone()
        for v in range(17,145):b.merge(torch.empty(0,32),v)
        self.assertAlmostEqual(b.W,W*.5,11);self.assertAlmostEqual(b.Q,Q*.25,11)
        close(b.M2,M*.5);close(b.mean,mu,exact=True);close(b.snapshot(),old,exact=True)
        self.assertEqual((b.n,b.images,b.version,b.eigh_calls),(128,16,16,1))
        self.assertLess(audit(b)['state_bytes'],1024**2);self.assertFalse(any(isinstance(v,(list,dict)) for v in vars(b).values()))
        with self.assertRaises(ValueError):b.merge(torch.empty(0,32),146)
    def test_rho_one_direct_R1_cumulative_equivalence(self):
        a=WeightedSubspace(1.);b=StreamingSubspace();g=torch.Generator().manual_seed(12)
        for i in range(1,42):
            x=torch.randn(0 if i%7==0 else 8,32,generator=g);a.merge(x,i);b.merge(x,i)
            close(a.mean,b.mean);close(a.M2,b.M2);self.assertEqual((a.n,a.images,a.version),(b.n,b.images,b.version))
            if a.U is not None:close(a.U@a.U.T,b.U@b.U.T)
        self.assertIs(type(Memory('R2_D').banks[0]),StreamingSubspace)
    def test_rank_zero_failure_and_snapshot_causality(self):
        b=WeightedSubspace()
        for i in range(1,17):b.merge(torch.ones(8,32),i)
        self.assertIsNone(b.snapshot());self.assertEqual(b.version,16)
        m=Memory('R2_DE');ready(m)
        with self.assertRaises(ValueError):m.snapshots(16)
        before=m.snapshots(17)
        for i in range(17,32):m.merge([torch.ones(8,32)]*4,i)
        close(before,m.snapshots(32),exact=True)
        with patch('torch.linalg.eigh',side_effect=RuntimeError('eigh fixture')):
            with self.assertRaises(RuntimeError):m.merge([torch.ones(8,32)]*4,32)
        with self.assertRaises(ValueError):WeightedSubspace().merge(torch.full((8,32),float('nan')),1)
    def test_pair_identity_with_far_center_and_subspace_delta(self):
        x=torch.randn(7,32,requires_grad=True);f0=x.detach().clone().requires_grad_();U=torch.eye(32)[:,:8].requires_grad_();mu=torch.ones(32)*40
        loss=paired_loss(x,f0,U);loss.backward();self.assertEqual(float(loss),0);self.assertEqual(float(x.grad.norm()),0)
        self.assertGreater(float(projection_residual(x.detach(),mu,U)),0);self.assertIsNone(f0.grad);self.assertIsNone(U.grad)
        a=torch.eye(32)[0:1].requires_grad_();b=torch.eye(32)[1:2]
        self.assertEqual(float(paired_loss(a,b,U)),0)
    def test_gradient_reductions_actual_rank_full_and_empty_zero(self):
        for k in (1,4,8):
            a=torch.randn(5,32,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);U=torch.eye(32,dtype=torch.float64)[:,:k].requires_grad_()
            d=unit(a)-unit(b.detach());expected=32/(32-k)*d[:,k:].square().sum(1).mean()
            close(paired_loss(a,b,U),expected);paired_loss(a,b,U).backward();self.assertGreater(float(a.grad.norm()),0);self.assertIsNone(b.grad);self.assertIsNone(U.grad)
            close(paired_loss(a,b,U,True),d.square().sum(1).mean())
            close(paired_loss(a,b,U,True),paired_loss(a,b,torch.roll(U,10,0),True))
        for n in (0,3):
            a=torch.zeros(n,32,requires_grad=True);l=paired_loss(a,a.detach(),torch.eye(32)[:,:8]);l.backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertEqual(float(l),0)
    def test_regions_same_pair_shuffle_RNG_and_F_shadow(self):
        f0=torch.randn(1024,32,requires_grad=True);fs=torch.randn(1024,32,requires_grad=True);q=torch.zeros(1024,2);q[:100,0]=1;q[:30,1]=1;views=q.expand(6,-1,-1).clone()
        de,full,shuffled=Memory('R2_DE'),Memory('R2_F'),Memory('R2_DE_S')
        ready(de);full.banks=copy.deepcopy(de.banks);shuffled.banks=copy.deepcopy(de.banks);before=rng()
        x,ids,a=de.prepare(f0,q,views,17);y,fids,b=full.prepare(f0,q,views,17);z,sids,c=shuffled.prepare(f0,q,views,17);close(before,rng(),exact=True)
        close(ids,fids,exact=True);close(x,y,exact=True);self.assertEqual(a,b);self.assertEqual(a,c)
        old=R1Memory('SHUFFLED');ox,oi,oa=old.prepare(f0,q,views,17);close(oi,sids,exact=True);close(ox,z,exact=True)
        self.assertFalse(torch.equal(ids[0],sids[0]));snaps=shuffled.snapshots(17)
        expected=torch.stack([paired_loss(fs[t%1024],f0[t%1024],s[1]) for t,s in zip(sids,snaps)]).mean()
        actual,diagnostic=shuffled.loss(fs,f0,sids,snaps);close(actual,expected);actual.backward();self.assertIsNone(f0.grad)
        ld,dd=de.loss(fs,f0,ids,de.snapshots(17));lf,df=full.loss(fs,f0,fids,full.snapshots(17));self.assertEqual([v['bank'] for v in dd],[v['bank'] for v in df])
        empty,_=full.loss(fs,f0,fids,[None]*4);self.assertEqual(float(empty),0)
        changed=[(s[0],torch.roll(s[1],4,0),s[2]) for s in full.snapshots(17)];close(lf,full.loss(fs,f0,fids,changed)[0])


class HostChecks(unittest.TestCase):
    def test_full_model_CPU_smoke_ready_all_arms_physical_budget(self):
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        seed_all(20260907);model,_=build_reference_model('fundus');state=copy.deepcopy(model.state_dict());del model
        with tempfile.TemporaryDirectory() as tmp,patch('dpa_ctta.b3_runtime.process_audit'):
            run.smoke(state,'cpu',Path(tmp),dict(fixture='CPU_RANDOM_WEIGHTS'),dict(seed=20260907),dict(fixture=True))
            result=json.loads((Path(tmp)/'smoke.completion.json').read_text())
            self.assertEqual(result['physical'],dict(forwards=112,backwards=14,base_adam=14,perturb=0,restore=0))
            self.assertEqual(set(result['evidence']),{'OLD','C',*SPECS})
            self.assertTrue(result['evidence']['C']['parity'])
        self.assertFalse(torch.cuda.is_initialized())
    def test_lambda_zero_old_C_and_absolute_cumulative_compatibility(self):
        x=pixels('fundus',0);old=host('C',OldC);z,_=old.step(x);saved=capture(old)
        for arm in SPECS:
            h=host(arm);ready(h.memory);h.visit=16;h.counts['base_adam']=16
            with patch('dpa_ctta.r2.host.EXTRA_WEIGHT',0.):q,a=h.step(x)
            close(z,q);close(saved,capture(h));self.assertEqual(h.pending,{});self.assertEqual(h.scalar_diagnostics,{})
        old=host('C_PCA_REGION',R1Host);new=host('C_PCA_REGION');ready(old.memory);ready(new.memory)
        for h in (old,new):h.visit=16;h.counts['base_adam']=16
        p,a=old.step(x);q,b=new.step(x);close(p,q);close(capture(old),capture(new));close(a['pca'],b['pca'])
    def test_all_arms_eight_calls_post_prediction_merge_isolation_cleanup(self):
        for arm in SPECS:
            h=host(arm);ready(h.memory);h.visit=16;h.counts['base_adam']=16;seen=[];merge=h.memory.merge
            def checked(vectors,visit):
                self.assertEqual(h.pending['forward'],8);self.assertIsNone(h.phase);self.assertEqual(h.counts['base_adam'],17);seen.append(visit);return merge(vectors,visit)
            with patch.object(h.memory,'merge',side_effect=checked),patch('dpa_ctta.source_io.read_pixels',side_effect=AssertionError('source')),patch('dpa_ctta.source_io.read_mask',side_effect=AssertionError('mask')),patch('dpa_ctta.source_io.source_proxy',side_effect=AssertionError('proxy')),patch('torch.autograd.grad',side_effect=AssertionError('extra gradient')):
                z,a=h.step(pixels('fundus',0))
            self.assertEqual(seen,[17]);self.assertEqual(a['counts'],dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0));self.assertEqual(a['pca']['basis_versions_used'],[16]*4)
            self.assertEqual(h.pending,{});self.assertEqual(h.scalar_diagnostics,{});json.dumps(a,allow_nan=False);self.assertFalse(z.requires_grad)
            with self.assertRaises(TypeError):h.step(pixels('fundus',0),domain='toy')
            with self.assertRaises(TypeError):h.step(pixels('fundus',0),torch.zeros(1))
        h=host('R2_DE')
        with patch.object(h.memory,'loss',side_effect=ValueError('fixture')):
            with self.assertRaises(ValueError):h.step(pixels('fundus',0))
        self.assertEqual(h.pending,{});self.assertEqual(h.scalar_diagnostics,{})


@contextmanager
def fixture(root):
    old=root/'old';out=root/'new';old.mkdir();out.mkdir()
    with complete_fixture(old,2) as (reg,oldpacket,ordered):
        # Keep all identities synthetic but bind the old runtime exactly.
        oldcode=oldpacket['binding']['code_sha'];newcode=plan.R1_CODE
        for file in old.rglob('*.json*'):
            file.write_text(file.read_text().replace(oldcode,newcode).replace('"backend": {','"backend": {"seed": 20260907,'))
        analyze.r1.recompute(old,reg)
        assets=dict(registration=reg,r1_result_directory=str(old))
        cfg=copy.deepcopy(plan.science());jobs=plan.matrix()['jobs']
        for j in jobs:j.update(records=12,forwards=96,backwards=12,adam=12)
        cfg['formal_budget'].update(new_records=240,forwards=1920,backwards=240,adam=240)
        identity=dict(run_id='c'*32,code_sha='d'*40,science_sha256=plan.SCIENCE_SHA,registration_digest=plan.registration_digest(reg))
        devices=[dict(index=i,uuid='CPU_FIXTURE_'+str(i),model='PROGRAMMATIC_CPU') for i in (6,7)]
        auth=dict(enabled=True,approved_code_sha=identity['code_sha'],approved_science_sha256=plan.SCIENCE_SHA,approved_registration_digest=identity['registration_digest'],external_review_reference='PROGRAMMATIC_TEST_ONLY_NOT_AUTHORIZATION',allowed_physical_gpu_ids=[6,7],max_workers=2,background_allowed=False)
        packet=dict(binding=identity,authorization=auth,assets=assets,out=str(out),devices=devices,jobs=jobs,schedule=plan.allocation(jobs,2))
        write(out/'packet.private.json',packet);write(out/'receipt.json',dict(binding=identity,devices=devices,jobs=jobs,schedule=packet['schedule'],formal_budget=cfg['formal_budget'],smoke_budget=cfg['per_gpu_smoke_budget']))
        entries=[];backend=dict(seed=20260907,device_name='PROGRAMMATIC_CPU')
        for i in range(2):
            p=out/('device'+str(i));p.mkdir();b=plan.binding(packet,i)
            write(p/'smoke.completion.json',dict(binding=b,status='MECHANICAL_SMOKE_COMPLETE',physical=dict(forwards=112,backwards=14,base_adam=14,perturb=0,restore=0),backend=backend,checkpoint_io=dict(bytes=7)))
            entries.append(dict(binding=b,phase='smoke',key=p.name,pid=200+i,pgid=200+i,status='EXITED',exit_code=0))
        for job,assignment in zip(jobs,packet['schedule']['assignments']):
            p=out/job['job_id'];p.mkdir();b=plan.binding(packet,assignment['worker'],job);rows=[];memory=Memory(job['arm'])
            for i,e in enumerate(ordered(reg,job['order']),1):
                r=scalar_row(e,'C_PCA_REGION',job['order'],i);r.update(binding=b,arm=job['arm']);memory.merge([torch.empty(0,32)]*4,i);r['pca']['banks']=memory.audit()
                r['r2']=dict(base_loss=.1,extra_loss=0.,weighted_extra_loss=0.,active_regions=0,region_energies=[],shadow_readiness_only=job['arm']=='R2_F');rows.append(r)
            (p/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            write(p/'completion.json',dict(binding=b,status='TRAJECTORY_COMPLETE',records=12,physical=dict(forwards=96,backwards=12,base_adam=12,perturb=0,restore=0),backend=backend,checkpoint_io=dict(bytes=7)))
            pid=200+len(entries);entries.append(dict(binding=b,phase='formal',key=p.name,pid=pid,pgid=pid,status='EXITED',exit_code=0))
        write(out/'matrix.processes.json',dict(binding=identity,status='COMPUTE_COMPLETE',exit_codes=[0]*22,processes=entries,active_seconds=1.,wall_seconds=1.))
        write(out/'processes.started.json',dict(binding=identity,processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]))
        with patch.object(analyze,'science',return_value=cfg),patch.object(analyze,'matrix',return_value=dict(jobs=jobs)),patch.object(analyze,'stream',side_effect=ordered):
            yield out,assets


class ExecutionChecks(unittest.TestCase):
    def test_disabled_actual_entry_and_frozen_twenty_jobs(self):
        m=plan.matrix();self.assertEqual(len(m['jobs']),20);self.assertEqual(len({j['job_id'] for j in m['jobs']}),20)
        self.assertEqual(sum(j['forwards'] for j in m['jobs']),312160);self.assertEqual(sum(j['adam'] for j in m['jobs']),39020)
        for n in (1,2,3):
            a=plan.allocation(m['jobs'],n);self.assertEqual(len(a['assignments']),20)
            self.assertEqual([r['worker'] for r in a['assignments']],[(list(SPECS).index(j['arm'])+j['order'])%n for j in m['jobs']])
        with patch('sys.argv',['neutral','--run']),patch('torch.load',side_effect=AssertionError('weights')),patch('subprocess.check_output',side_effect=AssertionError('GPU query')):
            with self.assertRaises(PermissionError):run.main()
        self.assertFalse(torch.cuda.is_initialized())
    def test_scalar_replay_live_weighted_refresh_and_corruptions(self):
        for arm in SPECS:
            m=Memory(arm);prior=None;g=torch.Generator().manual_seed(2)
            for visit in range(1,35):
                k=0 if visit in (8,19) else 8;used=[b.version if b.U is not None else None for b in m.banks]
                m.merge([torch.randn(k,32,generator=g)]*4,visit)
                p=dict(banks=m.audit(),basis_versions_used=used,input=dict(region_token_counts=[512]*4,sampled_region_counts=[k]*4,selected_region_counts=[k]*4,zero_vectors=[0]*4,assigned_bank_counts=[k]*4,missing_foreground=[False,False]),subloss=0.,base_loss=.1)
                for key,value in [('W',999),('Q',999),('raw_n',0),('images',999),('version',visit+1)]:
                    bad=copy.deepcopy(p);bad['banks'][0][key]=value
                    with self.assertRaises(ValueError):analyze.pca_scalars(bad,prior,arm,visit)
                prior=analyze.pca_scalars(p,prior,arm,visit)
    def test_twenty_job_bound_closeout_and_truncation_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with fixture(Path(tmp)) as (out,assets):
                result=analyze.recompute(out,assets)
                self.assertEqual(result['physical']['new_records'],240);self.assertEqual(result['historical_primary_records'],96);self.assertEqual(result['status'],'R2_EXPERIMENT_COMPLETE')
                self.assertTrue((out/'R2_EXPERIMENT_REPORT.md').is_file());self.assertFalse(result['assessment']['next_execution_authorized'])
                (out/'o3a4/records.jsonl').write_text('')
                with self.assertRaises(ValueError):analyze.recompute(out,assets)
                self.assertFalse(json.loads((out/'current_result.json').read_text())['valid']);self.assertFalse((out/'R2_EXPERIMENT_REPORT.md').exists())
    def test_bound_historical_metadata_and_new_weighted_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with fixture(Path(tmp)) as (out,assets):
                receipt,paths=analyze.historical_metadata(assets);self.assertEqual(len(paths),8)
                p=out/'o0a0/records.jsonl';original=p.read_text();rows=[json.loads(s) for s in original.splitlines()];rows[0]['pca']['banks'][0]['Q']=1;p.write_text('\n'.join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):analyze.recompute(out,assets)
                p.write_text(original)
                old=Path(assets['r1_result_directory'])/'o0a0/completion.json';d=json.loads(old.read_text());d['binding']['arm']='C_PCA_REGION';old.write_text(json.dumps(d))
                with self.assertRaises(ValueError):analyze.historical_metadata(assets)
