import copy,json,math,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from dpa_ctta.r1.recovery import *
from dpa_ctta.r1.streaming_pca import *
from dpa_ctta.r1.region_memory import Memory,grid,tokens,split_shuffle
from dpa_ctta.r1.host import Host,ARMS,segmentation
from dpa_ctta.r1.plan import authorize,matrix,science,registration_digest,digest,SCIENCE
from dpa_ctta.r1.run import claim,main
from dpa_ctta.r1.analyze import validate,recompute
from dpa_ctta.b1_host import Host as OldC
from dpa_ctta.b4_run import capture
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.p1_analysis import evaluate
from test_vptta_host import pixels


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,32,1,bias=False);self.bn=nn.BatchNorm2d(32);self.seg_head=nn.Conv2d(32,2,1)
    def forward(self,x):
        f=torch.relu(self.bn(self.conv(x)));return self.seg_head(f),[],f


def toy(arm):
    seed_all(20260907);return Host(arm,model=Toy())


def ready(memory):
    g=torch.Generator().manual_seed(8)
    for b in memory.banks:
        for i in range(1,17):
            x=torch.randn(8,32,generator=g);b.merge(x/x.norm(dim=1,keepdim=True),i)


class MathChecks(unittest.TestCase):
    def test_entropy_shared_region_and_empty(self):
        x=bernoulli_entropy_bits(torch.tensor([0.,.5,1.],dtype=torch.float64));self.assertTrue(torch.isfinite(x).all());self.assertAlmostEqual(float(x[1]),1)
        p=torch.full((3,2,16,16),.1);valid=torch.ones(16,16,dtype=torch.bool);valid[:8]=False;p[1:,:,:8]=.9
        value,groups=dense_sensitivity(p,valid);self.assertEqual(float(value),0);self.assertEqual(len(groups),2)
        self.assertIsNone(dense_sensitivity(p,torch.zeros_like(valid))[0])
        with self.assertRaises(ValueError):bernoulli_entropy_bits(torch.tensor([1.1]))
    def test_nested_masks_rng(self):
        before=rng();x=torch.ones(1,3,512,512);a,v,info=nested_erasure(x,7);b,w,j=nested_erasure(x,7);close(before,rng(),exact=True);close(a,b,exact=True)
        self.assertTrue(((a[1]==0)<= (a[2]==0)).all());self.assertEqual(info['small_pixels'],161**2);self.assertEqual(info['large_pixels'],228**2);self.assertEqual(int((~v).sum()),228**2);self.assertTrue(x.eq(1).all())
        self.assertEqual(local_seed('erasure',7),int.from_bytes(hashlib.sha256(b'R1:erasure:20260907:7').digest()[:8],'big')%(2**63-1))
    def test_controller_exact_threshold_floor_reset(self):
        c=TrendController();r=c.observe(0.);self.assertEqual(r['age'],1);self.assertEqual(r['best'],1e-6)
        c.age=48;c.ema=c.best=.01;self.assertFalse(c.observe(1.)['trigger']);self.assertTrue(c.observe(1.)['trigger']);self.assertIsNone(c.ema);self.assertEqual(c.age,0)
        c.age=49;c.ema=.01;c.best=.01;self.assertFalse(c.observe(.61)['trigger']);self.assertTrue(c.observe(.611)['trigger'])
        z=TrendController();self.assertTrue(all(not z.observe(.1)['trigger'] for _ in range(200)))
        self.assertEqual([i for i in range(1,1952) if periodic_due(i)],[257,513,769,1025,1281,1537,1793])
    def test_covariance_ready_snapshot_and_projector(self):
        g=torch.Generator().manual_seed(3);X=torch.randn(256,32,dtype=torch.float64,generator=g);X=X/X.norm(dim=1,keepdim=True);b=StreamingSubspace()
        for i,x in enumerate(X[:128].split(8),1):self.assertIsNone(b.snapshot());b.merge(x,i)
        snap=b.snapshot();self.assertEqual(snap[2],16);close(b.M2/127,torch.cov(X[:128].T));close(b.U.T@b.U,torch.eye(8,dtype=torch.float64));self.assertEqual(b.eigh_calls,1)
        for i,x in enumerate(X[128:].split(8),17):
            b.merge(x,i)
            if i<32:close(b.snapshot(),snap,exact=True)
        close(b.mean,X.mean(0));close(b.M2/255,torch.cov(X.T));self.assertEqual(b.version,32);self.assertEqual(b.eigh_calls,2)
        vals,U=torch.linalg.eigh(torch.cov(X.T));close(b.U@b.U.T,U[:,-8:]@U[:,-8:].T)
        self.assertFalse(any(isinstance(v,list) for v in vars(b).values()));self.assertLess(bank_audit(b)['state_bytes'],1024**2)
    def test_zero_rank_rank_cap_and_eigh_failure(self):
        b=StreamingSubspace()
        for i in range(1,17):b.merge(torch.ones(8,32),i)
        self.assertIsNone(b.snapshot());self.assertEqual(b.eigh_calls,1)
        z=StreamingSubspace();x=torch.zeros(8,32);x[:,0]=torch.arange(8)
        for i in range(1,17):z.merge(x,i)
        self.assertEqual(z.U.shape[1],1)
        with patch('torch.linalg.eigh',side_effect=RuntimeError('fixture')):
            for i in range(17,32):z.merge(x,i)
            with self.assertRaises(RuntimeError):z.merge(x,32)
    def test_tokens_overlap_reliable_and_shuffle_quota(self):
        q=torch.zeros(1024,2);q[:100,0]=1;q[:30,1]=1;views=q.expand(6,-1,-1).clone();views[1,3,0]=0
        region,rel=tokens(q,views);self.assertEqual([len(x) for x in region],[924,100,994,30]);self.assertFalse(rel[3]);self.assertEqual(sum(len(i) for i in region),2048)
        f=torch.randn(1024,32);before=rng();m=Memory('REGION');x,ids,a=m.prepare(f,q,views,1);s=Memory('SHUFFLED');y,si,b=s.prepare(f,q,views,1);close(before,rng(),exact=True)
        self.assertEqual(a['selected_region_counts'],b['assigned_bank_counts']);self.assertEqual([len(i) for i in ids],[len(i) for i in si]);self.assertEqual(a['selected_region_counts'],[32,32,32,30]);close(q,views[0],exact=True)
        global_m=Memory('GLOBAL');g,_,_=global_m.prepare(f,q,views,1);close(torch.cat(x).sort(dim=0).values,g[0].sort(dim=0).values)
        zero,_,audit=m.prepare(torch.zeros_like(f),q,views,2);self.assertTrue(all(len(v)==0 for v in zero));self.assertEqual(sum(audit['zero_vectors']),126)
    def test_grid_alignment_and_projection_gradient(self):
        x=torch.arange(512*512).reshape(1,1,512,512).float();g=grid(x);self.assertEqual(float(g[0]),float(x[0,0,8,8]));self.assertEqual(float(g[-1]),float(x[0,0,504,504]))
        m=Memory('REGION');ready(m);snap=m.snapshots(17);f=torch.randn(1024,32,requires_grad=True);ids=[torch.arange(256*i,256*(i+1)) for i in range(4)];loss=m.loss(f,ids,snap);loss.backward();self.assertGreater(float(f.grad.norm()),0)
        c=snap[0][0].clone().requires_grad_();u=snap[0][1].clone().requires_grad_();z=torch.randn(8,32,requires_grad=True);r=projection_residual(z,c,u);r.backward();self.assertIsNone(c.grad);self.assertIsNone(u.grad)
        close(r.detach(),projection_residual(z.detach(),c,-u));self.assertTrue(all(s[2]==16 for s in snap));self.assertRaises(ValueError,m.snapshots,16)


class HostChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(20260907);model,_=build_reference_model('fundus');cls.state=torch.load(os.environ['CHECKPOINT'],map_location='cpu',weights_only=True) if os.environ.get('CHECKPOINT') else copy.deepcopy(model.state_dict())
    def test_full_model_old_C_equivalence(self):
        seed_all(20260907);old=OldC('C',self.state);saved=[]
        for i in range(2):z,a=old.step(pixels('fundus',i));saved.append((z,capture(old),[p.grad.clone() for p in old.params]))
        old.finish(self.state);del old;seed_all(20260907);h=Host('C',self.state)
        for i in range(2):
            z,a=h.step(pixels('fundus',i));close(z,saved[i][0]);close(capture(h),saved[i][1]);close([p.grad for p in h.params],saved[i][2]);close(rng(),saved[i][1]['rng'],exact=True)
        self.assertEqual((len(h.names),sum(p.numel() for p in h.params)),(82,19136));h.finish(self.state)
    def test_real_feature_interface_and_PCA_ready_backward(self):
        seed_all(20260907);h=Host('C_PCA_REGION',self.state);ready(h.memory);h.visit=16;h.counts['base_adam']=16
        original=h.memory.loss;checked=[]
        def checked_loss(fs,ids,snap):
            loss=original(fs,ids,snap);g=torch.autograd.grad(loss,fs,retain_graph=True)[0];checked.append(float(g.norm()));return loss
        with patch.object(h.memory,'loss',side_effect=checked_loss):z,a=h.step(pixels('fundus',0))
        self.assertTrue(checked and checked[0]>0);self.assertEqual(z.shape,(1,2,512,512));self.assertEqual(a['pca']['basis_versions_used'],[16]*4);self.assertGreater(a['pca']['subloss'],0);self.assertEqual(h.pending,{});h.finish(self.state)
    def test_all_six_hosts_physical_and_state_independence(self):
        for arm in ARMS:
            h=toy(arm)
            if h.memory:ready(h.memory);h.visit=16;h.counts['base_adam']=16
            z,a=h.step(pixels('fundus',0));self.assertEqual(a['counts']['forwards'],11 if arm=='C_SENS' else 8);self.assertEqual(a['counts']['base_adam'],1);self.assertFalse(z.requires_grad);self.assertEqual(h.pending,{})
            self.assertFalse(any(isinstance(v,torch.Tensor) for v in a.values()))
        h=toy('C_PCA_GLOBAL');other=toy('C_PCA_GLOBAL');h.step(pixels('fundus',0));self.assertEqual(other.memory.banks[0].n,0)
    def test_recovery_affine_adam_rng_and_period(self):
        h=toy('C_PER256');h.step(pixels('fundus',0));rng_before=rng();private=copy.deepcopy(h.rng);ids=[id(p) for p in h.params];calls=h.counts.copy();h.recover()
        self.assertEqual(h.base.state,{});self.assertEqual(h.steps,0);self.assertEqual(h.visit,1);self.assertEqual(h.counts,calls);close(rng_before,rng(),exact=True);close(private,h.rng,exact=True);self.assertEqual(ids,[id(p) for p in h.params])
        for p,v in zip(h.params,h.initial):close(p,v,exact=True);self.assertIsNone(p.grad)
        h.visit=256;h.counts['base_adam']=256;z,a=h.step(pixels('fundus',1));self.assertTrue(a['reset_before_current']);self.assertEqual(a['optimizer_steps_since_reset'],1);self.assertEqual(a['total_adam_calls'],257)
    def test_labels_source_readers_private_rng_and_zero_extra_loss(self):
        a=toy('C');x=pixels('fundus',0);z,_=a.step(x);snapshot=capture(a);evaluate(z.sigmoid(),torch.zeros_like(z),'fundus');evaluate(z.sigmoid(),torch.ones_like(z),'fundus');close(snapshot,capture(a),exact=True)
        with self.assertRaises(TypeError):a.step(x,torch.ones_like(z))
        b=toy('C_PCA_REGION')
        ready(b.memory);b.visit=16;b.counts['base_adam']=16
        with patch('dpa_ctta.r1.host.SUBSPACE_WEIGHT',0.),patch('dpa_ctta.source_io.read_pixels',side_effect=AssertionError('source reader')),patch('dpa_ctta.source_io.read_mask',side_effect=AssertionError('mask reader')),patch('dpa_ctta.source_io.source_proxy',side_effect=AssertionError('proxy')):
            q,_=b.step(x)
        close(z,q);close(snapshot,capture(b));self.assertEqual(b.pending,{})
        self.assertEqual(segmentation(torch.ones(1,2,3,3)).shape,(1,2,3,3))


class ExecutionChecks(unittest.TestCase):
    def test_disabled_digest_commit_and_matrix(self):
        r={'target':[]};sha='a'*40;d=digest(SCIENCE);auth=dict(enabled=True,approved_code_sha=sha,approved_science_sha256=d,approved_registration_digest=registration_digest(r),external_review_reference='TEST_FIXTURE_ONLY',allowed_physical_gpu_ids=[7],max_workers=1,background_allowed=False)
        for k,v in [('enabled',False),('approved_code_sha','b'*40),('approved_science_sha256','0'*64),('approved_registration_digest','0'*64),('allowed_physical_gpu_ids',[7,7]),('external_review_reference',None)]:
            with self.assertRaises(PermissionError):authorize(dict(auth,**{k:v}),r,sha,d)
        self.assertEqual(authorize(auth,r,sha,d),[7]);m=matrix();self.assertEqual(len(m['jobs']),24);self.assertEqual(sum(j['forwards'] for j in m['jobs']),398004);self.assertEqual(sum(j['adam'] for j in m['jobs']),46824)
    def test_duplicate_claim_and_incomplete_closeout(self):
        with tempfile.TemporaryDirectory() as out:
            claim(out,'o0a0')
            with self.assertRaises(FileExistsError):claim(out,'o0a0')
            with self.assertRaises(FileNotFoundError):recompute(out,{})
            self.assertFalse((Path(out)/'public_aggregate.json').exists())
    def test_prefix_wrong_identity_counts_and_reset(self):
        e=dict(group_id='toy',sample_id='toy',domain='D',subset='remaining_dev');z=torch.zeros(1,2,3,3)
        r=dict(e,arm='C',order=0,global_visit=1,reset_count=0,total_adam_calls=1,optimizer_steps_since_reset=1,segment_age=1,reset_before_current=False,counts=dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0),prediction_fixed_before_label=True,metrics=evaluate(z,z,'fundus'),pca=None)
        validate([r],[e],'C',0)
        for k,v in [('global_visit',2),('total_adam_calls',0),('sample_id','wrong'),('segment_age',2)]:
            with self.assertRaises(ValueError):validate([dict(r,**{k:v})],[e],'C',0)
        with self.assertRaises(ValueError):validate([], [e],'C',0)


class CloseoutChecks(unittest.TestCase):
    def test_actual_default_CLI_refuses_before_weight_or_GPU_access(self):
        with patch('sys.argv',['neutral','--run']),patch('torch.load',side_effect=AssertionError('weight access')):
            with self.assertRaises(PermissionError):main()
        self.assertFalse(torch.cuda.is_initialized())
    def test_complete_scalar_matrix_and_truncation(self):
        from test_r1_fixes import complete_fixture
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with complete_fixture(out) as (reg,packet,ordered):
                result=recompute(out,reg)
                self.assertEqual(result['status'],'R1_EXPERIMENT_COMPLETE');self.assertEqual(result['physical']['records'],288)
                (out/'o3a5/records.jsonl').write_text('')
                with self.assertRaises(ValueError):recompute(out,reg)
                self.assertFalse((out/'public_aggregate.json').exists())
                self.assertFalse(json.loads((out/'current_result.json').read_text())['valid'])
