"""Procedural current tensors and synthetic bounded states only; no real assets."""
import copy,json,unittest
from unittest.mock import patch
import torch
from torch import nn
from dpa_ctta.r3 import kernels as k,plan,run
from dpa_ctta.r3.host import Host,ARMS
from dpa_ctta.r3.stats import Memory
from dpa_ctta.r3.contexts import Contexts,cpu_copy
from dpa_ctta.r3.displacement import flatten,probe_rows,transform
from dpa_ctta.r3.transport import commit as transport
from dpa_ctta.r3.teachers import density_target,graph_target,correction
from dpa_ctta.r1.host import Host as OldHost
from dpa_ctta.r1.region_memory import Memory as OldMemory,grid
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.host_diagnostic import rng,close
from dpa_ctta.b4_run import capture
from test_r1 import Toy
from test_vptta_host import pixels


from dpa_ctta.r3.testing import ready


def model():
    m=Toy()
    with torch.no_grad():
        m.seg_head.weight.zero_();m.seg_head.bias.fill_(-3)
        m.seg_head.weight[0,0]=10;m.seg_head.weight[1,1]=10
    return m


def host(arm,cls=Host):
    seed_all(20260907);return cls(arm,model=model())


def ready_host(h):
    if h.contexts:
        load=h.contexts.load
        def prepare(d):
            m,trace=load(d)
            if trace['created']:ready(m)
            return m,trace
        h.contexts.load=prepare
    elif h.memory is not None:ready(h.memory)


class HostTests(unittest.TestCase):
    def test_C_RP_two_step_old_path_parity(self):
        for arm,oldarm in [('C','C'),('RP','C_PCA_REGION')]:
            old=host(oldarm,OldHost);new=host(arm)
            if arm=='RP':ready(old.memory);ready(new.memory)
            for t in range(2):
                x=pixels('fundus',t);a,ta=old.step(x);b,tb=new.step(x)
                close(a,b,exact=True);close(capture(old),capture(new.legacy),exact=True)
                self.assertEqual(tb['r3']['counts']['network_forwards'],8)

    def test_all_families_actual_cold_and_ready_paths(self):
        for arm in ARMS[2:]:
            with self.subTest(arm=arm):
                for warm in (False,True):
                    h=host(arm)
                    if warm:ready_host(h)
                    frozen=None if h.reference is None else copy.deepcopy(h.reference.state_dict())
                    x=pixels('fundus',0);z,r=h.step(x);a=r['r3'];c=a['counts']
                    self.assertEqual(c['network_forwards'],9 if arm.startswith(('T_','S_')) else 8)
                    self.assertEqual(c['adam_calls'],1);self.assertEqual(c['loss_backward_calls'],1)
                    self.assertFalse(z.requires_grad);self.assertEqual(h.pending,{});self.assertEqual(h.phase,'IDLE')
                    self.assertEqual(c['jacobian_vjp_calls'],8 if arm.startswith('U_') and warm else 0)
                    if arm.startswith('U_') and warm:
                        self.assertEqual(a['update']['parameter_dim'],64);self.assertGreater(a['update']['rows'],0)
                        self.assertLessEqual(a['update']['new_norm'],a['update']['base_norm']+1e-10)
                        self.assertEqual(c['actual_parameter_replacements'],1)
                    if arm.startswith('T_'):self.assertEqual(a['teacher']['pair_ready'],[warm]*2)
                    if arm.startswith('G_'):
                        self.assertEqual(a['graph']['ready'],warm)
                        if warm:self.assertEqual(a['graph']['teacher_violations_after'],0)
                    if arm.startswith('M_'):self.assertEqual(a['transport']['frame_version'],1)
                    if arm.startswith('S_'):self.assertEqual(a['context']['active_slots'],1)
                    if frozen is not None:close(frozen,h.reference.state_dict(),exact=True)
                    json.dumps(a,allow_nan=False)

    def test_cold_no_loss_branches_match_C_and_augmentation_rng(self):
        old=host('C',OldHost);z,_=old.step(pixels('fundus',0));expected=capture(old)
        for arm in ARMS[2:]:
            h=host(arm);out,_=h.step(pixels('fundus',0))
            close(out,z,exact=True);close([p.detach() for p in h.params],expected['affine'],exact=True)
            close(h.rng,expected['rng'],exact=True)

    def test_prediction_before_merge_labels_private_and_failure_cleanup(self):
        for arm in ('T_LR','U_PCA','S_JOINT','M_TRANSPORT','G_PCA'):
            h=host(arm);events=[]
            original=Memory.merge
            def checked(memory,vectors,visit):
                self.assertEqual(h.phase,'COMMIT');self.assertEqual(h.total['adam_calls'],1);events.append(visit)
                return original(memory,vectors,visit)
            with patch.object(Memory,'merge',checked):h.step(pixels('fundus',0))
            self.assertEqual(events,[1])
            with self.assertRaises(TypeError):h.step(pixels('fundus',0),domain='forbidden')
        h=host('T_LR')
        with patch('dpa_ctta.r3.host.density_target',side_effect=ValueError('injected geometry failure')):
            with self.assertRaises(ValueError):h.step(pixels('fundus',0))
        self.assertEqual(h.pending,{});self.assertTrue(h.failed)
        with self.assertRaises(RuntimeError):h.step(pixels('fundus',0))

    def test_S_host_switch_does_not_replay_augmentation_or_copy_model(self):
        old=host('C',OldHost);h=host('S_JOINT');identity=(id(h.model),id(h.reference))
        descriptors=torch.eye(64,dtype=torch.float64)
        def fixture_descriptor(module,args,output):
            h.pending['descriptor']=descriptors[1 if h.visit==17 else 0]
        handle=h.reference.register_forward_hook(fixture_descriptor)
        try:
            for visit in range(1,19):
                old.step(pixels('fundus',visit%2));expected_rng=rng()
                _,r=h.step(pixels('fundus',visit%2));close(h.rng,expected_rng,exact=True)
                self.assertEqual((id(h.model),id(h.reference)),identity)
                self.assertEqual(r['r3']['counts']['adam_calls'],1)
                if visit==17:
                    self.assertTrue(r['r3']['context']['created']);self.assertEqual(r['r3']['context']['slot'],1)
                    saved=cpu_copy(h.contexts.slots[1]['adam'])
                if visit==18:
                    self.assertEqual(r['r3']['context']['slot'],0);self.assertTrue(r['r3']['context']['switched'])
                    self.assertEqual(r['r3']['context']['slot_updates'],[17,1]);close(saved,h.contexts.slots[1]['adam'],exact=True)
        finally:handle.remove()

    def test_programmatic_full_ResUNet_38_updates_all_families(self):
        import tempfile,os
        from pathlib import Path
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        from dpa_ctta.r3.execution import smoke,backend_policy
        seed_all(20260907);m,_=build_reference_model('fundus')
        with torch.no_grad():m.seg_head.weight.mul_(30)
        state=copy.deepcopy(m.state_dict());del m
        original=backend_policy()
        try:
            torch.use_deterministic_algorithms(False,warn_only=True)
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ),patch('dpa_ctta.b3_runtime.process_audit',side_effect=AssertionError('CPU smoke must not query GPU processes')):
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
                before=backend_policy()
                smoke(state,'cpu',Path(tmp),dict(fixture='PROGRAMMATIC_CPU_NOT_EXECUTION_AUTHORIZATION'),dict(seed=20260907,**before),dict(bytes=1))
                proof=json.loads((Path(tmp)/'smoke.completion.json').read_text())
                self.assertEqual(backend_policy(),before)
                self.assertEqual(proof['backend'],dict(seed=20260907,**before))
                self.assertEqual(proof['paired_comparison_backend'],dict(deterministic_algorithms=True,warn_only=False,cublas_workspace_config=None))
                print('SMOKE_CPU_BACKEND '+json.dumps(dict(before=before,comparison=proof['paired_comparison_backend'],restored=backend_policy()),sort_keys=True))
        finally:torch.use_deterministic_algorithms(original['deterministic_algorithms'],warn_only=original['warn_only'])
        total=proof['physical'];traces=proof['evidence']
        self.assertEqual(total['network_forwards'],316);self.assertEqual(total['adam_calls'],38)
        self.assertEqual(total['loss_backward_calls'],38);self.assertLessEqual(total['jacobian_vjp_calls'],24)
        for arm in ('U_PCA','U_RAND','U_SCALE'):
            self.assertGreater(traces[arm+':1']['counts']['jacobian_vjp_calls'],0)
            self.assertEqual(traces[arm+':1']['update']['parameter_dim'],19136)
        print('FULL_RESUNET_CPU_38_UPDATE_COUNTS '+json.dumps(total,sort_keys=True))
        if os.environ.get('TRACE_OUTPUT'):Path(os.environ['TRACE_OUTPUT']).write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
        self.assertFalse(torch.cuda.is_initialized())


class MechanismTests(unittest.TestCase):
    def test_sampling_equal_R1_and_local_rng(self):
        q=torch.zeros(1024,2);q[:500,0]=1;q[:100,1]=1;views=q.expand(6,-1,-1)
        raw=torch.randn(1024,32);before=rng();m=Memory()
        selected,ids,v,a=m.select(raw,q,views,7);vold,iold,aold=OldMemory('REGION').prepare(raw,q,views,7)
        close(v,vold,exact=True);close(ids,iold,exact=True);self.assertEqual(a,aold);close(before,rng(),exact=True)
        close(v,[k.unit(raw)[i].double() for i in selected],exact=True)
        self.assertEqual(float(grid(torch.arange(512*512).reshape(1,1,512,512).float())[0]),4104.)

    def test_Jacobian_finite_difference_unused_columns_and_no_32D_parameter_projection(self):
        m=model().double();from dpa_ctta.b1_host import configure
        names,ps=configure(m);x=torch.randn(1,3,32,32,dtype=torch.float64);raw=grid(m(x)[2]);memory=Memory();ready(memory)
        ids=[torch.arange(32)]*4;snap=memory.snapshots(1)
        A,defs,values=probe_rows(raw,ids,snap,ps)
        before=flatten(ps);g=torch.Generator().manual_seed(71);d=torch.randn(before.shape,generator=g,dtype=torch.float64);d=d/d.norm();eps=1e-5
        from dpa_ctta.r3.displacement import replace
        vals=[]
        for sign in (-1,1):
            replace(ps,before+sign*eps*d);v=k.unit(grid(m(x)[2]));vals.append(torch.stack([v[i].mean(0)@u for i,u in defs]).detach())
        torch.testing.assert_close((vals[1]-vals[0])/(2*eps),A@d,atol=2e-7,rtol=2e-5)
        self.assertEqual(A.shape,(8,64))
        with self.assertRaises(ValueError):k.constrain_displacement(before,torch.ones(8,32))

    def test_actual_Adam_moments_and_norm_control(self):
        for mode in ('PCA','RAND','SCALE'):
            p=nn.Parameter(torch.tensor([1.,2.,3.],dtype=torch.float64));o=torch.optim.Adam([p],lr=1e-4)
            p.square().sum().backward();before=flatten([p]);o.step();moment=copy.deepcopy(o.state_dict());proposal=flatten([p])-before
            A=torch.tensor([[1.,2.,0.]],dtype=torch.float64);protected,_=k.constrain_displacement(proposal,A)
            info=transform(before,[p],A,mode);close(moment,o.state_dict(),exact=True)
            expected=k.norm_matched_displacement(proposal,protected) if mode=='SCALE' else protected
            torch.testing.assert_close(p.detach()-before,expected,atol=1e-15,rtol=1e-10)
            self.assertEqual(int(o.state[p]['step']),1)

    def test_context_small_states_capacity_reuse_and_rng(self):
        for shared in (False,True):
            p=nn.Parameter(torch.ones(3));o=torch.optim.Adam([p],lr=1e-4);pool=Contexts([p],['bn.weight'],o,shared)
            descriptors=torch.eye(64,dtype=torch.float64);initial_rng=rng()
            for index in [0]*16+[1]*16+[2]*16+[3]+[0]:
                m,r=pool.load(descriptors[index]);old=[cpu_copy({k:v for k,v in s.items() if k!='memory'}) for s in pool.slots]
                o.zero_grad();p.square().sum().backward();o.step();pool.commit(descriptors[index],r)
                for j,s in enumerate(pool.slots):
                    if j!=r['slot']:close(old[j],{k:v for k,v in s.items() if k!='memory'},exact=True)
            self.assertEqual(len(pool.slots),3);self.assertEqual(pool.total_steps,50);close(initial_rng,rng(),exact=True)
            self.assertEqual(sum(s['steps'] for s in pool.slots),50)
            self.assertTrue(all(not any(isinstance(v,nn.Module) for v in s.values()) for s in pool.slots))

    def test_transport_all_live_cached_frames_and_post_storage(self):
        x=k.unit(torch.randn(1024,32));Q=torch.linalg.qr(torch.randn(32,32,dtype=torch.float64)).Q;y=x.double()@Q.T
        ids=[torch.arange(32)+i*32 for i in range(4)]
        memories=[]
        for mode in ('TRANSPORT','IDPOST','SHUFFLE'):
            m=Memory();ready(m);old=copy.deepcopy(m);trace=transport(m,x,y,ids,1,mode)
            self.assertEqual([b.n for b in m.banks],[160]*4);self.assertEqual(m.frame_version,1)
            self.assertEqual([b.version for b in m.banks],[0]*4)
            if mode=='IDPOST':close(m.banks[0].center,old.banks[0].center,exact=True)
            else:self.assertFalse(torch.equal(m.banks[0].center,old.banks[0].center))
            self.assertLess(trace['orthogonal_error'],1e-10);memories.append(m)
        self.assertFalse(torch.equal(memories[0].banks[0].M2,memories[2].banks[0].M2))
        m=Memory();ready(m);old=copy.deepcopy(m);Q=torch.linalg.qr(torch.randn(32,32,dtype=torch.float64)).Q;m.transport(Q,1)
        close(m.banks[0].center,Q@old.banks[0].center);close(m.banks[0].U,Q@old.banks[0].U)
        close(m.covariances[0][1],Q@old.covariances[0][1]@Q.T)

    def test_T_high_resolution_zero_pair_ready_and_graph_detached(self):
        q=torch.rand(1,2,512,512);raw=torch.randn(1024,32);m=Memory();ready(m);s=m.density_snapshots(1)
        cold,a=density_target(q,raw,[None]*4,'LR');close(cold,q,exact=True)
        partial,a=density_target(q,raw,[s[0],s[1],None,None],'ISO');close(partial[:,1],q[:,1],exact=True)
        close(correction(q,torch.zeros(1024,2)),q,exact=True)
        q.requires_grad_();raw.requires_grad_();views=grid(q.detach()).expand(6,-1,-1)
        for mode in ('PCA','ISO','ORDER'):
            out,a=graph_target(q,raw,views,s,mode);self.assertFalse(out.requires_grad);self.assertEqual(a['edges'],1984)
            self.assertEqual(a['teacher_violations_after'],0)
        with patch('dpa_ctta.r3.teachers.graph_weights',return_value=torch.zeros(1984,dtype=torch.float64)):
            out,_=graph_target(q,raw,views,s,'PCA')
        control,_=graph_target(q,raw,views,s,'ORDER');close(out,control,exact=True)

    def test_unused_probe_columns_and_zero_A_candidate_exact(self):
        p=nn.Parameter(torch.randn(32,dtype=torch.float64));unused=nn.Parameter(torch.ones(3,dtype=torch.float64))
        raw=p[None,:].expand(1024,-1);m=Memory();ready(m)
        A,_,_=probe_rows(raw,[torch.arange(8)]*4,m.snapshots(1),[p,unused])
        self.assertEqual(A.shape,(8,35));self.assertEqual(float(A[:,-3:].abs().sum()),0)
        o=torch.optim.Adam([p,unused],lr=1e-4);(p.square().sum()+unused.square().sum()).backward()
        before=flatten([p,unused]);o.step();candidate=flatten([p,unused]).clone()
        info=transform(before,[p,unused],torch.zeros(0,35,dtype=torch.float64),'PCA')
        close(candidate,flatten([p,unused]),exact=True);self.assertEqual(info['actual_parameter_replacements'],0)

    def test_current_reader_reused_mask_only_after_R3_commit(self):
        from dpa_ctta.r1 import run as pipeline
        for arm in ('T_LR','U_PCA','S_JOINT','M_TRANSPORT','G_PCA'):
            h=host(arm);events=[]
            def synthetic(row,kind):
                events.append(kind)
                if kind=='mask':
                    self.assertEqual(h.phase,'IDLE');self.assertEqual(h.visit,1)
                    self.assertEqual(h.pending,{})
                value=pixels('fundus',0) if kind=='image' else torch.zeros(1,2,512,512)
                return value,dict(bytes=1,read_verify_decode_seconds=0.)
            with patch.object(pipeline,'target',side_effect=synthetic),patch('dpa_ctta.p1_run.sync'):
                row=pipeline.current(h,{})
            self.assertEqual(events,['image','mask']);self.assertTrue(row['prediction_fixed_before_label'])

    def test_transport_insufficient_support_and_single_token_shuffle(self):
        m=Memory();ready(m);raw=torch.randn(1024,32);s=m.banks[0].center.clone()
        trace=transport(m,raw,raw*2,[torch.tensor([0])]*4,1,'SHUFFLE')
        self.assertTrue(trace['insufficient_support']);self.assertEqual(trace['shuffled_positions'],0)
        close(m.banks[0].center,s,exact=True)

    def test_matrix_and_disabled_entry_before_any_asset_access(self):
        m=plan.matrix();self.assertEqual(len(m['jobs']),85)
        for w in (1,2,3):
            assignments=m['rotation_examples'][str(w)]['assignments']
            self.assertEqual([r['worker'] for r in assignments],[(ARMS.index(j['arm'])+j['order'])%w for j in m['jobs']])
        with patch('sys.argv',['neutral','--run']),patch('torch.load',side_effect=AssertionError('weights')):
            with self.assertRaises(PermissionError):run.main()
