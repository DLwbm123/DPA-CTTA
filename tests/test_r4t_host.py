"""Procedural integration checks for time, ownership, evaluation and counters."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r4_three.host import Host,ARMS
from dpa_ctta.r4_three import evaluation
from dpa_ctta.r1.host import Host as OldHost
from dpa_ctta.r3.testing import ready
from dpa_ctta.source_pilot import seed_all
from dpa_ctta.host_diagnostic import close,rng
from dpa_ctta.b4_run import capture
from test_r3 import model
from test_vptta_host import pixels


def host(arm):
    seed_all(20260907);return Host(arm,model=model())


class KernelToy(torch.nn.Module):
    """Small network with exactly the two prescribed OIHW entry signatures."""
    def __init__(self):
        super().__init__();n=torch.nn
        self.res=n.Module();self.res.conv1=n.Conv2d(3,64,7,stride=2,padding=3,bias=False);self.res.bn1=n.BatchNorm2d(64)
        block=n.Module();block.conv2=n.Conv2d(64,64,3,padding=1,bias=False);block.bn2=n.BatchNorm2d(64)
        self.res.layer1=n.Sequential(n.Identity(),n.Identity(),block)
        self.project=n.Conv2d(64,32,1,bias=False);self.bn=n.BatchNorm2d(32);self.seg_head=n.Conv2d(32,2,1)
    def forward(self,x):
        x=torch.relu(self.res.bn1(self.res.conv1(x)));b=self.res.layer1[2]
        x=torch.relu(b.bn2(b.conv2(x)));f=torch.relu(self.bn(self.project(x)))
        return torch.nn.functional.interpolate(self.seg_head(f),size=(512,512),mode='bilinear',align_corners=False),[],f


class HostTests(unittest.TestCase):
    def test_kernel_coordinates_frozen_zero_gives_complete_C_trajectory(self):
        for arm in ('KDG','K_ALL','K_MAG','K_FREE'):
            seed_all(20260907);old=OldHost('C',model=KernelToy())
            seed_all(20260907);new=Host(arm,model=KernelToy())
            # Explicit test-only ablation. Production retains all coordinates.
            for k in new.kernels.values():k.requires_grad_(False)
            new.params=list(new.bn_params);new.base.param_groups[0]['params']=new.params
            for t in range(2):
                a,_=old.step(pixels('fundus',t));b,_=new.step(pixels('fundus',t));new.take_evaluation()
                close(a,b,exact=True);close([p.detach() for p in old.params],[p.detach() for p in new.params],exact=True)
                close(old.rng,new.rng,exact=True)

    def test_baseline_RP_exact_parity_and_actual_selection(self):
        for arm in ('C','RP'):
            seed_all(20260907);old=OldHost('C' if arm=='C' else 'C_PCA_REGION',model=model());new=host(arm)
            if arm=='RP':ready(old.memory);ready(new.memory)
            for t in range(2):
                a,ta=old.step(pixels('fundus',t));b,tb=new.step(pixels('fundus',t))
                close(a,b,exact=True);close(capture(old),capture(new.legacy),exact=True)
                payload=new.take_evaluation();aux=evaluation.auxiliary(payload,torch.zeros_like(payload['q']))
                self.assertEqual(tb['r4t']['counts']['network_forwards'],8)
                if arm=='RP':self.assertEqual([len(x) for x in payload['selected']],ta['pca']['input']['selected_region_counts'])
                else:self.assertEqual(aux['memory']['status'],'NOT_APPLICABLE')

    def test_teacher_first_step_parity_EMA_timing_and_frozen_teacher(self):
        for arm in ('MT','MT_RP','FT','FT_RP'):
            baseline=host('C');expected,_=baseline.step(pixels('fundus',0));baseline.take_evaluation()
            h=host(arm);initial=[p.clone() for p in h.teacher_params]
            z,row=h.step(pixels('fundus',0));payload=h.take_evaluation()
            close(z,expected);close(h.rng,baseline.rng,exact=True)
            for old,t,p in zip(initial,h.teacher_params,h.bn_params):
                close(t,old*.99+p*.01 if arm.startswith('MT') else old,exact=False)
                self.assertIsNone(t.grad);self.assertFalse(t.requires_grad)
            if h.memory:
                self.assertEqual(row['r4t']['pca']['basis_versions_used'],[None]*4)
                ready(h.memory)
            z,row=h.step(pixels('fundus',1));h.take_evaluation()
            self.assertEqual(row['r4t']['counts']['network_forwards'],9 if h.memory else 8)
            self.assertFalse(z.requires_grad)

    def test_graph_integrated_all_modes_and_empty_band(self):
        for arm in ARMS[10:]:
            h=host(arm);before=rng();z,row=h.step(pixels('fundus',0));payload=h.take_evaluation()
            self.assertEqual(row['r4t']['counts']['network_forwards'],8)
            self.assertEqual(row['r4t']['graph']['steps'],64)
            self.assertTrue(torch.equal(payload['qstar'][~payload['allowed']],payload['q'][~payload['allowed']]))
            self.assertFalse(payload['qstar'].requires_grad)
            aux=evaluation.auxiliary(payload,torch.zeros_like(payload['q']))
            for r in aux['graph']['transitions']:
                self.assertEqual(r['denominator'],sum(r[k] for k in ('wrong_to_correct','correct_to_wrong','correct_unchanged','wrong_unchanged')))
            self.assertIsNone(h.memory)

    def test_mask_read_after_commit_and_payload_release(self):
        for arm in ('RP','MT_RP','FT','G_BOUND'):
            h=host(arm);events=[]
            def read(entry,kind):
                events.append(kind)
                if kind=='mask':
                    self.assertEqual(h.phase,'IDLE');self.assertEqual(h.pending,{})
                    self.assertIsNone(h.evaluation_pending)
                    self.assertEqual(h.visit,1)
                return (pixels('fundus',0) if kind=='image' else torch.zeros(1,2,512,512)),dict(bytes=1,read_verify_decode_seconds=0.)
            with patch.object(evaluation,'target',side_effect=read),patch.object(evaluation,'sync'):
                row=evaluation.current(h,{})
            self.assertEqual(events,['image','mask']);self.assertTrue(row['auxiliary_after_state_commit'])

    def test_auxiliary_known_flips_Brier_and_no_feedback(self):
        q=torch.full((1,2,512,512),.2);qs=q.clone();qs[:,:,0,0]=.8;mask=torch.zeros_like(q);mask[:,:,0,0]=1
        allowed=torch.zeros_like(q,dtype=torch.bool);allowed[:,:,0,0]=True
        payload=dict(q=q,qstar=qs,allowed=allowed,reliable=~allowed,selected=None)
        before={k:v.clone() for k,v in payload.items() if isinstance(v,torch.Tensor)};state=rng()
        a=evaluation.auxiliary(payload,mask)
        self.assertEqual(a['graph']['transitions'][0]['wrong_to_correct'],1)
        self.assertEqual(a['graph']['transitions'][2]['denominator'],1)
        self.assertLess(a['qstar'][0]['brier'],a['q'][0]['brier'])
        for k,v in before.items():close(payload[k],v,exact=True)
        close(rng(),state,exact=True)

    @unittest.skipIf(os.environ.get('R4T_FOCUSED')=='1','full model in complete suite')
    def test_full_model_actual_32_updates_cpu_smoke(self):
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        from dpa_ctta.r4_three.execution import smoke,backend_policy
        seed_all(20260907);m,_=build_reference_model('fundus');state={n:v.detach().clone() for n,v in m.state_dict().items()};del m
        with tempfile.TemporaryDirectory() as tmp,patch('dpa_ctta.b3_runtime.process_audit',side_effect=AssertionError('CPU no GPU audit')):
            smoke(state,'cpu',Path(tmp),dict(fixture='PROGRAMMATIC_CPU'),dict(seed=20260907,**backend_policy()),dict(bytes=1))
            proof=json.loads((Path(tmp)/'smoke.completion.json').read_text())
            self.assertEqual(proof['physical'],dict(network_forwards=260,loss_backward_calls=32,adam_calls=32,jacobian_vjp_calls=0,actual_parameter_replacements=0))
            for arm in ('KDG','K_ALL','K_MAG','K_FREE'):
                row=proof['evidence'][arm+':0'];self.assertEqual(row['trainable_scalars'],dict(KDG=21283,K_ALL=21283,K_MAG=19264,K_FREE=23424)[arm])
                self.assertTrue(all(v['effective_change_l2']>0 for v in row['kernels'].values()))
            if os.environ.get('R4T_TRACE_OUTPUT'):Path(os.environ['R4T_TRACE_OUTPUT']).write_text(json.dumps(proof,indent=2)+'\n')
            print('FULL_MODEL_CPU '+json.dumps(dict(physical=proof['physical'],backend=proof['backend'],comparison=proof['paired_comparison_backend'])))
