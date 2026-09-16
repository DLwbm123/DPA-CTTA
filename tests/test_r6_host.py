import copy,json,unittest
from unittest.mock import patch
import torch
from dpa_ctta.r6_regional_consistency.host import Host
from dpa_ctta.r6_regional_consistency import evaluation
from dpa_ctta.r6_regional_consistency.loss import ARMS,PHYSICAL
from dpa_ctta.r1.host import Host as Old
from dpa_ctta.host_diagnostic import close,rng
from dpa_ctta.b4_run import capture
from dpa_ctta.source_pilot import seed_all
from test_r5_host import Toy,state
from test_vptta_host import pixels


def new(arm='C'):
    seed_all(20260907);return Host(arm,model=Toy())


class HostTests(unittest.TestCase):
    def test_actual_q_and_original_C_bitwise(self):
        seed_all(20260907);old=Old('C',model=Toy());h=new();observed=[]
        handle=h.model.register_forward_hook(lambda m,a,z:observed.append(z[0].detach().clone()))
        for i in range(2):
            z,_=old.step(pixels('fundus',i));v,t=h.step(pixels('fundus',i));p=h.take_evaluation()
            close(z,v,exact=True);close(capture(old),capture(h.core),exact=True)
            close(p['pre_logits'],observed[-8],exact=True);close(p['post_logits'],observed[-1],exact=True)
            from dpa_ctta.b1_host import official
            aligned=[observed[-8].cpu()]+[official().Rotate_and_Flip().inverse(z.cpu(),j) for j,z in enumerate(observed[-7:-2])]
            close(p['q'],torch.stack(aligned).sigmoid().mean(0),exact=True);p.clear()
            self.assertEqual(t['counts'],PHYSICAL);self.assertEqual(len(h.handles),2)
        handle.remove()
    def test_each_arm_GT_change_delay_omit_cannot_change_state(self):
        for arm in ARMS:
            a,b,c=new(arm),new(arm),new(arm)
            for i in range(2):
                za,_=a.step(pixels('fundus',i));zb,_=b.step(pixels('fundus',i));zc,_=c.step(pixels('fundus',i))
                with self.assertRaises(ValueError):a.step(pixels('fundus',i))
                before=state(a);random=rng()
                evaluation.score(a.take_evaluation(),torch.zeros(1,2,512,512));evaluation.score(b.take_evaluation(),torch.ones(1,2,512,512));c.take_evaluation().clear()
                close(random,rng(),exact=True);close(za,zb,exact=True);close(zb,zc,exact=True);close(before,state(a),exact=True);close(state(a),state(b),exact=True);close(state(b),state(c),exact=True)
            for key in ('domain','sample_id','subset','mask'):
                with self.assertRaises(TypeError):a.step(pixels('fundus',0),**{key:None})
    def test_evaluation_release_after_Adam_and_RNG_commit(self):
        h=new('R_BAL');events=[]
        def target(row,kind):
            events.append(kind)
            if kind=='mask':self.assertEqual(h.phase,'IDLE');self.assertEqual(h.core.steps,1);close(h.core.rng,rng(),exact=True)
            return (pixels('fundus',0) if kind=='image' else torch.zeros(1,2,512,512)),{}
        with patch.object(evaluation,'target',side_effect=target),patch.object(evaluation,'sync'):t,e=evaluation.current(h,{})
        self.assertEqual(events,['image','mask']);self.assertTrue(e['transaction_before_GT']);self.assertEqual(set(e['metrics']),{'pre','q','post'})
    def test_nonfinite_state_failure_and_production_no_VJP(self):
        for bad in ('parameter','state'):
            h=new()
            def corrupt(opt,*args):
                if bad=='parameter':h.params[0].data.fill_(float('nan'))
                else:opt.state[h.params[0]]['extra']=float('inf')
            h.base.register_step_post_hook(corrupt)
            with self.assertRaises(ValueError):h.step(pixels('fundus',0))
            self.assertTrue(h.failed);self.assertIsNone(h.payload)
            with self.assertRaises(ValueError):h.step(pixels('fundus',1))
        with patch('torch.autograd.grad',side_effect=AssertionError('VJP forbidden')):
            for arm in ARMS:
                h=new(arm);_,t=h.step(pixels('fundus',0));h.take_evaluation().clear();self.assertEqual(t['counts'],PHYSICAL)
    def test_full_random_ResUNet_old_C_and_four_arms_four_visits(self):
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        seed_all(20260907);model,_=build_reference_model('fundus');weights=copy.deepcopy(model.state_dict());del model
        saved=[];actual={k:0 for k in PHYSICAL}
        for arm in ('OLD',*ARMS):
            seed_all(20260907);h=Old('C',weights,'cpu') if arm=='OLD' else Host(arm,weights,'cpu');core=h if arm=='OLD' else h.core
            self.assertEqual((len(core.params),sum(p.numel() for p in core.params)),(82,19136))
            for i in range(4):
                z,t=h.step(pixels('fundus',i));value=dict(z=z.clone(),state=capture(core))
                if arm=='OLD':saved.append(value)
                elif arm=='C':close(value,saved[i],exact=True)
                if arm!='OLD':h.take_evaluation().clear();self.assertEqual(t['counts'],PHYSICAL)
            for k,j in [('network_forwards','forwards'),('loss_backward_calls','backwards'),('adam_calls','base_adam')]:actual[k]+=core.counts[j]
            h.finish(weights);del h,core
        self.assertEqual(actual,dict(network_forwards=160,loss_backward_calls=20,adam_calls=20,jacobian_vjp_calls=0))
        print('R6_FULL_RANDOM_MODEL '+json.dumps(actual))
    def test_small_model_against_independent_current_objective(self):
        import hashlib
        import torch.nn.functional as F
        from dpa_ctta.b1_host import Host as Core
        for arm in ('R_BAL','R_SCALE','R_SHUFFLE'):
            seed_all(20260907);ref=Core('C',model=Toy());h=new(arm);visit=[0]
            original=ref.opt.cal_consis_loss
            def criterion(z,q):
                visit[0]+=1;w=torch.ones_like(q,device='cpu',dtype=torch.float64)
                for c in range(2):
                    foreground=(q[0,c].detach().cpu()>=.5);nf=int(foreground.sum());nb=foreground.numel()-nf
                    if nf and nb:
                        ratio=min(8.,max(.125,nb/nf));v=foreground.numel()/(ratio*nf+nb)
                        w[0,c].fill_(v);w[0,c][foreground]=ratio*v
                w=w.to(z.dtype);res=(z.detach().sigmoid()-q).cpu().double();loss=F.binary_cross_entropy_with_logits(z,q,reduction='none')
                if arm=='R_BAL':return (w.to(z.device)*loss).mean()
                s0=res.square().sum((2,3));sw=(res*w.double()).square().sum((2,3))
                if arm=='R_SCALE':
                    scale=torch.ones_like(sw);active=s0>0;scale[active]=(sw[active]/s0[active]).sqrt()
                    return (loss.mean((2,3))*scale.to(z.device,z.dtype)).mean()
                shuffled=w.clone()
                for c in range(2):
                    seed=int.from_bytes(hashlib.sha256(f'R6_WEIGHT_PERM_V1|20260907|{visit[0]}|{c}'.encode()).digest()[:8],'big')%(2**63)
                    perm=torch.randperm(w[0,c].numel(),generator=torch.Generator().manual_seed(seed));shuffled[0,c]=w[0,c].flatten()[perm].reshape_as(w[0,c])
                sp=(res*shuffled.double()).square().sum((2,3));scale=torch.ones_like(sp);active=s0>0;scale[active]=(sw[active]/sp[active]).sqrt()
                return (scale.to(z.device,z.dtype).reshape(1,2,1,1)*shuffled.to(z.device)*loss).mean()
            ref.opt.cal_consis_loss=lambda data:original(data,criterion=criterion)
            for i in range(2):
                expected,_=ref.step(pixels('fundus',i));actual,t=h.step(pixels('fundus',i));h.take_evaluation().clear()
                close(expected,actual,exact=True);close(capture(ref),capture(h.core),exact=True)
    def test_zero_gradient_still_calls_Adam_with_existing_moments(self):
        h=new('R_BAL');h.step(pixels('fundus',0));h.take_evaluation().clear();before=[p.detach().clone() for p in h.params]
        original=h._criterion
        with patch.object(h,'_criterion',side_effect=lambda z,q:original(z,z.detach().sigmoid())):
            _,trace=h.step(pixels('fundus',1));h.take_evaluation().clear()
        self.assertEqual(trace['bn_gradient_l2'],0.);self.assertEqual(trace['counts'],PHYSICAL);self.assertEqual(h.core.steps,2)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,h.params)))
