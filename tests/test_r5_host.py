import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from dpa_ctta.r5_update_acceptance.host import Host,snapshot,rollback
from dpa_ctta.r5_update_acceptance import evaluation
from dpa_ctta.r5_update_acceptance.rule import PHYSICAL
from dpa_ctta.r5_update_acceptance.plan import science
from dpa_ctta.r1.host import Host as Old
from dpa_ctta.host_diagnostic import close,rng
from dpa_ctta.b4_run import capture
from dpa_ctta.source_pilot import seed_all
from test_vptta_host import pixels


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,4,1);self.bn=torch.nn.BatchNorm2d(4);self.head=torch.nn.Conv2d(4,2,1)
    def forward(self,x):return self.head(self.bn(self.conv(x))),[],None


def new(arm='C',**kw):
    seed_all(20260907);return Host(arm,model=Toy(),**kw)

def state(h):return dict(params=[p.clone() for p in h.params],adam=copy.deepcopy(h.base.state_dict()),rng=copy.deepcopy(h.core.rng))


class HostTests(unittest.TestCase):
    def test_toy_all_accept_old_C_exact_and_capture_contract(self):
        seed_all(20260907);old=Old('C',model=Toy());h=new();expected=[]
        for i in range(3):
            x=pixels('fundus',i);z,_=old.step(x);v,t=h.step(x);payload=h.take_evaluation()
            close(z,v,exact=True);close(capture(old),capture(h.core),exact=True)
            self.assertEqual(t['counts'],PHYSICAL);self.assertEqual(h.core.counts['forwards'],8*(i+1))
            for p in payload.values():self.assertFalse(p.requires_grad)
            self.assertEqual(v.dtype,torch.float32);self.assertEqual(v.shape,(1,2,512,512));payload.clear()
    def test_actual_q_target_pre_and_six_view_alignment(self):
        h=new();observed=[]
        handle=h.model.register_forward_hook(lambda m,args,z:observed.append(z[0].detach().clone()))
        original=h._criterion;target=[]
        def criterion(z,q):target.append(q.detach().clone());return original(z,q)
        with patch.object(h,'_criterion',side_effect=criterion):v,t=h.step(pixels('fundus',0))
        payload=h.take_evaluation();close(payload['pre_logits'],observed[0],exact=True);close(payload['trial_logits'],observed[7],exact=True);close(payload['q'],target[0],exact=True)
        from dpa_ctta.b1_host import official
        aligned=[observed[0].cpu()]+[official().Rotate_and_Flip().inverse(z.cpu(),i) for i,z in enumerate(observed[1:6])]
        close(payload['q'],torch.stack(aligned).sigmoid().mean(0),exact=True);self.assertEqual(len(observed),8);handle.remove();payload.clear()
    def test_reject_first_and_existing_Adam_state_next_visit_reference(self):
        for initialized in (False,True):
            h=new('C_VERIFY');reference=new('C_VERIFY')
            if initialized:
                h.step(pixels('fundus',0));h.take_evaluation().clear();reference.step(pixels('fundus',0));reference.take_evaluation().clear()
            before=state(h);ids=[id(p) for p in h.params];opt_id=id(h.base);groups=h.core.opt.param_groups
            saved=snapshot(reference.core);decide=h.rule.decide
            def reject(obs):
                z=decide(obs);z['accept']=False;return z
            with patch.object(h.rule,'decide',side_effect=reject):emit,t=h.step(pixels('fundus',1))
            payload=h.take_evaluation();close(emit,payload['pre_logits'],exact=True);payload.clear()
            # Independent reference: run candidate with original C, then explicit full restore.
            reference.step(pixels('fundus',1));reference.take_evaluation().clear();rollback(reference.core,saved)
            reference.totals['n_committed']-=1;reference.totals['n_rejected']+=1;reference.totals['parameter_restorations']+=1
            close(before['params'],[p for p in h.params],exact=True);close(before['adam'],h.base.state_dict(),exact=True)
            self.assertEqual(ids,[id(p) for p in h.params]);self.assertEqual(opt_id,id(h.base));self.assertIs(groups,h.base.param_groups)
            self.assertTrue(all(p.grad is None for p in h.params));self.assertEqual(t['totals']['n_rejected'],1)
            self.assertEqual(t['counts'],PHYSICAL);self.assertGreater(h.rule.visit,0)
            a,_=h.step(pixels('fundus',2));b,_=reference.step(pixels('fundus',2));h.take_evaluation().clear();reference.take_evaluation().clear()
            close(a,b,exact=True);close(state(h),state(reference),exact=True)
    def test_snapshot_buffers_extra_state_groups_and_no_alias(self):
        h=new('C_VERIFY');h.model.register_buffer('audit',torch.tensor([1.]))
        saved=snapshot(h.core);p=h.params[0];h.base.state[p]={'step':torch.tensor(2.),'extra':{'x':torch.ones(2)}};h.base.param_groups[0]['custom']={'x':[1]};h.model.audit.add_(3)
        with torch.no_grad():p.add_(1)
        rollback(h.core,saved);self.assertEqual(h.base.state,{});self.assertNotIn('custom',h.base.param_groups[0]);self.assertEqual(float(h.model.audit),1.)
        close(p,saved['params'][0],exact=True)
        h.base.state[p]={'step':torch.tensor(2.),'extra':{'x':torch.ones(2)}};h.base.param_groups[0]['custom']={'values':[7]}
        saved=snapshot(h.core);h.base.state[p]['extra']['x'].add_(4);h.base.param_groups[0]['custom']['values'][0]=9
        rollback(h.core,saved);self.assertEqual(h.base.param_groups[0]['custom'],{'values':[7]});self.assertTrue(torch.equal(h.base.state[p]['extra']['x'],torch.ones(2)))
        h.base.state[p]['extra']['x'].add_(1);self.assertTrue(torch.equal(saved['state'][p]['extra']['x'],torch.ones(2)))
    def test_half_only_lr_and_all_physical_counts(self):
        h=new('C_HALF');self.assertEqual(h.base.param_groups[0]['lr'],5e-5)
        for arm in ('C','C_HALF','C_RANDOM','C_VERIFY'):
            h=new(arm,**({'p_accept':0.5} if arm=='C_RANDOM' else {}))
            _,t=h.step(pixels('fundus',0));h.take_evaluation().clear();self.assertEqual(t['counts'],PHYSICAL);self.assertEqual(t['totals']['n_committed'],1)
    def test_evaluation_isolation_delayed_omitted_identity_rejected(self):
        a=new();b=new();c=new()
        for i in range(2):
            za,_=a.step(pixels('fundus',i));zb,_=b.step(pixels('fundus',i));zc,_=c.step(pixels('fundus',i))
            with self.assertRaises(ValueError):a.step(pixels('fundus',i))
            sa=state(a);oldrng=rng();evaluation.score(a.take_evaluation(),torch.zeros(1,2,512,512));evaluation.score(b.take_evaluation(),torch.ones(1,2,512,512));c.take_evaluation().clear()
            close(rng(),oldrng,exact=True);close(za,zb,exact=True);close(zb,zc,exact=True);close(sa,state(a),exact=True);close(state(a),state(b),exact=True);close(state(a),state(c),exact=True)
        with self.assertRaises(TypeError):a.step(pixels('fundus',0),domain='not_allowed')
    def test_evaluator_reads_GT_after_all_commits(self):
        h=new();events=[]
        def target(row,kind):
            events.append(kind)
            if kind=='mask':self.assertEqual(h.phase,'IDLE');self.assertIsNone(h.payload);self.assertEqual(h.rule.visit,1);self.assertEqual(h.totals['n_committed'],1)
            return (pixels('fundus',0) if kind=='image' else torch.zeros(1,2,512,512)),dict(bytes=1,read_verify_decode_seconds=0.)
        with patch.object(evaluation,'target',side_effect=target),patch.object(evaluation,'sync'):t,e=evaluation.current(h,{})
        self.assertEqual(events,['image','mask']);self.assertTrue(e['transaction_before_GT'])
    def test_nonfinite_state_hard_failure_no_resume(self):
        h=new();p=h.params[0]
        h.base.register_step_post_hook(lambda *a:p.data.fill_(float('nan')))
        with self.assertRaises(ValueError):h.step(pixels('fundus',0))
        self.assertTrue(h.failed);self.assertEqual(h.phase,'FAILED');self.assertIsNone(h.payload)
        with self.assertRaises(ValueError):h.step(pixels('fundus',1))
        h=new();h.base.register_step_post_hook(lambda opt,*a:opt.state[h.params[0]].update(extra=float('inf')))
        with self.assertRaises(ValueError):h.step(pixels('fundus',0))
        self.assertTrue(h.failed)
    @unittest.skipIf(os.environ.get('R5_FAST')=='1','explicit fast-only pass; full network in required final suite')
    def test_full_random_ResUNet_old_C_diag_verify_random_p1_four_steps(self):
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        seed_all(20260907);model,_=build_reference_model('fundus');weights={k:v.detach().clone() for k,v in model.state_dict().items()};del model
        old_outputs=[];actual={k:0 for k in PHYSICAL}
        for arm in ('OLD','C','C_VERIFY','C_RANDOM'):
            seed_all(20260907);h=Old('C',weights,'cpu') if arm=='OLD' else Host(arm,weights,'cpu',p_accept=1 if arm=='C_RANDOM' else None)
            self.assertEqual(sum(p.numel() for p in h.params),19136)
            for i in range(4):
                z,t=h.step(pixels('fundus',i));core=h if arm=='OLD' else h.core;value=dict(z=z.clone(),state=capture(core))
                if arm=='OLD':old_outputs.append(value)
                else:close(value,old_outputs[i],exact=True);h.take_evaluation().clear()
                for k,v in PHYSICAL.items():actual[k]+=v
            h.finish(weights);del h,core
        self.assertEqual(actual,dict(network_forwards=128,loss_backward_calls=16,adam_calls=16,jacobian_vjp_calls=0))
        print('FULL_MODEL_CPU '+json.dumps(dict(actual=actual,multi_step_exact_parity=True,checkpoint_reads=0)))
