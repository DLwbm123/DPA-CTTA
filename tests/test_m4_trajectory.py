import copy
import importlib
import unittest
from types import SimpleNamespace,MethodType
import numpy as np
import torch
from torch import nn
from test_m1 import TinyPrompt,TinyBN
from test_m2_episodes import old_fixture
from dpa_ctta.integrations.ctta_suite import checkout_root,native_imports
from dpa_ctta.offline.trajectory_dd import TrajectoryEpisode,tensor_batch,detach_state
from dpa_ctta.offline.adaptation_dd import OfflineEpisode,history_state,preprocess,cpu_tree
from dpa_ctta.medical_losses import medical_loss
from dpa_ctta.host_diagnostic import close
from dpa_ctta.m4_sequences import make_sequences,validate_sequences,target_order


def tiny():
    torch.manual_seed(31)
    with native_imports(checkout_root()[0],'fundus'):
        Memory=importlib.import_module('utils.memory').Memory
    h=SimpleNamespace(prompt=TinyPrompt(),model=nn.Sequential(TinyBN(),nn.Conv2d(3,1,1)),adabn=TinyBN,
        memory_bank=Memory(40,3),neighbor=16)
    h.optimizer=torch.optim.Adam(h.prompt.parameters(),lr=.01,betas=(.9,.99))
    h.model.eval().requires_grad_(False)
    e=TrajectoryEpisode.__new__(TrajectoryEpisode);e.host=h;e.clone=copy.deepcopy(h.model);e.task='fundus';e.device=torch.device('cpu')
    e.counts={k:0 for k in ['live_forwards','proxy_forwards','proxy_images','prompt_forwards','gradient_calls','differentiable_inner','source_visits','memory_pushes','retrievals']}
    e.memory=copy.deepcopy(h.memory_bank);e.memory._prepare_batch=MethodType(tensor_batch,e.memory)
    return e


def inputs():
    torch.manual_seed(71)
    return torch.rand(4,3,8,8), (torch.rand(4,1,8,8)>.5).float(),[(torch.rand(1,3,8,8),(torch.rand(1,1,8,8)>.5).float()) for _ in range(4)]


def warm(e):
    state=e.initial()
    for i in range(16):
        key=np.array([i+1,i*.1+1,i*.3+1],dtype=np.float32).reshape(1,3,1,1)
        e.memory.push(key,torch.ones(1,3,1,1)*(1+i*.01))
    state['memory']=dict(e.memory.memory);state['count']=16
    state['adam']=dict(step=16,exp_avg=torch.ones_like(state['prompt'])*.01,exp_avg_sq=torch.ones_like(state['prompt'])*.02)
    return state


class M4Tests(unittest.TestCase):
    def test_sequences_complete_marginals_and_reverse_blocks(self):
        for task in ['fundus','polyp']:
            old=old_fixture(task);original=copy.deepcopy(old);rows=make_sequences(old,task)
            self.assertTrue(validate_sequences(old,rows));self.assertEqual(rows,make_sequences(old,task));self.assertEqual(old,original)
            bad=copy.deepcopy(rows);bad[3]['original_episode']=bad[2]['original_episode']
            with self.assertRaises(AssertionError):validate_sequences(old,bad)
        rows=[dict(domain='a',sample_id=1),dict(domain='a',sample_id=2),dict(domain='b',sample_id=3)]
        self.assertEqual([r['sample_id'] for r in target_order(rows,1)],[3,1,2])

    def test_tensor_memory_native_selection_weights_overwrite_eviction(self):
        e=tiny();native=copy.deepcopy(e.host.memory_bank)
        torch.manual_seed(9)
        keys=torch.rand(46,3,1,1).numpy();values=torch.rand(46,3,1,1,requires_grad=True)
        for i in range(46):
            native.push(keys[i:i+1],values[i:i+1].detach().numpy().copy());e.memory.push(keys[i:i+1],values[i:i+1])
        self.assertEqual(list(native.memory),list(e.memory.memory));self.assertEqual(len(native.memory),41)
        close(native.get_neighbours(keys[[-1]],16)[0],e.memory.get_neighbours(keys[[-1]],16)[0])
        grad,=torch.autograd.grad(e.memory.get_neighbours(keys[[-1]],16)[0].sum(),values)
        self.assertGreater(float(grad.norm()),0)
        native.push(keys[-2:-1],values[-1:].detach().numpy());e.memory.push(keys[-2:-1],values[-1:])
        self.assertEqual(list(native.memory),list(e.memory.memory))

    def test_single_step_matches_original_M2_objective_and_gradient(self):
        initial,masks,queries=inputs()
        for arm,old in [('D4','D'),('L4','O')]:
            e=tiny();S=initial.clone().requires_grad_();state=history_state(e.host)
            legacy=OfflineEpisode.objective(e,old,S,masks,*queries[0],state);g0,=torch.autograd.grad(legacy,S);e.clear_graphs()
            loss,_,_=e.window(arm,S,masks,queries[:1],e.initial());g1,=torch.autograd.grad(loss,S)
            close(legacy.detach(),loss.detach());close(g0,g1)

    def test_four_step_native_forward_cold_and_warm_all_arms(self):
        initial,masks,queries=inputs()
        for iswarm in [False,True]:
            ref=tiny();start=warm(ref) if iswarm else ref.initial();h=ref.host
            h.memory_bank.memory={k:v.detach().numpy().copy() for k,v in start['memory'].items()}
            if start['adam']:h.optimizer.state[h.prompt.data_prompt]={k:torch.tensor(float(v)) if k=='step' else v.clone() for k,v in start['adam'].items()}
            traces=[];source=copy.deepcopy(h.model.state_dict())
            for i,(pixels,label) in enumerate(queries):
                x=preprocess(pixels,'fundus');_,key=h.prompt(x);key=key.detach().numpy()
                phi=h.memory_bank.get_neighbours(key,16)[0] if h.memory_bank.get_size()>=16 else torch.ones_like(h.prompt.data_prompt)
                h.prompt.update(phi);ref.count=start['count']+i+1
                ref.forward(h.model,x,h.prompt.data_prompt)
                host=sum(m.bn_loss for m in h.model.modules() if isinstance(m,TinyBN))
                syn=ref.forward(ref.clone,preprocess(initial,'fundus'),h.prompt.data_prompt,True)
                loss=host+.1*medical_loss(syn,masks,beta_boundary=0).region
                h.optimizer.zero_grad();loss.backward();h.optimizer.step()
                with torch.no_grad():pred=ref.forward(h.model,x,h.prompt.data_prompt)
                h.memory_bank.push(key,h.prompt.data_prompt.detach().numpy().copy())
                traces.append(dict(prompt=h.prompt.data_prompt.detach().clone(),adam=cpu_tree(h.optimizer.state[h.prompt.data_prompt]),memory={k:torch.tensor(v.copy()) for k,v in h.memory_bank.memory.items()},prediction=pred))
            for arm in ['D4','L4','T4']:
                e=tiny();S=initial.clone().requires_grad_();loss,last,trace=e.window(arm,S,masks,queries,start,True)
                for i,(got,want) in enumerate(zip(trace,traces)):
                    close(got['prediction'],want['prediction']);close(got['state']['prompt'],want['prompt'])
                    for k in ['exp_avg','exp_avg_sq']:close(got['state']['adam'][k],want['adam'][k])
                    self.assertEqual(got['state']['adam']['step'],int(want['adam']['step']))
                    self.assertEqual(list(got['state']['memory']),list(want['memory']));close(got['state']['memory'],want['memory'])
                    self.assertEqual(got['state']['count'],start['count']+i+1)
                close(e.host.model.state_dict(),source,exact=True)

    def test_T_L_values_labels_cross_step_paths_and_detach_boundary(self):
        initial,masks,queries=inputs();results={}
        for arm in ['L4','T4']:
            e=tiny();S=initial.clone().requires_grad_();loss,last,trace=e.window(arm,S,masks,queries,warm(e),True)
            grad,=torch.autograd.grad(loss,S);results[arm]=(loss.detach(),grad,trace)
            detached=detach_state(last);close(cpu_tree(last),cpu_tree(detached),exact=True)
            self.assertIsNone(detached['adam']['exp_avg'].grad_fn)
            self.assertTrue(all(v.grad_fn is None for v in detached['memory'].values()))
            e.clear_graphs()
        close(results['L4'][0],results['T4'][0]);close(results['L4'][2],results['T4'][2])
        self.assertGreater(float((results['T4'][1]-results['L4'][1]).norm()),1e-8)
        for arm in ['L4','T4']:
            e=tiny();S=initial.clone().requires_grad_();start=warm(e)
            first,state,pred=e.step(arm,S,masks,*queries[0],start)
            second,_,_=e.step(arm,initial.clone().requires_grad_(),masks,*queries[1],state)
            cross,=torch.autograd.grad(second,S,allow_unused=True)
            if arm=='L4':self.assertIsNone(cross)
            else:self.assertGreater(float(cross.norm()),0)
        e=tiny();S=initial.clone().requires_grad_();start=warm(e)
        l1,_,t1=e.window('T4',S,masks,queries,start,True);e.clear_graphs()
        l2,_,t2=e.window('T4',S,masks,[(x,1-y) for x,y in queries],start,True)
        self.assertNotEqual(float(l1),float(l2))
        for a,b in zip(t1,t2):close(a['state'],b['state'],exact=True);close(a['prediction'],b['prediction'],exact=True)

    def test_full_120_visits_only_stream_start_resets(self):
        initial,masks,queries=inputs();e=tiny();state=e.initial();S=initial.clone().requires_grad_()
        for w in range(30):
            loss,state,_=e.window('L4',S,masks,queries,state);torch.autograd.grad(loss,S)
            state=detach_state(state);e.clear_graphs()
            self.assertEqual(state['count'],(w+1)*4);self.assertEqual(state['adam']['step'],(w+1)*4)
        self.assertEqual(e.initial()['count'],0);self.assertEqual(e.counts['source_visits'],120)

    def test_paired_report_and_new_record_validation(self):
        import tempfile
        from pathlib import Path
        from dpa_ctta.m4_analysis import summarize,render_report
        from dpa_ctta.m4_run import validate_new_arm
        from test_source_pilot_release import complete_fixture
        fixture,expected=complete_fixture()
        rows=copy.deepcopy(fixture['B'])
        for row in rows:row.update(task='polyp',arm='T4',counts=dict(online_adam=1,memory_pushes=1))
        self.assertTrue(validate_new_arm(rows,expected,'polyp','T4'))
        bad=copy.deepcopy(rows);bad[0]['counts']['memory_pushes']=0
        with self.assertRaises(ValueError):validate_new_arm(bad,expected,'polyp','T4')
        def records(value):return [dict(domain='fixture',metrics=[dict(channel='polyp',dice=value,assd=1.,gt_empty=False,gt_full=False,pred_empty=False,pred_full=False)])]
        result=summarize({a:records(v) for a,v in dict(N=.1,A=.2,R=.2,D2=.3,O2=.4,O3=.5,D4=.6,L4=.7,T4=.8).items()},'polyp')
        self.assertAlmostEqual(result['task_comparisons_pp']['T4-L4'],10.)
        self.assertAlmostEqual(result['task_comparisons_pp']['T4-O3'],30.)
        with tempfile.TemporaryDirectory() as tmp:
            public=dict(status='FIXTURE',source={'polyp':result},target={'order0':{'polyp':result},'order1':{'polyp':result}},training={})
            audit=dict(execution_commit='fixture',scoring_cost={},updates_and_new_records={},gpu_seconds=0,private_output_bytes=0,reused_records_display_occurrences=0)
            render_report(Path(tmp),public,audit)
            text=(Path(tmp)/'M4_EXPERIMENT_REPORT.md').read_text()
            self.assertIn('T4-L4',text);self.assertIn('order1',text)
