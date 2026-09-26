"""One full 16k fit, checkpoint-isolated calibration and both validation conditions."""
import copy
import json
from pathlib import Path
from ..r8_ba.methods import CurrentMLP
from ..r7_shared.numerics import COUNTS
from .protocol import CHECKPOINTS,digest
from .storage import PhaseJournal,write_torch,load_torch,write_json
from .validation import Calibrator,episode
from .selection import checkpoint


class Validation:
    def __init__(self,segmenter,method,data,oracles,aligned,guard):
        self.segmenter,self.method,self.data,self.oracles=segmenter,method,data,oracles
        self.aligned,self.guard=aligned,guard;self.steps=0;self.rows=[]
    def step(self):
        row=episode(self.segmenter,self.method,self.data,self.oracles,self.steps,self.aligned,self.guard)
        self.rows.append(row);self.steps+=1;return row
    def snapshot(self):return dict(steps=self.steps,rows=self.rows,aligned=self.aligned)
    def restore(self,s):
        if s['aligned']!=self.aligned or s['steps']!=len(s['rows']) or not 0<=s['steps']<=64:raise ValueError('R9 val snapshot')
        self.steps=s['steps'];self.rows=copy.deepcopy(s['rows'])


def drive(worker,journal,limit,guard,failure=None,after_step=lambda _:None):
    journal.begin(worker,failure)
    while worker.steps<limit:
        guard();before=COUNTS.copy();row=worker.step() if hasattr(worker,'step') else worker.fit_step()
        if hasattr(worker,'optimizer'):
            import torch
            grads=[p.grad.detach().double().square().sum() for group in worker.optimizer.param_groups for p in group['params'] if p.grad is not None]
            row['gradient_norm_after_clip']=float(torch.stack(grads).sum().sqrt()) if grads else 0.
        journal.append(dict(step=worker.steps,trace=row,counts=dict(COUNTS-before)))
        after_step(worker)
        if worker.steps%50==0 or worker.steps==limit:journal.checkpoint(worker)
        guard()


def run(trainer,root,cal_oracles,val_oracles,guard,failure=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);identity=dict(binding=trainer.binding,recipe=trainer.recipe)
    selected=root/'fit_points.json';points=json.loads(selected.read_text()) if selected.exists() else {}
    used=False
    def recover(journal):
        nonlocal used
        if journal.root.exists() and not journal.completed():
            if used:raise ValueError('one recovery per job')
            used=True;return failure
        return None
    def save(worker):
        if worker.steps in CHECKPOINTS:
            path=root/f'fit.{worker.steps}.pt';state=worker.snapshot()
            sha=write_torch(path,state);points[str(worker.steps)]=sha;write_json(selected,points)
    fit=PhaseJournal(root,'fit',identity)
    if fit.completed() is None:
        if trainer.steps==4000:save(trainer)
        drive(trainer,fit,16000,guard,recover(fit),save)
        fit.complete(dict(steps=16000,points=points,parent=trainer.parent_snapshot))
    if set(points)!=set(map(str,CHECKPOINTS)):raise ValueError('R9 all four source fit archives required')
    validations={};deployed={}
    for step in CHECKPOINTS:
        state=load_torch(root/f'fit.{step}.pt',points[str(step)])
        if state['fit']['binding']!=trainer.binding or state['recipe']!=trainer.recipe:raise ValueError('fit archive identity')
        method=copy.deepcopy(trainer.method);method.load_state_dict(state['fit']['method']);method.freeze()
        for condition in ('uncal_aligned','uncal_legacy'):
            journal=PhaseJournal(root,f'{condition}.{step}',dict(identity,point=points[str(step)]))
            done=journal.completed()
            if done is None:
                worker=Validation(trainer.segmenter,method,trainer.data,val_oracles,condition.endswith('aligned'),guard)
                drive(worker,journal,64,guard,recover(journal));done=journal.complete(worker.rows)
        if not isinstance(method,CurrentMLP):
            cal=Calibrator(trainer.segmenter,method,trainer.data,cal_oracles,trainer.binding)
            journal=PhaseJournal(root,f'cal.{step}',dict(identity,point=points[str(step)]));done=journal.completed()
            if done is None:
                drive(cal,journal,256,guard,recover(journal));cal.method.freeze()
                sha=write_torch(root/f'cal.{step}.pt',dict(schema='R9_DEPLOYMENT_METHOD_V1',binding=trainer.binding,step=step,method=cal.method.state_dict(),method_digest=cal.method.digest()))
                done=journal.complete(dict(steps=256,sha256=sha))
            state=load_torch(root/f'cal.{step}.pt',done['result']['sha256']);method.load_state_dict(state['method']);method.freeze()
            deployed[str(step)]=dict(file=f'cal.{step}.pt',sha256=done['result']['sha256'],step=step)
        else:
            deployed[str(step)]=dict(file=f'fit.{step}.pt',sha256=points[str(step)],step=step)
        for condition in ('aligned','legacy'):
            journal=PhaseJournal(root,f'{condition}.{step}',dict(identity,artifact=deployed[str(step)]));done=journal.completed()
            if done is None:
                worker=Validation(trainer.segmenter,method,trainer.data,val_oracles,condition=='aligned',guard)
                drive(worker,journal,64,guard,recover(journal));done=journal.complete(worker.rows)
            if condition=='aligned':validations[step]=done['result']
    choice=checkpoint(validations)
    result=dict(schema='R9_SOURCE_COMPLETE_V1',identity=identity,source_seed=trainer.source_seed,fit_steps=16000,
                selection=choice,artifacts=deployed,fit_points=points)
    write_json(root/'complete.json',result);return result
