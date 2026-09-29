"""R10 atomic complete-round snapshots, fixed endpoints and source-only selection."""
import copy,json
from pathlib import Path
from .learning import Trainer
from .controller import Actor
from .evaluation import validate,choose_checkpoint,deployment_gap
from ..r9_current_first.storage import PhaseJournal,write_json,write_torch,load_torch


def run(trainer,root,guard,failure=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    journal=PhaseJournal(root,'fit',trainer.binding)
    pointsfile=root/'points.json';points=json.loads(pointsfile.read_text()) if pointsfile.exists() else {}
    limit=2000 if trainer.method=='WARM' else 4000
    done=journal.completed()
    if done is None:
        journal.begin(trainer,failure)
        while trainer.steps<limit:
            guard();row=trainer.step();journal.append(row)
            if trainer.method!='WARM' and trainer.steps in (500,1000,2000,4000):
                step=str(trainer.steps);name=f'actor.{step}.pt'
                sha=write_torch(root/name,dict(schema='R10_ACTOR_V1',binding=trainer.binding,actor=trainer.actor.state_dict(),seed=trainer.seed,method=trainer.method,round=trainer.steps))
                evaluation=validate(trainer.source,trainer.actor,trainer.seed,trainer.method=='SUP_STATIC',guard)
                points[step]=dict(artifact=dict(file=name,sha256=sha),validation=evaluation)
                write_json(pointsfile,points)
            if trainer.steps%50==0 or trainer.steps==limit:journal.checkpoint(trainer)
        journal.complete(dict(steps=trainer.steps))
    if done is not None:
        meta=json.loads((journal.root/'latest.json').read_text());trainer.restore(load_torch(journal.root/f"checkpoint.{meta['slot']}.pt",meta['sha256'])['snapshot'])
    if trainer.method=='WARM':
        name='actor.2000.pt';sha=write_torch(root/name,dict(schema='R10_ACTOR_V1',binding=trainer.binding,actor=trainer.actor.state_dict(),seed=trainer.seed,method='WARM',round=2000))
        result=dict(schema='R10_SOURCE_COMPLETE_V1',binding=trainer.binding,method='WARM',steps=2000,artifact=dict(file=name,sha256=sha))
    else:
        choice=choose_checkpoint(points);actor=Actor(trainer.seed)
        saved=load_torch(root/choice['artifact']['file'],choice['artifact']['sha256']);actor.load_state_dict(saved['actor']);actor.requires_grad_(False)
        gap=deployment_gap(trainer.source,actor,trainer.seed,trainer.method=='SUP_STATIC',guard)
        result=dict(schema='R10_SOURCE_COMPLETE_V1',binding=trainer.binding,method=trainer.method,steps=4000,selection=choice,points=points,deployment_gap=gap)
    write_json(root/'complete.json',result);return result
