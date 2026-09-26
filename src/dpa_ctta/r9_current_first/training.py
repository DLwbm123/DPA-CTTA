"""Deployment-aligned R9 fit; the LEGACY kernel is the untouched R8 trainer."""
import copy
import re
import torch
from ..r8_ba.trainer import SourceTrainer as LegacyTrainer, MAX_STEPS, lr_at, detached_state
from ..r8_ba.schedule import episode_styles, episode_roles
from ..r8_ba.methods import CurrentMLP
from ..r7_shared.source import simulate
from ..r7_shared.numerics import COUNTS, finite, seg_loss, temperature_initial


class SourceTrainer(LegacyTrainer):
    def __init__(self,*args,recipe,**kwargs):
        if MAX_STEPS!=16000:raise ValueError('R9 cannot inherit R8_SCOPE=SCREEN24')
        if recipe not in ('LEGACY','SELF','SELF_TASK'):raise ValueError('R9 fit recipe')
        super().__init__(*args,**kwargs)
        if any(p.requires_grad for p in getattr(self.segmenter,'model',torch.nn.Module()).parameters()):
            raise ValueError('source backbone must be frozen')
        self.recipe=recipe
        self.parent_snapshot=None

    def fit_step(self):
        if self.recipe=='LEGACY':return super().fit_step()
        if self.steps>=16000:raise ValueError('R9 fit exhausted')
        episode,chunk=divmod(self.steps,4)
        if chunk==0:self.state=self.method.initial()
        ids,curriculum,_=episode_styles('fit',episode,self.source_seed)
        roles,reused=episode_roles(self.data.folds['fit'],'fit',episode,ids,self.oracles.support_pairs,self.source_seed)
        losses=[]
        for visit in range(chunk*8,(chunk+1)*8):
            anchor=ids[visit]; _,query_name=roles[visit]
            query=self.data.get(query_name,'fit')
            # Exactly the original query simulator key; no new source roles or RNG path.
            image=simulate(query.image,self.style_bank[anchor],f'R8_FIT_QUERY|{self.source_seed}|{episode}|{visit}|{query_name}')
            with torch.no_grad():
                clean=None
                if self.method.group=='A' and self.recipe!='SELF_TASK':
                    _,cr,ct=self.segmenter(query.image,observe=True)
                _,raw,tokens=self.segmenter(image,observe=True)
                if self.method.group=='A' and self.recipe!='SELF_TASK':clean=self.method.observe(cr,ct)
            state,audit=self.method.update(raw,tokens,self.state)
            self.state=state
            logits=self.segmenter(image,self.method.ambient(state))
            loss=seg_loss(logits,query.label)
            if self.recipe!='SELF_TASK':
                zstar=None if isinstance(self.method,CurrentMLP) else self.method.project(self.oracles.values[:,anchor])
                loss=loss+self.method.fit_loss(audit,clean,zstar,state)
            losses.append(loss)
        loss=torch.stack(losses).mean();finite(loss)
        step=self.steps+1
        for group in self.optimizer.param_groups:group['lr']=lr_at(step)
        self.optimizer.zero_grad(set_to_none=True);loss.backward();COUNTS['source_backward_calls']+=1
        torch.nn.utils.clip_grad_norm_([p for p in self.method.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        self.optimizer.step();COUNTS['source_AdamW']+=1;finite(*self.method.parameters())
        self.steps=step
        self.state=detached_state(self.state) if step%4 else self.method.initial()
        return dict(step=step,loss=float(loss.detach()),curriculum=curriculum,group_reuse=reused,
                    query_visits=8,learning_rate=lr_at(step))

    def snapshot(self):
        return dict(schema='R9_FIT_SNAPSHOT_V1',recipe=self.recipe,parent=self.parent_snapshot,fit=super().snapshot())

    def restore(self,snapshot):
        if snapshot.get('schema')!='R9_FIT_SNAPSHOT_V1' or snapshot.get('recipe')!=self.recipe:
            raise ValueError('R9 fit identity')
        super().restore(snapshot['fit']);self.parent_snapshot=copy.deepcopy(snapshot['parent'])

    def fork_legacy(self,snapshot,parent_binding,parent_sha256):
        """Caller has verified the archive seal; never accept a deployment-only state."""
        if (self.recipe!='LEGACY' or self.steps!=0 or snapshot.get('steps')!=4000 or
            snapshot.get('binding')!=parent_binding or re.fullmatch('[0-9a-f]{64}',parent_sha256) is None or
            not snapshot.get('optimizer',{}).get('state') or 'rng' not in snapshot or
            snapshot.get('episode_state',{}).get('counter')!=0):raise ValueError('full pre-cal 4k snapshot required')
        if hasattr(self.method,'cal_raw') and not torch.equal(snapshot['method']['cal_raw'],temperature_initial()):
            raise ValueError('calibrated weights cannot continue fit')
        binding=self.binding; physical=COUNTS.copy()
        try:
            self.binding=parent_binding
            super().restore(snapshot)
        finally:
            self.binding=binding;COUNTS.clear();COUNTS.update(physical)
        self.parent_snapshot=dict(binding=parent_binding,sha256=parent_sha256,step=4000)
