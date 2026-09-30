"""Connected source objectives; exact old-policy rollout and two-epoch replay."""
import copy,math,random
import numpy as np
import torch
from .controller import detached
from .math import Memory,normal_log_prob,reference_kl,retention,group_advantage,clipped_surrogate
from ..r7_shared.numerics import finite,seg_loss
from ..r9_current_first.validation import dice


def soft_dice(logits,label):
    p=logits.sigmoid();y=label.to(p)
    return (2*(p*y).sum((-2,-1))+1e-6)/(p.sum((-2,-1))+y.sum((-2,-1))+1e-6)


def lr(step,total,peak,end):
    if step<100:return peak*(step+1)/100
    return end+(peak-end)*(1+math.cos(math.pi*(step-99)/(total-100)))/2


def optimizer(actor,rate):
    return torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=rate,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state((s['numpy'][0],np.array(s['numpy'][1],dtype=np.uint32),*s['numpy'][2:]));torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])


class Trainer:
    def __init__(self,actor,controller,source,seed,method,binding,reference=None,guard=lambda:None,total=None):
        self.actor,self.controller,self.source=actor,controller,source
        self.seed,self.method,self.binding=seed,method,binding;self.guard=guard
        if method not in ('WARM','SUP_STATIC','SUP_SEQ','SUP_RET','GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA'):raise ValueError('method')
        self.total=(2000 if method=='WARM' else 4000) if total is None else total
        if type(self.total) is not int or self.total<=100:raise ValueError('scheduler length')
        self.actor.train_writer(method not in ('WARM','SUP_STATIC','GR_CUR'))
        self.reference=copy.deepcopy(reference if reference is not None else actor).eval();self.reference.requires_grad_(False)
        self.opt=optimizer(actor,3e-4 if method=='WARM' else 3e-5)
        self.audit=[];self.steps=0;self.ema=.01 if method=='GR_RET_EMA' else None
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    def prediction(self,actor,item,state,epsilon=None,forced_write=None):
        self.guard();use,nxt,a=self.controller.act(actor,item['observation'],state,epsilon,forced_write)
        logits=self.source.segmenter(item['image'],self.controller.carrier.basis@use);finite(logits)
        self.audit.append((float(a['raw_action'][0].sigmoid().detach()),float(a['raw_action'][9].sigmoid().detach()) if forced_write is None else forced_write,float(nxt.m.detach().norm()),float(nxt.h.detach())))
        return logits,nxt,a
    def prefix(self,actor,episode,t):
        state=Memory.zero()
        with torch.no_grad():
            for v in range(t):
                self.guard();item=self.source.item('fit',self.seed,episode,v)
                _,state,_=self.controller.act(actor,item['observation'],state,forced_write=.5 if self.method=='GR_CUR' else None)
        return detached(state)
    def probe(self,actor,items,state,soft):
        vals=[]
        for item in items:
            logits,_,_=self.prediction(actor,item,state)
            vals.append(soft_dice(logits,item['label']).reshape(2) if soft else logits.new_tensor(dice(logits.sigmoid(),item['label'])[0]))
        return torch.stack(vals)
    def step(self):
        total=self.total
        if self.steps>=total:raise ValueError('training exhausted')
        rate=lr(self.steps,total,3e-4 if self.method=='WARM' else 3e-5,3e-5 if self.method=='WARM' else 3e-6)
        for g in self.opt.param_groups:g['lr']=rate
        self.audit=[]
        result=self.warm() if self.method=='WARM' else self.post()
        result['controller_diagnostics']={k:sum(row[i] for row in self.audit)/len(self.audit) for i,k in enumerate(('gain_mean','write_mean','state_norm_mean','mass_mean'))}
        self.steps+=1;self.guard();return dict(round=self.steps,lr=rate,**result)
    def update(self):
        params=[p for p in self.actor.parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        self.opt.step();finite(*params);self.guard();return float(norm)
    def warm(self):
        episode,chunk=divmod(self.steps,4);self.opt.zero_grad(set_to_none=True);losses=[]
        for v in range(chunk*8,(chunk+1)*8):
            item=self.source.item('fit',self.seed,episode,v,warm=True)
            logits,_,_=self.prediction(self.actor,item,Memory.zero(),forced_write=.5)
            loss=seg_loss(logits,item['label']);finite(loss);(loss/8).backward();losses.append(float(loss.detach()))
        return dict(loss=sum(losses)/8,grad_norm=self.update())
    def post(self):
        episode,t,items=self.source.window(self.steps,self.seed)
        old=copy.deepcopy(self.actor).eval();old.requires_grad_(False);pre=self.prefix(old,episode,t)
        epsilon=torch.randn(4,4,10)
        probes=self.source.probes('fit',self.seed,episode,t) if 'RET' in self.method else None
        with torch.no_grad():anchor=self.probe(old,probes,pre,self.method.startswith('SUP')) if probes else None
        if self.method.startswith('SUP'):return self.supervised(items,pre,epsilon,probes,anchor,t)
        return self.reinforce(old,items,pre,epsilon,probes,anchor,t)
    def supervised(self,items,pre,epsilon,probes,anchor,t):
        losses=[];kls=[];norms=[]
        for epoch in range(2):
            self.opt.zero_grad(set_to_none=True)
            for candidate in range(4):
                state=pre;task=[];kl=[]
                for v,item in enumerate(items):
                    if self.method=='SUP_STATIC':state=Memory.zero()
                    logits,state,a=self.prediction(self.actor,item,state,epsilon[candidate,v],.5 if self.method=='SUP_STATIC' else None)
                    task.append(soft_dice(logits,item['label']).mean())
                    obs=a['observation'].detach();mu=self.actor(obs)
                    kl.append(reference_kl(mu,self.reference(obs),active_dims=9 if self.method=='SUP_STATIC' else 10))
                ret=retention(anchor,self.probe(self.actor,probes,state,True),existing_history=t>0) if probes else 0.
                k=torch.stack(kl).mean();loss=1-torch.stack(task).mean()-.05*ret+.005*k
                finite(loss);(loss/4).backward();losses.append(float(loss.detach()));kls.append(float(k.detach()))
            norms.append(self.update())
        return dict(loss=sum(losses)/8,KL=sum(kls)/8,grad_norm=max(norms),epochs=2)
    def reinforce(self,old,items,pre,epsilon,probes,anchor,t):
        observations=[];actions=[];logps=[];rewards=[];active=9 if self.method=='GR_CUR' else 10
        with torch.no_grad():
            if self.method=='GR_CUR':
                state=pre;groups=[]
                for v,item in enumerate(items):
                    group=[];next_zero=None
                    for i in range(4):
                        logits,nxt,a=self.prediction(old,item,state,epsilon[i,v],.5)
                        group.append(sum(dice(logits.sigmoid(),item['label'])[0])/2)
                        observations.append(a['observation']);actions.append(a['raw_action']);logps.append(normal_log_prob(a['raw_action'],a['mean'],active_dims=active))
                        if i==0:next_zero=nxt
                    groups.append(torch.tensor(group));state=next_zero
                adv=torch.cat([group_advantage(g)[0] for g in groups]);rewards=torch.cat(groups)
            else:
                for i in range(4):
                    state=pre;task=[]
                    for v,item in enumerate(items):
                        logits,state,a=self.prediction(old,item,state,epsilon[i,v])
                        task.append(sum(dice(logits.sigmoid(),item['label'])[0])/2)
                        observations.append(a['observation']);actions.append(a['raw_action']);logps.append(normal_log_prob(a['raw_action'],a['mean']))
                    ret=float(retention(anchor,self.probe(old,probes,state,False),existing_history=t>0)) if probes else 0.
                    rewards.append(sum(task)/4+.05*ret)
                rewards=torch.tensor(rewards,dtype=torch.float64);a,self.ema=group_advantage(rewards,ema=self.ema);adv=a.repeat_interleave(4)
        obs=torch.stack(observations).detach();raw=torch.stack(actions).detach();old_lp=torch.stack(logps).detach()
        losses=[];clips=[];kls=[]
        for epoch in range(2):
            self.opt.zero_grad(set_to_none=True);mu=self.actor(obs)
            lp=normal_log_prob(raw,mu,active_dims=active);ratio=(lp-old_lp).exp();finite(ratio)
            if epoch==0 and not torch.allclose(ratio,torch.ones_like(ratio),rtol=1e-5,atol=1e-5):raise ValueError('pi_old does not match sampled policy')
            kl=reference_kl(mu,self.reference(obs),active_dims=active).mean()
            loss=clipped_surrogate(lp,old_lp,adv)+.005*kl;finite(loss);loss.backward();self.update()
            losses.append(float(loss.detach()));clips.append(float(((ratio<.8)|(ratio>1.2)).double().mean()));kls.append(float(kl.detach()))
        return dict(loss=sum(losses)/2,reward_mean=float(rewards.mean()),reward_std=float(rewards.std(unbiased=False)),zero_advantage_fraction=float((adv==0).double().mean()),KL=sum(kls)/2,clip_fraction=clips,ema=self.ema,epochs=2)
    def snapshot(self):
        return dict(schema='R10_TRAIN_V1',binding=self.binding,method=self.method,seed=self.seed,steps=self.steps,total=self.total,
                    actor=copy.deepcopy(self.actor.state_dict()),reference=copy.deepcopy(self.reference.state_dict()),optimizer=copy.deepcopy(self.opt.state_dict()),ema=self.ema,rng=rng_state(),boundary='complete_two_epoch_round' if self.method!='WARM' else 'complete_warmup_update')
    def restore(self,s):
        if s['schema']!='R10_TRAIN_V1' or (s['binding'],s['method'],s['seed'])!=(self.binding,self.method,self.seed):raise ValueError('training identity')
        if s.get('total',2000 if self.method=='WARM' else 4000)!=self.total:raise ValueError('scheduler identity')
        if not 0<=s['steps']<=self.total:raise ValueError('training cursor')
        self.actor.load_state_dict(s['actor']);self.reference.load_state_dict(s['reference']);self.opt.load_state_dict(s['optimizer']);self.steps=s['steps'];self.ema=s['ema'];restore_rng(s['rng'])
