"""Source-only schedules, immutable zero-modulation observations and probe groups."""
import hashlib
from functools import lru_cache
from collections import OrderedDict
import torch
from ..r8_ba.schedule import anchors,episode_styles,episode_roles
from ..r7_shared.source import simulate


class Source:
    def __init__(self,data,oracles,segmenter,controller,binding):
        self.data,self.oracles,self.segmenter,self.controller=data,oracles,segmenter,controller
        self.binding=binding;self.cache=OrderedDict();self.banks={f:anchors(f) for f in ('fit','cal','val')}
    @lru_cache(maxsize=16)
    def schedule(self,fold,episode,seed):
        ids,mode,_=episode_styles(fold,episode,seed)
        roles,_=episode_roles(self.data.folds[fold],fold,episode,ids,self.oracles[fold].support_pairs,seed)
        return ids,roles,mode
    def item(self,fold,seed,episode,visit,warm=False,group=None,anchor=None,probe_id=None):
        ids,roles,_=self.schedule(fold,episode,seed)
        group=roles[visit][1] if group is None else group;anchor=ids[visit] if anchor is None else anchor
        key=(self.binding,fold,seed,episode,visit,warm,group,anchor,probe_id)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        row=self.data.get(group,fold)
        if warm:simkey=f'R10_WARM_QUERY|{seed}|{episode}|{visit}|{group}'
        elif probe_id is not None:simkey=f'R10_PROBE_QUERY|{fold}|{seed}|{episode}|{visit}|{probe_id}|{group}'
        else:simkey=f'R10_QUERY|{fold}|{seed}|{episode}|{visit}|{group}'
        image=simulate(row.image,self.banks[fold][anchor],simkey)
        with torch.no_grad():
            _,raw,tokens=self.segmenter(image,observe=True)
            observation=self.controller.observe(raw,tokens)
        result=dict(image=image,label=row.label,observation=observation)
        self.cache[key]=result
        # Bounded source-only cache: four 32-visit episodes plus probe observations.
        while len(self.cache)>160:self.cache.popitem(last=False)
        return result
    def window(self,round,seed):
        episode=4*(round//32)+round%4;t=4*((round//4)%8)
        return episode,t,[self.item('fit',seed,episode,v) for v in range(t,t+4)]
    def probes(self,fold,seed,episode,t):
        ids,roles,_=self.schedule(fold,episode,seed)
        excluded={roles[v][1] for v in range(t,t+4)};result=[]
        for i,anchor in enumerate((ids[0],ids[t-1] if t else ids[0])):
            blocked=excluded|set(self.oracles[fold].support_pairs[anchor])
            candidates=[g for g in self.data.folds[fold] if g not in blocked]
            if not candidates:raise ValueError('insufficient distinct source probe groups')
            group=min(candidates,key=lambda g:hashlib.sha256(f'R10_PROBE|{fold}|{seed}|{episode}|{t}|{i}|{g}'.encode()).digest())
            excluded.add(group)
            result.append(self.item(fold,seed,episode,t,group=group,anchor=anchor,probe_id=i))
        return result
