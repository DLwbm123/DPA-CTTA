"""Source-cal kernel; policy selection remains explicit and unapproved by default."""
import random
import statistics as st
import numpy as np
import torch
from ..r8_ba.schedule import episode_styles,episode_roles,anchors,CURRICULA
from ..r8_ba.gradient_calibration import augmentation_seed
from ..r7_shared.source import simulate
from .gradient import STEPS,LATENT_LR,BN_LR
from .selection import balanced
from .validation import dice

POLICIES={'first_two_mean':(20260924,20260925),'first_only':(20260924,),
          'per_seed':tuple(range(20260924,20260929))}


def select(rows,policy):
    if policy not in POLICIES:raise ValueError('explicit LR source policy required')
    seeds=POLICIES[policy];expected={(a,lr,s,i) for a in STEPS for lr in (BN_LR if a=='BN_RESET_G1' else LATENT_LR) for s in seeds for i in range(64)}
    if len(rows)!=len(expected) or {(r['arm'],r['lr'],r['source_seed'],r['episode']) for r in rows}!=expected:raise ValueError('complete LR grid required')
    scores={};selected={}
    for a in STEPS:
        per_seed={s:{lr:balanced([r for r in rows if r['arm']==a and r['lr']==lr and r['source_seed']==s],4) for lr in (BN_LR if a=='BN_RESET_G1' else LATENT_LR)} for s in seeds}
        scores[a]=per_seed
        groups={str(s):per_seed[s] for s in seeds} if policy=='per_seed' else {'global':{lr:st.mean(per_seed[s][lr] for s in seeds) for lr in per_seed[seeds[0]]}}
        selected[a]={key:min(lr for lr,v in values.items() if max(values.values())-v<=1e-8) for key,values in groups.items()}
    return dict(schema='R9_LR_SELECTION_V1',policy=policy,selected_lr=selected,scores=scores)


def episode(host,data,oracles,index,seed,guard):
    if host.visits!=0 or oracles.fold!='cal' or not 0<=index<64:raise ValueError('source-cal episode identity')
    oracles.validate(data)
    # Four representative positions in each unchanged 32-visit source curriculum.
    schedule_index=4*(index%16)+index//16
    ids,mode,_=episode_styles('cal',schedule_index)
    roles,_=episode_roles(data.folds['cal'],'cal',schedule_index,ids,oracles.support_pairs)
    rng=augmentation_seed(index,seed);random.seed(rng);np.random.seed(rng);torch.manual_seed(rng)
    bank=anchors('cal');values=[];soft=[]
    for v in (0,8,16,24):
        guard();q=data.get(roles[v][1],'cal')
        image=simulate(q.image,bank[ids[v]],f'R9_GRAD_LR_QUERY|{index}|{v}|{roles[v][1]}')
        logits,_=host.step(image);h,s=dice(logits.sigmoid(),q.label);values.append(st.mean(h));soft.append(st.mean(s))
    return dict(episode=index,source_seed=seed,arm=host.arm,lr=host.lr,curriculum=mode,visits=4,hard_Dice=st.mean(values),soft_Dice=st.mean(soft))
