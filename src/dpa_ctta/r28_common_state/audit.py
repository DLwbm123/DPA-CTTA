"""Exact historical selector replay on stored SEARCH scalars; no new model calls."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def replay(rewards, dice):
    rewards, dice = np.asarray(rewards), np.asarray(dice)
    best = rewards == rewards.max()
    selected = int(rewards.argmax())
    return dict(any_nonzero=float(np.any(rewards!=0)), all_zero=float(np.all(rewards==0)),
        tied_max=float(best.sum()>1), first_selected=float(selected==0),
        reward_range=float(np.ptp(rewards)),
        top_gap=float(np.sort(rewards)[-1]-np.sort(rewards)[-2]),
        actual_gain_pp=100*(dice[selected]-dice.mean()),
        uniform_tie_gain_pp=100*(dice[best].mean()-dice.mean()),
        oracle_gain_over_random_pp=100*(dice.max()-dice.mean()),
        actual_regret_pp=100*(dice.max()-dice[selected]))


def audit(root, output):
    groups=defaultdict(list)
    for path in sorted((Path(root)/'scores/final').glob('*.private.jsonl')):
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                if r['role']!='SEARCH' or 'proposal_Dice64' not in r['metrics'][0]:continue
                d=r['diagnostics'];rewards=d['rewards']
                if r['condition']=='GREEDY' and int(np.argmax(rewards))!=d['selected']:
                    raise ValueError('historical selector differs from argmax')
                dice=np.mean([v['proposal_Dice64'] for v in r['metrics']],axis=0)
                groups[r['condition'],r['seed'],r['order'],r['domain']].append(replay(rewards,dice))
    rows=[]
    for (arm,seed,order,domain),g in sorted(groups.items()):
        rows.append(dict(condition=arm,seed=seed,order=order,domain=domain,n=len(g),
            **{k:float(np.mean([x[k] for x in g])) for k in g[0]}))
    output=Path(output);output.mkdir(exist_ok=True)
    with (output/'EXACT_SELECTOR_AUDIT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for arm in sorted({r['condition'] for r in rows}):
        g=[r for r in rows if r['condition']==arm]
        summary.append(dict(condition=arm,**{k:float(np.mean([r[k] for r in g])) for k in groups[next(k for k in groups if k[0]==arm)][0]}))
    (output/'EXACT_SELECTOR_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
