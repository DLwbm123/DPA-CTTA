"""Offline SEARCH-only candidate audit; exports aggregates, never private identities."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

COMPONENTS = ('cons', 'structural', 'retain', 'consistency', 'extent', 'topology', 'nesting', 'retention')


def selection_gain(scores, dice):
    # Tied maxima average their Dice so constant components cannot exploit index order.
    winners = scores == scores.max(axis=1, keepdims=True)
    selected = (dice*winners).sum(1)/winners.sum(1)
    return 100*(selected-dice.mean(1)), 100*(dice.max(1)-selected)


def audit(root, output):
    groups = defaultdict(list)
    for path in sorted((root/'scores/final').glob('*.private.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['role'] != 'SEARCH' or 'proposal_Dice64' not in row['metrics'][0]:
                    continue
                dice = np.mean([m['proposal_Dice64'] for m in row['metrics']],axis=0)
                components = row['diagnostics']['reward_components']
                groups[row['condition'],row['seed'],row['order'],row['domain']].append((dice,components))
    cells=[]
    for (arm,seed,order,domain),rows in sorted(groups.items()):
        dice=np.asarray([r[0] for r in rows]);dr=rankdata(dice,axis=1);dr-=dr.mean(1,keepdims=True)
        for component in COMPONENTS:
            sign=1 if component in ('cons','structural','retain') else -1
            scores=sign*np.asarray([[x[component] for x in r[1]] for r in rows])
            sr=rankdata(scores,axis=1);sr-=sr.mean(1,keepdims=True)
            denom=np.sqrt((sr*sr).sum(1)*(dr*dr).sum(1));valid=denom>0
            rho=(sr[valid]*dr[valid]).sum(1)/denom[valid]
            gain,regret=selection_gain(scores,dice)
            cells.append(dict(condition=arm,seed=seed,order=order,domain=domain,component=component,
                observations=len(rows),rank_estimable=int(valid.sum()),mean_Spearman=float(rho.mean()) if len(rho) else None,
                gain_vs_uniform_random_pp=float(gain.mean()),oracle_regret_pp=float(regret.mean()),
                positive_observations=int((gain>0).sum()),negative_observations=int((gain<0).sum())))
    summaries=[]
    for arm in sorted({r['condition'] for r in cells}):
        for comp in COMPONENTS:
            subset=[r for r in cells if r['condition']==arm and r['component']==comp]
            effects=[np.mean([r['gain_vs_uniform_random_pp'] for r in subset if r['seed']==s and r['order']==o])
                     for s in sorted({r['seed'] for r in subset}) for o in (0,1)]
            summaries.append(dict(condition=arm,component=comp,equal_domain_seed_order_gain_pp=float(np.mean(effects)),
                order0_gain_pp=float(np.mean(effects[::2])),order1_gain_pp=float(np.mean(effects[1::2])),
                positive_trajectories=sum(x>0 for x in effects),trajectories=len(effects),
                worst_domain_mean_pp=float(min(np.mean([r['gain_vs_uniform_random_pp'] for r in subset if r['domain']==d]) for d in {r['domain'] for r in subset})),
                role='SEARCH_ONLY_EXPOSED',resolution=64,tie_policy='uniform among equal maxima'))
    output.mkdir(exist_ok=True)
    for name,rows in [('REWARD_COMPONENT_CELLS.csv',cells),('REWARD_COMPONENT_SUMMARY.csv',summaries)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='COMPLETE',candidate_observations=sum(len(x) for x in groups.values()),
        components=list(COMPONENTS),cells=len(cells),summaries=len(summaries),online_training=False,role='SEARCH_ONLY_EXPOSED')))


if __name__=='__main__':
    dice=np.asarray([[.1,.2,.3,.4]])
    gain,regret=selection_gain(np.zeros_like(dice),dice)
    assert abs(gain[0])<1e-12 and abs(regret[0]-15)<1e-10
    assert selection_gain(dice,dice)[0][0]>0 and selection_gain(-dice,dice)[0][0]<0
    audit(Path(os.environ['AUDIT_INPUT']),Path(os.environ['AUDIT_OUTPUT']))
