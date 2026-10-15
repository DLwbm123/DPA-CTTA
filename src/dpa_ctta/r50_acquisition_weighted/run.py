"""Retained fixed matrix, same-round VRAM admission and independent CPU scoring."""
import json
import statistics
from collections import defaultdict
from pathlib import Path
from .method import Host
from ..r36_incremental_modules import run as retained
from ..r37_cw_orientation.run import public
from ..r49_joint_lso_factorial.run import signal
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout, diagnostics
from ..r10_12h_core.run import read, save

ID = 'R50_ACQUISITION_WEIGHTED'
PRIMARY = 'C_JS_ACQ_WEIGHTED'
CONTRASTS = [(PRIMARY, 'C'), (PRIMARY, 'C_JS_SHARED'), (PRIMARY, 'CW_LSO'),
             ('C_JS_SHARED', 'C'), ('CW_LSO', 'C')]


def paired_cells(domains):
    index={(r['condition'],r['seed'],r['order'],r['role'],r['domain'],r['channel']):r for r in domains}
    out=[];groups=defaultdict(list)
    for a,b in CONTRASTS:
        for key,row in index.items():
            other=index.get((b,*key[1:]))
            if key[0]!=a or key[3] not in ('SEARCH','SEALED_REVIEW') or other is None:continue
            delta=row['Dice_percent']-other['Dice_percent']
            out.append(dict(candidate=a,baseline=b,aggregation='individual_seed',seed=key[1],order=key[2],role=key[3],domain=key[4],channel=key[5],delta_pp=delta))
            groups[(a,b,*key[2:])].append(delta)
    for (a,b,order,role,domain,channel),values in groups.items():
        out.append(dict(candidate=a,baseline=b,aggregation='seed_mean',seed=None,order=order,role=role,domain=domain,channel=channel,delta_pp=statistics.mean(values)))
    return out


def decision(pairs, complete):
    by = {p['baseline']: p for p in pairs if p['candidate']==PRIMARY and p['role']=='SEARCH'}
    return dict(primary=PRIMARY, standalone_signal=complete and signal(by['C']),
                improvement_over_shared=complete and signal(by['C_JS_SHARED']),
                improvement_over_strong_control=complete and signal(by['CW_LSO']),
                review_descriptive_only=True, independent_generalization=False, automatic_followup=False)


def report(c, state):
    root=Path(c['output_root']); out=root/'public'
    resources=retained.base.ledger(c,state)
    jobs=read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done=[j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows=retained.scoring.load_rows(root,'FINAL',done,final=True)
    main,domains=aggregate(rows)
    csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    csvout(out/'ALL_PAIRED_CELLS.csv',paired_cells(domains))
    csvout(out/'MECHANISM_DIAGNOSTICS.csv',diagnostics(rows))
    by_domain=[]
    for domain in sorted({r['domain'] for r in rows}):
        by_domain.extend(dict(domain=domain,**d) for d in diagnostics([r for r in rows if r['domain']==domain]))
    csvout(out/'MECHANISM_BY_DOMAIN.csv',by_domain)
    csvout(out/'COST.csv',[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],wall_seconds=a['wall_seconds'],**a['cost']) for a in resources['attempts']])
    pairs, intervals, negative = [], [], []
    for role in ('SEARCH','SEALED_REVIEW'):
        for a,b in CONTRASTS:
            pair,ci,neg=bootstrap_pair(rows,a,b,c['seeds'],role=role)
            pairs.append(pair);intervals.extend(ci);negative.extend(neg)
    csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',intervals);csvout(out/'ALL_NEGATIVE_CELLS.csv',negative)
    final=decision(pairs,state['status']=='COMPLETE')
    save(out/'FINAL_DECISION.json',final)
    for name,value in [('RUN_STATE.json',state),('RESOURCE_LEDGER.json',resources),('CANDIDATES.json',c['candidates'])]:save(out/name,public(value))
    for name in ('FROZEN.json','MECHANICAL_TESTS.json','PROFILE_ADMISSION.json'):
        if (root/name).exists():save(out/name,public(read(root/name)))
    save(out/'COMPLETION_AUDIT.json',dict(formal_planned=24,formal_complete=sum(state['jobs'].get(j['id'])=='COMPLETE' for j in c['jobs']),
         scored_jobs=len(done),scored_rows=len(rows),expected_scored_rows=46824,
         all_formal_workers_retired=not any(read(p).get('active') for p in (root/'processes').glob('online_*.json')),
         source_image_reads=0,online_target_label_reads=0,all_attempts_charged=True))
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
    lines=['# R50 acquisition weighted structural filtering','',f"Status: **{state['status']}**.",'',
        'Primary: C_JS_ACQ_WEIGHTED minus C. C_JS_SHARED and unchanged CW_LSO are retained controls.',
        'A gain over C is distinct from improving the original filter or the strong control. Each claim has its own frozen gate.', '',
        '| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |','|---|---:|---:|']
    for candidate in c['candidates']:
        means=[]
        for role in ('SEARCH','SEALED_REVIEW'):
            values=[x['macro_Dice_percent'] for x in main if x['condition']==candidate['id'] and x['role']==role]
            means.append(f'{statistics.mean(values):.6f}' if values else 'NA')
        lines.append(f"| {candidate['id']} | {means[0]} | {means[1]} |")
    lines += ['',f"Frozen decisions: {json.dumps(final)}.",
        'Each positive comparison requires >=0.1pp, paired 95% CI lower>0, both orders positive, >=5/6 positive trajectories, imageweighted>=0 and worst seed-mean domain/channel/order cell>=-2pp.',
        'All 24 fresh trajectories and independent CPU scores are required; unscored or incomplete means NOT_EVALUABLE, not zero.',
        'All positive and negative domain/channel/order/seed cells and bootstrap intervals are retained. SEARCH/REVIEW are historically exposed; REVIEW is descriptive. No independent generalization or clinical/anatomical protection claim.',
        'The new matrix changes direction selection and effective strength. Derivatives describe a pseudo-objective; association cannot prove Dice preservation. No descent constraint, retuning, retry or next round.',
        f"GPU-worker {resources['gpu_worker_seconds']/3600:.6f}/48h; all profiles, initialization and failures billed. Original T0 retained.",
        'Only own source/protocol and anonymous aggregate results/cost/audit are public. Raw images, masks, content identities/hashes, predictions, model state and private logs remain private.']
    if state.get('reason'):lines += ['','Stopped reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    retained.ID=ID;retained.Host=Host;retained.report=report;retained.public=public
    retained.main()


if __name__=='__main__':main()
