"""Source-only terminal report; target outcomes explicitly not newly measured."""
import statistics,time
from collections import defaultdict
from pathlib import Path
from ..r10_12h_core.run import read,save,epoch,OPS
from ..r10_attribution.report import write_csv
from .view import NAME

def report(c,state):
    if state['status']=='RUNNING':raise ValueError('source mechanism pending')
    root=Path(c['output_root']);src=read(root/'SOURCE_COMPARISON.json') if (root/'SOURCE_COMPARISON.json').exists() else dict(new_rows=[],status='NOT_RUN');rows=c['source_reference']+src.get('new_rows',[]);write_csv(root/'SOURCE_EPISODES.csv',rows);summary=[]
    for name in ('C0','CV_H025','U10_H025',NAME):
        for mode in ['ALL']+sorted({r['mode'] for r in rows}):
            selected=[r for r in rows if r['condition']==name and (mode=='ALL' or r['mode']==mode)];values={}
            for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice'):
                groups=defaultdict(list)
                for r in selected:groups[r['mode']].append(r[k])
                values[k]=statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None
            values['support_fraction']=statistics.mean(r['eligible_fraction'] for r in selected) if name==NAME and selected else None;summary.append(dict(condition=name,mode=mode,episodes=len(selected),**values))
    write_csv(root/'SOURCE_COMPARISON.csv',summary);allmode={r['condition']:r for r in summary if r['mode']=='ALL'};complete=state['status']=='COMPLETE';a=allmode[NAME];b=allmode['C0'];q=allmode['CV_H025'];u=allmode['U10_H025'];decision=dict(source_complete=complete,targets='NOT_RUN_SOURCE_ONLY',new_target_visits=0,quarter_hard_equal_by_construction=True,source_hard_reference_parity=read(root/'SOURCE_HARD_PARITY.json') if (root/'SOURCE_HARD_PARITY.json').exists() else None,soft_half_gap_recovered=a['soft_Dice']>=(b['soft_Dice']+q['soft_Dice'])/2 if complete else None,source_soft_delta_vs_C0=a['soft_Dice']-b['soft_Dice'] if complete else None,source_soft_delta_vs_quarter=a['soft_Dice']-q['soft_Dice'] if complete else None,source_soft_delta_vs_U10=a['soft_Dice']-u['soft_Dice'] if complete else None,original_target_priority='Prior measured quarter+0.159763pp vsC0 below+0.5pp; no new target improvement possible from decision preservation',campaign_successor_round=4,automatic_next_round=False,stop_reason='Four authorized successors exhausted; close and pause after delivery',independent_confirmation=False)
    save(root/'DECISION.json',decision);attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))];ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=False,disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    lines=[f'# {c["experiment_id"]}',f'Status {state["status"]}; execution {c["code_sha"]}.','SOURCE_ONLY. Preserve native logits on native/quarter hard agreement, quarter logits on disagreement; no eligibility threshold search. No new target access/label/probability/scorer. Mathematical hard-mask preservation is not a new target measurement.','| source condition | hardDice % | softDice % |','|---|---:|---:|']
    for r in summary:
        if r['mode']=='ALL':lines.append(f'| {r["condition"]} | '+' | '.join('MISSING' if r[k] is None else f'{100*r[k]:.6f}' for k in ('hard_Dice','soft_Dice'))+' |')
    lines+=['```json',__import__('json').dumps(decision,indent=2),'```','Full source modes/OD/OC, support fractions and all old adverse results retained. Source softDice improvement does not establish calibration, unseen generalization, a new CTTA method or clinical utility. Historical target quarter75.239183%,C075.079420%,G77.229693% are context only; new target soft is NOT_RUN. No meaningful fake seed rerun. Four successor budget reached; close report and pause heartbeat after publication verification.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
