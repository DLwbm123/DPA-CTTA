"""Render completed independent M2 aggregates; never infer missing scores."""
import json
from pathlib import Path
import sys


def render(directory):
    directory=Path(directory)
    public=json.loads((directory/'public_aggregate.json').read_text())
    audit=json.loads((directory/'execution_audit.json').read_text())
    status='M2_EPISODE_COVERAGE_COMPARISON_COMPLETE'
    if public['status']!=status or audit['status']!=status or not audit['independent_CPU_recompute']:
        raise ValueError('A completed independent recomputation is required')
    arms=['N','A','R','D1','O1','D2','O2']
    lines=['# M2 episode coverage experiment report','',status,'',
        'Execution commit: '+audit['experiment_execution_commit'],
        '','All values below are Dice percent; paired differences are percentage points. Task means weight domains equally.',
        '','| Split / task / domain / channel | '+' | '.join(arms)+' |',
        '| --- | '+' | '.join(['---:']*7)+' |']
    for split in ['source','target']:
        for task,result in public[split].items():
            for domain,d in result['domains'].items():
                for channel in (['OD','OC','macro'] if task=='fundus' else ['polyp']):
                    values=[f"{d['arms'][a][channel]['dice_percent']['mean']:.6f}" if a in d['arms'] else 'NOT_AVAILABLE' for a in arms]
                    lines.append('| '+f'{split} / {task} / {domain} / {channel}'+' | '+' | '.join(values)+' |')
    lines+=['','## Target paired comparisons','','| Task / domain | O2-D2 | O2-O1 | D2-D1 | Interaction |','| --- | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'NOT_AVAILABLE' if x is None else f'{x:+.6f}'
    for task,result in public['target'].items():
        c=result['task_comparisons_pp']
        lines.append('| '+task+' / domain-equal mean | '+' | '.join(fmt(c[p]) for p in ['O2-D2','O2-O1','D2-D1'])+' | '+fmt(result['task_interaction_pp'])+' |')
        channel='macro' if task=='fundus' else 'polyp'
        for domain,d in result['domains'].items():
            values=[d['paired'][p][channel]['dice_delta_pp']['mean'] if isinstance(d['paired'][p],dict) else None for p in ['O2-D2','O2-O1','D2-D1']]
            inter=d['interaction'][channel]
            values.append(inter['dice_delta_pp']['mean'] if isinstance(inter,dict) else None)
            lines.append('| '+task+' / '+domain+' | '+' | '.join(map(fmt,values))+' |')
    lines+=['','## Execution and interpretation','',
        'New-run counts (including corrected smoke): '+json.dumps(audit['updates_and_new_records'])+'.',
        f"Reused M1 records: {audit['reused_M1_records']}; combined display: {audit['combined_display_records']}. History reused: {audit['history_reused']}.",
        f"GPU-stage wall seconds: {audit['gpu_seconds']:.3f}; private file bytes at recompute: {audit['private_output_bytes']}.",
        'Each of Fundus D2/O2 and Polyp D2/O2 completed 600 episodes. Source and target scoring and independent CPU reconstruction completed.',
        'The original failed smoke and authorized repair diagnostics are separate historical work, not included in the new-run counts above.',
        'Smoke paired checks used strict deterministic CUDA execution, restored afterward. Formal training/scoring retained M1 settings. Original tolerance and fixed seed were unchanged.',
        'Full paired medians, signs, adverse tails, OD/OC channels, common ASSD cohorts, source/target tables and training curves are in [public_aggregate.json](public_aggregate.json). Resource counts and backend scope are in [execution_audit.json](execution_audit.json).',
        'Interpret task and domain directions jointly. Single seed/order, previously exposed target domains, UNKNOWN patient/video associations, fixed off-policy history and one-step adaptation limit inference. Completion does not establish method superiority or SOTA.','']
    with (directory/'M2_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines))


if __name__=='__main__':render(sys.argv[1])
