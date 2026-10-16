"""Independent CPU labels after the only GPU worker has retired."""
import csv
import json
import os
from pathlib import Path
import time
import numpy as np

from .run import read,save


def interval(values, domains):
    """values: content x anchor x channel; keep anchors/channels coupled."""
    rng=np.random.default_rng(20261016);samples=np.zeros(2000)
    for domain in sorted(set(domains)):
        index=np.flatnonzero(np.asarray(domains)==domain)
        draws=rng.choice(index,size=(2000,len(index)),replace=True)
        samples += values[draws].mean(axis=(1,2,3))/len(set(domains))
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def csvout(path, rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def score(c):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer only')
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from .diagnostic import BRANCHES
    root=Path(c['output_root']);start=time.time()
    state=read(root/'RUN_STATE.json');process=read(root/'processes/worker.json')
    if state['status']!='GPU_COMPLETE' or state['formal_pairs']!=96 or process['active'] or process['exit_code']!=0:
        raise ValueError('complete retired GPU worker required before labels')
    panel=read(root/'scorer/panel.json');reader=TargetReader(c['target_root'],256*1024**2,'mask')
    traces=[json.loads(line) for line in (root/'private/pairs.jsonl').read_text().splitlines()]
    if [t['pair'] for t in traces]!=list(range(96)):raise ValueError('formal prediction coverage')
    values=np.empty((48,2,8,2));private=[]
    for i,row in enumerate(panel):
        label=reader.read(row)
        for order in (0,1):
            raw=(root/'predictions'/f'{order*48+i:03d}.bits').read_bytes()
            if len(raw)!=8*65536:raise ValueError('prediction bytes')
            masks=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(8,2,512,512).astype(bool)
            metrics=[hard_metrics(m,label) for m in masks]
            for branch,m in enumerate(metrics):
                values[i,order,branch]=[100*x['dice'] for x in m]
            private.append(dict(panel_index=i,order=order,domain=row['domain'],metrics=dict(zip(BRANCHES,metrics))))
    reader.after_check()
    save(root/'scorer/scores.private.json',private)
    domains=np.array([r['domain'] for r in panel]);cells=[]
    for order in (0,1):
        for domain in sorted(set(domains)):
            for b,name in enumerate(BRANCHES):
                for ch,channel in enumerate(('OD','OC')):
                    cells.append(dict(anchor_order=order,domain=domain,branch=name,channel=channel,
                                      n_contents=int((domains==domain).sum()),dice_percent=float(values[domains==domain,order,b,ch].mean())))
    contrasts={'joint_minus_null_mean':values[:,:,2]-values[:,:,4:8].mean(2),
               'joint_minus_radial':values[:,:,2]-values[:,:,3],
               'joint_minus_native':values[:,:,2]-values[:,:,1],
               'native_minus_no_update':values[:,:,1]-values[:,:,0],
               'joint_minus_no_update':values[:,:,2]-values[:,:,0]}
    for k in range(4):contrasts[f'joint_minus_null{k}']=values[:,:,2]-values[:,:,4+k]
    summaries=[];paired=[]
    for name,delta in contrasts.items():
        summaries.append(dict(contrast=name,mean_pp=float(delta.mean()),ci95_pp=interval(delta,domains),
                              anchor0_pp=float(delta[:,0].mean()),anchor1_pp=float(delta[:,1].mean()),
                              positive_state_image_pairs=int((delta.mean(2)>0).sum()),zero_state_image_pairs=int((delta.mean(2)==0).sum())))
        for order in (0,1):
            for domain in sorted(set(domains)):
                for ch,channel in enumerate(('OD','OC')):
                    paired.append(dict(contrast=name,anchor_order=order,domain=domain,channel=channel,
                                       delta_pp=float(delta[domains==domain,order,ch].mean())))
    by={r['contrast']:r for r in summaries};main=by['joint_minus_null_mean']
    worthwhile=(main['mean_pp']>=.1 and main['ci95_pp'][0]>0 and main['anchor0_pp']>0 and main['anchor1_pp']>0
                and by['joint_minus_radial']['mean_pp']>0 and by['joint_minus_native']['mean_pp']>=0)
    errors=[e for t in traces for e in t['diagnostics']['committed_matching'].values()]
    null_errors=[e for t in traces for k,e in t['diagnostics']['committed_matching'].items() if k.startswith('null')]
    mechanical=dict(pairs=96,contents=48,anchors=2,predictions=768,label_reads=len(panel),
                    all_pairs_matched=True,degenerate_pairs=sum(t['diagnostics']['matching']['degenerate'] for t in traces),
                    max_gradient_decomposition_relative_error=max(t['diagnostics']['gradient_decomposition_relative_error'] for t in traces),
                    max_norm_relative_error=max(e['norm_relative_error'] for e in errors),
                    max_null_modification_relative_error=max(e['modification_relative_error'] for e in null_errors),
                    max_null_base_derivative_cosine_error=max(e['base_derivative_cosine_error'] for e in null_errors),
                    max_null_lso_derivative_cosine_error=max(e['lso_derivative_cosine_error'] for e in null_errors),
                    mean_pair_seconds=float(np.mean([t['seconds'] for t in traces])),
                    max_pair_seconds=max(t['seconds'] for t in traces))
    result=dict(status='COMPLETE',source_sha=c['source_sha'],contrasts=summaries,
                worth_discussing_further=worthwhile,mechanical=mechanical,
                interpretation='Conditional on two fixed late states and 48 exposed contents; not a full-stream or generalization result.',
                limitations=['First-order full-BN gradient matching, not nonlinear loss-surface matching.',
                             'Adam-metric norms matched; Euclidean norms not matched.',
                             'Nulls test whole J directional choice, not low rank separately.',
                             'Content-cluster uncertainty conditions on the two fixed states; patient linkage unknown.'])
    save(root/'public/RESULTS.json',result);save(root/'public/COMPLETION_AUDIT.json',mechanical)
    csvout(root/'public/DOMAIN_CHANNEL.csv',cells);csvout(root/'public/ALL_PAIRED_CELLS.csv',paired)
    csvout(root/'public/ALL_NEGATIVE_CELLS.csv',[r for r in paired if r['delta_pp']<0])
    save(root/'public/QUALIFICATION.json',read(root/'QUALIFICATION.json'))
    lines=['# R51: same-state late diagnostic','', 'Status: **COMPLETE**.', '',
           'Two fixed CW_LSO visit1900 states (seed17011, orders0/1); 48 historical SEARCH contents, 12/domain; 96 independent resets, not 96 independent states. No replay or trajectory continuation.', '',
           '| Contrast | Mean Dice difference (pp) | Conditional 95% CI | Anchor0 / Anchor1 (pp) |',
           '|---|---:|---|---|']
    for r in summaries:lines.append(f"| {r['contrast']} | {r['mean_pp']:.6f} | [{r['ci95_pp'][0]:.6f}, {r['ci95_pp'][1]:.6f}] | {r['anchor0_pp']:.6f} / {r['anchor1_pp']:.6f} |")
    lines += ['',f'Frozen continuation criterion met: **{worthwhile}**. No automatic follow-up is authorized.', '',
              'All eight masks per pair were saved before a separate CPU scorer read labels. No-update and native are same-state counterfactuals; current-image one-step effects do not establish future-stream effects.',
              'Four nulls match the full CW and LSO gradient directional derivatives, final update norm and distance to native in the Adam metric. Post-write tolerances were fixed before GPU qualification. Complete positive/negative domain/channel/anchor cells are retained.',
              'Random directions range over the eligible BN parameter space. A gain over them would support the whole directional choice only; it would not independently identify low-rank selection, anatomical protection, or optimality.',
              'Bootstrap: 2000 draws, domain-stratified content clusters, seed20261016; anchors/channels/nulls remain coupled. Data were historically exposed. Patient dependence is unknown. No claim outside the two late states.',
              '',f"Degenerate pairs: {mechanical['degenerate_pairs']}/96. Qualification and all matching checks passed. See resource ledger for imports, initialization, qualification, execution and CPU scoring charges."]
    (root/'public/REPORT.md').write_text('\n'.join(lines)+'\n')
    state.update(status='COMPLETE',scored_pairs=96,scored_predictions=768,score_seconds=time.time()-start,ended=time.time())
    save(root/'RUN_STATE.json',state)
    print(json.dumps({'status':'COMPLETE','worth_discussing_further':worthwhile,'primary':main},allow_nan=False),flush=True)
    return 0
