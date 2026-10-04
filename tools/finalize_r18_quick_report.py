from pathlib import Path
import json,csv,statistics as st,math,time,subprocess,os
r=Path(os.environ['RUN_ROOT']);read=lambda p:json.loads(p.read_text());state=read(r/'RUN_STATE.json');receipt=read(r/'SCORER_RECEIPT.json');assert state['status']=='COMPLETE' and receipt['status']=='COMPLETE'
processes=[read(p) for p in (r/'processes').glob('*.json')]
for x in processes:
 s=Path(f'/proc/{x["pid"]}/stat');assert not(s.exists() and s.read_text().split()[21]==x['start_ticks'] and s.read_text().split()[2]!='Z'),'worker still active'
assert len(receipt['completed'])==6 and receipt['scalar_rows']==19510 and all(x['visits']==1951 and x['principal']==1695 for x in receipt['completed'])
assert receipt['new_rows']==1294 and receipt['preserved_prior_rows']==10412
out=r/'public'
# Remove obsolete WARM comparisons from the reused report template only.
for name in ['main.csv','domain-channel.csv','paired.csv']:
 p=out/name
 with p.open() as f:reader=csv.DictReader(f);fields=[v for v in reader.fieldnames if 'WARM' not in v];rows=[{k:v for k,v in x.items() if k in fields} for x in reader if 'WARM' not in x.get('condition','') and 'WARM' not in x.get('control','')]
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
# Aggregate persisted deployment diagnostics without publishing per-image identities.
c=read(r/'RESOLVED_CONFIG.json');lock=read(r/'EXPERIMENT_LOCK.json');diags=[]
for arm in ('RL_GRATA','RL_ORIGINAL','FIXED_GRATA'):
 for o in (0,1):
  d=r/'target'/f'{arm}_o{o}';manifest=read(Path(c['manifests'][o]['path']));traces=[json.loads(x) for x in (d/'visits.jsonl').open()];assert len(traces)==1951 and all(t['visit']==i+1 for i,t in enumerate(traces))
  groups={'ALL':traces}
  for domain in sorted({m['domain'] for m in manifest}):groups[domain]=[t for t,m in zip(traces,manifest) if m['domain']==domain and m['subset']=='remaining_dev']
  for domain,ts in groups.items():
   row=dict(condition=arm,order=o,domain=domain,n=len(ts))
   for k in ['gain','write','action_residual_norm','use_norm','memory_norm','mass']:
    vals=[t[k] for t in ts];row[k]=dict(mean=st.mean(vals),minimum=min(vals),maximum=max(vals),stdev=st.pstdev(vals))
   diags.append(row)
(out/'ACTION_DIAGNOSTICS.json').write_text(json.dumps(diags,indent=2))
main=list(csv.DictReader((out/'main.csv').open()));domains=list(csv.DictReader((out/'domain-channel.csv').open()));paired=list(csv.DictReader((out/'paired.csv').open()));summary=read(out/'SUMMARY.json')
means={a:st.mean(float(x['Dice_percent']) for x in main if x['condition']==a) for a in sorted({x['condition'] for x in main})}
gain=means['RL_GRATA']-means['G'];worst=min((x for x in domains if x['condition']=='RL_GRATA'),key=lambda x:float(x['delta_G_pp']))
ledger=read(r/'RESOURCE_LEDGER.json');campaign=read(out/'CAMPAIGN_COST.json');prior=read(r.parent/'RESOURCE_LEDGER.json')
ops=ledger['operations'];assert [ops[k] for k in ['model_forwards','backward_calls','optimizer_steps','vjp_calls']]==[93648,15608,7804,0]
assert all(x['status']=='COMPLETE' and x['cost']['image_accesses']==1951 for x in ledger['attempts'] if x['phase'].startswith('online_'))
assert sum(x['cost']['image_accesses'] for x in ledger['attempts'] if x['phase'].startswith('online_'))==11706
assert abs(campaign['total_gpu_seconds']-(ledger['gpu_seconds']+prior['gpu_seconds']))<1e-6
cpu=[x for x in ledger['attempts'] if x['phase']=='score'];online=[x for x in ledger['attempts'] if x['phase'].startswith('online_')];assert len(cpu)==2 and cpu[0]['status']=='FAILED' and cpu[1]['status']=='COMPLETE';assert max(x['ended'] for x in online)<min(x['started'] for x in cpu)
audit=dict(status='PASS',all_six_trajectories_complete=True,new_visits=11706,new_principal=10170,scored_rows_with_references=19510,principal_with_references=16950,source_target_isolation='source refit cancelled; target image-only workers; independent CPU scoring after all online workers retired',frozen_parameters='online_complete requires boundary frozen checks; native BN intentionally adapted, actor/carrier/nonBN unchanged',full_prefix_reused=10412,new_CPU_rows=1294,no_GPU_rerun=True,score_first_attempt_timeout_preserved=True,all_workers_retired=True,new_operations=ops,campaign_gpu_seconds=campaign['total_gpu_seconds'],wall_from_original_T0_seconds=state['ended']-c['origin']['T0_epoch'],quick_wall_seconds=state['ended']-min(x['started'] for x in online),three_hour_cap_met=state['ended']<=c['origin']['absolute_deadline_epoch'],exposed_development=True,patient_dependence='unknown',orders_independent=False,report_prepared_at=time.time())
(out/'FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2))
(out/'CPU_SCORE_REPAIR.json').write_text(json.dumps(dict(reason='initial CPU scoring task had insufficient 1200-second allowance',preserved_rows=10412,new_rows=1294,first_attempt_seconds=cpu[0]['wall_seconds'],resume_seconds=cpu[1]['wall_seconds'],GPU_seconds=0,metric_functions_unchanged=True,no_target_reexecution=True,absolute_deadline_unchanged=True),indent=2))
report=['# R18 quick full-RL + GraTa results','',f"Status: COMPLETE. All six new full trajectories and independent CPU scoring are complete. Execution `{c['code_sha']}`. Frozen config `{lock['payload']['config_sha256']}`.",'',f"The unchanged full RL policy plus GraTa achieved {means['RL_GRATA']:.6f}% domain/channel-macro Dice versus GraTa {means['G']:.6f}% ({gain:+.6f} percentage points). The predeclared priority signal requires at least +0.5 points and both orders positive; it was {'met' if next(x for x in summary if x['condition']=='RL_GRATA')['priority_signal'] else 'not met'}.",'','| Condition | Mean macro Dice (%) | Delta vs GraTa (pp) |','|---|---:|---:|']
for a in ['C0','G','RL_ORIGINAL','FIXED_GRATA','RL_GRATA']:report.append(f'| {a} | {means[a]:.6f} | {means[a]-means["G"]:+.6f} |')
report+=['',f"Worst combination-minus-GraTa domain/channel: {worst['domain']}/{worst['channel']}, order{worst['order']}, n={worst['n']}, {float(worst['delta_G_pp']):+.6f} pp. Losses larger than2 pp meet the predefined risk flag.",'','Each new condition has two complete1951-visit trajectories, each1695 primary observations. In total11706 new visits and10170 primary scores; C0/G reuse four exactly matched sealed historical full trajectories. Main scores average the four domains and two channels equally, then average the two orders. `main.csv`, `domain-channel.csv` and `paired.csv` report all orders, OD/OC, ASSD valid/undefined denominators and image-weighted paired outcomes; macro and image-weighted effects should both be read.','',
'RL_ORIGINAL and RL_GRATA use the SAME historical R10 POST GR_RET_EMA actor, frozen B64/global carrier, FiLM and full10-action use/write with cross-image m/q/h memory. GraTa completes its native current-image BN update before exact BN mirroring to the FiLM model; zero-FiLM logits equal native GraTa at every image. The policy then chooses FiLM use and memory write. GraTa evolution is independent of these actions. FIXED_GRATA fixes gain0.8, residual0, write0.5. Native GraTa seed20260907 and policy seed20260924 have distinct roles. Actor, carrier and non-BN weights are frozen; BN adapts by design. No target reward or online policy fitting.',
'', 'No new source retraining was used. The original source-preparation plan was cancelled before formal policy training or any target access;17 completed source-cache episodes remain private. Earlier qualification included discarded cost-probe updates and is fully billed. Those source caches do not enter the quick screen. New source-fit/validation metrics and WARM controls are NA/not registered. Learned-versus-fixed is a deployment diagnostic, not isolation of RL training from the actor\'s earlier supervised fitting. A negative result concerns this fixed-policy coupling and does not rule out a separately refitted policy.',
'',f"Quick GPU-worker cost: {ledger['gpu_seconds']:.6f}s. Prior qualification/cancelled source cost: {prior['gpu_seconds']:.6f}s. Combined campaign cost: {campaign['total_gpu_seconds']:.6f}s ({campaign['total_gpu_seconds']/3600:.4f} GPU hours). New online operations:93648 backbone forwards,15608 backwards,7804 Adam steps,0 VJP. Composite deployment costs11F/2BP/1Adam per image, original RL2F/0BP. Prior cancelled-stage operation counts are last-observed lower bounds, not exact totals.",
'',f"Total wall time from original19:45 Beijing T0: {audit['wall_from_original_T0_seconds']/60:.2f} minutes, within the180-minute cap. All GPU workers completed without failure or recovery. Initial CPU scorer timed out after its1200s task allocation; five complete scalar streams and657 rows of the last stream were preserved. Only the remaining1294 rows were scored once in a CPU-only continuation, with unchanged metric code and absolute cutoff. Original timeout, CPU time and recovery evidence remain recorded. No scientific experiment was rerun.",
'', 'All images belong to previously exposed development data. The orders contain the same content, patient dependence is unknown, and one fitted policy seed is not independent replication. No clinical or independent held-out claim. Target soft-probability metrics are NA because sealed target outputs are masks; private images, labels, content IDs, predictions, weights and credentials are not released. No successor is authorized; monitoring stops after verified publication.']
(out/'REPORT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(dict(means=means,summary=summary,worst=worst,paired=paired,cost=campaign,audit=audit),indent=2))
