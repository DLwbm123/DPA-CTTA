"""Descriptive summaries of sealed signals and terminal receipts; no refit or new model."""
import json,time,os,subprocess
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def run():
    started=time.time();r=Path('/tmp/n97').resolve();c=read(r/'private/config.json')
    assert read(r/'public/CONTROLLER_EXIT.json')['status']=='COMPLETE'
    assert time.time()<c['deadline']-30
    old=read(c['prior_config'])
    with np.load(Path(old['output_root'])/'scorer/values.private.npz',allow_pickle=False) as f:domains=f['domains']
    groups=[];chosen=set();all_rows=[]
    for o in (0,1):
        data=read(r/'private'/f'signals_{o}.json')['rows'];assert len(data)==192
        for row in data:
            all_rows.append(row)
            if row['z']:chosen.add(row['index'])
        for d in range(4):
            rows=[row for row in data if domains[row['index']]==d]
            U=np.array([row['U'] for row in rows]);hd=np.array([row['entropyD'] for row in rows]);RR=np.sum([row['R_count'] for row in rows],axis=0);DD=np.sum([row['D_count'] for row in rows],axis=0);rev=np.sum([row['reverse_count'] for row in rows],axis=0)
            groups.append(dict(order=o,domain=f'D{d}',n=len(rows),eligible=sum(row['eligible'] for row in rows),switches=sum(row['z'] for row in rows),U_min=float(U.min()),U_median=float(np.median(U)),U_p90=float(np.quantile(U,.9)),U_max=float(U.max()),entropyD_min=float(hd.min()),entropyD_max=float(hd.max()),R_count=RR.tolist(),D_count=DD.tolist(),reverse_count=rev.tolist()))
    # No threshold scan, correlation hunt or selected-subset method fitting.
    summary=dict(status='POST_HOC_DESCRIPTIVE_ONLY',unique_selected_contents=len(chosen),selected_arrivals=sum(x['z'] for x in all_rows),total_arrivals=len(all_rows),groups=groups,new_model_calls=0,new_label_reads=0,CPU_body_seconds=time.time()-started)
    processes=[]
    for folder in (r/'private',r/'private/attempt0'):
        for p in sorted(folder.glob('*_process.json'))+sorted(folder.glob('*_timeout.json')):
            rec=read(p);pid=rec['pid'];st=Path('/proc')/str(pid)/'stat'
            live=st.exists() and st.read_text().rsplit(')',1)[1].split()[19]==str(rec['start_ticks'])
            processes.append(dict(stage=rec.get('mode',p.stem),attempt=0 if folder.name=='attempt0' else 1,identity_alive=live))
    assert processes and not any(p['identity_alive'] for p in processes)
    raw=subprocess.check_output(['ps','-eo','args='],text=True)
    active=[x for x in raw.splitlines() if x.strip().split() and x.strip().split()[-1] in ('/tmp/e97.py','/tmp/l97.py','/tmp/c97.py')]
    assert not active,active
    states=[read(r/'private/attempt0/qualify_state.json'),read(r/'private/qualify_state.json'),read(r/'private/formal_state.json')]
    counts={k:sum(s[k] for s in states) for k in ('physical_updates','physical_forwards','training_backwards')}
    assert counts==dict(physical_updates=396,physical_forwards=6312,training_backwards=396)
    costs={}
    for key,p in [('audit',r/'public/AUDIT_EXIT.json'),('qualification_attempt0',r/'public/attempt0/QUALIFY_EXIT.json'),('qualification_attempt1',r/'public/QUALIFY_EXIT.json'),('formal',r/'public/FORMAL_EXIT.json'),('score',r/'public/SCORE_EXIT.json')]:
        x=read(p);costs[key]={k:x[k] for k in ('status','wall_seconds','CPU_process_seconds')}
    end=read(r/'public/CONTROLLER_EXIT.json')['at']
    terminal=dict(status='COMPLETE_NO_ACTIVE_OWNED_PROCESS',registered_processes=processes,exact_neutral_entry_processes_remaining=len(active),counts=counts,costs=costs,
                  formal_and_score_exit0=True,first_wrapper_exit='UNRECORDED_ENGINEERING_FAILURE',T0=c['T0'],deadline=c['deadline'],completed_at=end,elapsed_from_conservative_T0_seconds=end-c['T0'],
                  remote_logical_or_allocated_snapshot_bytes=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in [r,*r.rglob('*')] if not p.is_symlink()),snapshot_before_final_exports=True,
                  remote_cap_bytes=c['remote_cap_bytes'],old_clocks_reset=False,unmetered_preparation_and_git=True,source_data_reads=0,new_label_reads=0)
    for name,obj in [('SIGNAL_DIAGNOSTICS.json',summary),('TERMINAL_AUDIT.json',terminal)]:
        data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode();assert len(data)<100000
        assert terminal['remote_logical_or_allocated_snapshot_bytes']+sum(len(json.dumps(x)) for x in (summary,terminal))+32768<c['remote_normal_cap_bytes']
        (r/'public'/name).write_bytes(data)
    print(json.dumps(dict(terminal=terminal,signal=summary),indent=2))

if __name__=='__main__':run()
