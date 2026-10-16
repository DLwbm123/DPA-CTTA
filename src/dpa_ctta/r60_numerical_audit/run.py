"""Matched-policy replay of fixed R57/R58 controls; no scientific retuning."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
import numpy as np
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import csvout


def tables(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root']);started=time.time();state=dict(status='RUNNING_R57_REPLAY',started=started)
    save(root/'RUN_STATE.json',state);cost=0
    try:
        for stage,module in [('r57','dpa_ctta.r57_source_bn.run'),('r58','dpa_ctta.r58_bn_anchor.run')]:
            env=dict(os.environ,RUN_MODULE=module,RUN_CONFIG=str(root/stage/'CONFIG.json'),CUDA_VISIBLE_DEVICES='')
            with (root/'logs'/f'{stage}.log').open('a') as log:
                p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT)
                save(root/f'{stage}_PROCESS.json',dict(pid=p.pid,started=time.time(),active=True));rc=p.wait()
            save(root/f'{stage}_RETIRED.json',dict(returncode=rc,ended=time.time(),active=False));assert rc==0
            assert read(root/stage/'RUN_STATE.json')['status']=='COMPLETE'
            cost+=read(root/stage/'RETIREMENT.json')['ended']-read(root/stage/'WORKER.json')['started']
            state.update(status='RUNNING_R58_REPLAY' if stage=='r57' else 'COMPARING',GPU_seconds=cost);save(root/'RUN_STATE.json',state)
        lookup=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
        assert int(lookup[np.bitwise_xor(np.array([0,255],dtype=np.uint8),np.array([255,255],dtype=np.uint8))].sum())==8
        comparisons=[];out=root/'public'
        for method,stage in [('SOURCE','r57'),('BATCH','r57'),('MIX','r58')]:
            old=Path(c['old_'+stage]);new=root/stage
            a=tables(old/'public/FULL_RESULTS.csv');b=tables(new/'public/FULL_RESULTS.csv')
            oldmacro=mean(float(r['macro_Dice_percent']) for r in a if r['condition']==method and r['role']=='PRIMARY_ALL');newmacro=mean(float(r['macro_Dice_percent']) for r in b if r['condition']==method and r['role']=='PRIMARY_ALL')
            def cells(p):return {(r['seed'],r['order'],r['domain'],r['channel']):float(r['Dice_percent']) for r in tables(p/'public/DOMAIN_CHANNEL.csv') if r['condition']==method and r['role']=='PRIMARY_ALL'}
            ca,cb=cells(old),cells(new);assert ca.keys()==cb.keys();worst=max(abs(cb[k]-ca[k]) for k in ca)
            changed=0;changed_images=0
            with (old/(method+'.bits')).open('rb') as f,(new/(method+'.bits')).open('rb') as g:
                for _ in range(1951):
                    aa=f.read(65536);bb=g.read(65536);assert len(aa)==len(bb)==65536
                    count=int(lookup[np.bitwise_xor(np.frombuffer(aa,dtype=np.uint8),np.frombuffer(bb,dtype=np.uint8))].sum());changed+=count;changed_images+=count>0
                assert not f.read(1) and not g.read(1)
            comparisons.append(dict(method=method,old_macro=oldmacro,matched_macro=newmacro,delta_pp=newmacro-oldmacro,max_abs_cell_delta_pp=worst,changed_channel_pixels=changed,changed_images=changed_images,total_channel_pixels=1951*2*512*512))
        csvout(out/'NUMERICAL_SENSITIVITY.csv',comparisons)
        old_gates=[read(Path(c['old_r57'])/'public/DECISION.json')['incremental_over_source_supported'],read(Path(c['old_r58'])/'public/DECISION.json')['fixed_anchor_gate']]
        new_gates=[read(root/'r57/public/DECISION.json')['incremental_over_source_supported'],read(root/'r58/public/DECISION.json')['fixed_anchor_gate']]
        stable=old_gates==new_gates and all(abs(r['delta_pp'])<=.05 and r['max_abs_cell_delta_pp']<=.1 for r in comparisons)
        save(out/'DECISION.json',dict(numerical_stability_gate=stable,old_gates=old_gates,matched_gates=new_gates,independent_validation=False))
        for stage in ('r57','r58'):shutil.copytree(root/stage/'public',out/stage)
        (out/'REPORT.md').write_text('# R60 matched numerical-policy audit\n\nFixed stability gate: '+str(stable)+'\n\nThe three controls were replayed using historical C deterministic, TF32-disabled policy. This is numerical sensitivity validation, not another hyperparameter search or independent replication. Old predictions and results are retained. Interpret changes using NUMERICAL_SENSITIVITY and corrected per-stage reports; a small aggregate difference does not imply bitwise equality.\n')
        (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,GPU_seconds=cost,unique_control_images=5853);save(root/'RUN_STATE.json',state);save(out/'RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time(),GPU_seconds=cost);save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
