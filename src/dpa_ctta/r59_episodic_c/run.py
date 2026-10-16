"""Reset C model/optimizer history per arrival, preserving augmentation RNG progression."""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from ..r10_12h_core.run import read, save
from ..r7_target_screen.runner import TargetReader
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout
from ..r57_source_bn.run import control_fields, scoring_self_check


def reset_episode(h,initial):
    from ..r8_ba.rng import capture,restore
    current=capture();native=copy.deepcopy(h.native.rng)
    h.restore(initial);h.native.rng=native;restore(current)
    assert h.native.steps==0 and not h.native.base.state


def online(c,job):
    from ..r19_model_only.runtime import weights
    from ..r20_model_only_search.method import Host
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    root=Path(c['output_root']);qualify=os.environ.get('RUN_QUALIFY')=='1';dest=root/'target'/('QUAL_'+job['id'] if qualify else job['id']);dest.mkdir()
    ms=read(root/f'images_o{job["order"]}.json');assert len(ms)==1951
    assert all(set(m)=={'image_path','image_sha256','image_size'} for m in ms)
    h=Host(weights(c),dict(family='control',host='C',id='C',params={}),job['seed'],job['id'],device='cuda:0')
    initial=h.snapshot();reader=TargetReader(c['target_root'],256*1024**2,'image');started=time.time()
    with (dest/'predictions.bits').open('xb') as f:
        for i,m in enumerate(ms[:2] if qualify else ms):
            if i:reset_episode(h,initial)
            z,detail=h.step(reader.read(m));assert z.shape==(1,2,512,512) and torch.isfinite(z).all() and h.native.steps==1
            bits=np.packbits((z.sigmoid()>=.5).numpy()).tobytes();assert len(bits)==65536
            if i==0:
                with (Path(c['r50_root'])/'target'/f'C_s{job["seed"]}_o{job["order"]}'/'predictions.bits').open('rb') as reference:assert bits==reference.read(65536),'first-arrival C parity failed'
                save(dest/'PARITY.json',dict(first_arrival_matches_C=True))
            f.write(bits)
            if (i+1)%50==0:save(dest/'PROGRESS.json',dict(images=i+1,seconds=time.time()-started))
    reader.after_check();h.check_frozen(True);h.close()
    save(dest/'ONLINE_COMPLETE.json',dict(images=2 if qualify else 1951,target_label_reads=0,updates=2 if qualify else 1951,forwards=16 if qualify else 15608,first_arrival_matches_C=True,qualification_only=qualify))


def score(c):
    root=Path(c['output_root']);src=Path(c['r52_root']);reader=TargetReader(c['target_root'],256*1024**2,'mask');rows=[]
    assert all(read(root/'retirement'/f'{j["id"]}.json')['returncode']==0 for j in c['jobs'])
    for j in c['jobs']:
        dest=root/'target'/j['id'];assert read(dest/'ONLINE_COMPLETE.json')['images']==1951
        ms=read(src/'scorer'/f'FULL_o{j["order"]}.json');rs=[json.loads(x) for x in (src/'scorer/reused'/f'C_s{j["seed"]}_o{j["order"]}.private.jsonl').read_text().splitlines()];assert len(rs)==len(ms)==1951
        with (dest/'predictions.bits').open('rb') as f:
            for i,(r,m) in enumerate(zip(rs,ms)):
                assert (r['content'],r['visit'])==(m['image_sha256'],i+1)
                mask=np.unpackbits(np.frombuffer(f.read(65536),dtype=np.uint8)).reshape(2,512,512).astype(bool)
                rows.extend([r,dict(r,condition='EPISODIC_C',**control_fields(mask,reader.read(m)))])
            assert not f.read(1)
    reader.after_check();assert len(rows)==23412
    with (root/'scores.private.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    out=root/'public';main,domains=aggregate(rows);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    primary=[dict(r,role='PRIMARY_ALL') for r in rows if r['subset']=='remaining_dev']
    p,ci,negative=bootstrap_pair(primary,'EPISODIC_C','C',[20260907,17011,29009],role='PRIMARY_ALL');csvout(out/'PAIRED_SUMMARY.csv',[p]);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',ci);csvout(out/'NEGATIVE_CELLS.csv',negative)
    focus=[r['delta_pp'] for r in ci if r['domain']=='REFUGE_Valid' and r['channel']=='OC'];assert len(focus)==1
    passed=p['delta_pp']>=.1 and p['CI_low_pp']>0 and p['imageweighted_delta_pp']>=0 and p['worst_seed_averaged_cell_pp']>=-2 and min(p['order_delta_pp'])>0 and p['positive_trajectories']>=5 and focus[0]>=1
    save(out/'DECISION.json',dict(episodic_gate=passed,primary='EPISODIC_C minus C',refuge_valid_OC_delta_pp=focus[0],independent_validation=False))
    (out/'REPORT.md').write_text('# R59 episodic C history ablation\n\nFrozen gate: '+str(passed)+'\n\nAll six new trajectories and six reused C trajectories retained. Current-image update-before-predict remains; parameters and optimizer are reset before each new arrival while augmentation RNG continues. This jointly removes learned history and Adam history, not a separation of either cause. Development evidence only, no novel-method or independent-generalization claim.\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def main():
    torch.set_num_threads(2);scoring_self_check();c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root'])
    if os.environ.get('RUN_JOB'):
        online(c,next(j for j in c['jobs'] if j['id']==os.environ['RUN_JOB']));return
    started=time.time();state=dict(status='ONLINE',started=started);save(root/'RUN_STATE.json',state);cost=0
    try:
        for start in (0,3):
            running=[]
            for j,gpu in zip(c['jobs'][start:start+3],c['gpus']):
                env=dict(os.environ,RUN_JOB=j['id'],CUDA_VISIBLE_DEVICES=str(gpu));log=(root/'logs'/f'{j["id"]}.log').open('a');t=time.time()
                p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();running.append((j,p,t))
                save(root/'processes'/f'{j["id"]}.json',dict(pid=p.pid,started=t,gpu=gpu,active=True))
            failed=[]
            for j,p,t in running:
                rc=p.wait();elapsed=time.time()-t;cost+=elapsed;save(root/'retirement'/f'{j["id"]}.json',dict(returncode=rc,ended=time.time(),GPU_seconds=elapsed,active=False))
                if rc:failed.append(j['id'])
            if failed:raise RuntimeError('failed jobs: '+','.join(failed))
        state.update(status='CPU_SCORING',GPU_seconds=cost);save(root/'RUN_STATE.json',state);score(c)
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,scored_rows=23412);save(root/'RUN_STATE.json',state);save(root/'public/RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time(),GPU_seconds=cost);save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
