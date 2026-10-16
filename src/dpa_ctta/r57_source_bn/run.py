"""Retired stateless source/BN controls, then independent CPU scoring."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from ..r10_12h_core.run import read, save
from ..r7_target_screen.runner import TargetReader
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout
from ..r24_c_context.run import hard_metrics

WIDTH=65536


def online(c):
    from ..source_pilot import SourceOnlyHost
    from ..b1_host import configure
    from ..r19_model_only.runtime import weights
    torch.set_num_threads(2);torch.manual_seed(20260907)
    torch.backends.cudnn.benchmark=False
    root=Path(c['output_root']);rows=read(root/'images.json');assert len(rows)==1951
    assert all(set(r)=={'image_path','image_sha256','image_size'} for r in rows)
    state=weights(c);host=SourceOnlyHost('fundus',state,device='cuda:0');host.model.eval().requires_grad_(False)
    baseline=SourceOnlyHost('fundus',state,device='cuda:0');configure(baseline.model);baseline.model.requires_grad_(False)
    assert not any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in baseline.model.modules())
    assert all(not p.requires_grad for h in (host,baseline) for p in h.model.parameters())
    versions=[[(p,p._version) for p in h.model.parameters()] for h in (host,baseline)]
    reader=TargetReader(c['target_root'],256*1024**2,'image');initial=[];started=time.time()
    with (root/'SOURCE.bits').open('xb') as sf,(root/'BATCH.bits').open('xb') as bf:
        for i,row in enumerate(rows):
            x=reader.read(row)
            for j,(h,f) in enumerate(((host,sf),(baseline,bf))):
                z=h.step(x);assert z.shape==(1,2,512,512) and torch.isfinite(z).all()
                bits=np.packbits((z.sigmoid()>=.5).cpu().numpy()).tobytes();assert len(bits)==WIDTH;f.write(bits)
                if i==0:initial.append(bits)
            if (i+1)%100==0:save(root/'PROGRESS.json',dict(images=i+1,seconds=time.time()-started))
        # Stateless replay check: same first image after all others, no parameter writes.
        x=reader.read(rows[0])
        for j,h in enumerate((host,baseline)):
            assert np.packbits((h.step(x).sigmoid()>=.5).cpu().numpy()).tobytes()==initial[j]
            assert all(p._version==v for p,v in versions[j])
    reader.after_check();torch.cuda.synchronize()
    save(root/'ONLINE_COMPLETE.json',dict(images=1951,streams=2,target_label_reads=0,forwards=3904,stateless_replay_pass=True,GPU_seconds=time.time()-started))


def score(c):
    root=Path(c['output_root']);src=Path(c['r52_root'])
    assert read(root/'ONLINE_COMPLETE.json')['images']==1951
    assert read(root/'RETIREMENT.json')['returncode']==0
    manifest=read(src/'scorer/FULL_o0.json');index={m['image_sha256']:i for i,m in enumerate(manifest)};assert len(index)==1951
    assignment=read(src/'scorer/SPLIT.private.json');reader=TargetReader(c['target_root'],256*1024**2,'mask')
    values={};allrows=[]
    for method in ('SOURCE','BATCH'):
        assert (root/(method+'.bits')).stat().st_size==1951*WIDTH
        values[method]={}
        with (root/(method+'.bits')).open('rb') as f:
            for m in manifest:
                mask=np.unpackbits(np.frombuffer(f.read(WIDTH),dtype=np.uint8)).reshape(2,512,512).astype(bool)
                values[method][m['image_sha256']]=hard_metrics(mask,reader.read(m))
    reader.after_check()
    for seed in (20260907,17011,29009):
        for order in (0,1):
            ms=read(src/'scorer'/f'FULL_o{order}.json');rs=[json.loads(x) for x in (src/'scorer/reused'/f'C_s{seed}_o{order}.private.jsonl').read_text().splitlines()]
            assert len(rs)==len(ms)==1951
            for i,(r,m) in enumerate(zip(rs,ms)):
                assert (r['content'],r['visit'])==(m['image_sha256'],i+1)
                allrows.append(r)
                for method in ('SOURCE','BATCH'):
                    allrows.append(dict(condition=method,seed=seed,order=order,visit=i+1,content=m['image_sha256'],domain=m['domain'],subset=m['subset'],role=assignment.get(m['image_sha256'],'CONTEXT'),metrics=values[method][m['image_sha256']],diagnostics={}))
    assert len(allrows)==35118
    with (root/'scores.private.jsonl').open('x') as f:
        for r in allrows:f.write(json.dumps(r)+'\n')
    main,domains=aggregate(allrows);out=root/'public';csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    primary=[dict(r,role='PRIMARY_ALL') for r in allrows if r['subset']=='remaining_dev'];pairs=[]
    for a,b in [('C','SOURCE'),('BATCH','SOURCE'),('C','BATCH')]:
        pair,_,_=bootstrap_pair(primary,a,b,[20260907,17011,29009],role='PRIMARY_ALL');pairs.append(pair)
    csvout(out/'PAIRED_SUMMARY.csv',pairs)
    counts={}
    for r in primary:
        for m in r['metrics']:
            k=(r['condition'],r['seed'],r['order'],r['domain'],m['channel']);v=counts.setdefault(k,[0,0,0])
            for j,key in enumerate(('TP','FP','FN')):v[j]+=m[key]
    rows=[]
    for (method,seed,order,domain,channel),(tp,fp,fn) in sorted(counts.items()):
        rows.append(dict(method=method,seed=seed,order=order,domain=domain,channel=channel,TP=tp,FP=fp,FN=fn,area_ratio=(tp+fp)/(tp+fn)))
    csvout(out/'CONFUSION.csv',rows);lookup={(r['method'],r['seed'],r['order'],r['domain'],r['channel']):r for r in rows}
    diffs=[r['area_ratio']-lookup[('SOURCE',r['seed'],r['order'],r['domain'],r['channel'])]['area_ratio'] for r in rows if r['method']=='C' and r['domain']=='REFUGE_Valid' and r['channel']=='OC'];assert len(diffs)==6
    decision=dict(incremental_over_source_supported=mean(diffs)>=.05 and sum(x>0 for x in diffs)>=5,mean_area_excess=mean(diffs),positive_pairs=sum(x>0 for x in diffs),independent_validation=False)
    save(out/'DECISION.json',decision)
    (out/'REPORT.md').write_text('# R57 source and current-image BN controls\n\n'+json.dumps(decision,indent=2)+'\n\nSOURCE and BATCH are each one stateless stream reused across paired seeds/orders, not six independent runs. C retains its original update-before-predict convention. Bootstrap contrasts are descriptive and couple content across seeds/orders. No independent validation or algorithm improvement claim.\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def main():
    torch.set_num_threads(2);c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root'])
    if os.environ.get('RUN_PHASE')=='online':online(c);return
    started=time.time();state=dict(status='ONLINE',started=started);save(root/'RUN_STATE.json',state)
    try:
        env=dict(os.environ,RUN_PHASE='online',CUDA_VISIBLE_DEVICES=str(c['gpu']))
        p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env)
        save(root/'WORKER.json',dict(pid=p.pid,started=time.time(),active=True))
        rc=p.wait();save(root/'RETIREMENT.json',dict(returncode=rc,ended=time.time(),active=False));assert rc==0
        state['status']='CPU_SCORING';save(root/'RUN_STATE.json',state);score(c)
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,scored_rows=35118,GPU_seconds=read(root/'ONLINE_COMPLETE.json')['GPU_seconds']);save(root/'RUN_STATE.json',state);save(root/'public/RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time());save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
