"""Fixed half-source BN moments; no parameter updates or mixing-weight search."""
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


def mixed_bn(x,m,alpha=.5):
    current=x.mean((0,2,3));variance=x.var((0,2,3),unbiased=False)
    mu=(1-alpha)*m.running_mean+alpha*current
    var=(1-alpha)*m.running_var+alpha*variance+alpha*(1-alpha)*(current-m.running_mean).square()
    return (x-mu[None,:,None,None])*torch.rsqrt(var[None,:,None,None]+m.eps)*m.weight[None,:,None,None]+m.bias[None,:,None,None]


def self_check():
    m=torch.nn.BatchNorm2d(2).eval();x=torch.arange(32,dtype=torch.float32).reshape(1,2,4,4)/10
    assert torch.allclose(mixed_bn(x,m,0),m(x),atol=1e-6)
    native=torch.nn.functional.batch_norm(x,None,None,m.weight,m.bias,True,0,m.eps)
    assert torch.allclose(mixed_bn(x,m,1),native,atol=1e-6)
    assert torch.isfinite(mixed_bn(torch.ones_like(x),m)).all()
    scoring_self_check()


def online(c):
    from ..source_pilot import SourceOnlyHost
    from ..r19_model_only.runtime import weights
    root=Path(c['output_root']);ms=read(root/'images.json');assert len(ms)==1951
    assert all(set(m)=={'image_path','image_sha256','image_size'} for m in ms)
    h=SourceOnlyHost('fundus',weights(c),device='cuda:0');h.model.eval().requires_grad_(False)
    handles=[m.register_forward_hook(lambda m,args,out:mixed_bn(args[0],m)) for m in h.model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    assert handles;versions=[(p,p._version) for p in h.model.parameters()]
    reader=TargetReader(c['target_root'],256*1024**2,'image');started=time.time()
    with (root/'MIX.bits').open('xb') as f:
        for i,m in enumerate(ms):
            z=h.step(reader.read(m));assert z.shape==(1,2,512,512) and torch.isfinite(z).all()
            bits=np.packbits((z.sigmoid()>=.5).cpu().numpy()).tobytes();assert len(bits)==65536;f.write(bits)
            if i==0:first=bits
            if (i+1)%100==0:save(root/'PROGRESS.json',dict(images=i+1,seconds=time.time()-started))
        assert np.packbits((h.step(reader.read(ms[0])).sigmoid()>=.5).cpu().numpy()).tobytes()==first
    assert all(p._version==v for p,v in versions);reader.after_check()
    save(root/'ONLINE_COMPLETE.json',dict(images=1951,forwards=1952,stateless_replay_pass=True,target_label_reads=0))


def score(c):
    root=Path(c['output_root']);src=Path(c['r52_root']);prior=Path(c['r57_root'])
    assert read(root/'RETIREMENT.json')['returncode']==0 and read(prior/'RUN_STATE.json')['status']=='COMPLETE'
    assert (root/'MIX.bits').stat().st_size==1951*65536
    rows=[json.loads(x) for x in (prior/'scores.private.jsonl').read_text().splitlines()];assert len(rows)==35118
    reader=TargetReader(c['target_root'],256*1024**2,'mask');values={}
    with (root/'MIX.bits').open('rb') as f:
        for m in read(src/'scorer/FULL_o0.json'):
            mask=np.unpackbits(np.frombuffer(f.read(65536),dtype=np.uint8)).reshape(2,512,512).astype(bool)
            values[m['image_sha256']]=control_fields(mask,reader.read(m))
    reader.after_check();controls=[r for r in rows if r['condition']=='BATCH'];assert len(controls)==11706
    for r in controls:rows.append(dict(r,condition='MIX',**values[r['content']]))
    assert len(rows)==46824
    with (root/'scores.private.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    out=root/'public';main,domains=aggregate(rows);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    primary=[dict(r,role='PRIMARY_ALL') for r in rows if r['subset']=='remaining_dev'];pairs=[];negative=[]
    for baseline in ('BATCH','SOURCE','C'):
        p,_,n=bootstrap_pair(primary,'MIX',baseline,[20260907,17011,29009],role='PRIMARY_ALL');pairs.append(p);negative.extend(n)
    csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'NEGATIVE_CELLS.csv',negative);p=pairs[0]
    passed=p['delta_pp']>=.1 and p['CI_low_pp']>0 and p['imageweighted_delta_pp']>=0 and p['worst_seed_averaged_cell_pp']>=-2
    save(out/'DECISION.json',dict(fixed_anchor_gate=passed,primary='MIX minus BATCH',independent_validation=False))
    (out/'REPORT.md').write_text('# R58 fixed half-source BN anchor\n\nFrozen gate: '+str(passed)+'\n\nMIX/SOURCE/BATCH are each one stateless stream mapped across paired C orders/seeds; these are not independent replicated inference runs. Primary MIX-BATCH only; comparisons to C and SOURCE descriptive. No weight search, parameter update, independent validation or superiority over adaptive methods is established by this gate.\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def main():
    torch.set_num_threads(2);self_check();c=read(os.environ['RUN_CONFIG']);root=Path(c['output_root'])
    if os.environ.get('RUN_PHASE')=='online':online(c);return
    started=time.time();state=dict(status='ONLINE',started=started);save(root/'RUN_STATE.json',state)
    try:
        worker_started=time.time();p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=dict(os.environ,RUN_PHASE='online',CUDA_VISIBLE_DEVICES=str(c['gpu'])))
        save(root/'WORKER.json',dict(pid=p.pid,started=worker_started,active=True))
        rc=p.wait();ended=time.time();save(root/'RETIREMENT.json',dict(returncode=rc,ended=ended,active=False));assert rc==0
        state.update(status='CPU_SCORING',GPU_seconds=ended-worker_started);save(root/'RUN_STATE.json',state);score(c)
        state.update(status='COMPLETE',ended=time.time(),wall_seconds=time.time()-started,scored_rows=46824);save(root/'RUN_STATE.json',state);save(root/'public/RUN_STATE.json',state)
    except BaseException as e:
        state.update(status='FAILED',reason=str(e),ended=time.time());save(root/'RUN_STATE.json',state);raise


if __name__=='__main__':main()
