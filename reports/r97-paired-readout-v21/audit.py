"""Existing score/mask audit only. Does not import a model or read labels."""
import gzip,json,time
from pathlib import Path
import numpy as np
from signal_v21 import self_check

def read(p):return json.loads(Path(p).read_text())
def run(c,write):
    started=time.time();old=read(c['prior_config']);r91=Path(old['output_root'])
    with np.load(r91/'scorer/values.private.npz',allow_pickle=False) as f:v=f['values'];domains=f['domains']
    assert v.shape==(192,6,6,2,2,11) and np.isfinite(v[...,0]).all()
    assert np.bincount(domains).tolist()==[22,57,57,56]
    done=read(r91/'public/COMPLETION.json');assert done['PRE_K_reference_exact_all'] and done['Q0_P0_identity_exact']
    panels=[read(r91/'private'/f'screen_o{o}.json') for o in (0,1)]
    ids={r['image_sha256']:i for i,r in enumerate(panels[0])};assert len(ids)==192
    def unpack(raw):
        assert len(raw)==65536
        return np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool)
    with gzip.open(r91/'jobs/Q0.bits.gz','rb') as f:qs=[unpack(f.read(65536)) for _ in range(192)];assert not f.read(1)
    with open(old['reference_predictions']['static'],'rb') as f:
        p0=[]
        for _ in range(192):
            raw=f.read(131072);assert len(raw)==131072;p0.append(unpack(raw[65536:]))
        assert not f.read(1)
    eligibility=np.zeros((192,2),bool);reverse=np.zeros((2,2),np.int64);nr=reverse.copy();nd=reverse.copy()
    for o in (0,1):
        with open(old['reference_predictions'][f'20260907_{o}'],'rb') as f:
            for row in panels[o]:
                i=ids[row['image_sha256']];raw=f.read(131072);assert len(raw)==131072
                p,k=unpack(raw[:65536]),unpack(raw[65536:]);q=qs[i]
                e=p.astype(np.int8)-p0[i].astype(np.int8);d=k!=q;r=(e!=0)&d
                eligibility[i,o]=r.any();nr[o]+=r.sum((1,2));nd[o]+=d.sum((1,2))
                reverse[o]+=((e*(k.astype(np.int8)-q.astype(np.int8))<0)&r).sum((1,2))
            assert not f.read(1)
    # v is content, anchor, arm, raw/post, head, metric; the first two anchors are frozen seed.
    g=v[:,:2,4,:,:,0]-v[:,:2,2,:,:,0]
    def macro(x):return float(np.mean([x[domains==d].mean() for d in range(4)]))
    rows=[]
    for o in (0,1):
        for vi,name in enumerate(('raw','post')):
            gi=g[:,o,vi].mean(1)
            rows.append(dict(order=o,variant=name,input_oracle_pp=macro(np.maximum(gi,0)),eligible_oracle_pp=macro(eligibility[:,o]*np.maximum(gi,0)),class_relaxed_oracle_pp=macro(np.maximum(g[:,o,vi],0)),eligible_contents=int(eligibility[:,o].sum()),Q_minus_K_pp=macro(gi)))
    admitted=all(x['input_oracle_pp']>=.1 and x['eligible_oracle_pp']>=.1 for x in rows if x['variant']=='raw')
    result=dict(status='PASS_TO_QUALIFICATION' if admitted else 'STOP_ACTION_SPACE',admitted=admitted,
                minimum_development_space_pp=.1,oracle=rows,R_count=nr.tolist(),D_count=nd.tolist(),reverse_count=reverse.tolist(),
                R_coverage=np.divide(nr,nd,out=np.zeros_like(nr,dtype=float),where=nd>0).tolist(),
                reverse_fraction=np.divide(reverse,nr,out=np.zeros_like(nr,dtype=float),where=nr>0).tolist(),
                candidate_semantics='Original deterministic six geometric weak views; no new keys; actual GPU endpoints must match every existing bitstream before scoring',
                old_scores_reused=True,new_label_reads=0,new_model_calls=0,CPU_body_seconds=time.time()-started)
    write('public/DEFINITION_CHECK.json',self_check());write('public/ACTION_AUDIT.json',result)
    return result
