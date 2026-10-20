"""Frozen no-label certificate acceptance census; roots supplied by environment."""
import json,os,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from formula_qualification import worst_loss

def run():
    roots={k:Path(v).resolve() for k,v in json.loads(os.environ['CENSUS_ROOTS_JSON']).items()}
    assert set(roots)=={'p2','p5'}
    for r in roots.values():assert json.loads((r/'RESOURCE_LEDGER.json').read_text())['all_workers_retired']
    allowed=set()
    for r in roots.values():
        allowed.add((r/'private/predictions.json').resolve())
        allowed.update(x.resolve() for x in (r/'predictions').glob('*.bits'))
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p.is_relative_to(r) for r in roots.values()) and p not in allowed:
            raise PermissionError('prediction bits and indexes only')
    sys.addaudithook(audit)
    for r in roots.values():
        try:(r/'public/RESULTS.json').read_text()
        except PermissionError:pass
        else:raise AssertionError('outcome guard failed')
    lookup={k:{(v['case'],v['condition'],v['arm']):v['index'] for v in json.loads((r/'private/predictions.json').read_text())} for k,r in roots.items()}
    def mask(which,i,k,a):
        path=roots[which]/'predictions'/f"{lookup[which][i,k,a]:04d}.bits"
        raw=path.read_bytes()
        if len(raw)!=65536:raise ValueError('native packed binary shape')
        return np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
    arms=('SOURCE','SMOOTH','FILL_SOURCE','TOPO_SOURCE','TOPO_SMOOTH','FILL_SMOOTH')
    data={(a,ch):[] for a in arms for ch in range(2)};started=time.perf_counter()
    for i in range(8):
        for k in range(4):
            source=mask('p2',i,k,'SOURCE')
            for a in arms:
                proposal=source if a=='SOURCE' else mask('p5' if a=='FILL_SMOOTH' else 'p2',i,k,a)
                for ch in range(2):
                    s,p=source[ch],proposal[ch];A=int(s.sum());d=int((s&~p).sum());add=int((p&~s).sum());v,t=worst_loss(A,d,add)
                    data[a,ch].append((v,d,add))
    out=[]
    for (a,ch),q in data.items():
        assert len(q)==32
        out.append(dict(arm=a,channel=('OD','OC')[ch],events=32,accepted_at_one_pp=sum(v<=Fraction(1,100) for v,d,b in q),mean_bound_pp=float(sum(v for v,d,b in q)/32)*100,max_bound_pp=float(max(v for v,d,b in q))*100,removed_pixels_total=sum(d for v,d,b in q),added_pixels_total=sum(b for v,d,b in q)))
    return dict(status='PASS',threshold_pp=1,unique_inputs=32,rows=out,no_labels_outcomes_rgb_weights_model_cuda=True,scope='Acceptance eligibility only, not task performance or online selection',body_seconds=time.perf_counter()-started)
if __name__=='__main__':print(json.dumps(run()))
