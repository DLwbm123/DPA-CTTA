"""Export only paired cells from already published R94 aggregate references."""
import csv
import io
import json
import os
import signal
import time
from pathlib import Path

START = time.time()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('export alarm')))
signal.alarm(10)
root = Path(os.environ['ROOT']).resolve()
assert time.time() < float(os.environ['DEADLINE'])
public = root / 'public'
with (public / 'STATIC_REFERENCE_CELLS.csv').open() as f:
    rows = list(csv.DictReader(f))
key = lambda r: tuple(r[k] for k in ('variant','domain','seed','order','channel','metric'))
q = {key(r):r for r in rows if r['arm'] == 'Q0'}
result=[]
for r in rows:
    if r['arm'] != 'K':
        continue
    b = q[key(r)]
    assert r['n_contents'] == b['n_contents'] and r['n_NA'] == b['n_NA'] == '0'
    result.append(dict(variant=r['variant'],domain=r['domain'],seed=r['seed'],
                       order=r['order'],channel=r['channel'],metric=r['metric'],
                       K_minus_Q0=float(r['mean'])-float(b['mean']),n_contents=r['n_contents']))
assert len(result) == 1056
with (public / 'DOMAIN_HEAD.csv').open() as f:
    for r in csv.DictReader(f):
        a = [x['K_minus_Q0'] for x in result if all(x[k] == r[k] for k in ('variant','domain','channel','metric'))]
        assert len(a) == 6 and abs(sum(a)/6 - float(r['paired_mean_delta'])) < 1e-10
s = io.StringIO();w = csv.DictWriter(s,fieldnames=list(result[0]),lineterminator="\n");w.writeheader();w.writerows(result)
data=s.getvalue().encode()
used=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (root,*root.rglob('*')) if not p.is_symlink())
assert used+len(data)+16384 < 7*1024**2
(public / 'ANCHOR_DOMAIN_HEAD.csv').write_bytes(data)
(public / 'ANCHOR_EXPORT.json').write_text(json.dumps(dict(status='PASS',rows=1056,existing_aggregate_reconstruction_checks=176,body_seconds=time.time()-START,new_private_reads=0,new_statistics=False)))
