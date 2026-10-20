import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.ndimage import label

SEED = 20261018
SCORES = ('CROP_Q', 'CROP_LOGIT_Q', 'GEOMETRY_ONLY')
ARMS = ('SOURCE', 'B', 'ALL_COMMON', 'CROP_POS10', 'LOGIT_POS10', 'GEOM_POS10',
        'LOGIT_MATCH', 'GEOM_MATCH', 'MARGIN_MATCH', 'CYCLE_MATCH')


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def unpack(p):
    x = Path(p).read_bytes()
    assert len(x) == 65536
    return np.unpackbits(np.frombuffer(x, np.uint8)).reshape(2, 512, 512).astype(bool)


def components(s, b):
    return label((s != b).any(0), np.ones((3, 3), int))[0]


def stratum(r):
    return r['size'], r['direction'], r['distance_bucket']


def choose(rows, name, cap):
    selected = []
    for r in sorted(rows, key=lambda r: (-r['scores'][name], r['block_id'])):
        if r['scores'][name] > 0 and r['oc_pixels'] <= cap:
            selected.append(r['block_id'])
            cap -= r['oc_pixels']
    return selected


def matched(rows, chosen, name):
    selected = set(chosen)
    quota = Counter(stratum(r) for r in rows if r['block_id'] in selected)
    out = []
    for key, count in sorted(quota.items()):
        group = sorted((r for r in rows if stratum(r) == key), key=lambda r: r['block_id'])
        if name == 'CYCLE':
            out += [group[(i + 1) % len(group)]['block_id']
                    for i, r in enumerate(group) if r['block_id'] in selected]
        else:
            out += [r['block_id'] for r in sorted(group, key=lambda r: (-r['scores'][name], r['block_id']))[:count]]
    assert len(out) == len(set(out)) == len(chosen)
    assert Counter(stratum(r) for r in rows if r['block_id'] in set(out)) == quota
    return out


def aggregate(a):
    a = np.asarray(a, float)
    content = []
    for row in a:
        valid = np.isfinite(row)
        content.append(float(np.average(row[valid], weights=np.array([2, 1, 1, 1])[valid])) if valid.any() else np.nan)
    v = np.asarray(content)
    good = np.isfinite(v)
    ci = None
    if good.sum() == 8:
        ids = np.random.default_rng(SEED).integers(0, 8, (2000, 8))
        ci = np.quantile(v[ids].mean(1), [.025, .975]).tolist()
    return dict(mean=float(v[good].mean()) if good.any() else None, ci95=ci,
                defined_inputs=int(np.isfinite(a).sum()), defined_contents=int(good.sum()))


def auc(s, u, weighted):
    s, u = np.asarray(s), np.asarray(u)
    p, n = u > 0, u < 0
    if not p.any() or not n.any():
        return None
    weights = np.abs(u) if weighted else np.ones(len(u))
    pair = weights[p, None] * weights[None, n]
    credit = (s[p, None] > s[None, n]) + .5 * (s[p, None] == s[None, n])
    return float((credit * pair).sum() / pair.sum())


def audit_hook(origin, root, allowed, extra_denied=()):
    import sys
    allowed = {str(Path(p).resolve()) for p in allowed}
    denied = {str(Path(p).resolve()) for p in extra_denied}
    opened = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = str(Path(__import__('os').fsdecode(args[0])).resolve())
        if any(p == d or p.startswith(d + '/') for d in denied) or (p.startswith(str(origin) + '/') and p not in allowed):
            raise PermissionError('phase access boundary')
        mode = args[1]
        if isinstance(mode, str) and any(v in mode for v in 'wax+') and not p.startswith(str(root) + '/'):
            raise PermissionError('write outside audit root')
        opened.add(p)
    sys.addaudithook(hook)
    return opened
