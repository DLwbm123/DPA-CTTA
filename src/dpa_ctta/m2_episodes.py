"""Metadata-only deterministic M2 episode construction; no model/data I/O."""
from collections import Counter, defaultdict
from hashlib import sha256

TRANSFORMS = ('clean', 'gamma_0.7', 'gamma_1.5', 'blur_5_sigma_1')


def make_m2_episodes(old, task, seed=20260907):
    """Preserve M1 marginals, balance crossed exposures, use unique triples."""
    if task not in ('fundus', 'polyp') or len(old) != 600:
        raise ValueError('Expected one registered M1 600-episode task')
    expected_q = 40 if task == 'fundus' else 64
    for i, e in enumerate(old):
        if (e.get('episode') != i + 1 or type(e.get('query_index')) is not int
                or type(e.get('state_index')) is not int or not 0 <= e['query_index'] < expected_q
                or not 0 <= e['state_index'] < 32 or e.get('transform') not in TRANSFORMS):
            raise ValueError('Invalid registered episode identity/order')
    qc = Counter(e['query_index'] for e in old)
    sc = Counter(e['state_index'] for e in old)
    tc = Counter(e['transform'] for e in old)
    if len(sc) != 32 or len(qc) != (40 if task == 'fundus' else 64):
        raise ValueError('M1 identities/counts differ; do not silently resample')
    if tc != Counter({t: 150 for t in TRANSFORMS}):
        raise ValueError('Original transform marginal differs')

    def key(*parts):
        text = ':'.join(map(str, ('M2', seed, task, *parts)))
        return sha256(text.encode('utf-8')).hexdigest()

    def quotas(counts, tag):
        result, cursor = {}, 0
        for item in sorted(counts, key=lambda i: (key(tag, i), i)):
            base, extra = divmod(counts[item], 4)
            values = [base] * 4
            for j in range(extra):
                values[(cursor + j) % 4] += 1
            cursor = (cursor + extra) % 4
            result[item] = values
        if any(sum(v[t] for v in result.values()) != 150 for t in range(4)):
            raise ValueError('Unbalanced transform quota')
        return result

    qt, st = quotas(qc, 'query-quota'), quotas(sc, 'state-quota')
    seen_states = defaultdict(set)
    triples = []
    # Bipartite degree construction for each transform. No seed/score search.
    for t in range(4):
        remaining = {s: st[s][t] for s in sc}
        order = sorted(qc, key=lambda q: (-qt[q][t], key('row', t, q), q))
        for q in order:
            need = qt[q][t]
            available = [s for s in sc if remaining[s] > 0]
            available.sort(key=lambda s: (
                -remaining[s], s in seen_states[q], key('edge', t, q, s), s
            ))
            if len(available) < need:
                raise ValueError('Quota construction error; fix offline, no GPU')
            for s in available[:need]:
                triples.append((s, q, t))
                remaining[s] -= 1
                seen_states[q].add(s)
        if any(remaining.values()):
            raise ValueError('Unfilled state quota')

    triples.sort(key=lambda v: (key('playback', *v), v))
    rows = [dict(episode=i + 1, state_index=s, query_index=q,
                 transform=TRANSFORMS[t])
            for i, (s, q, t) in enumerate(triples)]
    if len(rows) != 600 or len(set(triples)) != 600:
        raise ValueError('Coverage or duplicate-triple error')
    if Counter(e['query_index'] for e in rows) != qc:
        raise ValueError('Query marginal changed')
    if Counter(e['state_index'] for e in rows) != sc:
        raise ValueError('State marginal changed')
    if Counter(e['transform'] for e in rows) != tc:
        raise ValueError('Transform marginal changed')
    if min(len(v) for v in seen_states.values()) < 4:
        raise ValueError('Insufficient crossed query-state coverage')
    for q in qc:
        if any(qt[q][t] == 0 for t in range(4)):
            raise ValueError('Missing query-transform pair')
    for s in sc:
        if any(st[s][t] == 0 for t in range(4)):
            raise ValueError('Missing state-transform pair')
    validate_coverage(old, rows)
    return rows


def validate_coverage(old, new):
    if len(old) != 600 or len(new) != 600:
        raise ValueError('Exactly 600 episodes required')
    for field in ('query_index', 'state_index', 'transform'):
        if Counter(e[field] for e in old) != Counter(e[field] for e in new):
            raise ValueError('Exposure marginal changed: ' + field)
    triples = [(e['state_index'], e['query_index'], e['transform']) for e in new]
    if len(set(triples)) != 600 or [e['episode'] for e in new] != list(range(1, 601)):
        raise ValueError('Duplicate triple or episode order mismatch')
    for field in ('query_index', 'state_index'):
        for index in {e[field] for e in old}:
            rows = [e for e in new if e[field] == index]
            counts = Counter(e['transform'] for e in rows)
            if set(counts) != set(TRANSFORMS) or max(counts.values()) - min(counts.values()) > 1:
                raise ValueError('Unbalanced crossed transform coverage')
            if field == 'query_index' and len({e['state_index'] for e in rows}) < 4:
                raise ValueError('Query sees fewer than four states')
    return True


def coverage_summary(rows):
    def histogram(values):
        return {str(k): v for k, v in sorted(Counter(values).items())}
    result = {'episodes': len(rows)}
    for field in ('query_index', 'state_index', 'transform'):
        result[field + '_exposure_distribution'] = histogram(Counter(e[field] for e in rows).values())
    for field in ('query_index', 'state_index'):
        groups = [[e for e in rows if e[field] == i] for i in sorted({e[field] for e in rows})]
        result[field + '_distinct_transforms_distribution'] = histogram(len({e['transform'] for e in g}) for g in groups)
        result[field + '_transform_quota_patterns'] = {str(k):v for k,v in sorted(Counter(tuple(sorted(Counter(e['transform'] for e in g).get(t,0) for t in TRANSFORMS)) for g in groups).items())}
        if field == 'query_index':
            result['query_distinct_states_distribution'] = histogram(len({e['state_index'] for e in g}) for g in groups)
    triples = Counter((e['state_index'], e['query_index'], e['transform']) for e in rows)
    result.update(unique_triples=len(triples), maximum_triple_repeat=max(triples.values()), triple_repeat_distribution=histogram(triples.values()))
    return result
