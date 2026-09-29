import hashlib
from ..r3.plan import stream
from ..r1.plan import registration_digest


def sequence(registration,order):
    if type(order) is int and order in range(5):rows=stream(registration,order)
    elif order=='LONG10':rows=stream(registration,0)*10
    elif order=='MIXED':
        tag=registration_digest(registration)
        rows=[r for _,r in sorted(enumerate(stream(registration,0)),key=lambda x:(hashlib.sha256(f"R10_MIXED_V1|{tag}|{x[1]['group_id']}".encode()).digest(),x[1]['manifest_index']))]
    else:raise ValueError('R10 order')
    cycles=10 if order=='LONG10' else 1
    if len(rows)!=1951*cycles or sum(r['subset']=='remaining_dev' for r in rows)!=1695*cycles:raise ValueError('stream coverage')
    return rows
