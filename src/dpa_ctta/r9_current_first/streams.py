"""Registered standard orders, new MIXED salt, and visit-preserving LONG10."""
import hashlib
from ..r3.plan import stream
from ..r1.plan import registration_digest


def sequence(registration,order):
    if type(order) is int and order in range(5):rows=stream(registration,order)
    elif order=='LONG10':rows=stream(registration,0)*10
    elif order=='MIXED':
        base=stream(registration,0);tag=registration_digest(registration)
        rows=[r for i,r in sorted(enumerate(base),key=lambda x:(hashlib.sha256(f"R9_MIXED_V1|{tag}|{x[1]['group_id']}".encode()).digest(),x[1]['manifest_index']))]
    else:raise ValueError('R9 registered stream only')
    cycles=10 if order=='LONG10' else 1
    if len(rows)!=1951*cycles or sum(r['subset']=='remaining_dev' for r in rows)!=1695*cycles:
        raise ValueError('R9 stream coverage')
    return rows


def visits(rows):
    return [dict(visit=i+1,cycle=i//1951+1,cycle_visit=i%1951+1,content=r['group_id']) for i,r in enumerate(rows)]
