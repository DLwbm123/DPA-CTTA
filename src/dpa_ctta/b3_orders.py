"""Frozen complete domain blocks; no selection or online rule changes."""
from collections import Counter
from itertools import permutations
from .p2_data import expected

ORDERS={0:('REFUGE','ORIGA','REFUGE_Valid','Drishti_GS'),1:('Drishti_GS','REFUGE_Valid','ORIGA','REFUGE'),2:('ORIGA','Drishti_GS','REFUGE','REFUGE_Valid'),3:('REFUGE_Valid','REFUGE','Drishti_GS','ORIGA')}
ARMS=('A','C','U','S','I')
COUNTS=dict(REFUGE=400,ORIGA=650,REFUGE_Valid=800,Drishti_GS=101)
BUDGET=dict(records=19510,forwards=132668,backwards=19510,base_adam=19510,perturb=0,restore=0)


def check_balance():
    domains=set(ORDERS[0])
    assert all(len(row)==4 and set(row)==domains for row in ORDERS.values())
    for i in range(4):assert Counter(row[i] for row in ORDERS.values())==Counter(domains)
    assert Counter(pair for row in ORDERS.values() for pair in zip(row,row[1:]))==Counter(permutations(domains,2))
    return dict(domain_position_balance=True,directed_adjacent_pairs=12,orders={str(k):list(v) for k,v in ORDERS.items()})


def build_stream(reg,order_id):
    if order_id not in ORDERS:raise ValueError('explicit order 0/1/2/3 required')
    base=expected(reg,'fundus',0)
    if len(base)!=1951 or len({r['group_id'] for r in base})!=1951 or Counter(r['domain'] for r in base)!=COUNTS:raise ValueError('frozen coverage')
    rows=[r for d in ORDERS[order_id] for r in base if r['domain']==d]
    if order_id in (0,1) and rows!=expected(reg,'fundus',order_id):raise ValueError('old within-block order drift')
    return rows
