"""Data-free extremal Dice-loss qualification; not an adaptation controller."""
from fractions import Fraction
import itertools,json,time
from pathlib import Path

def worst_loss(source_area,removed,added):
    """Exact maximum Dice(source,Y)-Dice(proposal,Y), empty-empty Dice=1."""
    A,d,a=source_area,removed,added
    if any(type(x) is not int or x<0 for x in (A,d,a)) or d>A:
        raise ValueError('nonnegative integer edit counts required')
    if A==0:return Fraction(int(a>0)),0
    B=A+a;c=A-d
    # The sign of f(t+1)-f(t) is nonincreasing on t >= 0.
    lo,hi=0,c
    while lo<hi:
        t=(lo+hi)//2
        sign=A*(B+t)*(B+t+1)-B*(A+d+t)*(A+d+t+1)
        if sign>0:lo=t+1
        else:hi=t
    t=lo
    return Fraction(2*(d+t),A+d+t)-Fraction(2*t,B+t),t

def dice(x,y):
    A=x.bit_count();G=y.bit_count()
    return Fraction(2*(x&y).bit_count(),A+G) if A+G else Fraction(1)

def run():
    started=time.perf_counter();counts={}
    n=5;checked=0;observed=0
    for s,p in itertools.product(range(1<<n),repeat=2):
        A=s.bit_count();d=(s&~p).bit_count();a=(p&~s).bit_count()
        bound,t=worst_loss(A,d,a)
        actual=max(dice(s,y)-dice(p,y) for y in range(1<<n))
        assert actual==bound,(s,p,actual,bound)
        checked+=1;observed+=1<<n
    counts['exhaustive_source_proposal_pairs']=checked
    counts['exhaustive_truth_comparisons']=observed
    for A in range(1,50):
        for d in range(A+1):
            assert worst_loss(A,d,0)[0]==Fraction(2*d,A+d)
        for a in range(30):
            assert worst_loss(A,0,a)[0]==Fraction(a,2*A+a)
    counts['deletion_only_cases']=sum(A+1 for A in range(1,50))
    counts['addition_only_cases']=49*30
    for A in range(1,30):
        for d in range(A+1):
            for a in range(15):
                v,t=worst_loss(A,d,a);B=A+a
                brute=max(Fraction(2*(d+j),A+d+j)-Fraction(2*j,B+j) for j in range(A-d+1))
                assert v==brute
    counts['binary_search_equals_exact_finite_scan']=sum(A+1 for A in range(1,30))*15
    for A,d,a in [(100,1,0),(100,0,1),(10,2,3),(4000,10,9)]:
        value,_=worst_loss(A,d,a)
        for k in (2,3,11):assert worst_loss(k*A,k*d,k*a)[0]>=value
        # Replication restricts integer t differently, so exact invariance is not claimed.
    for args in [(-1,0,0),(1,2,0),(1.0,0,0),(1,0,-1)]:
        try:worst_loss(*args)
        except ValueError:pass
        else:raise AssertionError(args)
    # Source-unary candidate scoring can prefer the smaller confident island.
    assert sum([3.,3.])>sum([.1]*10)
    # A connected bridge can beat every separately filled source component.
    assert sum([1.,-.1,1.])>max(1.,1.)
    # Every one-pixel deletion can be more harmful than an addition of equal count.
    assert worst_loss(100,1,0)[0]>worst_loss(100,0,1)[0]
    return dict(status='PASS',checks=counts,examples=[dict(source_area=A,removed=d,added=a,exact_bound=str(worst_loss(A,d,a)[0]),bound_pp=float(worst_loss(A,d,a)[0])*100) for A,d,a in [(100,1,0),(100,0,1),(100,1,1),(6000,10,10)]],no_real_data_labels_weights_model_cuda=True,body_seconds=time.perf_counter()-started)
if __name__=='__main__':
    result=run();Path(__file__).with_name('FORMULA_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
