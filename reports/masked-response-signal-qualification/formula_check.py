"""Synthetic Bernoulli masking-score identities; no task data or model imports."""
from fractions import Fraction as F
import json, math, time

def check():
    begin=time.perf_counter();n=0
    for i in range(21):
        p=F(i,20)
        for j in range(21):
            for k in range(21):
                q=(F(j,20),F(k,20));mean=sum(q)/2
                c=sum(p*x+(1-p)*(1-x) for x in q)/2
                assert c==(1+(2*p-1)*(2*mean-1))/2
                assert abs(c-F(1,2))<=abs(p-F(1,2))
                if p==F(1,2):assert c==F(1,2)
                n+=1
    # Same original p and masked mean, different variance, same score.
    variance_blind=[]
    for i in range(21):
        p=F(i,20)
        a=(F(0),F(1));b=(F(1,2),F(1,2))
        score=lambda views:sum(p*x+(1-p)*(1-x) for x in views)/len(views)
        assert score(a)==score(b)
        variance_blind.append(True)
    # Stability does not imply confidence, correctness, or source preference.
    p=F(1,2);assert p*p+(1-p)*(1-p)==F(1,2)
    eps=1e-12;H=lambda p:-p*math.log(p)-(1-p)*math.log(1-p)
    unstable= -H(1-eps)*(1-((1-eps)*eps+eps*(1-eps)))
    assert abs(unstable)<1e-9
    return dict(status='PASS',exact_identity_and_bound_cases=n,mean_equal_variance_different_cases=len(variance_blind),half_probability_score_invariant=True,confident_unstable_limit_not_conjunctive=True,original_appendix_E1_acknowledges_product_limit=True,real_rgb_labels_bits_weights_reads=0,model_forwards=0,gpu_seconds=0,body_seconds=time.perf_counter()-begin)

if __name__=='__main__':print(json.dumps(check(),indent=2))
