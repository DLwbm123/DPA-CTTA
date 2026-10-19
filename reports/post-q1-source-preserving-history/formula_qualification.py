"""Data-free source-preserving feature-history working-model qualification."""
import json
import math
import time


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sigmoid(x):return math.exp(-math.log1p(math.exp(-x))) if x>=0 else math.exp(x)/(1+math.exp(x))


def literal(w,b,points,query,kappa=3.):
    # Simple source-equivalent symmetric means, fixed I covariance.
    n=[0.,0.];s=[[0.]*len(w) for _ in range(2)]
    for x in points:
        p=sigmoid(dot(w,x)+b)
        for c,v in enumerate((1-p,p)):
            v*=max(p,1-p)
            n[c]+=v/len(points)
            for d,z in enumerate(x):s[c][d]+=v*z/len(points)
    means=[[(kappa*(-.5 if c==0 else .5)*v+s[c][d])/(kappa+n[c]) for d,v in enumerate(w)] for c in range(2)]
    return dot([a-b for a,b in zip(means[1],means[0])],query)-.5*(dot(means[1],means[1])-dot(means[0],means[0]))+b


class WorkingHistory:
    """Fixed first-target diagonal geometry, calibrated residual; not a clinical posterior."""
    def __init__(self,w,b,first,kappa=3.):
        self.w=list(w);self.b=b;self.kappa=kappa;d=len(w)
        self.mu=[sum(x[j] for x in first)/len(first) for j in range(d)]
        self.var=[sum((x[j]-self.mu[j])**2 for x in first)/len(first) for j in range(d)]
        self.active=[j for j in range(d) if self.var[j]>0]
        if not self.active:raise ValueError('no variable feature directions')
        self.n=[0.,0.];self.s=[[0.]*d for _ in range(2)];self.images=0

    def add(self,points):
        # Total fractional support <=1/image, including source reliability weights.
        for x in points:
            p=sigmoid(dot(self.w,x)+self.b);reliability=max(p,1-p)
            for c,v in enumerate((1-p,p)):
                mass=reliability*v/len(points);self.n[c]+=mass
                for j in self.active:self.s[c][j]+=mass*(x[j]-self.mu[j])
        self.images+=1

    def prediction(self,x):
        source=dot(self.w,x)+self.b
        if self.images==0:return source,source,source
        means=[[(self.kappa*(-.5 if c==0 else .5)*self.var[j]*self.w[j]+self.s[c][j])/(self.kappa+self.n[c]) for j in range(len(x))] for c in range(2)]
        prior=dot([self.w[j] for j in self.active],[x[j]-self.mu[j] for j in self.active])
        q=[];r=[]
        for c in range(2):
            distance=sum((x[j]-self.mu[j]-means[c][j])**2/self.var[j] for j in self.active)
            h=1+1/(self.kappa+self.n[c]);q.append(-.5*distance);r.append(-.5*distance/h-len(self.active)*math.log(h)/2)
        return source,source-prior+q[1]-q[0],source-prior+r[1]-r[0]


def run():
    start=time.perf_counter();maxerr=0.;cases=0
    # Gaussian scores exactly reproduce source before adding evidence, in dimensions 1/2/32.
    for d in (1,2,32):
        w=[(j+1)/(d+2) for j in range(d)]
        for b in (-2.,0.,1.3):
            for t in range(16):
                x=[(t+j)/17 for j in range(d)]
                q0=-.5*sum((x[j]+w[j]/2)**2 for j in range(d))
                q1=-.5*sum((x[j]-w[j]/2)**2 for j in range(d))
                err=abs(q1-q0+b-(dot(w,x)+b));maxerr=max(maxerr,err);cases+=1
                assert err<1e-12
    # Initial equivalence alone does not make literal history updates coordinate-equivariant.
    xs=[[0.],[.5],[1.],[2.]];w=[1.7];b=-1.;query=[.5]
    source=sigmoid(dot(w,query)+b);literal_cases=[]
    for a,shift in ((1.,0.),(.25,0.),(4.,0.),(1.,2.)):
        ww=[w[0]/a];bb=b-ww[0]*shift;points=[[a*x[0]+shift] for x in xs];x=[a*query[0]+shift]
        assert abs(sigmoid(dot(ww,x)+bb)-source)<1e-12
        literal_cases.append(dict(scale=a,translation=shift,source=source,proposal=sigmoid(literal(ww,bb,points,x))))
    assert max(x['proposal'] for x in literal_cases)-min(x['proposal'] for x in literal_cases)>.01
    # First-target centered diagonal geometry + source residual fixes positive diagonal affine gauge.
    points=[[0.,0.,2.],[1.,2.,2.],[2.,1.,2.],[3.,4.,2.]]
    w=[.7,-.4,.2];b=-.3;query=[1.5,3.,5.];base=WorkingHistory(w,b,points)
    assert base.prediction(query)==(dot(w,query)+b,)*3
    base.add(points);expected=base.prediction(query);covariant=0
    for a,shift in (([.2,3.,2.],[2.,1.,4.]),([5.,.3,1.],[0.,3.,2.]),([1.,1.,1.],[5.,4.,3.])):
        ww=[v/scale for v,scale in zip(w,a)];bb=b-dot(ww,shift)
        transform=lambda x:[scale*v+t for v,scale,t in zip(x,a,shift)]
        new=WorkingHistory(ww,bb,[transform(x) for x in points]);new.add([transform(x) for x in points])
        observed=new.prediction(transform(query));err=max(abs(x-y) for x,y in zip(observed,expected));assert err<1e-11
        maxerr=max(maxerr,err);covariant+=1
    # Repeating every pixel does not add fractional image support or change geometry/predictions.
    repeated=WorkingHistory(w,b,points*17);repeated.add(points*17)
    err=max(abs(x-y) for x,y in zip(repeated.prediction(query),expected));assert err<1e-11
    assert max(abs(x-y) for x,y in zip(repeated.n,base.n))<1e-12
    assert sum(base.n)<=1. and base.var[2]==0 and 2 not in base.active
    # Orthogonal feature history can add information beyond identical frozen logits/histograms.
    a=[[0.,0.],[1.,1.],[2.,2.]];bb=[[0.,2.],[1.,1.],[2.,0.]]
    left=WorkingHistory([1.,0.],-1.,a);right=WorkingHistory([1.,0.],-1.,bb)
    left.add(a);right.add(bb);pa=left.prediction([1.,2.]);pb=right.prediction([1.,2.])
    assert all(dot([1.,0.],x)-1==dot([1.,0.],y)-1 for x,y in zip(a,bb))
    assert pa[0]==pb[0]==0 and pa[1]>0>pb[1]
    # Equal per-class support collapses uncertainty correction to one temperature about prior odds.
    qodds=.8;prior_odds=-.3;h=1+1/5
    equal_support_r=prior_odds+(qodds-prior_odds)/h
    assert abs(equal_support_r-.6166666666666667)<1e-12
    return dict(status='PASS',scope='DATA_FREE_MATHEMATICAL_QUALIFICATION_ONLY',source_initialization_cases=cases,maximum_numeric_error=maxerr,
        literal_history_gauge_counterexample=literal_cases,affine_equivariance_cases=covariant,
        image_replication_factor=17,image_unit_total_support=sum(base.n),replication_prediction_max_error=err,
        zero_variance_channel_passthrough=True,orthogonal_history_example=dict(source_probability=.5,left_proposal=sigmoid(pa[1]),right_proposal=sigmoid(pb[1]),left_evaluator=sigmoid(pa[2]),right_evaluator=sigmoid(pb[2]),same_source_logit_histories=True),
        equal_class_support_temperature_example=dict(q_logodds=qodds,prior_logodds=prior_odds,r_logodds=equal_support_r,h=h),
        new_gpu_runs=0,model_forwards=0,image_label_weight_reads=0,gpu_admission=False,
        limitations='Synthetic algebra only, no real-data efficacy or runtime certificate; current labels not used to choose parameters',body_seconds=time.perf_counter()-start)

if __name__=='__main__':print(json.dumps(run(),indent=2))
