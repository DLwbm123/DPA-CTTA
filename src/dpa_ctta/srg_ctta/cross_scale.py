"""Oracle-conditioned diagnostic scales, always anchored on the query mean."""
import torch
from torch.nn import functional as F

ARMS=('SOURCE','CURRENT_RB','CROSS_SCALE','RELATIVE_SCALE')
GROUPS=((0,1,2,3),(4,5,6,7))


def supporters(query):
    if query not in range(8):raise ValueError('fixed eight contents')
    return GROUPS[1] if query in GROUPS[0] else GROUPS[0]


def pool(means,variances):
    if means.shape!=variances.shape or means.ndim!=2 or means.shape[0]!=4 or not torch.isfinite(means).all() or not torch.isfinite(variances).all() or (variances<0).any():
        raise ValueError('four finite equally weighted image moments')
    mu=means.double().mean(0);within=variances.double().mean(0);between=(means.double()-mu).square().mean(0)
    return mu,within+between,within,between


def anchored_output(bn,h,query_mean,target_variance):
    if not torch.isfinite(target_variance).all() or (target_variance<0).any():raise ValueError('invalid pooled variance')
    ss=(bn.running_var.double()+bn.eps).sqrt();st=(target_variance+bn.eps).sqrt()
    mu=query_mean-st/ss*(query_mean-bn.running_mean.double())
    return F.batch_norm(h,mu.to(h),target_variance.to(h),bn.weight,bn.bias,False,0.,bn.eps)


def relative_output(source_output,original_variance,current_variance,eps):
    ratio=((original_variance+eps)/(current_variance+eps)).sqrt()
    if not torch.isfinite(ratio).all() or (ratio<=0).any():raise ValueError('invalid relative scale')
    if torch.equal(ratio,torch.ones_like(ratio)):return source_output.clone()
    mean=source_output.double().mean((0,2,3),keepdim=True)
    result=(mean+ratio[None,:,None,None]*(source_output.double()-mean)).to(source_output)
    if not torch.isfinite(result).all():raise ValueError('nonfinite scaled output')
    return result


def mean_check(source,changed):
    a=source.double().mean((0,2,3));b=changed.double().mean((0,2,3))
    error=float((a-b).abs().max());scale=max(1.,float(a.abs().max()),float(b.abs().max()))
    if not torch.isfinite(b).all() or error>5e-5*scale:raise ValueError('query mean anchor failed')
    return error
