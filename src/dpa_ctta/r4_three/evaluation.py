"""One-way post-commit evaluator. No host/model/optimizer references accepted."""
import time
import torch
from ..r1.assets import target
from ..r1.region_memory import grid
from ..p1_analysis import evaluate
from ..p1_run import sync


def quality(q,mask):
    if q.shape!=mask.shape or q.shape!=(1,2,512,512) or q.requires_grad:raise ValueError('detached current probability and mapped mask')
    if not torch.all((mask==0)|(mask==1)) or not torch.isfinite(q).all() or bool(((q<0)|(q>1)).any()):raise ValueError('probability/binary ground truth')
    result=[]
    for c,channel in enumerate(('OD','OC')):
        p=q[0,c].double();g=mask[0,c].bool();h=p>=.5;error=(p-g.double()).square()
        n=g.numel();ng=int(g.sum());np=int(h.sum());intersection=int((h&g).sum())
        fg=float(error[g].sum());bg=float(error[~g].sum())
        result.append(dict(channel=channel,total_pixels=n,gt_foreground=ng,gt_background=n-ng,
                           pred_foreground=np,intersection=intersection,dice=2*intersection/(np+ng) if np+ng else 1.,
                           brier_sum=fg+bg,brier=(fg+bg)/n,foreground_brier_sum=fg,background_brier_sum=bg,
                           foreground_brier=fg/ng if ng else None,background_brier=bg/(n-ng) if ng<n else None))
    return result


def auxiliary(payload,mask):
    q=payload['q'];result=dict(q=quality(q,mask),memory=dict(status='NOT_APPLICABLE'))
    if payload['selected'] is not None:
        g=grid(mask).bool();regions=[]
        for r,ids in enumerate(payload['selected']):
            correct=int((g[ids,r//2]==bool(r%2)).sum())
            regions.append(dict(region=r,selected_tokens=len(ids),correct_tokens=correct,incorrect_tokens=len(ids)-correct))
        result['memory']=dict(status='ACTUAL_COMMITTED_STUDENT_PRE_TOKENS',regions=regions)
    if 'qstar' in payload:
        qs=payload['qstar'];result['qstar']=quality(qs,mask);h=q>=.5;hs=qs>=.5;g=mask.bool();old=h==g;new=hs==g
        delta=(qs-q).abs();allowed=payload['allowed'];reliable=payload['reliable'];rows=[]
        for c,channel in enumerate(('OD','OC')):
            for name,support in [('all',torch.ones_like(g,dtype=torch.bool)),('allowed',allowed),('hard_flip',h!=hs)]:
                m=support[0,c];a=old[0,c];b=new[0,c]
                rows.append(dict(channel=channel,scope=name,denominator=int(m.sum()),wrong_to_correct=int((m&~a&b).sum()),
                                 correct_to_wrong=int((m&a&~b).sum()),correct_unchanged=int((m&a&b).sum()),wrong_unchanged=int((m&~a&~b).sum())))
        result['graph']=dict(transitions=rows,allowed_counts=[int(x.sum()) for x in allowed[0]],
                             reliable_counts=[int(x.sum()) for x in reliable[0]],reliable_correct=[int(x.sum()) for x in (old&reliable)[0]],
                             changed_counts=[int(x.sum()) for x in (qs!=q)[0]],mean_abs_change=float(delta.mean()),
                             outside_allowed_max_change=float(delta[~allowed].max()) if (~allowed).any() else 0.,
                             reliable_max_change=float(delta[reliable].max()) if reliable.any() else 0.,
                             q_nested_violations=int((q[:,1]>q[:,0]).sum()),qstar_nested_violations=int((qs[:,1]>qs[:,0]).sum()))
        if result['graph']['outside_allowed_max_change']!=0 or result['graph']['reliable_max_change']!=0:raise ValueError('teacher support violation')
    return result


def current(host,row):
    begin=time.monotonic();pixels,rgb_io=target(row,'image');sync();started=time.monotonic()
    z,trace=host.step(pixels);sync();seconds=time.monotonic()-started
    if z.requires_grad or host.phase!='IDLE' or host.pending:raise ValueError('state not committed before evaluator')
    payload=host.take_evaluation()
    try:
        mask,mask_io=target(row,'mask')
        metrics=evaluate(z.sigmoid(),mask,'fundus');extra=auxiliary(payload,mask)
        return dict(**trace,metrics=metrics,auxiliary=extra,prediction_fixed_before_label=True,
                    auxiliary_after_state_commit=True,host_seconds=seconds,pipeline_seconds=time.monotonic()-begin,
                    asset_io=dict(image=rgb_io,mask=mask_io))
    finally:payload.clear()
