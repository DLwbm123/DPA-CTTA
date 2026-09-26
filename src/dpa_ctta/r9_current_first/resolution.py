"""Resolve all symbolic slots from frozen source-only choices; never inspect target scores."""
from .protocol import parse_recipe,SPEC
GRAD={'B_G1','B_G3','B_RESET_G3','COLD_G1','COLD_G3','BN_RESET_G1'}
NATIVE={'N_SOURCE_EVAL','C0_CURRENT_STATS','VPTTA_NATIVE','C_CTTA_FIXED_LR','G_CTTA_RELEASE_TRANSFER'}


def resolve(slot,recipe_selection,source_receipts):
    arm=slot['arm'];out=dict(slot=slot['id'],arm=arm,seed=slot['seed'],order=slot['order'],alpha=1.,diagnostic=None,gradient=None,source_job=None)
    if arm in NATIVE:return out
    if arm=='BN_RESET_G1':out['gradient']=arm;return out
    route='B' if arm.startswith(('COLD','B_')) else 'A'
    if slot['phase']=='SCREEN24_FIXED_DIAG':
        out['source_job']=f"{route}_FULL_{slot['seed']}";out['source_origin']='screen24';out['step']=4000
    else:
        recipe=arm
        if arm in GRAD or arm in {f'{r}_{d}' for r in ('A','B') for d in ('RESET','PRECAL','PRED_ONLY','ISTA20')}:
            recipe=f'{route}_SELECTED_FULL'
        route,kind,mode=parse_recipe(recipe,recipe_selection['recipes'])
        symbolic=recipe if 'SELECTED' in recipe else (f'{route}_{kind}_{mode}' if route!='MLP' else 'MLP_'+kind)
        # First two selected models are the original discovery jobs, not fresh duplicates.
        if 'SELECTED' in symbolic and slot['seed'] in (20260924,20260925):symbolic=symbolic.replace('SELECTED',kind)
        job=f"FIT_{symbolic}_{slot['seed']}"
        receipt=source_receipts[job];step=16000 if slot['checkpoint']=='fixed_step16000' else receipt['selection']['step']
        out.update(source_job=job,source_origin='r9',step=step,artifact=receipt['artifacts'][str(step)])
    if arm in GRAD:out['gradient']=arm
    elif '_OUTSCALE_' in arm:out['alpha']=float(arm.rsplit('_',1)[1]);out['diagnostic']='OUTSCALE'
    elif arm.split('_',1)[-1] in ('RESET','PRECAL','PRED_ONLY','ISTA20'):out['diagnostic']=arm.split('_',1)[-1]
    return out
