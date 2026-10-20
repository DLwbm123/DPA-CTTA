"""E6-C label utilities, only after score worker retirement. Never imported by it."""
import os,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image
from .signal_common import read,save,unpack,retired
from .signal_math import components,NAMES,auc
from .signal_oracle import solve

def withdrawal_utility(P,G,T,a,t):
    before=Fraction(2*T,P+G) if P+G else Fraction(1)
    den=P+G+a;after=Fraction(2*(T+t),den) if den else Fraction(1)
    return after-before

def aggregate(values):
    v=np.asarray(values,float);content=[]
    for row in v:
        ok=np.isfinite(row);w=np.array([2,1,1,1])[ok]
        content.append(float(np.average(row[ok],weights=w)) if len(w) else None)
    valid=[v for v in content if v is not None]
    ci=None
    if len(valid)==8:
        ids=np.random.default_rng(20261018).integers(0,8,(2000,8));ci=np.quantile(np.asarray(content)[ids].mean(1),[.025,.975]).tolist()
    return dict(mean=float(np.mean(valid)) if valid else None,ci95=ci,defined_inputs=int(np.isfinite(v).sum()),defined_contents=len(valid)),content

def score(c):
    root=Path(c['output_root']);started=time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and retired(root/'processes/worker.json')
    assert read(root/'RUN_STATE.json')['status']=='SCORES_COMPLETE'
    assert read(root/'FULL_REPEATABILITY.json')['all_scores_anchors_uids_and_coverage_exact']
    rows=read(root/'private/scores.json');panel=read(root/'scorer/panel.json');assert len(rows)==32 and len(panel)==8
    assert {(r['case'],r['condition']) for r in rows}=={(i,k) for i in range(8) for k in range(4)}
    bykey={(r['case'],r['condition']):r['blocks'] for r in rows};values={(name,weight):np.full((8,4),np.nan) for name in NAMES for weight in ('BLOCK','UTILITY','PIXEL')}
    private=[];common_space=np.zeros((8,4));full_space=np.zeros((8,4));total=common=pixels=common_pixels=moved=changed=0;zero_utility=positive=negative=0;reasons={};definitions=[]
    subgroup_values={}
    for i,item in enumerate(panel):
        with Image.open(item['mask_path']) as im:
            assert list(im.size)==item['image_size'];g=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))
        y=g==0
        for k in range(4):
            s=unpack(root/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(root/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');groups,sizes,od,oc=components(s,b)
            rr=bykey[i,k];expected=set((np.flatnonzero((od==0)&(oc>0))+1).tolist());assert {r['block_id'] for r in rr}==expected
            d=s[1].astype(np.int64)-b[1].astype(np.int64);n=len(sizes)
            area=np.bincount(groups.ravel(),weights=d.ravel(),minlength=n+1)[1:].astype(np.int64)
            tp=np.bincount(groups.ravel(),weights=(d*y).ravel(),minlength=n+1)[1:].astype(np.int64)
            P,G,T=int(b[1].sum()),int(y.sum()),int((b[1]&y).sum());Db=Fraction(2*T,P+G) if P+G else Fraction(1)
            utilities=[];valid=[]
            for r in rr:
                j=r['block_id'];j0=j-1
                u=withdrawal_utility(P,G,T,int(area[j0]),int(tp[j0]))
                actual=b[1].copy();mask=groups==j;actual[mask]=s[1][mask]
                dd=int(actual.sum()+y.sum());check=Fraction(2*int((actual&y).sum()),dd) if dd else Fraction(1);assert check-Db==u
                total+=1;pixels+=r['oc_pixels'];reasons[r['reason']]=reasons.get(r['reason'],0)+1
                if r['common']:
                    assert all(r['scores'][name] is not None and np.isfinite(r['scores'][name]) for name in NAMES)
                    common+=1;common_pixels+=r['oc_pixels'];valid.append(r);utilities.append(float(u));moved+=r['stratified_from_block']!=j;changed+=r['scores']['FEATURE']!=r['scores']['STRATIFIED_ROLL'];zero_utility+=u==0;positive+=u>0;negative+=u<0
                private.append(dict(case=i,condition=k,block_id=j,utility_numerator=u.numerator,utility_denominator=u.denominator,common=r['common']))
            utilities=np.asarray(utilities);w_pixel=np.asarray([r['oc_pixels'] for r in valid]);w_utility=np.abs(utilities)
            for name in NAMES:
                scores=[r['scores'][name] for r in valid]
                for weight,w in [('BLOCK',None),('UTILITY',w_utility),('PIXEL',w_pixel)]:
                    value=auc(scores,utilities,w);values[name,weight][i,k]=np.nan if value is None else value
            definitions.append(dict(case=i,condition=k,positive=int((utilities>0).sum()),negative=int((utilities<0).sum()),tied=int((utilities==0).sum())))
            for field in ('size_bucket','direction'):
                for key in sorted({r[field] for r in valid}):
                    keep=np.array([r[field]==key for r in valid]);uu=utilities[keep]
                    for name in NAMES:
                        a=subgroup_values.setdefault((field,key,name),np.full((8,4),np.nan));value=auc([r['scores'][name] for r,yes in zip(valid,keep) if yes],uu);a[i,k]=np.nan if value is None else value
            common_ids={r['block_id'] for r in valid}
            for ids,array in [(expected,full_space),(common_ids,common_space)]:
                changes=tuple((int(area[j-1]),int(tp[j-1])) for j in sorted(ids));best,_,_=solve(P,G,T,changes);array[i,k]=100*float(best-Db)
    public_auc=[];contrasts=[]
    for (name,weight),v in values.items():
        a,_=aggregate(v);public_auc.append(dict(score=name,weight=weight,**a))
    for name in NAMES[1:]:
        for weight in ('BLOCK','UTILITY','PIXEL'):
            a,_=aggregate(values['FEATURE',weight]-values[name,weight]);contrasts.append(dict(comparison='FEATURE-'+name,weight=weight,**a))
    def space(v):
        vv=v[:,[0,1,2,3,0]];ids=np.random.default_rng(20261018).integers(0,8,(2000,8));return dict(delta_pp=float(vv.mean()),ci95_pp=np.quantile(vv[ids].mean((1,2)),[.025,.975]).tolist())
    coverage=space(common_space);full=space(full_space)
    prior=read(Path(c['prior_space_root'])/'public/RESULTS.json');assert abs(full['delta_pp']-prior['primary']['delta_pp'])<=1e-12
    primary=next(r for r in contrasts if r['comparison']=='FEATURE-LOGIT_MATCH' and r['weight']=='BLOCK');feature=next(r for r in public_auc if r['score']=='FEATURE' and r['weight']=='BLOCK')
    good=lambda v:v['mean'] is not None and v['ci95'] is not None and v['ci95'][0]>0
    utility=next(r for r in contrasts if r['comparison']=='FEATURE-LOGIT_MATCH' and r['weight']=='UTILITY')
    controls=[r for r in contrasts if r['weight']=='BLOCK' and r['comparison']!='FEATURE-LOGIT_MATCH']
    gates=dict(common_oracle=coverage['delta_pp']>=.2 and coverage['ci95_pp'][0]>0,defined_inputs=feature['defined_inputs']>=24 and feature['defined_contents']==8,feature_auc=feature['mean'] is not None and feature['mean']>=.6,feature_vs_logit=good(primary) and primary['mean']>=.05,strong_controls=all(good(r) for r in controls),utility_nonreverse=utility['mean'] is not None and utility['mean']>=0,permutation_nonidentity=common>0 and moved/common>=.5)
    strata=[]
    for (field,key,name),v in subgroup_values.items():a,_=aggregate(v);strata.append(dict(field=field,stratum=key,score=name,**a))
    results=dict(signal_qualification_only=True,selector_trained=False,selector_applied=False,network_updated=False,history_used=False,continual_benefit_tested=False,signal_qualified=all(gates.values()),gates=gates,auc=public_auc,contrasts=contrasts,primary=primary,common_oracle_space=coverage,full_od_lock_space=full,coverage=dict(all_oc_only_blocks=total,common_blocks=common,all_oc_edit_pixels=pixels,common_edit_pixels=common_pixels,common_positive_utility=positive,common_negative_utility=negative,common_zero_utility=zero_utility,reasons=reasons,permutation_positions_moved=moved,permutation_values_changed=changed),strata=strata)
    save(root/'scorer/utility.private.json',private);save(root/'scorer/definition.private.json',definitions);save(root/'public/RESULTS.json',results)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',registered_label_files=8,unique_inputs=32,oc_only_utilities=total,auc_rows=len(public_auc),contrast_rows=len(contrasts),common_space_exactly_solved=True,body_seconds=time.time()-started))

if __name__=='__main__':score(read(os.environ['DIAG_CONFIG']))
