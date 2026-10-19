"""Independent fallback scorer: original OS exit gate remains unmet, never set to zero."""
from .topology_closed_run import *
from .storage_recovery import completion_certificate

def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_erosion,distance_transform_edt
    validate(c);root=Path(c['output_root']);started=time.time();ret=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code'] is None
    admission=read(root/'RECONSTRUCTION_ADMISSION.json');assert admission['status']=='PASS' and admission['pidfd_exit_observed'] and not admission['original_exit0_gate_restored']
    assert completion_certificate(root)
    status=read(root/'RUN_STATE.json');assert status['status']=='GPU_COMPLETE' and status['predictions']==384 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['all_arms_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    prior=Path(c['state_prior_root']);oldrows=read(prior/'private/predictions.json');old={(v['case'],v['condition'],v['arm']):v for v in oldrows}
    parity=0
    for row in rows:
        if row['arm'] in STATIC:
            v=old[row['case'],row['condition'],row['arm']]
            assert (root/'predictions'/f"{row['index']:04d}.bits").read_bytes()==(prior/'predictions'/f"{v['index']:04d}.bits").read_bytes()
            parity+=1
    assert parity==160
    save(root/'STATIC_PARITY.json',dict(status='PASS',bit_arrays=parity,scope='Only scorer after worker retirement reads sealed P2 bits; no metrics reused'))
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==384 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    masks={(r['case'],r['condition'],r['arm']):unpack(root/'predictions'/f"{r['index']:04d}.bits")[1] for r in rows}
    metrics={};counts={};individual=[]
    for row in rows:
        pass # no user CPU quota
        i,k,a=row['case'],row['condition'],row['arm'];p=masks[i,k,a];base=masks[i,k,'SOURCE'];gt=labels[i]
        mm,rr=metric(p,gt,base);metrics[i,k,a]=(mm,rr);individual.append(dict(**row,metrics=mm,ratios=rr))
        for ch in range(2):
            y=gt[ch];surface=y^binary_erosion(y,border_value=0)
            bd=distance_transform_edt(~surface)<=3 if surface.any() else np.zeros_like(y)
            for region,mask in [('all',np.ones_like(y)),('foreground',y),('background',~y),('boundary3px',bd)]:
                counts[i,k,a,ch,region]=region_counts(p[ch],y,base[ch],mask)
    save(root/'scorer/individual.private.json',individual)
    def finite(v):
        a=[float(x) for x in v if x is not None and np.isfinite(x)]
        return (float(np.mean(a)) if a else None),len(a)
    cubes={};absolute=[];channels=[];regions=[]
    for a in ARMS:
        cube=np.array([[[metrics[i,k,a][0][ch]['dice_percent'] for ch in range(2)] for k in (0,1,2,3,0)] for i in range(8)])
        cubes[a]=cube
        weighted=[metrics[i,k,a] for i in range(8) for k in (0,1,2,3,0) for _ in range(2)]
        assd,n=finite([m['assd_pixel'] for mm,rr in weighted for m in mm]);hd,_=finite([m['hd95_pixel'] for mm,rr in weighted for m in mm]);vc,vn=finite([rr['vcdr_abs_error'] for mm,rr in weighted])
        nesting=sum(int((masks[i,k,a][1]&~masks[i,k,a][0]).sum()) for i in range(8) for k in (0,1,2,3,0) for _ in range(2))
        absolute.append(dict(arm=a,dice_percent=float(cube.mean()),od_dice_percent=float(cube[:,:,0].mean()),oc_dice_percent=float(cube[:,:,1].mean()),assd_pixel=assd,hd95_pixel=hd,vcdr_mae=vc,surface_defined=n,surface_total=160,vcdr_defined=vn,vcdr_total=80,empty_predictions=sum(m['pred_empty'] for mm,rr in weighted for m in mm),nesting_violation_pixels=nesting,unique_inputs=32,logical_readouts=80,unique_condition_macro_dice_percent=float(np.mean([metrics[i,k,a][0][ch]['dice_percent'] for i in range(8) for k in range(4) for ch in range(2)]))))
        for k in range(4):
            for ch,name in enumerate(('OD','OC')):
                channels.append(dict(arm=a,condition=k,channel=name,dice_percent=float(np.mean([metrics[i,k,a][0][ch]['dice_percent'] for i in range(8)]))))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','background','boundary3px'):
                u=[counts[i,k,a,ch,region] for i in range(8) for k in range(4)]
                cf=np.array([[counts[i,k,a,ch,region]['corrected']/counts[i,k,a,ch,region]['pixels'] if counts[i,k,a,ch,region]['pixels'] else np.nan for k in (0,1,2,3,0)] for i in range(8)])
                hf=np.array([[counts[i,k,a,ch,region]['spoiled']/counts[i,k,a,ch,region]['pixels'] if counts[i,k,a,ch,region]['pixels'] else np.nan for k in (0,1,2,3,0)] for i in range(8)])
                net=summary(a+':'+name+':'+region+':NET_PIXEL_PP',(cf-hf)*100) if np.isfinite(cf).all() and np.isfinite(hf).all() else None
                regions.append(dict(arm=a,channel=name,region=region,corrected_pixels_unique=sum(t['corrected'] for t in u),spoiled_pixels_unique=sum(t['spoiled'] for t in u),disagreement_pixels_unique=sum(t['disagreement'] for t in u),region_pixels_unique=sum(t['pixels'] for t in u),macro_corrected_fraction=finite(cf.flatten())[0],macro_spoiled_fraction=finite(hf.flatten())[0],macro_net_pp=net['delta_pp'] if net else None,net_ci_low_pp=net['ci95_pp'][0] if net else None,net_ci_high_pp=net['ci95_pp'][1] if net else None,macro_defined=int(np.isfinite(cf).sum()),macro_total=40))
    contrasts=[]
    for a in ARMS[1:]:
        for name,ch in [('ALL',None),('OD',0),('OC',1)]:
            delta=cubes[a]-cubes['SOURCE'];d=delta if ch is None else delta[:,:,ch]
            contrasts.append(summary(a+'-SOURCE:'+name+'_DICE',d))
        contrasts.append(summary(a+'-SOURCE:RETURN_DICE',cubes[a][:,4]-cubes['SOURCE'][:,4]))
    for control in ARMS:
        if control in ('SOURCE','TIB'):continue
        contrasts.append(summary('TIB-'+control+':OC_DICE',cubes['TIB'][:,:,1]-cubes[control][:,:,1]))
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    controls={a:next(v for v in contrasts if v['comparison']=='TIB-'+a+':OC_DICE')['ci95_pp'][0]>0 for a in ('PERMUTED','INTEGRITY','COR','SMOOTH','FILL_SOURCE','TOPO_SOURCE','TOPO_SMOOTH','CURRENT_BN','CW1_RESET','CW2_RESET')}
    save(root/'public/RESULTS_RECONSTRUCTED.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,mechanism_primary_pass=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0,advance_to_new_protocol=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0 and all(controls.values()) and next(v['worth_followup'] for v in checks if v['arm']=='TIB'),static_only=True,parameter_updates=c['expected_optimizer_steps'],continual_benefit_tested=False,strong_control_checks=controls,unique_inputs=32,unique_predictions=384,logical_stream_readouts=960,patient_independence=False,execution_exit0_certified=False,independent_scoring_reconstruction=True,scope='Per-image RESET target optimization, not CTTA; native CW1/CW2 use current-image BN semantics; eight exposed contents, no independent patients'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'RECONSTRUCTED_SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=384,absolute_rows=12,region_rows=96,condition_channel_rows=96,contrast_rows=len(contrasts),cpu_seconds=time.time()-started))


