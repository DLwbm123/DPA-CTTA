import os,sys,json,time
from pathlib import Path
import numpy as np
import scipy,skimage
from bend import measure,score,outer,EPS

def tube(points,amp=.6):
    yy,xx=np.indices((128,128));d=np.full((128,128),np.inf)
    for a,b in zip(points,points[1:]):
        a=np.array(a,float);b=np.array(b,float);v=b-a;t=np.clip(((yy-a[0])*v[0]+(xx-a[1])*v[1])/np.dot(v,v),0,1)
        d=np.minimum(d,np.sqrt((yy-a[0]-t*v[0])**2+(xx-a[1]-t*v[1])**2))
    return .8-amp*np.exp(-d*d/(2*1.5**2))

def run(c):
    root=Path(c['root']);begin=time.perf_counter();passno=int(os.environ['AUDIT_MODE'][-1]);denied=[]
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('network denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p)
        if '/root-migrated/CTTA/data/' in s or '/root-migrated/CTTA/models/' in s or 'foveal-partition-signal-e8-migration' in s or p.is_relative_to(root/'private/literature'):
            denied.append(s);raise PermissionError('synthetic only')
    sys.addaudithook(audit)
    try:(root/'private/literature/ICPR2010.pdf').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('access gate')
    od=np.ones((128,128),bool);tests=[];outputs={}
    def check(name,condition):
        assert condition,name;tests.append(name)
    empty=measure(np.full((128,128),.8),od);check('constant_abstention',empty['bend'].sum()<=EPS)
    straight=measure(tube([(64,12),(64,116)]),od);check('straight_path_detected',len(straight['points'])>10);check('straight_has_no_bend',straight['bend'].sum()<=EPS)
    bend=measure(tube([(20,30),(64,64),(20,98)]),od)
    check('known_bend_detected',bend['curvature'].max(initial=0)>=np.pi/6)
    peak=bend['points'][np.argmax(bend['curvature'])];check('known_bend_localized',np.linalg.norm(peak-[64,64])<=5)
    t=measure(np.minimum(tube([(64,12),(64,116)]),tube([(16,64),(64,64)])),od)
    check('junction_detected',t['junction_points']>0)
    for name,e in (('junction',t),('endpoint',straight)):
        forbidden={tuple(p) for p in e['forbidden']};check(name+'_support_excluded',all(tuple(p) not in forbidden for p in e['points']))
    short=measure(tube([(64,60),(64,67)]),od);check('short_path_abstention',short['bend'].sum()<=EPS)
    low=measure(tube([(20,30),(64,64),(20,98)],amp=1e-5),od);check('low_contrast_abstention',low['bend'].sum()<=EPS)
    # Unit evidence tests the sign and contour definition without a real image.
    unit=dict(points=np.array([[64,64]]),bend=np.ones(1),vessel=np.ones(1),shift=np.ones(1))
    b=np.zeros((128,128),bool);b[10:80,20:100]=1;good=np.zeros_like(b);good[10:65,20:100]=1
    q,s,_=score(unit,b,good);check('correct_energy_direction',q>0 and s);check('reverse_direction',score(unit,good,b)[0]<0)
    check('identical_zero',score(unit,b,b)[0]==0)
    hole=b.copy();hole[40:45,50:55]=0;island=b.copy();island[2:4,2:4]=1
    check('hole_not_outer_boundary',np.array_equal(outer(hole),outer(b)));check('island_not_outer_boundary',np.array_equal(outer(island),outer(b)))
    check('empty_candidate_abstains',score(unit,b,np.zeros_like(b))[0]==0)
    saved={k:v.copy() for k,v in bend.items() if isinstance(v,np.ndarray)}
    score(bend,b,good);check('candidate_does_not_change_evidence',all(np.array_equal(v,bend[k]) for k,v in saved.items()))
    check('evidence_arrays_readonly',all(not v.flags.writeable for v in saved.values()) is False and all(not v.flags.writeable for v in bend.values() if isinstance(v,np.ndarray)))
    check('shift_preserves_weights',np.array_equal(np.sort(bend['bend']),np.sort(bend['shift'])))
    check('shift_nontrivial_on_bend',bend['permuted_weights_changed']>0)
    check('no_model_import','torch' not in sys.modules)
    for name,e in (('empty',empty),('straight',straight),('bend',bend),('junction',t),('short',short),('low',low)):
        for key,v in e.items():
            if isinstance(v,np.ndarray):outputs[name+'_'+key]=v
    if passno==1:np.savez(root/'private/SYNTHETIC_REPLAY.npz',**outputs)
    else:
        old=np.load(root/'private/SYNTHETIC_REPLAY.npz',allow_pickle=False);check('independent_exact_replay',set(old.files)==set(outputs) and all(np.array_equal(v,old[k]) for k,v in outputs.items()))
    result=dict(status='PASS',source_sha=c['source_sha'],pass_number=passno,checks=tests,check_count=len(tests),synthetic_families=6,
        real_rgb_reads=0,label_reads=0,weight_reads=0,model_calls=0,cuda_initialized=False,access_negative_checks=len(denied),
        libraries=dict(numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__),synthetic_body_seconds=time.perf_counter()-begin,
        task_signal_or_clinical_validity_established=False)
    (root/'public'/f'QUALIFICATION{passno}.json').write_text(json.dumps(result,indent=2)+'\n')
