"""Fixed, source-only likelihood ranking; no neural model or adaptation claim."""
import os, json, signal, time, random, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

R = Path(os.environ['WORK_ROOT'])
C = json.loads((R / 'config.private.json').read_text())
COUNTS = {'source_images_read': 0, 'source_masks_read': 0, 'dev_images_read': 0, 'target_images_read': 0,
          'neural_model_calls': 0, 'network_training_updates': 0, 'synthetic_scores': 0, 'probe_scores': 0, 'completed_probe_views': 0}
STOP = False


def stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)


def write(name, value):
    p = R / name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(p)


def check_stop():
    if STOP: raise InterruptedError('Budget stop; no automatic probe replay')


from scipy.ndimage import gaussian_filter

def candidates(y):
    od, oc = y
    return {'TRUE': y, 'OD_SHRINK': np.stack([binary_erosion(od, iterations=2) | oc, oc]),
            'OD_EXPAND': np.stack([binary_dilation(od, iterations=2), oc]),
            'OC_SHRINK': np.stack([od, binary_erosion(oc, iterations=2)]),
            'OC_EXPAND': np.stack([od, binary_dilation(oc, iterations=2) & od])}


def severity(y, wrong, head):
    a, b = y[head], wrong[head]
    return 100 - 200 * (a & b).sum() / max(int(a.sum() + b.sum()), 1)


def read(row):
    dr = Path(C['data_root']); n = C['size']
    x = np.asarray(Image.open(dr / row['image']).convert('RGB').resize((n, n), Image.Resampling.BICUBIC), dtype=np.float64)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    z = np.asarray(Image.open(dr / row['mask']).convert('L').resize((n, n), Image.Resampling.NEAREST))
    assert set(np.unique(z)) <= {0, 128, 255}
    COUNTS['source_images_read'] += 1; COUNTS['source_masks_read'] += 1
    return x, np.stack([z < 255, z == 0])



def gradients(x):
 return [np.gradient(gaussian_filter(x[:,:,k],1)) for k in range(3)]
def measure(g,mask,synthetic=False):
 COUNTS['synthetic_scores' if synthetic else 'probe_scores']+=1
 d=distance_transform_edt(~mask)-distance_transform_edt(mask);gy,gx=np.gradient(gaussian_filter(d,1));norm=np.sqrt(gy*gy+gx*gx).clip(min=1e-8);ny,nx=gy/norm,gx/norm;band=np.abs(d)<=1
 assert band.any()
 normal=np.sqrt(sum((iy*ny+ix*nx)**2 for iy,ix in g));tangent=np.sqrt(sum((-iy*nx+ix*ny)**2 for iy,ix in g))
 return {'normal':float(normal[band].mean()),'tangent':float(tangent[band].mean())}
def aggregate(rows,qualified):
 result={'qualified_contents':sum(qualified.values()),'planned_contents':12,'views':len(rows),'heads':{},'diagnostic_only':True,'counts':COUNTS.copy()};gate=len(qualified)==12 and all(qualified.values()) and len(rows)==24;orientation=True
 for h,head in enumerate(['OD','OC']):
  wrong=[head+'_SHRINK',head+'_EXPAND'];minimum=[];average=[]
  for i in sorted(qualified):
   v=[r['scores']['TRUE'][h]['normal']-r['scores'][name][h]['normal'] for r in rows if r['content']==i for name in wrong];minimum.append(min(v));average.append(float(np.mean(v)))
  cells={view:{name:float(np.mean([r['scores']['TRUE'][h]['normal']-r['scores'][name][h]['normal'] for r in rows if r['view']==view])) for name in wrong} for view in ['original','photometric']}
  direct={view:float(np.mean([r['scores']['TRUE'][h]['normal']-r['scores']['TRUE'][h]['tangent'] for r in rows if r['view']==view])) for view in ['original','photometric']}
  wins=sum(v>1e-6 for v in minimum);okay=wins>=9 and all(v>1e-6 for c in cells.values() for v in c.values());gate=gate and okay;orientation=orientation and all(v>1e-6 for v in direct.values())
  result['heads'][head]={'worst_direction_content_mean':float(np.mean(minimum)),'median':float(np.median(minimum)),'q05':float(np.quantile(minimum,.05)),'win_tie_loss':[wins,sum(abs(v)<=1e-6 for v in minimum),sum(v< -1e-6 for v in minimum)],'average_margin_not_gate':float(np.mean(average)),'direction_view_margins':cells,'true_normal_minus_tangent':direct,'ranking_gate':bool(okay),'truth_rank_counts':{str(rank):sum(1+sum(r['scores'][name][h]['normal']>r['scores']['TRUE'][h]['normal']+1e-6 for name in wrong)==rank for r in rows) for rank in [1,2,3]}}
 result.update(local_boundary_gate=bool(gate),orientation_support=bool(orientation),decision='Retain only for separately controlled prediction experiment' if gate and orientation else 'Reject fixed local-gradient observation; no sigma/bandwidth/weight search');return result
def main():
 assert not (R/'detailed.private.json').exists(),'No automatic replay'
 yy,xx=np.indices((96,96));y=np.stack([(xx-48)**2+(yy-48)**2<32**2,(xx-48)**2+(yy-48)**2<16**2]);x=np.repeat((.1+.3*y[0]+.4*y[1])[:,:,None],3,axis=2);g=gradients(x);cs=candidates(y)
 synth={k:[measure(g,z,True) for z in m] for k,m in cs.items()}
 for h,head in enumerate(['OD','OC']):
  assert all(synth['TRUE'][h]['normal']>synth[n][h]['normal'] for n in [head+'_SHRINK',head+'_EXPAND']);assert synth['TRUE'][h]['normal']>synth['TRUE'][h]['tangent']
 write('SELF_CHECK.json',{'synthetic_only':True,'boundary_operator_and_counterfactual_ranking':True,'counts':COUNTS.copy()})
 split=json.loads((Path(C['parent_root'])/'split.private.json').read_text());assert len(split['fit'])==79
 indices=random.Random(28417).sample(range(79),12);rows=[];qualified={}
 write('DATA_RECEIPT.json',{'source_probe_count':12,'calibration_count':0,'new_dev_images':0,'target_images':0,'network_calls':0,'patient_identity':'UNKNOWN','same_r05_source_probe':True})
 for i in indices:
  check_stop();x,y=read(split['fit'][i]);cs=candidates(y);qualified[i]=all(m[0].any() and m[1].any() and not (m[1]&~m[0]).any() and severity(y,m,0 if n.startswith('OD') else 1)>=.25 for n,m in cs.items() if n!='TRUE')
  for view,image in [('original',x),('photometric',(x**1.25*np.array([1.08,.96,.94])).clip(0,1))]:
   check_stop();g=gradients(image);scores={n:[measure(g,z) for z in m] for n,m in cs.items()};rows.append({'content':i,'view':view,'qualified':bool(qualified[i]),'scores':scores});COUNTS['completed_probe_views']+=1;write('detailed.private.json',rows)
 write('REPORT.json',aggregate(rows,qualified));write('TRAINING_FINISHED.json',{'status':'COMPLETE','counts':COUNTS,'ended_unix':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException as e:write('FAILURE.json',{'type':type(e).__name__,'message':str(e),'counts':COUNTS,'unix':time.time()});raise
