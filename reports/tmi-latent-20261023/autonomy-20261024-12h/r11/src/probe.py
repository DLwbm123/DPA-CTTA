import os,json,time,signal
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,binary_erosion
from scipy.spatial import cKDTree
R=Path(os.environ['WORK_ROOT']);C=json.loads((R/'config.private.json').read_text());STOP=False
COUNT={'source_masks':0,'source_images':0,'dev':0,'target':0,'neural':0,'training':0,'synthetic_contents':0,'processed_contents':0}
def stop(*args):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def check():
 if STOP:raise InterruptedError('Budget stop; no automatic replay')
def write(n,d):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def encode_decode(y):
 pts=np.argwhere(y[1]);assert len(pts)>0,'Empty cup';cy,cx=pts.mean(0);center=(int(round(cy)),int(round(cx)));assert y[0][center] and y[1][center],'Center outside anatomy'
 angles=np.arange(64)*2*np.pi/64;rad=np.arange(0,np.hypot(*y.shape[1:])+1,.25);rr=[]
 for mask in y:
  vals=map_coordinates(mask.astype(float),[cy+np.sin(angles)[:,None]*rad,cx+np.cos(angles)[:,None]*rad],order=1,mode='constant',cval=0)
  out=vals<.5;assert out.any(1).all() and not out[:,0].any(),'Ray invalid';first=out.argmax(1);rr.append((rad[first]+rad[first-1])/2)
 modified=int(np.sum(rr[1]>rr[0]));rr[1]=np.minimum(rr[1],rr[0]);yy,xx=np.indices(y.shape[1:]);angle=np.mod(np.arctan2(yy-cy,xx-cx),2*np.pi);distance=np.hypot(yy-cy,xx-cx);z=np.stack([distance<=np.interp(angle,angles,v,period=2*np.pi) for v in rr]);return z,modified

def metrics(a,b):
 d={}
 for head,x,y in zip(['od','oc'],a,b):
  d[head+'_dice']=float(200*(x&y).sum()/max(int(x.sum()+y.sum()),1));xx=np.argwhere(x^binary_erosion(x));yy=np.argwhere(y^binary_erosion(y))
  if len(xx) and len(yy):
   ds=np.r_[cKDTree(xx).query(yy)[0],cKDTree(yy).query(xx)[0]];d[head+'_hd95']=float(np.quantile(ds,.95));d[head+'_assd']=float(ds.mean())
  else:d[head+'_hd95']=d[head+'_assd']=None
 def ratio(z):
  def height(x):
   rows=np.where(x.any(1))[0];return int(rows[-1]-rows[0]+1) if len(rows) else 0
  return height(z[1])/max(height(z[0]),1)
 d['vcdr_abs_delta']=abs(ratio(a)-ratio(b));d['empty']=bool(not a[0].any() or not a[1].any());d['cup_outside_gt1pct']=bool((a[1]&~a[0]).sum()/max(int(a[1].sum()),1)>.01);return d

def main():
 assert not (R/'detailed.private.json').exists(),'No replay';yy,xx=np.indices((256,256))
 for y in [np.stack([(xx-128)**2+(yy-128)**2<=70**2,(xx-128)**2+(yy-128)**2<=32**2]),np.stack([((xx-125)/75)**2+((yy-130)/65)**2<=1,((xx-132)/35)**2+((yy-125)/28)**2<=1])]:
  z,_=encode_decode(y);v=metrics(z,y);assert v['od_dice']>98 and v['oc_dice']>98 and not v['empty'] and not v['cup_outside_gt1pct'];COUNT['synthetic_contents']+=1
 write('SELF_CHECK.json',{'synthetic_only':True,'circle_eccentric_ellipse_passed':True,'counts':COUNT.copy()})
 rows=json.loads((Path(C['parent_root'])/'split.private.json').read_text())['fit'];assert len(rows)==79;out=[]
 write('DATA_RECEIPT.json',{'planned_source_masks':79,'source_images':0,'dev':0,'target':0,'network':0,'patient_identity':'UNKNOWN','truth_center_diagnostic_only':True})
 for i,row in enumerate(rows):
  check();mask=np.asarray(Image.open(Path(C['data_root'])/row['mask']).convert('L').resize((256,256),Image.Resampling.NEAREST));COUNT['source_masks']+=1;assert set(np.unique(mask))<={0,128,255};y=np.stack([mask<255,mask==0])
  try:z,n=encode_decode(y);v={'content':i,'valid':True,'corrected_rays':n,**metrics(z,y)}
  except AssertionError as e:v={'content':i,'valid':False,'reason':str(e)}
  out.append(v);COUNT['processed_contents']+=1;write('detailed.private.json',out)
 valid=[x for x in out if x['valid']];report={'contents':79,'valid':len(valid),'heads':{},'counts':COUNT.copy(),'truth_center_not_deployment':True};okay=len(valid)==79
 for h in ['od','oc']:
  a=[x[h+'_dice'] for x in valid];stats={'mean':float(np.mean(a)),'median':float(np.median(a)),'q05':float(np.quantile(a,.05)),'min':min(a)};passed=stats['mean']>=99 and stats['q05']>=98 and stats['min']>=95;stats['passed']=passed;okay=okay and passed
  for k in ['hd95','assd']:
   q=[x[h+'_'+k] for x in valid if x[h+'_'+k] is not None];stats[k]={'finite':len(q),'total':79,'mean':float(np.mean(q)),'q95':float(np.quantile(q,.95)),'max':max(q)}
  report['heads'][h]=stats
 v=[x['vcdr_abs_delta'] for x in valid];report['vcdr']={'mean':float(np.mean(v)),'max':max(v),'over_005':sum(x>.05 for x in v)};report['empty_or_outside']=sum(x['empty'] or x['cup_outside_gt1pct'] for x in valid);report['corrected_rays']=sum(x['corrected_rays'] for x in valid);okay=okay and max(v)<=.05 and report['empty_or_outside']==0;report['representation_gate']=bool(okay);report['decision']='Eligible only for separate prediction-initialized study' if okay else 'Reject fixed common-center 64-ray representation; no ray/center/threshold search'
 write('REPORT.json',report);write('TRAINING_FINISHED.json',{'status':'COMPLETE','counts':COUNT,'representation_gate':bool(okay),'ended_unix':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException as e:write('FAILURE.json',{'type':type(e).__name__,'message':str(e),'counts':COUNT,'unix':time.time()});raise
