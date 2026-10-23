import os,json,time,importlib.util,random
from pathlib import Path
import torch
R=Path(os.environ['WORK_ROOT']);C=json.loads((R/'config.private.json').read_text())
spec=importlib.util.spec_from_file_location('corrector',Path(C['audit_model_root'])/'src/train.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);m=a.m
@torch.no_grad()
def main():
 assert not (R/'runs').exists(),'No automatic replay'
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=json.loads((Path(C['parent_root'])/'split.private.json').read_text())['fit'];assert len(rows)==79
 dr=Path(C['data_root']);fit=[]
 for row in rows:
  a.check();x=m.np.asarray(m.Image.open(dr/row['image']).convert('RGB').resize((256,256),m.Image.Resampling.BICUBIC),dtype=m.np.float32);x=(x-x.min())/max(float(x.max()-x.min()),1e-6)
  mask=m.np.asarray(m.Image.open(dr/row['mask']).convert('L').resize((256,256),m.Image.Resampling.NEAREST));assert set(m.np.unique(mask))<={0,128,255}
  y=m.np.stack([mask<255,mask==0]).astype(m.np.float32);fit.append((torch.tensor(x.transpose(2,0,1).copy()),torch.tensor(y),None))
 m.write(R/'DATA_RECEIPT.json',{'source_images':79,'source_masks':79,'new_dev_images':0,'target_images':0,'training_updates':0,'patient_identity':'UNKNOWN'})
 for seed in [10225,10226]:
  a.check();torch.manual_seed(seed);base=m.Model().cuda().eval();ck=torch.load(Path(C['parent_root'])/f'runs/{seed}/latest.pt',map_location='cpu',weights_only=False);assert ck['phase']=='meta' and ck['step']==500;base.load_state_dict(ck['model']);del ck;base.requires_grad_(False)
  ck=torch.load(Path(C['audit_model_root'])/f'runs/{seed}/latest.pt',map_location='cpu',weights_only=False);assert ck['completed_paired_episode']==1000
  nets={k:a.Corrector().cuda().eval() for k in ['DETACHED','FULL']}
  for k,n in nets.items():n.load_state_dict(ck['models'][k]);n.requires_grad_(False)
  del ck;before=[p.detach().clone() for p in base.parameters()];run=R/'runs'/str(seed);run.mkdir(parents=True)
  n=a.evaluate(base,nets,fit,run);assert n==316 and all(torch.equal(v,p) for v,p in zip(before,base.parameters()))
  m.write(run/'completion.json',{'source_contents':79,'conditions':316,'optimizer_steps':0,'base_frozen':True,'readout_weights_frozen':True})
  del before,base,nets;torch.cuda.empty_cache()
 m.write(R/'TRAINING_FINISHED.json',{'status':'COMPLETE','kind':'FROZEN_SOURCE_AUDIT','completed_seeds':[10225,10226],'source_conditions':632,'extra_counts':a.EXTRA,'base_counts':m.COUNTS,'ended_unix':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException as e:m.write(R/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'unix':time.time()});raise
