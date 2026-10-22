"""Bounded source-fit-only alignment diagnostic; never reads target data."""
import os,json,sys,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
r=Path(os.environ['WORK_ROOT']);cfg=json.loads((r/'config.private.json').read_text());torch.set_num_threads(4)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load('core',cfg['expert_module']);module=load('fusion',str(r/'code/train.py'));fit,_=core.load_data(cfg)
sys.path.insert(0,cfg['backbone_code']);from networks.ResUnet import ResUnet
base=ResUnet(resnet='resnet34',num_classes=2,pretrained=False).cuda().eval();base.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True))
expert=core.Model().cuda().eval();expert.load_state_dict(torch.load(cfg['expert_checkpoints']['10223'],map_location='cpu',weights_only=True)['model'])
x=torch.stack([v[0] for v in fit[:4]]).cuda();y=torch.stack([v[1] for v in fit[:4]]).cuda()
with torch.no_grad():
 z=core.base_forward(base,x);zf=core.base_forward(base,x.flip(-1));_,ps=module.experts(core,base,expert,x,None)
 assert torch.allclose(ps[:,:,2],zf.sigmoid().flip(-1),atol=1e-6)
 assert torch.allclose(core.dice(zf.flip(-1),y),core.dice(zf,y.flip(-1)),atol=1e-5)
 report={'source_fit_count':4,'target_read':0,'aligned_helper_matches_raw':True,'dice_alignment_equivalence':True,'original_dice':core.dice(z,y).cpu().tolist(),'aligned_flip_dice':core.dice(zf.flip(-1),y).cpu().tolist(),'unrealigned_flip_dice':core.dice(zf,y).cpu().tolist(),'counts':core.COUNTS,'interpretation':'Small fit diagnostic only; not independent performance evidence.'}
(r/'FLIP_ALIGNMENT_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
