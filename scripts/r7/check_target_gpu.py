"""Finite random full-network target qualification. No real images/checkpoint."""
import copy,io,json,os,subprocess,sys,tempfile,time,traceback
from pathlib import Path
import torch
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/r7'))
from common import segmenter,method,pixels
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_shared.context import environment,tensor_digest,tensors
from dpa_ctta.r7_shared.preparation import prepared_artifact
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_source_prep.runner import configure_backend,load_artifact
from dpa_ctta.r7_target_screen.runner import ARMS,physical,digest
from dpa_ctta.b1_host import Host
from dpa_ctta.source_pilot import seed_all

def main():
 torch.set_num_threads(2);configure_backend(dict(device='cuda:0'));start=time.monotonic()
 r=dict(status='FAILED',code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),arms=ARMS,physical_GPU_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])],checks=[],real_pixel_reads=0,real_checkpoint_loads=0,external_review='NOT_RUN')
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=root,text=True).strip()
 before=COUNTS.copy();base_counts=dict(forwards=0,backwards=0,Adam=0);g=None
 def check(n):r['checks'].append(n)
 try:
  g=segmenter(full=True).to('cuda:0');r['execution_backend']=environment(g)['execution_backend'];original=tensor_digest(tensors(g.model))
  # Historical C constructor uses the original unmodified algorithm on all devices.
  def base(device,n):
   seed_all(20260907);s=segmenter(full=True);s.close();h=Host('C',device=device,model=s.model)
   z=None
   try:
    for i in range(n):z,t=h.step(pixels(i));physical(t,'C_BASE')
    return z.cpu()
   finally:
    base_counts['forwards']+=h.counts['forwards'];base_counts['backwards']+=h.counts['backwards'];base_counts['Adam']+=h.counts['base_adam']
    for handle in h.handles:handle.remove()
  a=base('cuda:0',3);b=base('cuda:0',1);c=base('cpu',1);torch.testing.assert_close(b,c,rtol=.003,atol=.0003);check('C_BASE_8F_1B_1Adam_and_CPU_GPU')
  h=OnlineHost(g)
  for i in range(3):z,t=h.step(pixels(i));physical(t,'C0')
  x=OnlineHost(g);z1,_=x.step(pixels());z2,_=x.step(pixels());assert torch.equal(z1,z2);check('C0_1F_repeat')
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp)
   for name in ARMS[2:]:
    m=method(name[0],name.endswith('STATIC'));m.freeze();p=prepared_artifact(g,m);buf=io.BytesIO();torch.save(dict(schema=p['schema'],weights=p['weights'],method_digest=p['method_digest']),buf);raw=buf.getvalue();ctx=json.dumps(p['binding']).encode()
    (tmp/(name+'.pt')).write_bytes(raw);(tmp/(name+'.context.json')).write_bytes(ctx)
    inv={name:dict(file=name+'.pt',bytes=len(raw),training_asset_file_sha256=digest(raw),context_file_sha256=digest(ctx),context_sha256=p['binding']['sha256'])}
    host=load_artifact(g,name,tmp,inv)
    for i in range(3):z,t=host.step(pixels(i));physical(t,name)
    assert host.visits==3 and host.state['counter']==3
    if name.endswith('STATIC'):
     fresh=load_artifact(g,name,tmp,inv);single,_=fresh.step(pixels(2));assert torch.equal(z,single)
    host._check_frozen(boundary=True);check(name+'_three_visits_loader_state')
  assert tensor_digest(tensors(g.model))==original and all(p.grad is None for p in g.model.parameters());check('frozen_backbone')
  total=dict(COUNTS-before);r['counts']=dict(forwards=total['backbone_forwards']+base_counts['forwards'],backwards=base_counts['backwards'],Adam=base_counts['Adam'],VJP=0,AdamW=0)
  assert r['counts']==dict(forwards=87,backwards=5,Adam=5,VJP=0,AdamW=0)
  r['status']='PASSED'
 except BaseException as e:r['first_error']=dict(type=type(e).__name__,message=str(e));traceback.print_exc()
 finally:
  if g is not None:g.close()
  r.update(wall_seconds=time.monotonic()-start,observed_method_counts=dict(COUNTS-before),observed_base_counts=base_counts)
  Path(os.environ['GPU_QUALIFICATION_RESULT']).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 raise SystemExit(r['status']!='PASSED')
if __name__=='__main__':main()
