"""CPU-only review probes for DLwbm123/DPA-CTTA@54911f9.

Not a benchmark or training runner. Uses only small synthetic tensors and the
repository's TinySource fixture. Runs no real-data, checkpoint, or GPU operation.
Invoke: python DPA_CTTA_REVIEW_PROBES_54911f9.py --repo /path/to/DPA-CTTA --output result.json
On the audited commit, some results intentionally expose unsafe behavior; they are
observations, not tests which should pass after a fix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
sys.path[:0] = [str(root / 'src'), str(root / 'tests')]
import torch
from common import make_online, make_anchor, make_atlas
from dpa_ctta.descriptors import FrozenDescriptor
from dpa_ctta.manifold import LatentFiLM
from dpa_ctta.atlas import DistilledPotentialAtlas
from dpa_ctta.types import AtlasResult
from dpa_ctta.proximal import closed_form_update
from dpa_ctta.offline.oracle import optimize_oracle_state

torch.set_num_threads(2)
torch.manual_seed(29)
assert not torch.cuda.is_initialized()
expected = {
 'src/dpa_ctta/types.py':'f761848717b82f38d2f0d392efca21c3b91f47e7',
 'src/dpa_ctta/proximal.py':'a2884f75dc6b06886f575928239a588ce3ebd876',
 'src/dpa_ctta/manifold.py':'008cadf9d9fb192a91498faa2be190c95081d181',
 'src/dpa_ctta/online.py':'64eeb379828e5ba9c7270879075c8d83313d996b',
 'src/dpa_ctta/hooks.py':'b9ebf6d76910296415891b6163ad47250033616d',
 'src/dpa_ctta/losses.py':'a8c3ac5e5c07d2a94bdc93eb5942be0c92b0ccfd',
 'src/dpa_ctta/offline/oracle.py':'aa20d573f9910277e8158768933a45f96fe632fc',
 'src/dpa_ctta/descriptors.py':'c1e8481a7f65caa614da97bd2d28583d4ef2f875',
 'src/dpa_ctta/anchors.py':'a1739f72826ff93d3e2720ccf6c04cacc3ba0b2f',
 'src/dpa_ctta/atlas.py':'0213e30e3b569f0b73f1cf7c681f88c98d11677b',
 'tests/common.py':'f1d86e6e4fa83e7a4dd188d355d7d4d597b2da3f'}
verified=[]
for path, sha in expected.items():
    content=(root/path).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if sha!=actual:
        raise RuntimeError(f'Review pin mismatch in {path}: expected {sha}, got {actual}')
    verified.append({'path':path,'git_blob_sha1':actual})
report = {
 'reviewed_commit':'54911f9e1aff4cc0338dfac0264e67456024d503',
 'scope':'11 blob-verified files; synthetic CPU probes, not the 143-test or real-model integration suite',
 'environment':{'python':platform.python_version(),'torch':torch.__version__},
 'same_environment_as_publication':False,
 'server_accessed':False, 'real_data_accessed':False, 'checkpoint_loaded':False,
 'research_training_started':False, 'source_subset':verified, 'probes':{}}
p=report['probes']
# P1: normal finite proximal behavior.
prev=torch.randn(1,16,dtype=torch.float64)
precision=0.001+torch.rand(1,16,dtype=torch.float64)
rhs=torch.randn(1,16,dtype=torch.float64)
inputs=AtlasResult(torch.ones(1,1,dtype=torch.float64),precision,rhs)
out=closed_form_update(prev,inputs,1.0,0.001)
solution=torch.linalg.solve(torch.diag((precision+1.0)[0]),(rhs+prev)[0])[None]
other=prev+torch.randn_like(prev)
otherout=closed_form_update(other,inputs,1.0,0.001)
ratio=float((out.state-otherout.state).norm()/(prev-other).norm())
p['finite_proximal']={'max_solve_error':float((solution-out.state).abs().max()),'history_lipschitz_ratio':ratio,'bound':out.contraction_bound,'bound_satisfied':ratio<=out.contraction_bound+1e-12}
# P2: actual eight-row source-statistic projection is effectively rank two.
rank_rows={}
for channels in (32,64):
    desc=FrozenDescriptor('polyp' if channels==32 else 'fundus',channels)
    sv=torch.linalg.svdvals(desc.feature_projection.double())
    rank_rows[str(channels)]={'shape':list(desc.feature_projection.shape),'singular_values':sv.tolist(),'rank_at_relative_1e-5':int((sv>sv[0]*1e-5).sum())}
p['descriptor_projection_rank']=rank_rows
# P3: one-channel FiLM at three levels exposes at most six linear coordinates.
films=[LatentFiLM(1,16).double() for _ in range(3)]
coef=torch.cat([torch.cat((m.gamma_basis,m.beta_basis),0) for m in films],0)
_,sv,vh=torch.linalg.svd(coef,full_matrices=True)
null=vh[-1]
p['polyp_scalar_modulation_bound']={'coefficient_shape':list(coef.shape),'rank':int(torch.linalg.matrix_rank(coef)),'latent_dim':16,'null_dimension_at_least':10,'null_vector_coefficient_max_abs':float((coef@null).detach().abs().max()),'scope':'mathematical bound from the published one-channel injection config; not a pretrained PraNet run'}
# P4: unchecked rhs and previous state; malformed rhs silently broadcasts.
p['proximal_invalid_inputs']={}
for field in ('rhs','previous'):
    badprev=prev.clone(); badrhs=rhs.clone()
    if field=='rhs': badrhs[0,0]=float('nan')
    else: badprev[0,0]=float('nan')
    try:
        result=closed_form_update(badprev,AtlasResult(inputs.weights,precision,badrhs),1.,.001)
        p['proximal_invalid_inputs'][field]={'rejected':False,'returned_state_finite':bool(torch.isfinite(result.state).all())}
    except Exception as error:
        p['proximal_invalid_inputs'][field]={'rejected':True,'exception_type':type(error).__name__}
try:
    result=closed_form_update(prev,AtlasResult(inputs.weights,precision,torch.ones(1,1,dtype=torch.float64)),1.,.001)
    p['proximal_invalid_inputs']['broadcast_rhs']={'rejected':False,'returned_shape':list(result.state.shape)}
except Exception as error:
    p['proximal_invalid_inputs']['broadcast_rhs']={'rejected':True,'exception_type':type(error).__name__}
# P5: exception injected only in final adapted prediction leaves state committed.
x=torch.linspace(0,1,3*16*16).reshape(1,3,16,16)
a=make_online();before=a.z_prev.clone();counter=[0]
original=a.injected_model.forward

def fail_last(*call_args,**kwargs):
    counter[0]+=1
    if counter[0]==len(a.atlas.anchors)+2:
        raise RuntimeError('REVIEW_INJECTED_FINAL_FORWARD_FAILURE')
    return original(*call_args,**kwargs)
try:
    try:
        with patch.object(a.injected_model,'forward',side_effect=fail_last): a.step(x)
    except RuntimeError as error:
        caught=str(error)
    p['final_prediction_exception']={'exception':caught,'forward_count':counter[0],'state_unchanged':bool(torch.equal(before,a.z_prev)),'state_delta_norm':float((before-a.z_prev).norm())}
finally: a.injected_model.close()
# P6: NaN state center results in persistent NaN without rejection.
a=make_online()
try:
    with torch.no_grad(): a.atlas.anchors[0].state_center[0]=float('nan')
    try:
        result=a.step(x)
        p['nan_center_online']={'rejected':False,'state_finite':bool(torch.isfinite(a.z_prev).all()),'logits_finite':bool(torch.isfinite(result.logits).all())}
    except Exception as error:
        p['nan_center_online']={'rejected':True,'type':type(error).__name__,'state_finite':bool(torch.isfinite(a.z_prev).all())}
finally: a.injected_model.close()
# P7: frozen-anchor inference still executes a full source pass for every anchor.
a=make_online()
try:
    anchors=[make_anchor('polyp',.02*i)[0] for i in range(8)]
    old=a.atlas
    a.atlas=DistilledPotentialAtlas(anchors,old.descriptor,old.scaler,.7)
    n=[0]
    handle=a.injected_model.source_model.register_forward_pre_hook(lambda *_:n.__setitem__(0,n[0]+1))
    try:
        a.step(x); first=n[0]; a.step(x); second=n[0]-first
    finally: handle.remove()
    p['frozen_anchor_forward_count']={'K':8,'first_step_source_forwards':first,'second_step_source_forwards':second,'note':'forward call count, not latency or full-size FLOPs'}
finally: a.injected_model.close()
# P8: local state-only oracle accumulates gradients on unrelated trainable FiLM bases.
a=make_online();w=a.injected_model
try:
    label=torch.zeros(1,1,16,16);label[:,:,4:12,4:12]=1
    bases=[param for module in w.adapters for param in (module.gamma_basis,module.beta_basis)]
    weights_before=[item.detach().clone() for item in bases]
    state1=optimize_oracle_state(w,x,label,steps=2)
    grads1=[None if item.grad is None else item.grad.clone() for item in bases]
    state2=optimize_oracle_state(w,x,label,steps=2)
    ratios=[float(item.grad.norm()/grad.norm()) for item,grad in zip(bases,grads1) if grad is not None and grad.norm()>0]
    p['oracle_gradient_accumulation']={'basis_tensors_with_nonzero_grad':sum(item.grad is not None and bool(torch.count_nonzero(item.grad)) for item in bases),'basis_tensor_count':len(bases),'weights_unchanged':all(torch.equal(a_,b.detach()) for a_,b in zip(weights_before,bases)),'second_to_first_gradient_norm_ratios':ratios,'same_initial_state_same_result':bool(torch.equal(state1,state2)),'scope':'two two-step synthetic unit probes, not source training'}
finally: a.injected_model.close()
# P9: source descriptor is numerically history-independent; finite normal path works.
a=make_online()
try:
    queries=[]
    for previous in (0.,3.,-2.):
        a.z_prev.fill_(previous)
        logit,feature=a._zero_pass(x)
        queries.append(a.atlas.descriptor(x,logit.sigmoid(),feature).detach())
    p['history_independent_descriptor']={'maximum_difference':max(float((q-queries[0]).abs().max()) for q in queries)}
finally: a.injected_model.close()
report['cuda_initialized']=torch.cuda.is_initialized()
assert not report['cuda_initialized']
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'environment':report['environment'],'probes':p,'cuda_initialized':False},indent=2,allow_nan=False))
