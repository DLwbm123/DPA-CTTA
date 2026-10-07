"""One runnable check: bounded actions, rollback, unchanged controls and cap semantics."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from .method import Host, residual_action, ARMS
from ..r25_parameter_policy.method import Host as Previous
from ..r20_model_only_search.tests import Tiny
from ..r20_model_only_search import runtime
from ..r19_model_only.method import equal


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    h=torch.randn(1,8,4,4); output=h+.1*torch.randn_like(h)
    assert torch.equal(residual_action(h,output,torch.zeros(4)),output)
    action=torch.tensor([100.,-100.,.3,-.3])
    result=residual_action(h,output,action)
    assert torch.isfinite(result).all() and not torch.equal(result,output)
    assert (result-h).norm() <= 2.118*(output-h).norm()
    try: residual_action(h,output,torch.tensor([float('nan')]*4))
    except ValueError: pass
    else: raise AssertionError('nonfinite action accepted')
    model=Tiny(); x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
    cfg=lambda arm:dict(id=arm,family=arm,host='C',params={},components={})
    for arm in ARMS:
        a=Host(None,cfg(arm),20260907,'test',device='cpu',model=copy.deepcopy(model))
        if arm in ('C','ANCHOR','REPEAT5'):
            b=Previous(None,cfg(arm),20260907,'test',device='cpu',model=copy.deepcopy(model))
            for _ in range(2):
                za,_=a.step(x);zb,_=b.step(x)
                assert torch.equal(za,zb),arm
                for key in ('parameters','adam','native_rng','adapter'):
                    assert equal(a.snapshot()[key],b.snapshot()[key]),(arm,key)
            b.close()
        else:
            a.step(x); start=a.snapshot()
            z,t=a.step(x.flip(-1)); end=a.snapshot()
            a.restore(start); z2,t2=a.step(x.flip(-1))
            assert torch.equal(z,z2) and equal(end,a.snapshot()),arm
            assert t['diagnostics']==t2['diagnostics'] and a.visits==2 and a.native.steps==2
            assert t['diagnostics']['residual_action_enabled']
            assert len(t['proposal_masks_b64'])==4
        a.check_frozen(True); a.close();print('PASS',arm,flush=True)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/'live').mkdir()
        c=dict(output_root=td,origin=dict(gpu_worker_cap_seconds=None,disk_cap_bytes=1024**2))
        with patch.dict('os.environ',{'RUN_DEADLINE':'9999999999'}),patch.object(runtime,'amounts',return_value=(1000,{})):
            runtime.Guard(c,'online_test').observe()
            c['origin']['gpu_worker_cap_seconds']=900
            try:runtime.Guard(c,'online_test').observe()
            except TimeoutError:pass
            else:raise AssertionError('finite cap must still be enforced')
    print(json.dumps(dict(status='PASS',tests=8,checks=['bounded action and zero parity','finite validation',
        'three controls exact parity','three action arms exact resume','one committed branch',
        'frozen model','unlimited GPU cap','finite legacy cap'],online_labels=False)))


if __name__=='__main__':main()
