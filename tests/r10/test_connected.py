import copy,unittest
import torch
from dpa_ctta.r8_ba.methods import R8B
from dpa_ctta.r10_use_write_rl.controller import Actor,Controller,Host
from dpa_ctta.r10_use_write_rl.math import Memory,use_and_write
from dpa_ctta.r10_use_write_rl.learning import Trainer


def carrier():
    torch.manual_seed(9);b=R8B(torch.eye(1024,64,dtype=torch.float64),.3)
    b.observer.fit_scaler(torch.randn(8,134),'fit');b.freeze();return b


class ToySegmenter:
    def __init__(self):self.calls=0;self.fail=False
    def __call__(self,image,v=None,observe=False):
        self.calls+=1
        if self.fail:raise OSError(5,'injected prediction EIO')
        if observe:return torch.zeros(1,2,512,512),image,torch.zeros(64,64)
        x=v[:64].reshape(2,32).mean(-1).float() if v is not None else torch.zeros(2)
        return x[None,:,None,None].expand(1,2,512,512)


class ToySource:
    def __init__(self,c):
        self.segmenter=ToySegmenter();self.c=c
        self.items=[dict(image=torch.randn(134),label=(torch.rand(1,2,512,512)>.5).float(),observation=c.observe(torch.randn(134),torch.zeros(64,64))) for _ in range(8)]
    def item(self,fold,seed,episode,visit,warm=False):return self.items[visit%8]
    def window(self,k,seed):return 0,4,self.items[:4]
    def probes(self,*args):return self.items[4:6]


class Connected(unittest.TestCase):
    def test_real_B_full_reset_equivalence(self):
        b=carrier();c=Controller(b,torch.ones(64,dtype=torch.float64));actor=Actor(1)
        for reset in (False,True):
            state=Memory.zero();original=b.initial()
            for _ in range(6):
                raw=torch.randn(134);tokens=torch.randn(64,64)
                if reset:state=Memory.zero()
                o=c.observe(raw,tokens);u,state,_=c.act(actor,o,state,forced_write=1.,identity_use=True)
                original,_=b.update(raw,tokens,original,'RESET_HISTORY' if reset else None)
                self.assertTrue(torch.equal(u,original['z']))
                self.assertTrue(torch.equal(state.q,original['d']))
    def test_current_prediction_and_atomic_failure(self):
        b=carrier();c=Controller(b,torch.ones(64,dtype=torch.float64));a=Actor(2);s=ToySegmenter();raw=torch.randn(134)
        observation=c.observe(raw,torch.zeros(64,64));m=Memory.zero()
        u0,m0,_=c.act(a,observation,m,forced_write=0.)
        u1,_,_=c.act(a,observation,m,forced_write=1.)
        self.assertTrue(torch.equal(s(raw,b.basis@u0),s(raw,b.basis@u1)))
        for key in ('m','q','h'):self.assertTrue(torch.equal(getattr(m,key),getattr(m0,key)))
        host=Host(s,c,a,'fixture');host.step(raw);before=host.snapshot();s.fail=True
        with self.assertRaises(OSError):host.step(raw)
        self.assertEqual(host.visits,before['visits']);self.assertTrue(torch.equal(host.state.m,before['memory']['m']))
    def test_all_training_paths_and_exact_resume(self):
        for method in ('WARM','SUP_STATIC','SUP_SEQ','SUP_RET','GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA'):
            c=Controller(carrier(),torch.ones(64,dtype=torch.float64));source=ToySource(c)
            actor=Actor(3);t=Trainer(actor,c,source,3,method,'test');before=copy.deepcopy(actor.state_dict());frozen=copy.deepcopy(c.carrier.state_dict())
            result=t.step();self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
            for k,v in frozen.items():self.assertTrue(torch.equal(v,c.carrier.state_dict()[k]))
            writer_changed=any(not torch.equal(v,actor.state_dict()[k]) for k,v in before.items() if k.startswith('write.'))
            if method in ('WARM','SUP_STATIC','GR_CUR'):self.assertFalse(writer_changed)
            if method in ('SUP_SEQ','SUP_RET'):self.assertTrue(writer_changed)
            snap=t.snapshot()
            import tempfile
            from pathlib import Path
            from dpa_ctta.r9_current_first.storage import write_torch,load_torch
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'snapshot.pt';sha=write_torch(path,snap);snap=load_torch(path,sha)
            r1=t.step();weights=copy.deepcopy(actor.state_dict());t.restore(snap);r2=t.step()
            self.assertEqual(r1,r2)
            for k,v in weights.items():self.assertTrue(torch.equal(v,actor.state_dict()[k]))
    def test_writer_immediate_zero_delayed_nonzero(self):
        c=Controller(carrier(),torch.ones(64,dtype=torch.float64));a=Actor(4)
        o=c.observe(torch.randn(134),torch.zeros(64,64));u,nxt,_=c.act(a,o,Memory.zero())
        u.square().sum().backward(retain_graph=True)
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in a.write.parameters()))
        a.zero_grad();later,_,_=c.act(a,o,nxt);later.square().sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in a.write.parameters() if p.grad is not None),0.)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
