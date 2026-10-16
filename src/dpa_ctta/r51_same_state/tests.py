"""Generated CPU fixtures only: geometry, full gradients, and original-step parity."""
import copy
import unittest
import torch

from ..r20_model_only_search.tests import Tiny
from ..r37_cw_orientation.method import Host, candidates
from ..r19_model_only.method import equal
from .geometry import proposals, check_committed
from .diagnostic import one_pair, mask, assert_native_equivalence, BRANCHES


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_retry_preserves_failure_and_subtracts_budget(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .run import controller, save, read
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prior=dict(mode='worker',status='FAILED',gpu_worker_seconds=2.554733991622925,cpu_worker_seconds=0.)
            save(root/'PRIOR_RESOURCE_LEDGER.json',dict(attempts=[prior],all_workers_retired=True,gpu_worker_seconds=prior['gpu_worker_seconds']))
            for name in ('CPU_CHECK','IMPORT_CHECK'):save(root/(name+'.json'),dict(status='PASS'))
            save(root/'RUN_STATE.json',dict(status='GPU_COMPLETE'))
            c=dict(output_root=directory,source_sha='frozen',prior_gpu_worker_seconds=2.555,
                   gpu_cap_seconds=1800,cpu_cap_seconds=900,gpu=dict(physical_id=7,uuid='assigned'))
            def child(config,mode,cap,gpu):
                self.assertEqual(cap,1797.445 if gpu else 900)
                return dict(mode=mode,status='COMPLETE',gpu_worker_seconds=10. if gpu else 0.,cpu_worker_seconds=0. if gpu else 1.)
            with patch('subprocess.check_output',return_value='7, assigned, 10000'), patch('dpa_ctta.r51_same_state.run.run_child',side_effect=child):
                self.assertEqual(controller(c),0)
            ledger=read(root/'RESOURCE_LEDGER.json')
            self.assertEqual(ledger['attempts'][0],prior)
            self.assertEqual(ledger['gpu_worker_seconds'],12.555)
            self.assertEqual(len(ledger['attempts']),3)

    def test_matching_double_and_degenerate(self):
        gen=torch.Generator().manual_seed(49)
        n=torch.randn(64,generator=gen,dtype=torch.float64)
        h=torch.exp(torch.randn(64,generator=gen,dtype=torch.float64))
        g=torch.randn(64,generator=gen,dtype=torch.float64)
        s=torch.randn(64,generator=gen,dtype=torch.float64)
        j=n*.6+torch.randn(64,generator=gen,dtype=torch.float64)*.05
        out,info=proposals(n,j,h,g,s)
        self.assertFalse(info['degenerate'])
        for k,v in out.items():check_committed(n,j,v,h,g,s,radial=k=='radial')
        out,info=proposals(n,n*.5,h,g,s)
        self.assertTrue(info['degenerate'])
        self.assertTrue(torch.equal(out['null0'],n*.5))

    def test_reject_invalid_geometry(self):
        n=torch.ones(8);h=torch.ones(8)
        with self.assertRaises(ValueError):proposals(n,n,h*0,n,n)
        with self.assertRaises(ValueError):proposals(n,n*2,h,n,n)
        with self.assertRaises(ValueError):check_committed(n,n*.5,n*.8,h,n,n)

    def test_cluster_interval_keeps_constant_effect(self):
        import numpy as np
        from .score import interval
        self.assertEqual(interval(np.full((48,2,2),.125),['a']*12+['b']*12+['c']*12+['d']*12),[.125,.125])

    def test_full_gradients_and_original_step(self):
        torch.manual_seed(2026)
        h=Host(None,candidates()[3],17011,'synthetic',device='cpu',model=Tiny())
        x=torch.linspace(.02,.98,3*512*512).reshape(1,3,512,512)
        try:
            h.step(x.roll(7,-1));anchor=h.snapshot()
            h.restore(anchor);original,_=h.step(x);reference=h.snapshot()
            h.restore(anchor);again,_=h.step(x)
            self.assertTrue(torch.equal(original,again));self.assertTrue(equal(reference,h.snapshot()))
            h.restore(anchor);outputs,details,vectors=one_pair(h,x)
            self.assertEqual(tuple(outputs),BRANCHES)
            self.assertTrue(torch.equal(outputs['native'],mask(original)))
            assert_native_equivalence(reference,h.snapshot())
            self.assertLess(details['gradient_decomposition_relative_error'],1e-5)
            self.assertEqual(vectors['base_gradient'].numel(),sum(p.numel() for p in h.native.params))
            self.assertEqual(len(details['committed_matching']),5)
            h.restore(anchor);outputs2,_,vectors2=one_pair(h,x)
            self.assertTrue(all(torch.equal(outputs[k],outputs2[k]) for k in BRANCHES))
            self.assertTrue(equal(vectors,vectors2))
        finally:h.close()


if __name__=='__main__':unittest.main()
