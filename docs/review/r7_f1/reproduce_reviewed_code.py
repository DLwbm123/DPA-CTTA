"""Expected to FAIL on reviewed82aca: Small CPU fixture, not full ResUNet.
Run against the original implementation checkout, with tests/r7 on sys.path.
"""
import copy,json,unittest
import torch
from common import segmenter,method,pixels
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_shared.preparation import inference_from_tensors
from dpa_ctta.r7_shared.numerics import COUNTS
class RequiredRejections(unittest.TestCase):
    def fixture(self):
        a,b=segmenter(),segmenter();m=method('B');m.freeze()
        with torch.no_grad():b.model.seg_head.bias.add_(1)
        return a,b,m
    def test_fresh_rejects_different_head(self):
        a,b,m=self.fixture()
        with self.assertRaises(ValueError):inference_from_tensors(b,'B',False,m.basis,copy.deepcopy(m.state_dict()),m.digest())
        a.close();b.close()
    def test_restore_rejects_different_head(self):
        a,b,m=self.fixture();h=OnlineHost(a,m);h.step(pixels());packet=h.save_state();other=OnlineHost(b,copy.deepcopy(m))
        with self.assertRaises(ValueError):other.load_state(packet)
        a.close();b.close()
if __name__=='__main__':
    torch.set_num_threads(2);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RequiredRejections))
    print(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),counts=dict(COUNTS),backward=0,optimizers=0)))
    raise SystemExit(int(not result.wasSuccessful()))
