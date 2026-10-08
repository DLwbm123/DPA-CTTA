"""True skip versus half/full Adam parameter steps from one frozen state."""
import numpy as np
from ..r28_common_state.method import Host as Previous

ARMS=('FULL','ZERO','HALF')


class Host(Previous):
    def __init__(self,*args,**kwargs):
        self.strength=1.
        super().__init__(*args,**kwargs)

    def _lr(self,opt,args,kwargs):
        super()._lr(opt,args,kwargs)
        for group in opt.param_groups:group['lr']*=self.strength
        self.diag['BN_lr']=opt.param_groups[0]['lr']
        self.diag['adapter_lr']=opt.param_groups[1]['lr']

    def evaluate(self,x,strength):
        if strength not in (0.,.5,1.):raise ValueError('unregistered update strength')
        before=self.snapshot()
        try:
            if strength==0:
                self.action.zero_();self._prepare_prototypes(x)
                return self._readonly(self.native.model,x).cpu()
            self.strength=strength
            return self.step(x)[0]
        finally:
            self.restore(before);self.strength=1.

    def probe_strength(self,x):
        return np.stack([(self.evaluate(x,s).numpy()[0]>=0) for s in (1.,0.,.5)])
