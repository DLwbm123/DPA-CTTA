"""Attempt-local measured operations, independent of restored logical counters."""
import time
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook
from .protocol import CAPS


class Meter:
    def __init__(self,model,guard=lambda cost:None):
        self.model,self.guard=model,guard;self.cost=dict.fromkeys(CAPS,0);self.handles=[]
    def attach(self,model):
        handle=model.register_forward_pre_hook(lambda *_:self.tick('model_forwards'));self.handles.append(handle);return handle
    def tick(self,key):
        self.cost[key]+=1;self.check()
    def check(self):
        self.cost['gpu_seconds']=time.monotonic()-self.started
        self.guard(self.cost)
    def __enter__(self):
        self.started=getattr(self,'start_override',time.monotonic())
        self.backward,self.grad=torch.autograd.backward,torch.autograd.grad
        def backward(*a,**kw):self.tick('backward_calls');return self.backward(*a,**kw)
        def grad(*a,**kw):self.tick('vjp_calls');return self.grad(*a,**kw)
        torch.autograd.backward,torch.autograd.grad=backward,grad
        self.handles=[register_optimizer_step_pre_hook(lambda *_:self.tick('optimizer_steps'))]
        if self.model is not None:self.attach(self.model)
        try:self.check()
        except BaseException:
            self.__exit__(None,None,None);raise
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        torch.autograd.backward,torch.autograd.grad=self.backward,self.grad
        self.cost['gpu_seconds']=time.monotonic()-self.started
