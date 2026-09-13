"""At most three affine/Adam/bank states, one live model and one global RNG."""
import copy,time
import torch
from .kernels import DescriptorRouter
from .stats import Memory
from .displacement import flatten,replace


def state_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(state_bytes(v) for v in value.values())
    if isinstance(value,list):return sum(state_bytes(v) for v in value)
    return 0


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


class Contexts:
    def __init__(self,params,names,optimizer,shared=False):
        self.params,self.names,self.optimizer=params,names,optimizer
        self.shared=shared;self.source=flatten(params).cpu().clone()
        self.router=DescriptorRouter();self.slots=[];self.active=None;self.total_steps=0

    def load(self,descriptor):
        started=time.monotonic();index,created,distance=self.router.choose(descriptor)
        old_threshold=None;overflow=False
        if self.router.entries:
            ds=[float((descriptor.double()-e.centre).square().sum()) for e in self.router.entries]
            e=self.router.entries[min(range(len(ds)),key=ds.__getitem__)]
            old_threshold=max(1e-6,e.distance_mean+3*(e.distance_m2/max(e.count-1,1))**.5)
            overflow=len(self.slots)==3 and e.count>=16 and distance>old_threshold
        if created:
            if len(self.slots)>=3:raise ValueError('context capacity')
            self.slots.append(dict(affine=None if self.shared else self.source.clone(),adam={},steps=0,memory=Memory()))
        switched=self.active is not None and index!=self.active
        replacements=0
        if not self.shared and (created or index!=self.active):
            s=self.slots[index];replace(self.params,s['affine'].to(self.params[0]));replacements=1
            self.optimizer.state.clear()
            for name,p in zip(self.names,self.params):
                if name in s['adam']:
                    self.optimizer.state[p]={k:(v.clone().to(p.device) if isinstance(v,torch.Tensor) and k!='step' else cpu_copy(v)) for k,v in s['adam'][name].items()}
        self.active=index
        return self.slots[index]['memory'],dict(slot=index,created=created,switched=switched,distance=distance,
            old_threshold=old_threshold,overflow=overflow,actual_parameter_replacements=replacements,load_seconds=time.monotonic()-started)

    def commit(self,descriptor,trace):
        s=self.slots[self.active];s['steps']+=1;self.total_steps+=1
        if not self.shared:
            s['affine']=flatten(self.params).cpu().clone()
            s['adam']={name:cpu_copy(self.optimizer.state[p]) for name,p in zip(self.names,self.params)}
        self.router.commit(descriptor,trace['slot'],trace['created'],trace['distance'])
        if sum(x['steps'] for x in self.slots)!=self.total_steps:raise ValueError('context/global counts')
        expected=self.total_steps if self.shared else s['steps']
        if any(int(self.optimizer.state[p]['step'])!=expected for p in self.params):raise ValueError('selected Adam counter')
        trace.update(slot_assigned_counts=[e.count for e in self.router.entries],slot_updates=[x['steps'] for x in self.slots],
            optimizer_steps=[self.total_steps] if self.shared else [x['steps'] for x in self.slots],active_slots=1,
            slots=len(self.slots),model_copies=2,stored_affine_scalars=0 if self.shared else self.source.numel()*len(self.slots),
            context_tensor_bytes=sum(state_bytes(s['affine'])+state_bytes(s['adam'])+sum(b['state_bytes'] for b in s['memory'].audit()) for s in self.slots)+state_bytes(self.source)+sum(state_bytes(e.descriptor_sum) for e in self.router.entries))
