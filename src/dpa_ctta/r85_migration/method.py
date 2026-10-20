from ..r85_loss_aggregation.method import CELLS,SEEDS,style,self_check,oracle,Previous
from ..r85_loss_aggregation.method import Host as FactorialHost

def Host(state,cell,seed,identity,**kwargs):
    return Previous(state,'NATIVE',seed,identity,**kwargs) if cell=='NATIVE' else FactorialHost(state,cell,seed,identity,**kwargs)
