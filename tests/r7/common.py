import torch
from torch import nn
from dpa_ctta.r7_shared.network import Segmenter
from dpa_ctta.r7_shared.preparation import new_method
from dpa_ctta.r7_shared.source import Record,SourceData,Oracles,split,support_pair

class Small(nn.Module):
    """Explicit small fixture only; full ResUNet tests are separate."""
    def __init__(self):
        super().__init__();self.res=nn.Module();self.res.conv1=nn.Conv2d(3,64,1)
        self.up1=nn.Sequential(nn.Conv2d(64,256,1),nn.BatchNorm2d(256))
        self.up3=nn.Sequential(nn.Conv2d(256,256,1),nn.BatchNorm2d(256));self.seg_head=nn.Conv2d(256,2,1)
    def forward(self,x):
        x=torch.nn.functional.adaptive_avg_pool2d(x,(8,8));h=self.res.conv1(x);h=self.up1(h);h=self.up3(h)
        return torch.nn.functional.interpolate(self.seg_head(h),(512,512),mode='bilinear',align_corners=False)

def pixels(i=0):return torch.rand((1,3,512,512),generator=torch.Generator().manual_seed(31+i))
def basis(rank):return torch.linalg.qr(torch.randn(1024,rank,dtype=torch.float64,generator=torch.Generator().manual_seed(92)))[0]
def method(group,static=False):
    m=new_method(group,basis(16 if group=='A' else 32),static)
    m.observer.fit_scaler(torch.randn(6,134,generator=torch.Generator().manual_seed(44)),'fit');return m

def obs(i=0):
    g=torch.Generator().manual_seed(87+i);return torch.randn(134,generator=g),torch.randn(64,64,generator=g)

def segmenter(full=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        if full:
            from dpa_ctta.integrations.ctta_suite import build_reference_model
            m,_=build_reference_model('fundus')
        else:m=Small()
    return Segmenter(m)

def source_fixture():
    groups=[f'procedural_{i}' for i in range(48)];folds=split(groups);records=[]
    # Procedural identity metadata are distinct. Shared tensor construction does
    # not stand in for real independent patients; it tests API/fold enforcement.
    x=pixels();y=(x[:,:2]>.5).float()
    for fold,gs in folds.items():
        records.extend(Record(g,fold,x,y) for g in gs)
    data=SourceData(records,folds);oracles={}
    for fold in folds:
        n=128 if fold=='fit' else 32
        oracles[fold]=Oracles(fold,.01*torch.randn(1024,n,generator=torch.Generator().manual_seed(n)),tuple(support_pair(data,fold,i) for i in range(n))).validate(data)
    return data,oracles
