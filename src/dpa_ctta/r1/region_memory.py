"""Same selection pool, single/global or quota-matched semantic/shuffled banks."""
import torch
from .recovery import local_seed
from .streaming_pca import StreamingSubspace,projection_residual,bank_audit


def grid(x):
    if x.ndim!=4 or x.shape[0]!=1:raise ValueError('single BCHW tensor')
    H,W=x.shape[-2:];a=((torch.arange(32,device=x.device)+.5)*H/32).long();b=((torch.arange(32,device=x.device)+.5)*W/32).long()
    return x[0,:,a[:,None],b[None,:]].reshape(x.shape[1],1024).T


def split_shuffle(ids,visit,salt):
    counts=[len(i) for i in ids];pool=torch.cat(ids);g=torch.Generator(device='cpu').manual_seed(local_seed(salt,visit))
    return list(pool[torch.randperm(len(pool),generator=g)].split(counts))


def tokens(q,views):
    q=q.detach().cpu();views=views.detach().cpu()
    if q.shape!=(1024,2) or views.shape!=(6,1024,2):raise ValueError('OD/OC six-view grid')
    hard=q>=.5;reliable=((q<=.1)|(q>=.9)) & ((views>=.5)==hard).all(0)
    # Token ID c*1024+u, so OD/OC repeat a feature explicitly.
    region=[torch.nonzero(hard[:,c]==bool(b)).flatten()+c*1024 for c in (0,1) for b in (0,1)]
    reliable= reliable.T.reshape(-1)
    return region,reliable


class Memory:
    def __init__(self,mode):
        if mode not in ('GLOBAL','REGION','SHUFFLED'):raise ValueError('bank mode')
        self.mode=mode;self.banks=[StreamingSubspace() for _ in range(1 if mode=='GLOBAL' else 4)]
    def snapshots(self,visit):
        values=[b.snapshot() for b in self.banks]
        if any(s is not None and s[2]>=visit for s in values):raise ValueError('current item in own basis')
        return values
    def prepare(self,f0,q,views,visit):
        if f0.shape!=(1024,32):raise ValueError('32-dimensional penultimate feature grid')
        region,rel=tokens(q,views);selected=[]
        for index,ids in enumerate(region):
            ids=ids[rel[ids]];g=torch.Generator(device='cpu').manual_seed(local_seed('sample_'+str(index),visit))
            if len(ids)>32:ids=ids[torch.randperm(len(ids),generator=g)[:32]]
            selected.append(ids)
        sampled=[len(i) for i in selected];raw=f0.detach().cpu();norm=raw.norm(dim=1,keepdim=True);v=raw/norm.clamp_min(1e-6)
        zero=[int((norm[i%1024,0]==0).sum()) for i in selected]
        selected=[i[norm[i%1024,0]>0] for i in selected];counts=[len(i) for i in selected]
        if self.mode=='GLOBAL':selected=[torch.cat(selected)];loss_ids=[torch.cat(region)]
        elif self.mode=='SHUFFLED':selected=split_shuffle(selected,visit,'memory_shuffle');loss_ids=split_shuffle(region,visit,'loss_shuffle')
        else:loss_ids=region
        vectors=[v[ids%1024].double() for ids in selected]
        return vectors,loss_ids,dict(region_token_counts=[len(i) for i in region],sampled_region_counts=sampled,selected_region_counts=counts,assigned_bank_counts=[len(i) for i in selected],zero_vectors=zero,missing_foreground=[len(region[i])==0 for i in (1,3)])
    def loss(self,fs,ids,snapshots):
        if fs.shape!=(1024,32):raise ValueError('strong feature grid')
        terms=[projection_residual(fs[i.to(fs.device)%1024],s[0],s[1]) for i,s in zip(ids,snapshots) if len(i) and s is not None]
        return torch.stack(terms).mean() if terms else fs.new_zeros(())
    def merge(self,vectors,visit):
        for bank,x in zip(self.banks,vectors):bank.merge(x,visit)
    def audit(self):
        values=[bank_audit(b) for b in self.banks]
        if sum(v['state_bytes'] for v in values)>=1024**2:raise ValueError('bounded PCA state exceeded')
        return values
