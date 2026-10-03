"""Current-image connected structure tracking; no labels or cross-image state."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from scipy import ndimage as ndi

LEVELS=(.40,.45,.50,.55,.60)
NEIGHBORS=np.ones((3,3),bool)

def hard(z):return (z.detach().float().sigmoid()>=.5).cpu().numpy()[0]

def boundary_band(mask,radius):
    out=[]
    for a in mask:
        if not a.any() or a.all():out.append(np.zeros_like(a));continue
        edge=a^ndi.binary_erosion(a,structure=NEIGHBORS,border_value=0)
        out.append(ndi.distance_transform_edt(~edge)<=radius)
    return np.stack(out)

def topology(mask):
    return [(int(ndi.label(a,NEIGHBORS)[1]),int(ndi.label(ndi.binary_fill_holes(a)&~a,NEIGHBORS)[1])) for a in mask]

def containment(mask):return int((mask[1]&~mask[0]).sum())

def largest_filled(mask):
    out=[]
    for a in mask:
        lab,n=ndi.label(a,NEIGHBORS)
        if not n:out.append(a.copy());continue
        sizes=np.bincount(lab.ravel());sizes[0]=0
        # scipy's raster-order labels fix ties at the first top-left component.
        out.append(ndi.binary_fill_holes(lab==int(np.argmax(sizes))))
    return np.stack(out)

@dataclass
class Candidate:
    channel:int
    direction:int
    indices:np.ndarray
    values:torch.Tensor
    parent:dict
    threshold:float|None
    matches:list
    semantic:float

    def record(self):
        return dict(channel=self.channel,direction=self.direction,support=self.indices.tolist(),
                    parent=self.parent,threshold=self.threshold,matches=self.matches,
                    semantic=self.semantic if math.isfinite(self.semantic) else None,pixels=len(self.indices))

class Tracker:
    def __init__(self,b,editable,reliable,q):
        self.b=b;self.mask=hard(b);self.editable=editable;self.reliable=reliable
        self.q=None if q is None else q.detach().cpu().numpy()[0]
        self.prob=b.detach().float().sigmoid().cpu().numpy()[0]
        self.levels=[]
        for tau in LEVELS:
            masks=self.prob>=tau
            self.levels.append([ndi.label(a,NEIGHBORS)[0] for a in masks])

    def corresponding(self,parent,channel):
        nparent=int(parent.sum());matches=[]
        if not nparent:return matches
        for tau,labs in zip(LEVELS,self.levels):
            lab=labs[channel];ids,intersection=np.unique(lab[parent],return_counts=True)
            sizes=np.bincount(lab.ravel());options=[]
            for i,n in zip(ids,intersection):
                if i:
                    iou=float(n/(nparent+sizes[i]-n));options.append((iou,int(i)))
            if options:
                iou,i=max(options,key=lambda x:(x[0],-x[1]))
                if iou>=.5:matches.append(dict(threshold=tau,component=i,IoU=iou))
        return matches

    def candidates(self,z,tau=None):
        new=hard(z);out=[]
        for c in range(2):
            parents={-1:ndi.label(self.mask[c],NEIGHBORS)[0],1:ndi.label(new[c],NEIGHBORS)[0]}
            for direction in (-1,1):
                change=(new[c]!=self.mask[c])&self.editable[c]&(new[c]==(direction==1))
                lab,n=ndi.label(change,NEIGHBORS)
                for k in range(1,n+1):
                    support=lab==k;indices=np.flatnonzero(support)
                    pl=parents[direction];ids,counts=np.unique(pl[support],return_counts=True)
                    options=[(int(count),int(i)) for i,count in zip(ids,counts) if i]
                    pid=max(options,key=lambda x:(x[0],-x[1]))[1] if options else 0
                    parent=pl==pid if pid else np.zeros_like(support)
                    matches=self.corresponding(parent,c)
                    semantic=float(np.mean(direction*(2*self.q[c][support]-1))) if self.q is not None else -math.inf
                    values=z[0,c].flatten()[torch.from_numpy(indices)].detach().clone()
                    out.append(Candidate(c,direction,indices,values,
                        dict(component=pid,pixels=int(parent.sum()),grid=list(support.shape),
                             origin='proposed_foreground' if direction==1 else 'baseline_foreground'),tau,matches,semantic))
        return out

    def select(self,candidates,kind='VERIFY',simple_reference=None):
        out=self.b.clone();mask=self.mask.copy();occupied=np.zeros_like(mask);accepted=0
        rejects={};selected=[]
        order=sorted(candidates,key=lambda x:(-x.semantic,-len(x.matches),x.channel,int(x.indices[0]),-x.direction, x.threshold if x.threshold is not None else .5))
        for candidate in order:
            c=candidate.channel;idx=candidate.indices;reason=None
            if occupied[c].ravel()[idx].any():reason='conflict'
            elif not self.editable[c].ravel()[idx].all():reason='outside_band'
            elif self.reliable[c].ravel()[idx].any():reason='protected_seed'
            elif kind=='VERIFY' and (not math.isfinite(candidate.semantic) or candidate.semantic<.05):reason='semantic'
            elif kind=='VERIFY' and len(candidate.matches)<3:reason='stability'
            elif kind=='SIMPLE' and not (simple_reference[c].ravel()[idx]==(candidate.direction==1)).all():reason='simple_morphology'
            if reason is None:
                trial=mask.copy();trial[c].ravel()[idx]=candidate.direction==1
                if np.any(trial!=self.mask,axis=0).sum()>.02*trial.shape[-1]*trial.shape[-2]:reason='edit_budget'
                elif containment(trial)>containment(mask):reason='containment'
                elif kind=='VERIFY' and any(a>b for nt,ot in zip(topology(trial),topology(mask)) for a,b in zip(nt,ot)):reason='fragments_or_holes'
            if reason is not None:rejects[reason]=rejects.get(reason,0)+1;continue
            mask=trial;out[0,c].flatten()[torch.from_numpy(idx)]=candidate.values
            occupied[c].ravel()[idx]=True;accepted+=1;selected.append(candidate.record())
        protected=torch.from_numpy(~occupied).unsqueeze(0)
        if not torch.equal(out[protected],self.b[protected]):raise ValueError('unselected raw logits changed')
        return out,dict(candidates=len(candidates),accepted=accepted,rejections=rejects,
                       edited_pixels=int(np.any(mask!=self.mask,axis=0).sum()),selected=selected)

    def structural(self):
        candidates=[]
        for tau in LEVELS:
            if tau==.5:continue
            shifted=torch.where(torch.from_numpy(self.editable).unsqueeze(0),self.b-math.log(tau/(1-tau)),self.b)
            candidates.extend(self.candidates(shifted,tau))
        out,record=self.select(candidates);record['candidate_records']=[c.record() for c in candidates]
        return out,record
