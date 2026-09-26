"""Independent 256-step calibration; deployment-aligned source evaluation."""
import torch
from ..r8_ba.calibration import Calibrator as LegacyCalibrator
from ..r8_ba.schedule import anchors,episode_roles,episode_styles
from ..r7_shared.source import simulate


class Calibrator(LegacyCalibrator):
    def step(self):
        if self.steps>=256:raise ValueError('R9 calibration exhausted')
        return super().step()

    def run(self):
        while self.steps<256:self.step()
        self.method.freeze();return self.method

    def snapshot(self):return dict(schema='R9_CAL_SNAPSHOT_V1',cal=super().snapshot())

    def restore(self,snapshot):
        if snapshot.get('schema')!='R9_CAL_SNAPSHOT_V1' or not 0<=snapshot['cal']['steps']<=256:
            raise ValueError('R9 calibration snapshot')
        super().restore(snapshot['cal'])


def dice(probability,label):
    if probability.shape!=label.shape or not torch.all((label==0)|(label==1)):
        raise ValueError('binary source label shape')
    dim=(0,2,3);p=probability.double();y=label.double();a=(p>=.5).double()
    den=a.sum(dim)+y.sum(dim)
    hard=torch.where(den>0,2*(a*y).sum(dim)/den.clamp_min(1),torch.ones_like(den))
    soft=(2*(p*y).sum(dim)+1e-6)/(p.sum(dim)+y.sum(dim)+1e-6)
    return hard.tolist(),soft.tolist()


@torch.no_grad()
def episode(segmenter,method,data,oracles,index,aligned=True,guard=lambda:None):
    oracles.validate(data)
    if oracles.fold!='val' or not 0<=index<64 or method.stage!='online' or oracles.amplitude!=segmenter.amplitude or method.amplitude!=segmenter.amplitude:raise ValueError('source validation binding')
    ids,mode,severity=episode_styles('val',index)
    roles,reused=episode_roles(data.folds['val'],'val',index,ids,oracles.support_pairs)
    bank=anchors('val');state=method.initial();hard=[];soft=[];proxy=[]
    for visit,(anchor,(sn,qn)) in enumerate(zip(ids,roles)):
        guard();query=data.get(qn,'val')
        q=simulate(query.image,bank[anchor],f'R8_VAL_QUERY|{index}|{visit}|{qn}')
        image=q if aligned else simulate(data.get(sn,'val').image,bank[anchor],f'R8_VAL_SUPPORT|{index}|{visit}|{sn}')
        _,raw,tokens=segmenter(image,observe=True)
        state,_=method.update(raw,tokens,state)
        p=segmenter(q,method.ambient(state)).sigmoid()
        h,s=dice(p,query.label);hard.append(h);soft.append(s)
        proxy.append(float((method.code(state)-method.project(oracles.values[:,anchor])).square().mean()))
    return dict(episode=index,curriculum=mode,severity=severity,visits=32,aligned=aligned,
                hard_Dice=sum(sum(r) for r in hard)/64,soft_Dice=sum(sum(r) for r in soft)/64,
                hard_OD=sum(r[0] for r in hard)/32,hard_OC=sum(r[1] for r in hard)/32,
                soft_OD=sum(r[0] for r in soft)/32,soft_OC=sum(r[1] for r in soft)/32,
                proxy_MSE=sum(proxy)/32,group_reuse=reused)
