"""B1 C host with only the consistency target/loss changed."""
import torch
import torch.nn.functional as F
from .b1_host import Host as CHost,official,finite
from .b2_interval import targets,interval_kl,diagnostics


class Host(CHost):
    def __init__(self,arm,state=None,device='cpu',model=None):
        if arm not in ('zero','U','S','I'):raise ValueError('B2 arm')
        super().__init__('C',state,device,model)
        self.variant=arm;self.opt.cal_consis_loss=self.consistency

    def consistency(self,data):
        api=official();x=torch.from_numpy(data['data']).float().to(self.device)
        weak=api.Rotate_and_Flip();predictions=[]
        with torch.no_grad():
            pred,_=self.opt.model(x);predictions.append(pred.detach().cpu())
            for factor in (0,1,2,3,4):
                pred,_=self.opt.model(weak(x,factor));pred=weak.inverse(pred,factor);predictions.append(pred.detach().cpu())
            probabilities=torch.stack(predictions).sigmoid()
            q,s,r,lo,hi=targets(probabilities,self.variant,self.steps+1)
        self.base.zero_grad()
        augmented=api.augmentation_strong_style(data)
        augmented=api.normalize_image_to_0_1(torch.from_numpy(augmented).float().to(self.device))
        z,_=self.opt.model(augmented)
        loss,v,active,minimum=interval_kl(z,lo.to(self.device),hi.to(self.device))
        finite(loss)
        self.last=dict(interval_loss=float(loss.detach()),point_bce=float(F.binary_cross_entropy_with_logits(z.detach(),q.to(self.device))),minimum_raw_kl=float(minimum),channels=diagnostics(q,s,r,lo,hi,active))
        loss.backward()
        return loss

    def normalized_step(self,x):
        before=[p.detach().clone() for p in self.params]
        logits,meta=super().normalized_step(x)
        grad=sum(p.grad.detach().double().square().sum() for p in self.params).sqrt()
        update=sum((p.detach().double()-v.double()).square().sum() for p,v in zip(self.params,before)).sqrt()
        meta['diagnostics'].update(bn_gradient_l2=float(grad),adam_update_l2=float(update),zero_gradient=bool(grad==0))
        return logits,meta
