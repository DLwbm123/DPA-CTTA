"""P1 current-image anchor; native student code plus one loss at its first forward."""
import ast
import copy
import math
import torch
import torch.nn.functional as F
from .hosts.vptta import VPTTAHost,model_input_from_pixels
from .integrations.ctta_suite import checkout_root,native_imports


def check_probability(p):
    if not torch.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid probability')


def anchor_kl(logits,q0):
    if logits.shape!=q0.shape or not torch.isfinite(logits).all():raise ValueError('anchor shape/nonfinite')
    q=q0.detach();check_probability(q)
    bce=F.binary_cross_entropy_with_logits(logits,q)
    entropy=-(torch.xlogy(q,q)+torch.xlogy(1-q,1-q)).mean()
    return bce-entropy,bce,entropy


def ensemble(q,p):
    if q.shape!=p.shape:raise ValueError('ensemble shape')
    check_probability(q);check_probability(p)
    return .5*q.detach()+.5*p.detach()


def anchor_step(path,adabn):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VPTTA')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loop=next(n for n in run.body if isinstance(n,ast.For));texts=[ast.unparse(n) for n in loop.body]
    body=copy.deepcopy(loop.body[texts.index('self.model.eval()'):next(i for i,t in enumerate(texts) if t.startswith('self.memory_bank.push('))+1])
    loss_hits=0;forward_hits=0
    class Change(ast.NodeTransformer):
        def visit_Assign(self,n):
            nonlocal loss_hits,forward_hits
            if ast.unparse(n.targets[0])=='loss':
                if ast.unparse(n.value)!='bn_loss / times':raise ValueError('native loss drift')
                loss_hits+=1;n.value=ast.parse('self._anchor_loss(bn_loss / times, pre_output)',mode='eval').body
            elif ast.unparse(n)=='_ = self.model(prompt_x)':
                forward_hits+=1;n.targets=[ast.Name('pre_output',ast.Store())]
            return n
        def visit_Expr(self,n):
            nonlocal forward_hits
            if ast.unparse(n)=='self.model(prompt_x)':
                forward_hits+=1;return ast.Assign([ast.Name('pre_output',ast.Store())],n.value)
            return n
    body=[Change().visit(n) for n in body]
    if (loss_hits,forward_hits)!=(1,1):raise ValueError('native first-forward structure drift')
    if any(isinstance(n,ast.Name) and n.id in {'y','data','path'} for stmt in body for n in ast.walk(stmt)):raise ValueError('label reference')
    fn=ast.parse('def step(self,x):\n pass').body[0];fn.body=body+[ast.Return(ast.Name('pred_logit',ast.Load()))]
    ns={'torch':torch,'AdaBN':adabn};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns)
    return ns['step']


class SourceAnchorHost(VPTTAHost):
    def __init__(self,task,source_state,device='cpu',weight=.1):
        if not math.isfinite(weight) or weight<0:raise ValueError('anchor weight')
        super().__init__(task,source_state=source_state,device=device)
        self.anchor_weight=weight;self.q0=None;self.last_anchor=None
        with native_imports(checkout_root()[0],task) as directory:self._sa_step=anchor_step(directory/'vptta.py',self.adabn)
    def _anchor_loss(self,host,output):
        logits=output[0] if isinstance(output,(tuple,list)) else output
        kl,bce,entropy=anchor_kl(logits,self.q0)
        self.last_anchor=dict(kl=float(kl.detach()),bce=float(bce.detach()),entropy=float(entropy.detach()))
        return host if self.anchor_weight==0 else host+self.anchor_weight*kl
    def step(self,pixel_rgb,q0):
        if pixel_rgb.requires_grad or len(pixel_rgb)!=1:raise ValueError('one current image without labels')
        self.q0=q0.detach()
        try:return self._sa_step(self,model_input_from_pixels(pixel_rgb,self.task).to(self.device))
        finally:self.q0=None
