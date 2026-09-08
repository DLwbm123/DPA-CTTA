"""Native std values/first derivative, stable analytic double backward.

PyTorch 2.2 std_backward divides by 2*std before masking std==0. That
unselected 0/0 poisons double backward; tiny positive std also underflows
its squared denominator. Use the factored analytic Hessian, no epsilon.
"""
import ast
import inspect
import textwrap
from types import MethodType
import torch


class SpatialStdGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gradient,result):
        ctx.save_for_backward(x,gradient,result)
        zero=result==0
        denominator=torch.where(zero,torch.ones_like(result),result*2)
        grad_var=(gradient/denominator).masked_fill(zero,0)
        # Exact native first-gradient order; total derivative supplied below.
        return (2.0/(x.shape[2]*x.shape[3]-1))*grad_var*(x-x.mean((2,3),keepdim=True))
    @staticmethod
    def backward(ctx,direction):
        x,gradient,result=ctx.saved_tensors
        zero=result==0
        scale=torch.where(zero,torch.ones_like(result),result)
        n=x.shape[2]*x.shape[3]-1
        normalized=(x-x.mean((2,3),keepdim=True))/scale
        projection=(normalized*direction).sum((2,3),keepdim=True)/n
        centered=direction-direction.mean((2,3),keepdim=True)
        hessian=(gradient*(centered-normalized*projection)/(n*scale)).masked_fill(zero,0)
        # Includes the complete dependence of std on x; do not count it again
        # through the saved result. No std squared/cubed float32 intermediate.
        return hessian,projection.masked_fill(zero,0),None


class FiniteSpatialStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        result=x.std((2,3),keepdim=True)
        ctx.save_for_backward(x,result)
        return result
    @staticmethod
    def backward(ctx,gradient):
        x,result=ctx.saved_tensors
        return SpatialStdGradient.apply(x,gradient,result)


def install_finite_std(model,adabn):
    """Only isolated offline AdaBN instances; pinned source and online host untouched."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(adabn.forward)));hits=0
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal hits
            if isinstance(node.func,ast.Attribute) and node.func.attr=='std':
                if ast.unparse(node)!="x.std((2, 3), keepdims=True)":raise ValueError('unexpected native std contract')
                hits+=1;return ast.copy_location(ast.parse('finite_spatial_std(x)',mode='eval').body,node)
            return self.generic_visit(node)
    tree=Replace().visit(tree)
    if hits!=1:raise ValueError('native AdaBN must contain exactly one spatial std')
    namespace=dict(adabn.forward.__globals__,finite_spatial_std=FiniteSpatialStd.apply)
    exec(compile(ast.fix_missing_locations(tree),'<M4-native-AdaBN-finite-std>', 'exec'),namespace)
    for module in model.modules():
        if isinstance(module,adabn):module.forward=MethodType(namespace['forward'],module)
