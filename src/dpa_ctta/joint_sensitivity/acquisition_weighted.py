"""C-background acquisition-weighted structural filter; no experiment entry point.

For native Adam displacement d and metric H=sqrt(v_hat)+eps, set
X=H^(-1/2) acquisition, Y=H^(-1/2) anatomy, and retain the R46 basis
Q=orth([X/||X||_F, Y/||Y||_F]) with its original Gram rank cutoff.
A=Q^T X X^T Q / ||X||_F^2; S=Q^T Y Y^T Q / ||Y||_F^2.
R=A+0.01 I; T=R^(1/2) S R^(1/2); K=(I+T)^(-1).
Submit H^(-1/2)[w+Q(K-I)Q^T w], w=H^(1/2)d.
The historical filter uses R^(-1/2); its default is preserved.

Fixed strength=1, rank<=4, unchanged Sensor, two probes and four VJPs.
Enabled: 10 forwards / 5 reverse traversals / 1 Adam step per image;
disabled: exact C path, 8 / 1 / 1. No W, LSO or descent constraint.
The base-loss gradient is reused for native and actual-write derivative
diagnostics only; neither derivatives nor labels control the update.

This soft weighting changes both direction selection and effective strength.
It is not an exact subspace intersection or evidence of anatomical protection.
CPU qualification uses generated fixtures, not real-model or Dice evaluation.
"""
from .factorial import Host as FactorialHost, candidates as factorial_candidates


def candidates():
    c = factorial_candidates()[0]
    return [c, dict(c, id='C_JS_ACQ_WEIGHTED', joint_basis='shared', joint_strength=1.,
                    joint_filter='acquisition_weighted')]


class Host(FactorialHost):
    conditions = staticmethod(candidates)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_weighted = self.config.get('joint_filter') == 'acquisition_weighted'
