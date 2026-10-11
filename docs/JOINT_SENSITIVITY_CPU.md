# CW_LSO joint acquisition/anatomical sensitivity: minimal implementation

Status: CPU qualification only. No GPU profile, experiment registration, launch,
data evaluation, or claim of segmentation improvement. Existing R44 is unaffected.

## Scope and interface

`dpa_ctta.joint_sensitivity.method.Host` extends the retained CW_LSO Host.
`candidates()` returns unchanged C, W, CW_LSO and one CW_LSO_JS implementation
configuration. This factory is not an experiment queue. There is no runner,
campaign, seed selection or automatic monitoring for this module.

`joint_strength=0` delegates directly to the original Host before constructing
probes or changing RNG/state. The enabled setting is 1. Existing CW_LSO settings
are required: host C, W variance temperature 0.1 and boundary boost 2, LSO 0.1,
Adam learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8, weight decay 0.
Only the existing BN affine parameters adapt. No new model or persistent bank.

## Evidence and four vector-Jacobian products

Let q and V be the detached mean and variance of the original six aligned weak
sigmoid predictions. OD/OC remain independent nested sigmoid channels.
F(z) is sigmoid logits, 4x average pooling, then replicate-padded 3x3 smoothing.
Reuse the existing Sobel structure tensor and doubled-angle orientation operator.

On the pooled grid, B is max-pooled support of the existing radius-3 boundary of
q>=0.5; N_c=sum B_c. A pooled cell is nesting-valid only when none of its 4x4
pixels satisfies q_OC>=0.5 and q_OD<0.5. With t the existing structure-tensor
directional magnitude of pooled q, evidence is

    E_c = B_c * max(0,1-4*avg_pool(V_c,4))
          * t_c/max(max_spatial(t_c),1e-6) * nesting_valid * [t_c>2e-6].

E and all reference values are detached. The normalization uses boundary support
N, not sum E; unreliable evidence therefore remains weak. No ground-truth labels,
domain metadata, clinical shape atlas, or new pseudo-label edit is introduced.

Original logits z0 provide the reference field F0 and orientation o0. Sobel norm
of F0 is clamped below at 1e-3. Using fixed independent Rademacher signs for the
displacement and two orientation components, for each channel:

    d(z) = 4*(F(z)-stopgrad(F0))/max(||Sobel(F0)||,1e-3)
    a(z) = 4*(o(z)-stopgrad(o0))
    s_c(z) = sum sqrt(E_c/(N_c+1e-8)) * (omega_d*d + omega_a dot a).

The signs use a private CPU generator seeded 0, leaving stream RNG unchanged.
The original reference orientation is computed with tracing enabled before
detachment, matching the sensor's arithmetic at z0. Although s(z0)=0, its VJP
is nonzero where structure evidence supports a response.

Two fixed acquisition probes operate on pixels before native Fundus min-max
normalization, preserving exact 0/1 endpoints after logit clamping at 1e-6:

1. Contrast: sigmoid(1.10*logit(x)).
2. Channel balance: sigmoid(logit(x)+(0.05,0,-0.05)).

At the same pre-update parameters compute four separate `autograd.grad` calls:

    b_OD = grad_theta s_OD(z0); b_OC = grad_theta s_OC(z0)
    b_sum = (b_OD+b_OC)/sqrt(2)
    a_k = grad_theta[(s_OD(z_k)+s_OC(z_k))/sqrt(2)] - b_sum, k=1,2.

Reuse the original forward graph for the first two VJPs; process each photo
graph sequentially. These VJPs do not accumulate optimizer `.grad`. Photo passes
save/restore RNG, so native strong augmentation retains its original stream.

## Actual Adam proposal filter

The unchanged CW_LSO loss performs one backward and one Adam step. Let
Delta_A=theta_Adam-theta_before and H=diag(sqrt(v_hat)+1e-8), using the just-updated
bias-corrected second moment. Adam moments and its step counter remain intact.

    X = H^(-1/2) [a_1,a_2]; Y = H^(-1/2) [b_OD,b_OC]
    Q = orth(X)
    A = (Q^T X)(Q^T X)^T / ||X||_F^2
    S = (Q^T Y)(Q^T Y)^T / ||Y||_F^2
    R = (A+0.01 I)^(-1/2)
    T = R S R; K = (I+joint_strength*T)^(-1)
    w_A = H^(1/2) Delta_A
    Delta = H^(-1/2) [w_A + Q(K-I)Q^T w_A]
    theta_new = theta_before + Delta.

Orthogonalization uses the 2x2 Gram eigensystem. Retain eigenvalues strictly
above max(1e-12,1e-6*largest_eigenvalue). Rank zero keeps the actual Adam proposal
without subtract/add roundtrips. When ||Y||=0, S=0. Invalid/nonfinite vectors or
nonpositive metrics are rejected. Small matrix arithmetic uses float64; the
final proposal uses parameter dtype. No dense parameter-by-parameter matrix.

The H-whitened component outside span(X) is unchanged, and K contracts the
component within it. Thus attenuation depends jointly on acquisition and
anatomical sensitivity; this is not a general gradient norm cap or a new loss.
It is a local projected sensitivity estimate, not a complete Jacobian or proof
of anatomical preservation. Weak/incorrect predictions can produce weak evidence.

## Calls and cost

| Operation per image | CW_LSO | Enabled joint sensitivity |
|---|---:|---:|
| Total model forwards | 8 | 10 |
| Forwards with gradients enabled | 1 | 4 |
| Backward/VJP traversals | 1 | 5 |
| Adam steps | 1 | 1 |

Enabled forward order: original traced weak view, five no-grad geometric weak
views, two traced photo probes, traced native strong view, final no-grad output.
The original forward is reused, not repeated. Four auxiliary VJPs plus one base
loss backward give five traversals. Zero evidence does not change this schedule.

Baseline cost is 7*F_no_grad+F_grad+B; enabled cost is
6*F_no_grad+4*F_grad+5*B plus field and O(parameter_count*2) algebra. If all
forward costs are approximately equal and B/F is 1.5-2.5, the estimated ratio is
1.84-2.14x. This is an analytical estimate, not measured GPU latency or memory.

## CPU reproduction

Use a clean pinned GraTa dependency at commit
33ae20d664f305af34739ec54a5bec7da53ffa0b and batchgenerators 0.25.2. A Tiny BN
segmentation fixture and generated inputs exercise the native augmentation path;
there is no checkpoint or dataset loading. Set DPA_GRATA_ROOT to that dependency,
PYTHONPATH to this checkout's src, CUDA_VISIBLE_DEVICES to the empty string, and
run Python unittest modules:

    dpa_ctta.joint_sensitivity.tests
    dpa_ctta.r36_incremental_modules.tests
    dpa_ctta.r37_cw_orientation.tests
    dpa_ctta.r44_spectral_entropy.tests

See reports/joint-sensitivity-cpu/REPORT.md for the actual qualification receipt.
