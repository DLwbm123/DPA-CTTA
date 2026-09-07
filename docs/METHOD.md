# DPA-CTTA v0 method

## Frozen manifold

The source segmenter parameters and batch-normalization state stay frozen. One state `z` in R^16 controls three discovered decoder/tail outputs:

```text
h_l(z) = h_l * exp(A_gamma_l z) + A_beta_l z
```

The implementation bypasses FiLM arithmetic at exactly zero, so `z=0` returns the original output tensor unchanged. FiLM is a mechanism, not an innovation claim.

## Descriptor and anchors

The online descriptor uses only the current image and the frozen source model's zero-state feature/logits. It combines opponent-color mean/std, three radial Fourier energies, a fixed projection of early-feature mean/std, and differentiable per-class area, centroid, second moments, and boundary energy. Fundus adds soft OC/OD area ratio and nesting violation; Polyp adds soft compactness.

Each anchor stores trainable synthetic image logits, synthetic mask logits, state center `m_k`, local map `B_k`, and diagonal precision raw values. Its descriptor is recomputed from `sigmoid(image_logits)`, `sigmoid(mask_logits)`, and a frozen-source feature; there is no free descriptor parameter.

## Association and proximal update

For scaled descriptors `q` and `q_k`:

```text
w_k = softmax(-||D(q-q_k)||^2 / tau)
mu_k(q) = m_k + B_k(q-q_k)
p = sum_k w_k p_k
b = sum_k w_k p_k * mu_k(q)
z_t = (b + lambda z_(t-1)) / (p + lambda)
```

Here `p_k = softplus(precision_raw_k) + mu`, with a machine-epsilon lower clamp on the softplus term to preserve strict numerical inequality. The state-history contraction bound is `lambda / (lambda + mu)`.

## Deferred experiments

M0 source-supervised oracle states, M1 full-source potential recovery, D0 anchor distillation, and all target evaluations are not results of this repository version. Their functions exist only to make the interfaces reviewable before data access.
