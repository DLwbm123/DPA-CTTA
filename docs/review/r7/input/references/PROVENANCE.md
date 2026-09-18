# 文献来源与迁移边界
核对日期：2026-09-18。三组均为本项目新提案，不是原论文官方复现；论文录用不等于迁移后会有效。

## RP-GSSM — ICML 2026
Maximum-Likelihood Learning of Latent Dynamics Without Reconstruction

原始来源：https://discovery.ucl.ac.uk/id/eprint/10226242/

迁移边界：A borrows recognition-to-latent evidence and causal filtering; not the paper's exact recognition-parametrized normalized maximum-likelihood model.

## MC-TTDG — ICLR 2026
Test-time Domain Generalization for Image Super-resolution

原始来源：https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad9da32791fc6772868bc3eaf6f66c59-Abstract-Conference.html

迁移边界：A soft codebooks are task adaptations; not pixel nearest-neighbor transfer or voting-based original method.

## Low-rank Laplace — AISTATS 2026
Low Rank Based Subspace Inference for the Laplace Approximation of Bayesian Neural Networks

原始来源：https://proceedings.mlr.press/v300/faller26a.html

迁移边界：A uses predictive-covariance-inspired subspace in a pooled-logit GGN surrogate; no full-network target posterior guarantee.

## WARP — ICLR 2026
Weight-Space Linear Recurrent Neural Networks

原始来源：https://proceedings.iclr.cc/paper_files/paper/2026/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html

迁移边界：B uses bounded recurrence on low-rank adapter coordinates and appearance differences, not arbitrary root-network weight recurrence on raw input differences.

## Sparse coding — UAI 2026
Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalization

原始来源：https://proceedings.mlr.press/v337/barin-pacela26a.html

迁移边界：B uses finite5step elastic-net correction; no compressed-sensing recovery theorem assumed for medical features.

## Bridge — CVPR 2026
Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization

原始来源：https://openaccess.thecvf.com/content/CVPR2026/html/Hong_Bridge_Basis-Driven_Causal_Inference_Marries_VFMs_for_Domain_Generalization_CVPR_2026_paper.html

迁移边界：C uses query-conditioned low-rank bases; no proven front-door adjustment, VFM transplantation or causal anatomy disentanglement.

## GUIDE — UAI 2026
Gradual Uncertainty Refinement via Noise-Driven Curriculum: A Post-Hoc Meta-Model for Robust Uncertainty Quantification

原始来源：https://proceedings.mlr.press/v337/barker26a.html

迁移边界：C uses residual-variance calibration with a source noise curriculum, not the original evidential classification head or its reported guarantees.

## 不作出的声明
- 不宣称拼接已构成TMI创新；需要相邻工作比较与证据。
- 不将 source proxy oracle 称为真实域状态，不将源码收缩界称为分割准确性或抗遗忘保证。
- 不称A为完整RP-GSSM实现，不称C为前门因果解耦，不称校准方差为已分离的epistemic/aleatoric uncertainty。
- 不宣称三组是仅改C的公平消融。它们有额外源端准备，故必须有同预算的独立STATIC源端训练控制与C0。
- 新设计不继承R6的否定/肯定结论，R6-D也没有证明某一种历史机制。
