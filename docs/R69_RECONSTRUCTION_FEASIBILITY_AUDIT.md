# R69 preparation: observable reconstruction residual feasibility

**No compatible reconstruction/acquisition residual is established for the current Fundus path. No experiment registered or launched.** This is a bounded interface and primary-source audit, not an efficacy test or a statement that all reconstruction approaches fail.

## Actual interfaces

`integrations/ctta_suite.py:build_reference_model` builds the source-only reference model. The previously audited deployed class is `networks.ResUnet_TTA.ResUnet`; the local VPTTA OPTIC dependency source was inspected again. Its upsampling blocks end in a 32-channel feature map and `Conv2d(32, num_classes, 1)` segmentation head. Forward returns `(seg_output, sfs, head_input)`. For Fundus, the segmentation output has two channels. The existing decoder is therefore a segmentation decoder, not an RGB reconstruction decoder. Accessible skips/features are not a trained inverse from features to measured pixels.

The online interface in `r7_target_screen/runner.py:image_records` supplies image path, image identity and image size. `TargetReader` reads image pixels; `hosts/vptta.py:model_input_from_pixels` accepts three-channel resized [0,1] images and performs per-image min-max normalization. This inspected call chain supplies neither raw optical sensor measurements nor a calibrated acquisition operator mapping disc/cup segmentation to RGB. This is an interface claim, not an exhaustive inventory of every external dataset or server file. No remote scan or checkpoint loading was necessary.

## Two concrete primary-source mechanisms

1. [Test-Time Training with Masked Autoencoders](https://arxiv.org/pdf/2209.07522), sections 3–4: the method begins with an MAE-pretrained encoder and reconstruction decoder plus a separately trained task head. At test time it minimizes masked-pixel reconstruction loss and then predicts through the task head. Our checkpoint does not expose that trained reconstruction pathway. Adding a random RGB head or swapping in an MAE would not faithfully reuse this mechanism; source training/model replacement requires a separate scope decision.
2. [Test-Time Training Can Close the Natural Distribution Shift Performance Gap in Deep Learning Based Compressed Sensing](https://arxiv.org/html/2204.07204v2), sections 2 and 4: known measurement matrix A and measurements y support a data-consistency residual y-Af(A-dagger y). The network outputs an image; self-supervision is also included during training. Its validation uses held-out measured frequencies. Our two-channel segmentation output and processed RGB input do not instantiate these variables. Treating RGB pixels as y with an invented segmentation-to-image A is an unverified generative assumption, not measured data consistency.

These readings establish prerequisites, not Fundus effectiveness. The MAE HTML endpoint was unavailable; the primary PDF sections were accessible and inspected. MRI full-text HTML was inspected. Neither paper was reproduced.

## Excluded shortcuts and evidence correction

- Identity image reconstruction independent of segmentation parameters supplies no segmentation gradient. A differentiable segmentation-conditioned renderer would need its own justified observation model; it is not provided by naming a loss reconstruction.
- Inverse-transform consistency between predictions is self-prediction consistency already used by the native ensemble, not a new acquired observation.
- Pixel edges or within-mask color homogeneity are possible priors, but illumination/vessel edges and cup/disc ambiguity prevent treating them as observed anatomical truth. This is a design concern, not a measured rejection of every such prior.
- Earlier handoffs called R61 an image-energy rejection. The inspected [R61 evidence note](R61_EVIDENCE_DESIGN.md) documents feature-bank overlap and cautions about prompt-free image boundary cues. It does **not** report a completed RGB reconstruction experiment. Retain this distinction. R66/R67 supply the actual negative results for their specified candidate-derived feature-boundary signal only.

## Decision and continuation

Close the proposed drop-in reconstruction/acquisition residual under the existing input/model interface. No new decoder, source training, model download, raw measurement acquisition, target fitting or GPU job follows. The existing old-method research authorization and half-hour schedule remain active; new-method work remains in the separate designated conversation.

Next bounded task: inspect the primary TeSLA formulation and compare its teacher/augmentation feedback against native GraTa, R20 T, R61 and R62 before considering any implementation. The question is whether its automatic adversarial augmentation exposes a genuinely different task-relevant signal under this checkpoint, or merely requires new augmentation training/teacher machinery and repeats a closed family. Do not infer compatibility from published benchmark scores. If it is redundant or requires assets outside this scope, record the limitation without a token experiment. No performance claim or next-run promise is made.
