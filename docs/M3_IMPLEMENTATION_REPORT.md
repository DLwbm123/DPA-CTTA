# M3 conditioned proxy — implementation checkpoint

NOT_RUN at this implementation checkpoint. This is one finite user-authorized M3 experiment, not an independent reviewer signoff or evidence of effectiveness.

Start: M2 release `bc9d622d69d01bf4d1387de1c28cc4aa6f2e5983`; branch `experiment/m3-conditioned-proxy-v1`. External dependency remains `dbff0d985c6c95345d9fb78f5b1daef57b392564`. M1/M2 scientific files and configurations are unchanged.

## Fixed implementation

The independent `condition_proxy` follows the supplied unshifted conjugate-symmetric FFT window and zero-amplitude convention: rho=.5, beta=.01, epsilon=1e-8, backward norm, no padding, final clamp. It detaches the current RGB donor and retains the image-to-T gradient. Clipping and L2 statistics are detached scalar observations. No FFT cache or extra dependency is needed.

An M3 host subclass replaces only its temporary proxy input before the inherited native step; it retains the immutable original S and mask. The donor is the current raw RGB image. Current-image preprocessing, native prompt FFT, one Adam, source gradients, clone isolation and memory push remain inherited. Zero replay bypasses conditioning/clone construction through the original constructor; zero strength directly calls the old static path. All four scoring arms share this same class and configuration.

The offline subclass preserves the M2 D/O operations and native Adam. Its only scientific change is T(S, detached transformed query) before proxy preprocessing. The functional Adam returns are optionally captured for smoke before the outer image update; the paired actual-Adam checks reuse them. No second functional inner is silently added. Formal DD starts from original Real, not D2/O2.

## Asset and execution contract

The registration overlay references M2's frozen episode files and original M1 history/data. It never calls a selector or sampler. The two O2T files are first bound to the saved M2 final digests; subsequent stages reuse their metadata. Source and target labels/identities and tensors remain private. Existing UNKNOWN patient/video linkage is a limitation, not a new gate.

Training: Fundus D3/O3 then Polyp D3/O3, 600 episodes each, save 0/100/300/600. All four finals freeze before scoring R3/D3/O3/O2T. New scoring is 208 source + 896 target = 1,104. Old 1,932 records are reused for 3,036 combined records, not rerun. The CPU reconstruction verifies exact identity, order, channel, moment step and transform/update coverage before generating all nine predeclared paired comparisons and the report.

Smoke: 16 online / 4 outer / 2 functional inner, procedural queries/labels and history index 16. Strict deterministic scope is limited to paired static-rho0 and shared functional/actual checks; the latter necessarily includes generating the reused O3 functional result. Formal execution uses its separate original M2 environment. T input checks, CPU tests and evaluator-only calls do not add optimizer updates.

GPU scope is one physical GPU 4–7, prefer 7 if sufficient free memory; coexist without changing other processes. Six-hour active-stage and 2 GiB private-output caps. No automatic restart, score gate, parameter search or M4.

## Verification so far

31 selected local tests passed (1.199 seconds, exit 0), including eight M3 tests plus the relevant M1/M2/data/score tests. They cover symmetry/band boundaries, no-FFT identity, donor isolation, zero-spectrum gradient, finite meta-gradients, functional/actual Adam, immutable proxy/mask, new donor use, exact inherited math at identity, same algorithm for four arms, frozen-list reuse, score completeness and report generation. Full historical regressions are REUSED evidence and were not rerun. Deployment checks and the one GPU smoke are pending.

## Attribution and interpretation

This is an FDA-style partial amplitude mixing variant, not the full FDA pipeline or a novelty claim for proxy stylization. Prior work includes [Yang and Soatto, FDA, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html) and [Kang et al., Leveraging Proxy of Training Data for Test-Time Adaptation, ICML 2023](https://proceedings.mlr.press/v202/kang23a.html).

Judge O3 against D3, R3 and O2T as well as A/N; preserve domain cancellations and source degradation. D3-D2/O3-O2 include both training and deployment changes. There is no D2T arm. Single seed/order, previous target exposure, off-policy history, one-step adaptation and UNKNOWN patient linkage preclude independent-patient significance, SOTA or privacy claims. Phase retention and nonzero gradients do not prove medical semantics or utility.
