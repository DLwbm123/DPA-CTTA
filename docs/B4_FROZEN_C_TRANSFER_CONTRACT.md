# B4 frozen transfer contract

This task adds only current-statistics zero-update C0 on the existing 1,951 Fundus and 1,802 Polyp groups, and two complete Polyp C trajectories. Existing B1/B2/B3/P2 implementation and assets remain frozen. All data are exposed development data, not independent validation.

C0 uses source affine weights, standard BN current-image statistics and the same non-BN training configuration as C. It has no optimizer, gradients, augmentation or evolving RNG/state. Canonical scalar records are mapped by identity on CPU: 3,753 real records, 11,408 references across orders. Mapping is not additional GPU work or independent repetition.

Polyp C uses the registered BKAI checkpoint and standard-BN PraNet. Its final lateral_map_2 Tensor retains batch/channel dimensions. Actual enumeration yields 153 participating BN layers and 66,218 trainable affine scalars. Source convolution/head parameters remain fixed. Each trajectory starts at source with empty Adam and seed 20260907; no intra-stream reset.

| Interface | Frozen behavior |
|---|---|
| Original and weak inputs | RGB [0,1], then original CPU ImageNet channel normalization exactly once |
| Strong input | Independent RGB numpy copy; official GraTa style; official min-max in CPU RGB; then CPU ImageNet normalization once |
| Target | Six separate batch=1 forwards, inverse geometry, detached CPU sigmoid then mean |
| Update | Single-channel mean BCEWithLogits; Adam 1e-4, (0.9,0.999), eps 1e-8, weight decay 0 |
| Output | Eighth forward on unaugmented original image after one backward/Adam |
| Fundus bridge | Calls unchanged B1 C path, including its existing numerical/RNG behavior |

The Polyp preprocessing is an explicit task interface transfer, not byte-identical Fundus preprocessing or an official GraTa Polyp reproduction. No original algorithm claim is made.

Five new CPU checks cover interface arithmetic, normalization, independent reference/state parity, C0 state/identity, and scalar closeout. Old tests are reused historically, not added to the new test count. The single registered-weight GPU smoke uses a separately recomposed Polyp reference and unchanged Fundus reference, two images each; C0 forward/reverse and source-eval adapter checks complete exactly 76 forwards, 8 backwards and 8 Adam calls. Strict deterministic smoke scope is restored; formal work runs in a fresh process using the old backend. First-C versus C0 alignment reuses captured smoke output, with no extra forward.

Formal budget: 7,357 scalar records, 32,585 forwards, 3,604 backwards/Adam, zero perturb/restore/source/DD training. With smoke: 32,661 forwards, 3,612 backwards/Adam. Execution is sequential on one UUID-bound GPU from user-authorized 3-7, chosen by available memory. Six active hours and 256 MiB private output are hard caps. An engineering/resource failure preserves its prefix and stops without automatic retry. Predictions precede evaluator mask access; raw maps and adapted checkpoints are not persisted.

CPU closeout checks identities, coverage, physical/reference accounting and Dice reconstructed from pixel counts. ASSD statistics reuse stored scalars on shared valid cohorts. Report both Polyp orders and all Fundus orders, domains, channels and existing subsets; domain-equal task means, paired tails, undefined/empty/full counts and measured costs. Source/current-statistics differences are not a unique causal decomposition. No performance gate or automatic next experiment is authorized. B2/B3 interval experiments remain closed.
