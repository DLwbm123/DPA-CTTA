# ENGRAM memory-module transfer: frozen one-round pilot

Registered 2026-10-18 before new target inference. User authorized one different ICLR module, without RDC, small controlled run, followed by analysis and GitHub delivery. No automatic tuning or successor.

## Question and fixed arms

Can associative recall recover useful parameter adjustments when an imaging domain returns with different images? Two arms, each source-initialized with empty Adam moments: **NATIVE** (retained CW_LSO) and **ENGRAM** (same host plus memory). No RDC objective, feature affinity, agent selection or rank loss is called. Historic RDC files remain in repository history only.

Independent implementation of ENGRAM equations (2)-(4), Algorithm 2, Table 10 from the inspected [ICLR 2027 submission](https://openreview.net/pdf?id=bkwlR0ND2g); not an accepted-paper or full reproduction claim.

## Memory mechanism

Store orthonormal parameter directions U, coefficient map M, previous unit key and bias-corrected growth accumulator. Source vector contains existing BN affine coordinates only. Before the host update, add `U M (key_current - key_previous)` in place to the current BN parameters. This is the paper's Appendix A difference-of-reads implementation of full residual carry. Do not restore/reset Adam or alter its loss, learning rate, prediction timing, W or LSO.

After the complete host step, let y be the BN vector minus source. Write `M += 0.1 * outer(U.T y - M key, key)`. The just-completed model is untouched. Update the running out-of-basis adjustment with averaging factor0.9. After at least10 arrivals since initialization/growth, append a twice-reorthogonalized direction if its bias-corrected norm exceeds0.25 times norm(y), numerical norm exceeds1e-12, and rank<min(64,parameter dimension). Append a zero row to M so growth leaves existing reads unchanged. Reset growth accumulator/count only on append. Memory algebra uses float64; actual host parameters retain their original dtype. These are fixed paper defaults plus a declared numerical-zero threshold.

**Descriptor transfer difference:** reuse a frozen copy of the source segmenter's first convolution, rather than downloading/training a VGG encoder. Apply the same per-image model input normalization; take each channel's spatial variance (unbiased=False), floor1e-6, log, subtract the mean across channels and L2-normalize. Degenerate vectors use the first unit coordinate. No source images or labels fit descriptor statistics. One extra stem computation per current image, no extra complete segmentation forward. Source-standardized VGG log-variance descriptors in the paper are not reproduced. Thus a negative result cannot reject the full paper; key anatomy/style confounding remains possible.

## Small recurrence stream without image repetition

Reuse all192 exposed development contents from the prior pilot; no new images or scans. Partition each domain by its existing order0 list: floor(n/2) images in cycle0, remainder in cycle1, fixing membership for both orders. This gives95 first-cycle and97 second-cycle distinct images. Preserve within-domain order separately for each existing order. Each cycle follows the original domain order (order0 REFUGE, ORIGA, REFUGE_Valid, Drishti_GS; order1 reversed). Each image is encountered exactly once per trajectory. Domain/cycle metadata is used only in offline stream preparation and scoring; online rows contain image path/identity/size only. No boundary signal is passed to memory.

One seed20260907, two orders, two arms,192 arrivals: **4 trajectories,768 predictions**. Fresh NATIVE controls on these streams; do not reuse old scores or compare their absolute Dice with the RDC streams. Each arm retains state from cycle0 into cycle1.

## Checks and decision

CPU self-checks: constant-key corrections exactly zero after growth, write preserves current model, orthonormal basis, explicit compensated read equals difference-of-reads, nonzero correction for changed keys after learning, state roundtrip, unit-key write error contraction by0.9. Import checks include the existing protocol JSON dependencies.

Real-model qualification: repeat complete state and output for both arms on the first image of each order; NATIVE exactly matches the retained host. Additionally, run12 fixed-key steps against the retained host on the same qualification image and require exact host state/output equality after every step and at least one growth event. This is engineering validation; duplicated qualification input is not included in scientific scoring. Frozen non-BN parameters and unchanged host8F/1B/1Adam are checked; ENGRAM separately executes one frozen stem per arrival. Memory writes must leave the completed BN vector exactly unchanged.

Primary: **cycle1 ENGRAM−NATIVE**, equal domains, OD/OC and orders. Require >=+0.2 Dice pp, paired content-stratified95% lower bound>0, both order effects>0, worst domain/order/channel cell>=−2pp, and overall192-image macro effect>=0. Bootstrap2000 draws with fixed seed20261016, repeated content/orders/channels coupled. Cycle0 and overall effects are descriptive. Record absolute arm scores, all overall cells, per-cycle effects, recall magnitude, key movement, rank/growth, host update magnitude and timing. No post-hoc primary switch or descriptor tuning.

GPU worker sees no labels; separate CPU scoring after successful GPU exit requires all4 receipts and768 predictions. Exposed development cohort, unknown patient linkage, one seed, one return per domain: not independent/general long-horizon validation.

## Budget and delivery

GPU cap1800 seconds including qualification; scorer cap300 seconds; output cap2GiB. Admit only if slowest qualified step times768 times1.5 plus consumed qualification fits the cap. Existing detached process-group controller enforces cap and preserves failed costs. One round only, no scientific retry. Normalized descriptors/memory settings/stream memberships are frozen before inference.

After completion, inspect coverage and retirement, analyze both benefits and negative cells, publish own code/protocol/anonymous aggregate results/costs/report through the mandated proxy. Never publish images, masks, checkpoints, per-image identifiers/predictions/scores, private paths or third-party PDFs.
