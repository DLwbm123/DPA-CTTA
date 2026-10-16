# R52 five-method unified development comparison

Authorized after the five-method comparison plan on 2026-10-16. No new datasets, tuning, retries, seed expansion or automatic follow-up.

## Frozen matrix and reuse

C, CW_LSO, GraTa (G), VPTTA, MGIPT; seeds 20260907, 17011, 29009; orders 0 and 1. Shared registered RIM_ONE_r3 ResUNet34 checkpoint. Every stream contains 1951 current-image arrivals at 512x512, batch 1; post-update predictions; method state persists across domain boundaries and resets between trajectories. Order 0 is REFUGE, ORIGA, REFUGE_Valid, Drishti_GS; order 1 reverses domain blocks while preserving within-domain order.

Reuse 12 complete R50 C/CW_LSO and six complete R20 G trajectories only after checking checkpoint registration, identical manifests and split assignments, method/seed identity, completion and retirement receipts, and paired per-content hard counts. Environment provenance is retained; historical runtimes cannot establish a controlled speed advantage. Old P2 VPTTA is historical reference only because scoring occurred after each image rather than after retirement of the online workers. Run six VPTTA and six MGIPT full streams: 23412 new formal arrivals, plus 12 profile arrivals per method. The final matrix has 30 trajectories and 58530 scored rows.

VPTTA reuses the existing native AST image-step adapter, learning rate .05, Adam betas (.9,.99), one step, memory size 40, neighbor 16, prompt alpha .01 and native BN warm-up. MGIPT reuses the prior paper_equation implementation: prompt sizes 3/5/7, instance sizes 1/3/5, seven maximum steps per instance scale with native consistency selection, global one step per scale, Adam reset per prompt/sample, learning rate .05, teacher EMA .1 old/.9 new and confidence-weighted four-logit fusion. Preserve these native costs; do not equalize update counts. MGIPT is explicitly named an equation implementation, not exact released_legacy reproduction. Both strict-load the shared checkpoint; no source images or target labels are available to online workers.

## Outcomes and gates

Primary scoring subset is remaining_dev (1695 contents); macro averages equal-weight domains and OD/OC, then equal-weight three seeds and two orders. SEARCH (1017), historically exposed SEALED_REVIEW (678), and all 1951 are secondary. All are development evidence, not independent validation. Hard threshold >=.5; Dice is 2TP/(prediction+GT), both empty =1; retain existing mask decoding and independent CPU scorer.

Prespecified contrasts are CW_LSO minus C, G, VPTTA, MGIPT. Report all four, all domain/channel/order cells, ordinary paired 95% content-bootstrap intervals and 98.75% Bonferroni intervals (four two-sided comparisons, 20000 resamples, seed 20261016). Same content's seeds/orders move together, stratified by domain. Patient dependence remains unknown. A contrast passes only with delta >=.1pp, both orders positive, >=5/6 positive trajectories, nonnegative imageweighted delta, worst seed-mean domain/channel/order cell >=-2pp and adjusted CI lower >0. All-four superiority requires all four; incomplete matrix is NOT_EVALUABLE.

## Execution and resources

Use existing detached scheduler, current-image filesystem boundary, GPU admission, resource ledger, durable journal and CPU retirement barrier. Profiles are fresh disposable states and never continue into formal streams. Mechanical tests and profile admission must pass before formal runs. GPUs 4/5/6/7, at most four workers, native precision and existing Python3.10/Torch2.6/CUDA12.4 environment. Cap 48 GPU-worker hours including profiles/initialization/failures, 22h online, 24h total wall, 32GiB private output. If the fixed matrix does not fit, stop without pruning, retuning or retry. Persist all failure receipts. No recovery is authorized.

After all new online workers retire, score on CPU, combine audited historical scalars and produce the fixed report. Record forwards, backwards, optimizer updates, seconds per image and profiled peak memory; historical costs are descriptive only. No automatic monitoring or next experiment. On confirmed completion, publish own source, protocol and anonymous aggregates to the corresponding GitHub repository through the configured proxy; exclude images, masks, individual identities, predictions, weights and third-party source.
