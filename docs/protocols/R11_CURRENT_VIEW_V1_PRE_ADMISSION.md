# R11_CURRENT_VIEW_V1 — frozen current-image view pilot

Question: can fixed current-image prediction averaging improve frozen current-statistics C0, without the source-trained B carrier or RL actor that harmed the previous paired short-stream result?

This is an established test-time augmentation baseline, **not a new CTTA method**. Medical precedent: [Wang et al., test-time augmentation uncertainty](https://arxiv.org/abs/1807.07356). Negative or small effects are retained. No conclusion about all RL follows.

## Fixed conditions and fair comparison

- CV_H2: identity and horizontal flip; CV_FLIP4: identity, horizontal, vertical, and both flips. No rotation, interpolation, resize, view search, trained model, memory, B or actor.
- Reuse exact frozen checkpoint, 512×512 preprocessing, current-statistics evaluation BatchNorm, independent OD/OC sigmoid channels and original scorer. Inverse-align each view; arithmetic mean float32 sigmoid probabilities; clamp [1e-6,1-1e-6], logit to original sealed scorer API. Single-view wrapper returns original logits exactly. All parameters and BN buffers immutable.
- Source: original seed 20260924, 16 validation episodes (4 per actual mode), each 32 visits, both fixed conditions; 1024 new source visits, 3072 backbone forwards. Reuse paired original C0 source rows; no checkpoint or view selection from source scores.
- Target: original two registered orders, same 1024 contents/order, 888 remaining_dev principal visits/order, all four domains. Four new trajectories (4096 visits, 3552 principal rows), 12288 backbone forwards. Exact sealed C0 and GraTa results reused with matching manifest/scalar seals. No old target reruns or new seeds.
- Image-only online stage; independent CPU labels/scorer only after sealed trajectory. Target reports embargoed until whole fixed matrix terminal. Temporary probabilities retired only after sealed scoring. Retain all scalar/trace/failure evidence privately.

## Qualification, budget and execution

32 prespecified source visits: episode 0/16/32/48, visits 0/4/8/12/16/20/24/28. Identity wrapper must exactly match unchanged C0 logits/probabilities; verify fixed BN/parameter stamps, finite two-channel output, inverse-flip alignment and snapshot offsets. CPU executable checks cover alignment, probability averaging (not logit averaging), immutable view identity, snapshot and lock rejection. Physical model meter must count exactly 2F or 4F per logical arrival, zero backward/optimizer/VJP.

T0 2026-10-01 22:22:56.911 Beijing. Preserve prior original package cost 29712.181510498747 seconds. New cap 5400 seconds including preparation and closure, below original 12-hour cumulative cap. Preflight cutoff 22:52:56; normal compute cutoff 23:22:56; hard compute cutoff 23:37:56; absolute process exit 23:52:56. Source stage cap 900s; target stage cap 1800s. Admission requires measured source projection ≤900s, target ≤1800s and complete matrix within remaining normal deadline, factor 1.3 plus prior longest independent scorer and read/fsync/loading margin. No scope reduction after seeing target scores.

GPU5 only, subject to actual VRAM check, neutral process argv. Stable per-experiment exclusion; one infrastructure-only equivalent target recovery, ≤900s reserve, shared hard deadline. No science retry, source engineering retry, seed/arm expansion or automatic follow-on. Partial/incomplete execution is missing evidence, never a negative finding. Final 900s are closing reserve; process-group watchdog bounds absolute exit. Background run, one brief startup check, no ongoing monitor.

## Prespecified interpretation and delivery

Report each condition/order, domain-equal OD/OC/macro Dice, pooled descriptive mean, every domain/channel, paired distributions, last-quarter descriptions and worst domain/channel degradation. Priority scale: mean gain vs C0 ≥0.5 percentage point with both orders positive. This is a descriptive development signal, not significance; preserve smaller positive and negative effects. GraTa gap remains explicit. Two orders share contents and deterministic policies; they are correlated, not independent replication. Patient dependence unknown, source simulator is development evidence. No broad/clinical or novel-method claim.

After actual completion, collect anonymous source/results/receipts and report, commit and push source-only public results to the project GitHub using required proxy, then verify remote commit and anonymous access. Private content IDs, patient data, raw images/masks, probabilities, host paths and credentials excluded. No extra experiment or monitor authorized.
