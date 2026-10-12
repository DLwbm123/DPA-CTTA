# R44: causal spectral entropy on retained CW_LSO

Frozen before profiles, online execution and scoring. Primary: CW_LSO_SE.

## Hypothesis and source

ICLR 2026, Cheng et al., *Rethinking Model Calibration through Spectral Entropy Regularization in Medical Image Segmentation* ([official paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/c6e954799a0218f6d341ad5cbfb58999-Paper-Conference.pdf)), Eq. 1–10 and Appendix A.6. The paper uses supervised 3D softmax segmentation and a batch temporal spectral-energy reference. This experiment is an own unlabeled 2D binary-channel adaptation, not reproduction. Hypothesis: a causal lower entropy reference may discourage collapse to an overly concentrated spatial confidence spectrum while retaining CW_LSO gains. Neither noise suppression, clinical anatomy, calibration improvement nor causal explanation is guaranteed by spectral entropy.

## Fixed algorithm

Keep original C, W and CW_LSO. Candidate adds only lambda=0.01 times squared positive entropy deficit, averaged over independent OD/OC channels, to CW_LSO loss. No SCNP/GC/GP/RL, labels, domain decisions, additional views, models, forward/backward passes or optimizer steps.

For each channel of the existing strong-view logits z, use confidence |z| (equivalent, up to fixed scale, to max of centered binary logits [-z/2,z/2]); do not max across nested OD/OC. Float32 orthonormal FFT2, energy |FFT|^2. Unshifted fftfreq indices produce exactly equivalent radial shells to fftshift coordinates. Five bands (k*rmax/5,(k+1)*rmax/5], rmax=min(H,W)/2. DC is excluded per original open lower shell bound, outer corners excluded; no windowing or resolution change. Band powers S, P=S/(sum S+1e-6), H=-sum P log(P+1e-6). Constant/zero non-DC spectra yield finite zero entropy.

Reference is entropy of mean detached band powers from previous min(t-1,50) arrivals, separately per channel, before current image. First image uses detached current entropy so loss and gradient are zero. L_SE=mean_c relu(H_ref_c-H_current_c)^2. Commit detached current powers once after successful native update, evict oldest after 50; reference is causal and never includes current sample. Snapshot/restore includes the full buffer. Zero strength has exact original output/state behavior. Fixed lambda=.01,K=5,W=50 follow paper Table 4 base setting; no parameter scan. This causal history and binary confidence conversion differ from original supervised batch method.

## Matrix, budget and barriers

C/W/CW_LSO/CW_LSO_SE; seeds 20260907,17011,29009; two original orders; 1951 images each: 24 fresh complete trajectories, 46824 independent CPU-scored observations. No borrowing old trajectories. Same data/model/preprocessing/LR/Adam/BN adaptation/W weights/LSO0.1; no new evaluation set. Physical GPUs 0/1 only, maximum two workers, no interference with other workloads.

New immutable T0 before all GPU profiles: 22h online,24h wall,48 GPU-worker hours,32GiB. Profile and failure costs count. Four 12-arrival profiles must all PASS; whole fixed matrix must pass conservative dual-worker wall/GPU admission. No pruning, repeats, scientific retries, retuning, seeds, late selection or budget resets. Formal memory admission uses matching same-round PASS profile peak with max(1.2*peak,peak+512MiB) free; invalid/missing profile rejected. Cold profile retains original 5GiB peak guard. All online workers retire before independent CPU label scoring. Neutral parent/worker process names mandatory.

## Frozen success and delivery

SEARCH primary versus C/W each>=0.3pp and versus CW_LSO>=0.1pp. Each comparison requires both orders positive,at least5/6 trajectories positive,imageweighted>=0,worst seed-mean domain/channel/order cell>=-2pp. Do not replace primary or relax gates. SEARCH and REVIEW historically exposed; REVIEW descriptive only, not independent validation or later tuning.

Report all positive/negative aggregates and entropy/target/gap/active frequency/band powers/reference counts by channel, spectral loss and coefficient, base/LSO loss, BN and existing regional logit gradient norms. Associations are not causal mechanisms; entropy is largely amplitude invariant and does not directly enforce retinal geometry.

Retain every attempt, failure and cost. Publish own source/protocol and anonymous aggregate full results, report/audit on corresponding experiment branch using explicit local proxy. Exclude images, labels, per-content rows/predictions/scores, content/patient identifiers including hashes, model state, private paths/logs and third-party PDF. Verify remote SHA and anonymous report HTTP; update local/remote DELIVERY and cumulative ledger. Hourly necessary monitor, healthy silent; pause after completion and verified delivery regardless of scientific gate. This authorization does not start R45.
