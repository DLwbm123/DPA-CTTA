# R21: fixed short regional-prototype pilot

Authorized 2026-10-06: shortest practical test of C + W + delayed regional correction.
This is a new bounded pilot, not a continuation or budget reset of R20.

- Four arms: C, CW, CR, CWR. W is R20 W06 at fixed C learning rate 1e-4.
- R reuses the source-checkpoint frozen up3 features and the R20 three-class
  (background, optic-disc rim, optic cup) prototype machinery. No external weights.
- Each class stores up to 8 image-level prototypes, expires after 128 arrivals,
  requires 32 feature tokens with source/flip confidence >=0.9. Read past, write current last.
- Require >=2 historical entries in every class. Cosine temperature 0.1.
- Connected hard-disagreement regions >=32 pixels with mean historical top-class
  probability >=0.7 receive soft correction alpha=clip(confidence-0.5,0,0.5).
- Loss mixes whole-image C/W with per-region normalized C/W, coefficient 0.25;
  reliability remains outside regional normalization. These fixed engineering choices
  are uncalibrated and are not claimed to guarantee correctness.
- One seed 20260907; two existing orders; same 192 content identities: all 22
  SEARCH Drishti images, 57 ORIGA, 57 REFUGE, 56 REFUGEValid, selected by
  sorted existing content identifier within domain, retaining existing order afterward.
- Eight fresh trajectories, one BN-only Adam update per arrival. Four GPU workers
  on 4/5/6/7 if available memory suffices. No retries, tuning, extra seeds or expansion.
- Online deadline 10 minutes after launch; overall budget 15 minutes and at most
  40 GPU-worker minutes; disk cap 2 GiB. Any incomplete arm makes the pilot incomplete.
- Runtime reuses current-image-only filesystem audit and neutral subprocess entry.
  Only SEARCH masks are opened by CPU scorer after every online worker retires.
- Report macro Dice by domain/channel/order, image-weighted Dice, time/image,
  pseudo-target corrected/damaged pixels and the low-view-variance (<=0.001) subset.
- Promising signal: CWR-CW >=0.2 pp, positive in both orders, worst cell >=-2 pp,
  corrected > damaged, and CWR > C. Otherwise effectiveness is not established.
- This historically exposed, one-seed, short SEARCH stream is a screening test,
  not independent validation. Patient linkage is unknown. No auto follow-on.
- Publish source, registration, aggregate results and report; exclude images,
  masks, per-image identities, checkpoints, downloaded papers and private bindings.
