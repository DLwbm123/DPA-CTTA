# State and label isolation

| Phase | Allowed inputs and mutable values | Release / reset |
|---|---|---|
| Metadata Stage I | JSON registration; design bytes; procedural CPU tensors only | No RGB/mask/checkpoint loading and no GPU discovery |
| Source oracle (future tensor backend) | Same-fold two support groups and labels; ambient v Adam16 | Separate fold-tagged oracle; query excluded from fitting |
| Basis/scaler preparation | Fit oracle only; disjoint fit cov/probe groups; fit observations only | Fixed basis/scaler; no target statistics |
| Source FULL task fit | Four support observations update recurrent state; each query is a different group | One combined backward per four-time episode; query output/loss never commits state |
| Source STATIC task fit | Same architecture, seed, schedule and optimizer budget | Reset before **every** source visit; distinct parameter instance |
| Source calibration | Cal fold; frozen task modules and backbone | A tau only; B kappa only; C reliability MLP only;256 fixed steps |
| Source validation | Val fold; final weights and64 fixed episodes | Report proxy error/query loss/soft Dice, no selection or optimizer |
| Target observer | One current preprocessed image; frozen observer/modules | Temporary134 statistics and64×64 tokens only |
| Target candidate update | A m/P; B z/d; C z, prior visit count | No labels/domain/source/identity/future argument |
| Target final forward | Current image with candidate FiLM state | Commit only after state/logits finite and valid; then evaluator may receive logits |
| Target error | First exception and incurred call counts | FAILED terminal; no retry, no state commit, no evaluation release |
| New trajectory | Same frozen source assets with exact binding | Initial state; no cross-arm mutable state or RNG consumption |

`OnlineHost` holds no label/evaluator loader. Tests change query pixels and query labels independently, restore saved host state, create/delete unrelated GT/future tensors, and verify identical next outputs and state. These are API/causality tests, not claims about clinical independence. Full-network tests verify frozen backbone tensors and absent backbone gradients after source micro-training and online visits.

State save/load requires matching module/basis/scaler/calibration digest and ablation. A validates full symmetric SPD covariance. All source algorithms retain differentiability through float64 casts; online steps explicitly run under no_grad. Static history reset preserves total visit count for audit.

Real source manifest/split/checkpoint/trained modules and prior source-checkpoint subject exposure are PENDING. No target data is substituted. Source supervision represents new information and additional cost, not a free extension of historical C. No original results, gates or source files are changed.
