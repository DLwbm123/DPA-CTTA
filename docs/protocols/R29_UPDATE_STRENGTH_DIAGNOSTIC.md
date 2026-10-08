# R29: update-strength headroom on the same96 frozen states

Authorized2026-10-08 continuation of the discussed candidate-space diagnostic.
R28's four residual candidates offered +0.080122pp local oracle headroom,below the
registered0.3pp allocation threshold. R29 changes the action family,not the reward.
Use exactly the same96 prespecified SEARCH pre-arrival snapshots,three seeds and
two orders. Do not regenerate or select states using scores. Reuse the same A100
host/environment and GPUs0,1,2. No full trajectory rerun,RL or reward fitting.

Three arms from each identical R28 snapshot:
- FULL: unchanged ANCHOR one-step adaptation and final prediction.
- ZERO: one current-image forward with original BN batch-stat behavior and current
  image context; skip all backward/Adam operations. Restore all temporary context,
  counters,buffers and RNG afterward. It is not an Adam update with learning rate0.
- HALF: identical augmented inputs/gradient/Adam moments as FULL,with both BN and
  adapter step learning rates multiplied by0.5 after the inherited fixed-LR hook.
  Effective BN LR5e-5,adapter LR1.5e-4. No residual amplitude perturbation.

All three branches are diagnostic and discarded. Each FULL512px hard mask must
exactly match the corresponding saved R28 ANCHOR mask,or fail the run. Verify exact
input-state restoration. Save only three masks per state and label-free metadata
privately. Separate CPU scoring may read96 SEARCH labels only after all six jobs
retire successfully. Three one-state real GPU smoke jobs precede formal admission.
Reuse existing supervised subprocess,physical counters,receipts and failure ledger.

Primary diagnostic512px hardDice:joint OD/OC within state,equal domain/seed/order.
Report fixed ZERO-FULL and HALF-FULL deltas and the joint OD/OC oracle over
FULL/ZERO/HALF. This oracle includes FULL,so its headroom is nonnegative by design;
R28's four-candidate oracle excluded FULL. Do not compare these two raw oracle
numbers as an equal-cardinality/equal-definition test. Preserve every cell,negative
domain/channel,order and seed. Tied oracle selects FULL then ZERO then HALF for
bookkeeping only; no label-driven online policy or oracle trajectory is run.

Headroom allocation gate remains mean>=0.3pp,both orders positive,>=5/6 positive
seed/order groups. Even passing it does not establish an executable selector or
closed-loop gain. Any further short-horizon or fixed-strength full-trajectory test
needs a separate frozen protocol and evidence-based decision. No automatic sweep
of additional strengths,states,seeds or metrics; no conditional pruning of this run.

Mechanical checks cover FULL parity,true skip,half-sized BN/adapter parameter
steps with unchanged Adam moments,exact rollback and deterministic replay.
No cumulative GPU budget cap;24h task/7day supervisor safety deadline and64GiB
output guard remain. Previous private states and assets read-only; no other jobs
modified. No scientific retry for weak results. Preserve all engineering failures.
All data are development-exposed,patient/ROI provenance unresolved,not independent
confirmation. Publish code/protocol/aggregates/costs only,never private states,
images,masks,labels,identities,checkpoint weights,credentials or private paths.
