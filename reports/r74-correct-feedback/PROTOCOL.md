# R74: one correct feedback event, frozen future predictions

Registered before new target inference or feedback-label decoding. This is a supervised mechanism diagnostic, NOT an unsupervised CTTA performance result or a mathematical performance upper bound. The user authorized exactly this minimum experiment. No automatic extension follows a positive or negative outcome.

## Frozen assets and scope

Use the six R50 CW_LSO states after visit1900: seeds20260907,17011,29009 crossed with orders0,1. One feedback image per state is the original arrival1901; future evaluation is exactly arrivals1902 through1951 inclusive, H=50. Do not replay the historical prefix, select feedback by errors, change windows, add states or substitute the older48-content panel. R50 source checkpoint and original numerical policy remain unchanged.

Order0's feedback and50 future contents are Drishti_GS; order1's are REFUGE. Each future window has50 distinct archived content identities with zero overlap with its own1901-image prefix. The two future windows do not overlap, but each appears in the other order's historical prefix. Three seeds reuse the same contents. There are six states, two distinct feedback images and100 distinct future images, NOT six independent feedback patients. Patient linkage is UNKNOWN; archived group_id equals content identity. This tests late-state same-domain image transfer, not cross-patient, cross-domain, four-domain robustness or the REFUGE_Valid deficit. Development exposure persists.

## Exactly three branches

NO_UPDATE: no optimization at arrival1901. NATIVE: one unchanged CW_LSO update. CORRECT: one update with only the weighted BCE target replaced by the current feedback image's binary OD/OC truth. Preserve the original six-view W, original-pseudo-target LSO with coefficient0.1, BN parameter ownership, Adam history, learning rate, appearance augmentation and paired RNG. Strong-style augmentation has no spatial transform; feedback truth uses the existing512px nearest-neighbor label decoder and independent nested binary channels. No softmax normalization, norm matching, LR change, BCE-only condition or partial-correction condition.

Start every branch from the identical complete1900 state. Capture the actual BCE targets and original-image output probabilities for current-image diagnostics. CORRECT changes target correctness AND confidence/gradient magnitude; this is the total intervention effect, not isolated direction causality. Failure does not identify BN capacity separately from the retained objective/optimizer.

After the one feedback event, freeze each branch's learning state. Predict each of the50 future images read-only, restoring that same branch state before each image. Current-image BN computation follows the native path; buffers, modes, RNG, optimizer, gradients and counters must not accumulate across probe images. No future gradient update or continual adaptation is performed. Restore the original anchor after each state group.

## Label and state safeguards

The GPU worker may decode ONLY the two authorized feedback labels, reused across seeds/qualification. Future manifests given to it contain image fields only. The existing filesystem guard denies scorer files and unpermitted label/image assets. Future labels are decoded only by a separate CPU scorer after successful worker retirement and all918 output masks are present. No score controls runtime branch selection.

Preserve the journal's existing checkpoint integrity check when loading six copied states. Check archive identity/context/visit binding and fixed manifest counts. Qualification for each state: restored full state exact; NATIVE matches the unmodified retained host's full state/output and archived arrival1901 hard mask; repeat NATIVE and CORRECT from the anchor with exact output/full-state equality; six weak predictions, native pseudo-target and augmentation RNG identical across update branches; captured CORRECT target exactly equals authorized truth. Read-only prediction repeatability and state preservation are checked. Qualification cost counts inside the GPU cap. CPU self-checks test BCE replacement gradients including identity substitution, independent-channel label decoding, gate arithmetic, and aggregate dimensions. Mechanical failure is NOT an efficacy result.

## Fixed endpoints and decision

Primary: future CORRECT minus NATIVE hard-Dice percentage points. Average50 images and OD/OC within each state, then equal three seeds and two orders. This is a two-tail endpoint, not a four-domain macro average. Future CORRECT minus NO_UPDATE and NATIVE minus NO_UPDATE are reported with all state/order/channel cells. Feedback-image BCE diagnostics, Brier(all/GTforeground/GTbackground/radius3 boundary), Dice, FP/FN and actual parameter-update norms are descriptive; current-image improvement never substitutes for future benefit. Empty regional Brier is NA, not zero; no required future Dice cell is omitted.

The allocation signal requires ALL:

1. Primary mean >=+0.05pp.
2. Each order's mean >0.
3. At least5/6 state means >0.
4. Worst order/channel mean after averaging seeds and contents >=-0.5pp.
5. Overall future CORRECT minus NO_UPDATE >=0.

These are resource-allocation thresholds, not clinical importance claims. A2000-draw within-order content-paired bootstrap, seed20261018, keeps seeds/channels/branches coupled and reports conditional95% intervals only; intervals are not an extra gate and do not model feedback-patient uncertainty. Publish all negative cells. Report100 distinct future contents and two feedback events explicitly. No full-stream or independent-generalization claim.

## Budget and completion

One GPU worker, hard budget1200 process-wall seconds including imports/loading/qualification; scorer hard budget300 process-wall seconds. Existing controller terminates before the caps. Before formal work, measured qualification timing must conservatively admit the12 formal updates and900 future forwards within the remaining GPU time; no automatic cap increase or retry. CPU preparation/check cost is recorded separately. Planned formal outputs:18 current masks plus900 future masks;12 updates, approximately1002 formal forwards. Store current probabilities/targets only, not all future float probabilities. New output cap256MiB includes receipts/logs/predictions/current diagnostic arrays, excluding the code snapshot and copied input checkpoint/manifest assets.

After retirement, score, analyze and publicly deliver own code, this protocol, anonymous aggregate cells and receipts. Raw images, labels, states, current probabilities, predictions and content identities stay private. Any failure retains costs and partial files. No automatic additional controls, continuous adaptation, horizon/seed expansion, BCE-only test, parameter-opening or successor experiment.
