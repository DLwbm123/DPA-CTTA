# R32: history origin and subsequent within-domain adaptation

## Authorization and evidence
The user approved continuing the targeted diagnostic after reviewing the R30 domain audit. This authorizes offline reanalysis and a frozen counterfactual diagnosis, not a new controller, LR grid, RL training, extra patient data or new labels. Earlier hourly follow-up and no cumulative GPU-hour limit remain applicable.

R30 continuous minus episodic C is+12.581827pp ORIGA but-4.444417pp REFUGE_Valid, with both REFUGE_Valid channels negative in every seed/order group. Original domain order0 is REFUGE/ORIGA/REFUGE_Valid/Drishti_GS; order1 reverses domain blocks. Within-domain image order is identical. Existing paired scores show REFUGE_Valid order0 harm already in the first quarter, whereas order1 harm grows across quarters. These post-hoc trends motivate diagnosis; changing image difficulty prevents interpreting the trends alone as causal forgetting or gradient conflict. Old method reports are retained, and the former 'no supported next diagnostic' decision is superseded by this specific evidence and user authorization.

## Unchanged updater and assets
Use the source checkpoint, architecture,512px preprocessing, target-image manifests, and native C implementation from R30. Current-image BN has no running mean/variance; adaptive state is BN affine plus Adam m/v/local step. Six weak views, one strong consistency backward, one Adam step LR1e-4, final-original readout. All non-BN parameters stay frozen. No head adaptation or conditional adapter in diagnostic branches. Seeds20260907,17011,29009; both original orders. Keep C_CONT as fixed reference and ANCHOR as additional reference, never choose baselines by cell outcomes. Existing R30 masks provide fixed reference outputs; no reference is refitted.

## Fixed query sets and histories
For each order and each of ORIGA and REFUGE_Valid, reserve the first64 domain images as SAME64 history. Query ALL remaining images, starting at within-domain position65:586 ORIGA and736 REFUGE_Valid. The integer64 is fixed once, fitting every preceding block (smallest101 images); there is no length sweep or outcome-selected query subset. All roles are processed, only the existing authorized SEARCH and legacy REVIEW are scored; CONTEXT never enters denominators. Query images and contents are disjoint from both matched histories. Do not replace or wrap tail images.

Each case uses four states:

| State | Construction before shared query tail |
|---|---|
| SOURCE | Source BN affine, empty Adam, zero history updates, current-image BN mode |
| SAME64 | Source plus64 native C updates on the first64 query-domain images |
| CROSS64 | Source plus64 native C updates on the final64 images of the immediately preceding domain |
| NATIVE | Unchanged C replay of the entire original prefix including the same first64 query-domain images |

SAME64/CROSS64 use identical initial source state,64 unique images,64 updates, Adam step64 and paired augmentation RNG for each history position. Their composition differs intentionally. CROSS64 is REFUGE before ORIGA and ORIGA before REFUGE_Valid in order0; REFUGE_Valid before ORIGA and Drishti_GS before REFUGE_Valid in order1. These are different history treatments across orders, not replications of one universal cross-domain history. Interpret each ordered domain pair separately before any pooled summary.

NATIVE reproduces the original prefix exactly, including its larger Adam age/history length. NATIVE versus SAME64 tests the effect of adding the actual earlier prefix before the common64 images; it does NOT separately identify prefix length, optimizer age, or domain identity. SOURCE is a zero-history reference, not a matched64-update control.

## Future-rule factorial
Fork every state into UPDATE and HOLD, using the same query images in the same order. UPDATE applies native C once on each query. HOLD performs only original-image inference with fixed adaptive parameters and Adam, retaining current-image BN statistics. No inference mode switch or stale running statistics is introduced. Both branches start with exactly the same pre-update first-query prediction.

All UPDATE branches receive the native reference's per-query augmentation RNG input, and must end with the reference RNG output. Skipping updates cannot shift later augmentation draws. SAME64/CROSS64 history draws are likewise paired. Record original-image pre-update hard masks on every UPDATE query with read-only state/RNG isolation; HOLD predictions are their own pre/post output. This extra diagnostic forward is charged separately and does not influence adaptation. Do not fabricate soft probabilities, calibration or entropy from hard masks.

Native full replay must match every R30 C_CONT hard mask; NATIVE/UPDATE queries must match the corresponding reference masks after the added read-only probe. SOURCE/HOLD must match R30 S_BATCH masks. HOLD must leave parameters, buffers, Adam and optimizer-local clock unchanged; global processed-image count advances. Physical forwards, backward calls and Adam steps are explicitly reconciled.

Matrix:3 seeds x2 orders x2 query domains x4 histories x2 future rules=96 query branches. Six independent worker jobs each replay1951 native arrivals, warm256 matched-history arrivals, and process10576 branch-query arrivals. Total11706 reference replay arrivals,1536 history updates,63456 query arrivals;44970 total native Adam updates,423216 model forwards before profiles. Approximate hard-mask storage6.24GB plus traces;64GiB output protection. No branch pruning after scores.

## Labels, domain information and execution
True domains are used only to construct these offline diagnostic interventions and stratify scores. The C updater receives image tensors, not domain IDs, true Dice or labels. The diagnostic itself is not a deployable method or a domain detector. Online manifests contain opaque case IDs and fixed positions; scorer metadata remains inaccessible. Historical reference masks may be read only for exact replay checks, not update decisions. All six workers must retire successfully before any new CPU score/label release. Existing R30 score reanalysis is separate, prior development evidence.

Generated-input tests cover paired RNG replay, unchanged native UPDATE trajectory after pre-probes, frozen HOLD learning state, SOURCE/HOLD versus batch-stat source, overlap rejection, score coverage/contrasts and future-window bins. Three real A100 profiles each include8 native replay checks,64 SAME and64 CROSS updates, an8-query RNG calibration, and8 queries for each of8 branches. Profiles have no label access or scientific selection role. Profile NATIVE uses its short8-step replay state for timing only; formal NATIVE uses the full original prefix. Only formal branches are scored.

Use GPUs0/1/2 if live free memory exceeds measured peak plus margin; do not alter other processes. SharedGPU0 is permitted. No cumulative GPU cap. Seven-day wall, six-day online,22h worker timeout,64GiB output cap and20% wall reserve are engineering protections. Actual profiles determine admission and ETA. Repair retains original T0, failed receipts and costs; no performance-based retry or arm removal. Neutral executable/argument names are checked after launch.

## Estimands and interpretation
Primary diagnostic contrasts, reported separately for each query domain and order:

1. SAME64_HOLD-CROSS64_HOLD: effect of matched history content with subsequent learned state held fixed.
2. UPDATE-HOLD within each fixed starting history: effect of accumulating query-domain updates, on identical query images.
3. Their difference-in-differences: whether the update effect depends on history content.
4. NATIVE_HOLD-SAME64_HOLD: effect of the actual earlier prefix before the common64-image prefix, with the stated length/age confounding.
5. Each non-source HOLD-SOURCE_HOLD: the accumulated history's effect relative to no learned history on identical future images. SAME64-HOLD versus SOURCE-HOLD also measures the first64 same-domain updates' aggregate effect on this held query set; it does not reconstruct each of the first64 images' own causal trajectory.

Report full query-tail mean, fixed first64 query mean, four equal-position query bins, both channels, individual seeds and orders, and every negative cell. Pre/post one-step differences are auxiliary; they must not be substituted for the accumulating UPDATE-HOLD effect. Always show comparisons to native C_CONT and fixed ANCHOR on the SAME query subset; C_EPISODIC is an additional explanatory reference. Query-tail scores are not whole-stream scores. No selecting a new primary endpoint, favorable domain, history length or baseline after results.

These are descriptive counterfactual effects for specified histories and this source/model. Domain identity, class prevalence, appearance and annotation conventions may co-vary; SAME64/CROSS64 does not isolate a biological domain property. Larger or order-specific effects can motivate another separately justified protocol, but no result by itself authorizes a learned selector, hidden use of domain labels, LR sweep or RL. No arbitrary aggregate gate erases an ordered-pair effect. A failed diagnostic is not proof all histories are harmless.

SEARCH and legacy REVIEW are development-exposed, and histories/query sets share patients of UNKNOWN linkage; content-disjoint does not mean patient-disjoint. ROI provenance remains UNKNOWN. Results are not independent confirmation or clinical evidence. Publish source, protocol, post-hoc audit, registration, all aggregates/negative cells, operation/cost receipts and report. Exclude images, masks, per-content scores/identities, snapshots, weights, private paths and credentials. Verify proxy-only GitHub push and anonymous report access on completion.
