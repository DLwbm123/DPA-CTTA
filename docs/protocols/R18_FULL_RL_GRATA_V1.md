# R18: full RL use/write FiLM plus native GraTa

Frozen before new GPU qualification or target access. One finite experiment authorized by the user's instruction to test worthwhile combinations, with the original full RL framework retained. Parent R17 is closed and its costs remain independent. No successor or parameter search.

## Hypothesis and coupling

Native GraTa's entropy-gradient alignment and dynamic learning rate can improve the features presented to the original controller, while source-trained RL use/write decisions can add value beyond GraTa and a fixed carrier policy. Native G first completes its original current-image update. Copy only its BN affine parameters into an identical checkpoint/architecture FiLM model. Its zero-FiLM current-image logits must exactly equal native G's final logits on every visit. The unchanged B64/global/.3 carrier proposes the code; the original 193-input actor chooses gain, eight residual components and write gate; FiLM produces the final prediction; m64/q32/h memory is committed across images and domains. Both states reset per order.

GraTa evolution is independent of RL actions in this coupling. This does not test end-to-end policy control of the GraTa optimizer. Non-BN backbone, carrier, source normalization and target actor are frozen. Only native BN affine updates and recurrent controller memory change at target time. Current image only; no target labels/rewards or future-image reads. Composite physical inference costs 11 backbone forwards, 2 backwards and 1 Adam update per image; original full RL costs 2 forwards, no backbone backward/update. Extra mirror forwards are explicitly counted.

## Fixed matrix

| Condition | Actor | GraTa | New full streams |
|---|---|---|---:|
| C0 | none | no | 0; sealed reference |
| G | none | native | 0; sealed reference |
| RL_ORIGINAL | sealed R10 POST GR_RET_EMA | no | 2 |
| RL_GRATA | new source WARM then GR_RET_EMA | native | 2 |
| FIXED_GRATA | all actions fixed: gain .8, residual 0, write .5 | native | 2 |
| WARM_GRATA | same new WARM endpoint before RL | native | 2 |

All eight new streams have 1951 online visits/1695 primary observations: 15608 new visits/13560 primary. Complete original RL is rerun because its historical short stateful stream cannot be composed into a full trajectory. C0/G sealed R17 references retain original matching full native trajectories, exact manifest/checkpoint/registration/scalar provenance. Native G seed 20260907 matches the reference; actor/source seed 20260924 matches original actor architecture. These are different seed roles. No other seed or candidate.

## Source fitting and qualification

Reuse the frozen source data roles, B carrier, projection and fit-only scale. Source fit and validation groups remain disjoint and separate from target. Each fold has 16 fixed episodes: `i+16*m` for i in (0,4,8,12), m in (0,1,2,3), 32 visits each. Native G resets per episode. Cache only source BN states, observations and scalar diagnostics; regenerate source images by their exact simulator keys. These action-independent source trajectories are reused across policy training. Four fit episodes made in qualification are reused, never recomputed.

WARM uses 1000 supervised updates of eight source queries with zero initial memory and writer .5, original AdamW/cosine/gradient-clipping recipe. GR_RET_EMA then uses 1024 four-query rounds, four sampled candidates, Gaussian std .35, the original four-step Dice plus .05 retention reward, EMA-normalized grouped advantage, clipped likelihood ratio and .005 warm-reference KL, two optimizer epochs per round. All ten actions, including write, train in the RL stage. Retention probes use the BN state after the final query of the four-step window, held equal for the before/after-memory comparison. Only source labels supply rewards. Target actor is deterministic and frozen.

This bounded source pool uses ordinary R10_QUERY simulator keys also for WARM, and maps training episode indices modulo 16. It differs from historical R10's larger schedule; do not attribute RL_ORIGINAL versus RL_GRATA solely to GraTa. The new WARM and fixed-action controls address RL's added value. Source endpoints are fixed (1000/1024), no checkpoint selection. Report all three 16-episode validation hard/soft OD/OC, gains, writes and memory norms; source scores do not select target conditions.

Preflight: CPU all-action fixedness and memory recurrence, BN-copy isolation, finite balanced pool; real source old-RL exact output/state parity; zero-FiLM composite exact native-G parity; full native BN/Adam/RNG plus RL-memory snapshot continuation exact output/state; immutable parameter checks. Four source fit episodes form the reusable cache. Two WARM and two RL cost-probe updates are discarded and billed; source writer gradient must connect. Qualification failures stop before targets; no automatic qualification repeat. An identified implementation error may only be corrected explicitly with preserved evidence/cost and an unchanged T0, never interpreted as a negative scientific result.

## Scoring and decision

No new target score may be read before all eight registered trajectories are terminal and workers retired. An independent CPU scorer reads labels only then. Complete mask seals, paired content/role/arrival, immutable identities and actual costs precede interpretation. Report all conditions/orders/domains OD/OC, primary denominator, valid/undefined ASSD, macro and image-weighted paired effects, worst domain/channel, topology/containment where available, source hard/soft and action diagnostics. Target probability metrics are NA, not reconstructed from masks.

Primary priority signal: RL_GRATA minus G average at least +0.5 percentage points, both orders positive. Domain/channel losses greater than 2 pp mark risk. To credit RL, additionally require both-order gains over WARM_GRATA and FIXED_GRATA; report magnitude even if not met. Compare RL_GRATA to original RL separately. All data were exposed development data, two orders share content, patient dependence is unknown, one fitted policy seed is not independent confirmation. No clinical/new held-out claim, no cherry-picking.

## Budget and operations

Conservative T0 Oct 4 2026 19:45 Beijing remains fixed. Normal computation ends Oct 5 18:45, hard computation 19:15, absolute 19:45. Maximum 43200 GPU-worker seconds (36000 normal +7200 recovery reserve), preflight 3600/source10800/target21600, max two concurrent workers, physical GPUs4–7 only, 8GiB private artifacts/cache. Real free VRAM >=5GiB, actual NFS mount/free space and read/write probe required. All argv neutral. No other process interference. Keep unique phase/attempt cost, failures included; replace live estimate with final actual cost.

After measured preflight, 1.3 safety-factor cost admission covers remaining 896 source cache visits, 1000 WARM updates, 1024 RL rounds, 1536 validation predictions and 600s setup; targets cover two original RL and six combined full streams, .15s/image overhead and 60s/job setup. Full matrix only: if source/target/total/wall admission fails, stop as NOT_RUN_BUDGET and deliver the source qualification package. Do not shorten streams, delete arms or reduce training after seeing measurements.

Reuse finite detached watch/supervisor, identity leases, private environment, physical Meter, append-only target journal and CPU scorer. At most one evidenced infrastructure recovery per physical target, exact BN/Adam/RNG/memory/cursor/sealed-prefix restore. No source automatic retry, no performance-triggered retry. Hourly quiet monitoring, repair only within frozen scope. Finish by sanitized source/config/results/report commit and proxy-only GitHub push, verify remote SHA and anonymous REPORT. Then pause monitoring, no successor.

## User amendment: three-hour wall limit (October 4, before target access)

The user explicitly tightened the total duration to three hours. This supersedes the original 24-hour allowance and the requirement to finish every extra comparison. T0 remains October4 19:45 Beijing: GPU computation stops by22:25, absolute execution stops by22:45, with20 minutes reserved for scoring/closing. Already consumed time and costs remain charged. The ongoing source worker and scientific configuration are retained without restart or changes to training steps, actor, metrics or data. A separate identity-checked deadline guard and entry deadline clamp enforce the shorter window for existing and future workers. The original full-matrix estimate no longer promises completion within this shorter window. Unfinished/never-started conditions are reported explicitly; complete streams alone retain full denominators, no truncated stream is presented as a full result. This is a user budget amendment, not an infrastructure fault or permission to retune.

## User amendment: fast fixed-policy effectiveness screen, no source retraining

Before any new target access, the user requested rapid effectiveness validation and questioned the need for source retraining. The obsolete source-preparation phase was stopped with its17 completed cache episodes and actual costs preserved. It had not started policy optimization; no target stream ran. It is superseded, not a scientific failure.

The quick stage uses the same historical full POST GR_RET_EMA actor for both RL_ORIGINAL and RL_GRATA; no new actor/head/carrier fit or source checkpoint selection. Native GraTa and the exact qualified mirror/controller/FiLM path are unchanged. The source BN/cache/training data generated earlier are NOT used by this quick target screen. Qualified interface/parity/resume checks are reused, no GPU qualification rerun.

Fixed matrix: RL_GRATA, RL_ORIGINAL and FIXED_GRATA, each2 complete1951/1695 streams; matching complete G and C0 sealed references. 11706 new target visits/10170 primary observations. Main combination pair runs first, then original method, then fixed-policy pair. No WARM condition. All target scores remain sealed until all6 registered trajectories are terminal. Primary threshold is unchanged versus G; combined minus original and combined minus fixed are additional diagnostics. Negative evidence concerns this unchanged-policy coupling; it cannot exclude a separately source-refitted policy. Learned-policy versus fixed does not isolate RL training from earlier supervised training.

No shortened trajectories, target-based condition selection, new seed, parameter or source reward. Reuse frozen actor asset/seals and score alignment. Before targets, require1.3 profile projection for all6 streams divided by two workers to fit remaining GPU-compute wall time. No infrastructure retry is scheduled in this shortened screen. All earlier R18 phase costs remain in combined campaign accounting. OriginalT0 remains19:45, GPU cutoff22:25 and absolute22:45 Beijing October4. Source preparation is retained privately and reported as user-cancelled; later conditions that miss cutoff remain explicitly incomplete.
