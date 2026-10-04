> **Status update:** The user removed time budgets and the full matrix has now started. The budget-stop account below is historical; see [STARTUP.md](STARTUP.md). New effectiveness results are pending.

# R19: model-only GraTa validation and history checks

**NOT_RUN_BUDGET.** Implementation and mechanical qualification passed. The eight formal trajectories did not start, and no new target labels or scores were read. This is a resource-admission result, not evidence for or against the methods.

| Stage | Evidence |
|---|---|
| Implemented | G, G_HALF, G_VAL and G_MEM; execution commit `bf4db71ad923ffccfc1cca53fd988c6eb7e5c2c2` |
| CPU tested | 8 generated-input tests passed in 5.268 s |
| Model-only asset audit | Supplied segmentation checkpoint only; 19,136 trainable BN-affine parameters |
| Real mechanical qualification | PASS on 4 sequential unlabeled arrivals; all 16 recorded checks true |
| Formal online matrix | NOT_RUN_BUDGET; 0 / 15,608 visits |
| Independent scoring | NOT_RUN; 0 / 13,560 principal observations |
| Worker retirement | Supervisor, watchdog and GPU worker exited; no active R19 worker |
| Effectiveness | NA for every new method, domain, channel and order |

## Why execution stopped

The frozen allocation was 0.5 GPU hours for profile, 5 for ordinary full trajectories, 1 for sparse diagnostics, and 1.5 reserved for recovery (8 total). The measured profile, including a 1.3 factor, 0.20 s per arrival for I/O and 60 s per trajectory for startup/closure, projected:

| Allocation | Projected GPU-worker hours | Frozen allowance |
|---|---:|---:|
| Ordinary eight full trajectories | 6.798266 | 5.0 |
| Additional sparse diagnostics | 0.183131 | 1.0 |
| Already charged profile | 0.009050 | 0.5 |
| Total including already charged profile | 6.990447 | 6.5 normal, excluding recovery reserve |

Although the total projection is below the absolute 8-hour cap, it exceeds both the ordinary-trajectory allocation and the 6.5-hour normal budget. The runner did not silently spend the 1.5-hour recovery reserve or move phase allocations. These are GPU-worker hours, not a wall-clock promise. Profile timing used only four images and is a conservative admission estimate, not a full-run speed benchmark. No shorter stream, removed arm, changed threshold or second profile was used to obtain admission.

## What passed

Generated-input tests checked faithful native GraTa logits/Adam/random state, read-only state isolation, forced rejection returning the pre-update output and restoring BN/Adam/gradients, exact next-image continuation from a snapshot, empty-memory equivalence, half-step learning-rate scaling, diagnostic isolation, and arrival-based memory expiry. The real-model check matched new G to native G on all four arrivals in logits and persistent state, verified C0 initial native output and finite fixed DS inference, and checked synthetic next-image restoration on the actual model. These tests do not establish full historical G score reproduction; that remains NOT_RUN.

The first CPU constructor audit encountered an interpreter dependency directory omitted from the filesystem allowlist. The exact runtime directory was added; the CPU-only failure and its zero GPU/target-access cost were retained. No scientific condition changed, no source image qualification was performed, and no GPU qualification was repeated.

## Model-only and data limits

The only learned asset loaded was the supplied segmentation checkpoint (SHA `88b7d8902d23fb1c15b27668e0bf02599f8298aa45c38ae90279ae5a1d42bdf0`). No source images, source labels, source features/scaler/Fisher, actor, B carrier, auxiliary correction head or additional pretrained model were loaded. Native GraTa uses current-image BN statistics and only updates BN affine parameters; all other parameters remain frozen. Every formal trajectory would start from the original checkpoint and empty Adam/memory state.

Existing 800×800 ROI inputs were retained after the user delegated that decision. The original crop-center provenance is **UNKNOWN**. Any later findings are conditional on the provided ROIs; neither this audit nor the code proves the original ROI preparation was label-free. Online inference does no label-derived recropping. The host receives the current image tensor, without domain or role fields; raw file metadata stays in the input capability. The separate scorer directory is denied to online workers.

Each registered order contains 1,951 arrivals: 1,695 remaining_dev, 128 legacy_dev and 128 p1_extension_dev. All were previously exposed development content; both orders use the same images and patient dependence is unknown. They are not independent patient replications. Only the first four order-0 arrivals were accessed for mechanical checks, sequentially, without labels. Historical C0/DS/G scalar seals, checkpoint/manifest identity and exact content/role/order pairing were checked for potential reference reuse. C0 denotes historical current-image-statistics inference, not source-running-statistics inference; DS is fixed horizontal-flip inference. Historical hard-mask metrics cannot supply soft probability metrics.

## Costs and limits

Actual GPU-worker cost: **32.581206560 seconds**. Physical counts: **286 forwards, 46 backward calls, 23 optimizer steps, 0 VJP calls**. Four real image reads were reused across the four mechanical hosts and native reference; two extra generated-input visits tested actual-model continuation. The profile also measured additional history forwards where memory was empty, avoiding an underestimated G_MEM projection. All candidate, half-diagnostic, rejected-update and mechanical work is included. Formal target visits and target-label reads remain zero.

The 8-condition rows in `main.csv` and every domain/channel row are explicit NOT_RUN/NA, rather than zero Dice or fabricated performance. Zero scored observations do not mean zero undefined-ASSD cases; the latter is NA. Source-side performance is not applicable. No R20 or other successor was launched.

The frozen scientific protocol and configuration accompany the code. The raw checkpoint, images, labels, content identifiers, per-image outputs, memory and machine paths remain private. Public delivery contains code, configuration, aggregated mechanical evidence, cost ledger and this negative resource-admission result.

Public source commit: `3461e1a0edb23660a3190c3689437032511c81d7`. Before publication, three private server-directory literal occurrences were replaced with equivalent paths derived from the configured output directory. Path equivalence was checked; scientific and scoring behavior did not change and no computation was repeated. The execution receipt retains the original private commit above. See `SOURCE_EXPORT.json`.
