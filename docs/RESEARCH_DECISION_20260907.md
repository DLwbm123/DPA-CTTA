# Research decision — 2026-09-07

Status: design decision only. CODE_REVIEW_PENDING; no training or evaluation.

The existing Distilled Potential Atlas remains an implementation-only research
ablation. Its current files and APIs are preserved. The conceptual name
`potential_atlas` describes that role; no package rename or new mode was implemented.
There is no experimental evidence that dataset distillation (DD) causes an Atlas
benefit. Gradient connectivity establishes a computational path, not necessity
or improved segmentation. A contraction bound on the state recurrence guarantees
neither accuracy nor prevention of segmentation collapse.

The preferred investigation starts from a previously validated medical CTTA host,
then tests a medical loss independently, then tests supervised source-proxy
rehearsal. The supplied plan selects VPTTA; its historical performance numbers
are supplied context and were not re-evaluated in this publication. A real coreset
is a privileged mechanism control, not dataset distillation or a source-free
deployment result. Only a useful proxy-supervision mechanism warrants DD work.

An optional later direction optimizes synthetic images through an actual online
update and evaluates region/boundary loss on a different source query image under
the same style. Fixed source masks, native optimizer state, input gradients,
buffer side effects, and source/target label separation must be explicit. This
path is proposed, has no implementation in this snapshot, and has no results.
If DD adds no value relative to a simpler host or medical-loss branch, remove DD
from the main method regardless of this repository's name.

Test-style injection into condensed source images is prior art. Kang et al.,
[Leveraging Proxy of Training Data for Test-Time Adaptation, ICML 2023](https://proceedings.mlr.press/v202/kang23a.html),
already describe condensed images stylized using unlabeled test samples for
supervised adaptation. That combination alone is not a novelty claim here.
The official abstract was checked for this publication; the remaining references
in the supplied plan are a reading/review agenda, not a completed literature audit.

No SOTA, privacy guarantee, clinical result, or performance improvement is claimed.
The [revised plan](REVISED_EXPERIMENT_PLAN.md) is retained as supplied; it is not
this turn's training instruction. Its description of an empty remote is historical
pre-publication context. All M0, source pilots, coreset training, DD, target mini,
full-target, and multi-seed experiments remain pending independent code review
and explicit user approval of a specific commit and experiment configuration.
