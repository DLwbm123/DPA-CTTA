# R28 common-state diagnostic: complete, C gate not met

All six ANCHOR reference replays completed1,951 arrivals and16 fixed SEARCH probes
each:11,706 arrivals,96 scored states. Online replay ended2026-10-08 21:58:33
Asia/Shanghai,CPU scoring21:58:42. All formal/scoring processes exited0. The label
release followed retirement of every replay; online label reads were zero.

|512px same-state comparison|Equal domain/seed/order mean (pp)|
|---|---:|
|Four-candidate joint OD/OC oracle minus ANCHOR|+0.080122|
|Actual first-argmax nesting choice minus ANCHOR|+0.000840|
|Uniform candidate expectation minus ANCHOR|+0.006544|
|Oracle minus actual choice (selection regret)|+0.079282|

The oracle gains are +0.069984pp and +0.090260pp in the two orders,positive in6/6
seed/order groups. Nevertheless the registered >=0.3pp mean headroom gate fails.
Stage C was not launched. This allocation gate is not a statistical significance
claim. All domain/seed/order and channel outcomes are retained in the tables.

On these96 prespecified ANCHOR states,the residual candidate set offers limited
average immediate headroom,and the nesting selector captures little of it. Even
perfect local selection among these four candidates would not meet the allocation
threshold on this sample. This does NOT bound full-stream closed-loop gains,prove
all states lack useful candidates,or identify accumulated updates as the cause of
R27's failure. The actual selector is also below uniform expectation by0.005704pp.
The96 states are diagnostic observations,not96 independent patients.

The prior64px exact-selector audit remains phase A,with a different state/sample
and resolution. Its numbers should not be subtracted from these512px results as a
controlled resolution effect. No labels were used to select online updates; oracle
labels were read only by the offline scorer. SEARCH/legacy REVIEW are development-
exposed. Patient linkage,ROI provenance and independent confirmation remain open.

Total runtime from original T0:0.627506 wall hours;1.419942 GPU-worker hours;
0.001569 CPU-worker hours. These include all GPU profiles and the three initial
infrastructure failures; stage A CPU audit cost is separately recorded in
AUDIT_RECEIPT.json. The failed profiles executed zero forward/backward calls and
were repaired by making the pinned Git object database self-contained. No formal
attempt failed or was retried. See COST.csv and COMPLETION_AUDIT.json.

Next decision: do not expand the nesting selector or launch short-horizon C on
this evidence. The previously discussed zero/native/reduced-update-strength action
family is a more relevant next diagnostic target than a more complex reward learner.
Its implementation and protocol must be inspected and frozen separately before
execution; no such follow-up has been launched by this report. Retain both the
limited-headroom and ranking-loss findings rather than treating either as the sole
cause. No independent efficacy or paper-ready positive claim is established.

Runtime code:efb2bf758925d725f4550f63edfaef3fef7ed152. Public delivery excludes images,
labels,identities,per-state masks,snapshots,model weights,credentials and private paths.
