# R28: distinguish candidate headroom from selector regret

Authorized2026-10-08 after R27 completion. This is a mechanism diagnostic, not a
new efficacy claim or reward sweep. Host is the authorized A100 server, physical
GPUs0,1,2 only. No cumulative GPU cap; retain operation/runtime cost, 64GiB output
limit,24h per replay and7-day supervisory safety deadline. No other tasks changed.

A: one offline pass through existing R27 SEARCH candidate scalars. Report exact
nonzero/all-zero rates, exact maximum ties, first selection, reward range/top gap,
64px actual first-argmax gain/regret and uniform-tie expectation separately. For
RANDOM4 states this is a counterfactual greedy selector replay, not its executed
random choice. Aggregate equally over domains,seeds,orders; preserve all cells.
No labels are opened beyond the existing scored SEARCH records.

B: six complete ANCHOR reference replays: seeds20260907,17011,29009 x two existing
orders,1,951 arrivals each. Freeze four SEARCH occurrences per domain/order at
25%,50%,75%,100% of that domain's SEARCH occurrence list (ceil(q*n)-1). Four domains
x four points x two orders x three seeds=96 states. Select by manifest order only,
not observed rewards,errors or labels. Historical content is development-exposed.
At each pre-arrival snapshot evaluate one ANCHOR update and four residual-action
updates under identical native RNG. Four actions are independent N(0,.3^2) vectors
in R^4, matching the untrained R27 distribution. Use separate per-visit candidate
RNG keyed by seed,order,visit; no selector draw advances candidate noise.
Use the original exp(.75*tanh(action)) residual multiplier. Report negative soft
nesting at64px and512px and full-resolution violation fraction. Actual choice is
first argmax of64px reward; no epsilon/tolerance or label-guided tie rule.

Store five512px hard masks and pre-arrival snapshots privately. Restore exactly
the ANCHOR post-update state,including parameters,Adam,BN/native RNG,adapter,
history/memory and temporary context; diagnostic candidates never enter the
reference trajectory. Labels cannot be opened by GPU replay workers. Only after
all six replays successfully retire may a separate CPU scorer read the96 fixed
SEARCH labels. Preserve all failures and costs; failed profile stops admission.
Three real GPU smoke jobs(two arrivals,one probe each) precede the six replays.

Use512px hardDice,empty/empty=1,mean OD/OC within a state,then equal domain and
seed/order aggregation. A joint OD/OC oracle chooses among the FOUR candidates;
ANCHOR is separate,so oracle-minus-ANCHOR may be negative. Report oracle gain,
actual selector gain,uniform-random expectation and selection regret with the
identity actual gain=oracle gain-selection regret. Preserve domain/channel cells,
both orders and all seeds; no seed/metric substitution. This local oracle is not
an upper bound on closed-loop CTTA or a full-stream efficacy measurement.

C is conditional,not automatically launched: B must show mean oracle headroom
>=0.3pp,positive headroom in both orders and>=5/6 positive seed/order combinations.
These are resource-allocation gates on96 diagnostic states,not significance or
confirmation gates. If passed,freeze a separate short-horizon branch protocol
before generating further predictions,using the saved states and identical future
inputs/native RNG. If failed,report candidate limitation on the sampled ANCHOR
states and assess the previously proposed update-strength action family; do not
claim no other states/actions could work. Do not blindly swap reward components.
No new RL policy,DPO or covariance learning. No tuning within R28.

A100/PyTorch environment differs from historical3090 runs: B compares candidates
within the same state/hardware,not byte-for-byte R27 trajectory replication.
Independent validation,patient grouping and ROI provenance remain unresolved.
Publish code,protocol,aggregates,costs and negative results only; no private paths,
identities,images,masks,snapshots,labels,credentials or checkpoint weights.
