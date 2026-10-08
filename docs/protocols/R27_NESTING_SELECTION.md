# R27: fixed nesting-only residual candidate selection

R26 completed36/36 but failed its development gate. Its bounded residual actions
increased candidate Dice separation~21.6x without useful full-reward ranking.
A SEARCH-only audit compared eight predefined components and retained all outcomes.
Negative soft nesting showed small positive selection gains in both orders and5/6
trajectories for each candidate family; worst-domain effects remained negative.
This is development-selected motivation,not independent evidence or a guarantee.

Three arms:ANCHOR,RANDOM4,GREEDY. Use R26's bounded four-group residual-output actions,
unchanged four-candidate Gaussian distribution,optimizer,rollback and one committed
update. ANCHOR is unchanged. RANDOM4 selects randomly; GREEDY selects minimum mean
relu(p_OC-p_OD),computed at64px on candidate probabilities. Equal rewards keep the
first sampled candidate,as in the existing selector. The offline audit used uniform
tied-max expectation; that is a descriptive expectation,not a replay guarantee.
All original reward components remain logged; legacy_retain retains the original
combined score while retain now means the negative nesting selector score.
No learned policy update,covariance,DPO,new assets or source data are introduced.

Freeze18 trajectories:three arms x seeds20260907,17011,29009 x two orders,each1951
arrivals (35,118 total). No extra seeds,performance pruning,label feedback or changes
within this round. All3 real engineering profiles must pass before formal admission.
All online workers retire before CPU scoring; retain every failure and negative
result. Primary GREEDY; fixed contrasts GREEDY minus ANCHOR and RANDOM4.

Development signal requires >=0.3pp over ANCHOR,positive mean over RANDOM4,positive
both-order effects and>=5/6 positive trajectories against both comparators,
nonnegative image-weighted delta against ANCHOR and worst seed-averaged domain/channel
delta>=-2pp. No alternative primary if the gate fails. Main512px hardDice averages
domain/OD-OC equally then seeds/orders;empty/empty=1. Preserve image-weighted metrics,
all domains/channels,negative cells,paired content bootstrap,costs and failures.
64px candidate diagnostics are not main metrics. Anatomy alone admits empty or
incorrect nested masks; bounded residuals/native training do not eliminate this risk.

All SEARCH/legacy SEALED_REVIEW content is development-exposed. The reward was chosen
from R26 SEARCH diagnostics after R26 final results were observed. Patient linkage
and ROI provenance UNKNOWN. No independent or clinical efficacy claim,novelty not
established. A later paper needs appropriate independent confirmation.

No cumulative GPU-worker cap(JSON null),but record all costs. Per-task hang guards,
64GiB storage and seven-day supervisory safety timeout remain. GPU4–7 only,no other
tasks modified. Original T0 immutable;at most one evidenced transient I/O/network
recovery,no retry for numerics or weak effects. Publish code/protocol/aggregates and
all outcomes via configured GitHub proxy;never publish private images,labels,masks,
identities,weights,credentials or paths. Hourly monitoring continues across rounds.
