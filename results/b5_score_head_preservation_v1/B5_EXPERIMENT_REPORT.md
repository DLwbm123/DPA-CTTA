# B5 score-head preservation: launch status

B5_RUNNING — formal computation has started; no completed efficacy conclusion is available yet.

Startup snapshot: 506 H0 canonical scalar records persisted; detached launcher alive; no failure artifact; full process/GPU command-name audit passed. This is not a completion claim.

Execution commit: `3dc5decf75d627d75ec1afd77fe5abb9dcbdb81b`.

The task tests exactly ra4_conv5.bn, ra3_conv4.bn and ra2_conv4.bn in the pinned PraNet. F freezes their six affine scalars while retaining current-image statistics; H additionally uses their frozen source statistics; H0 uses H forward policy with all source parameters fixed and no optimizer. F/H retain 150 updating BN layers, 66,212 affine scalars and 300 parameter tensors. The full B4 Polyp update step, loss, augmentation and Adam recipe remain unchanged. B4 remains a valid negative transfer result.

Seven new CPU checks and one relevant B4 normalization/geometry/target-detachment check passed in the original server environment (8 current tests, zero failures/errors/skips). The single registered-source GPU smoke passed with exactly 100 forwards and 12 backwards/Adam calls, including frozen-head gradient connectivity and old B4 all-adapt equivalence. Formal execution is sequential: H0 canonical, order0 F/H, order1 F/H. It has no performance gate and no automatic retry.

Formal budget: 9,010 scored records, 59,466 forwards, 7,208 backwards/Adam. Including smoke: 59,566 forwards and 7,220 backwards/Adam. H0 creates 1,802 independent predictions mapped to 3,604 CPU reference positions. The detached task uses physical GPU 7, within the user-authorized 3-7 range; the recorded UUID is bound for all stages. It continues independently of this conversation and SSH connection. Six active hours and 512 MiB private output are caps, not timing promises.

All 1,802 Polyp groups and old paired controls are inherited by exact identity from completed B4/P2 registrations. All data remain exposed development data; remaining_dev is not unused validation and two orders are not independent cohorts. Final per-domain/subset metrics, paired tails, pixel counts, head scalar observations and costs require complete formal work and independent CPU closeout.

Public scope is source/config/tests and aggregate acceptance/results evidence. Private identities, paths and individual scalar records stay on the server. Target images, masks, prediction/activation maps and adapted model/optimizer snapshots are not saved or published. There are no new Fundus runs, DD/SA/G/U/S/I, fusion, radius/LR/layer searches or automatic next experiments.
