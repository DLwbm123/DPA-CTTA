# R63 source-moment feedback audit — no experiment registered

This bounded implementation/evidence audit follows R62. It adds no predictions, target-label reads or efficacy result. Source BN moments are information from training, distinct from current pseudo-labels, so they merit a separate compatibility check.

## What the existing mechanism actually supplies

The inspected local VPTTA OPTIC `utils/convert.py::AdaBN` takes saved source channel means/variances and blends them with detached current batch moments. Its coefficient is m=1/(sqrt(sample_num)/5+1). It then penalizes the mean absolute deviations of per-image channel means and standard deviations from that blended reference, averaged across BN layers. `vptta.py` optimizes an input prompt for one step and uses the blended moments for normalization. This is the mechanism already exercised by the [R52 VPTTA adapter](protocols/R52_FIVE_METHOD_COMPARISON.md), not an untested correction source newly invented here. No third-party source is copied into this report.

For batch size one, the mean mismatch value equals (1-m)*abs(source_mean-current_mean). The reference's current contribution is detached, however: its optimization gradient is a surrogate with the reference held fixed, not the total derivative of that simplified value. Variance uses a square root after blending variances and is not the same linear expression. Merely grafting the visible scalar expression into another host would not reproduce the native computation.

The [current C/CW host](../src/dpa_ctta/b1_host.py) explicitly sets running_mean/running_var to None and disables tracking. A new source-moment objective would therefore need a separately preserved reference and a specified feature path. Restoring running statistics in that host would also change normalization/readout; it is not a loss-only intervention. [R30](protocols/R30_STATE_HISTORY.md) already compares source versus current-image normalization, and R52 already evaluates native prompt adaptation.

## Why the proposed scalar is not yet a candidate selector

Let D(x) be the source-moment discrepancy measured in a fixed encoder on the original current image. For two candidate masks y1 and y2 from that same x, D(x) is identical: D does not depend on y. It cannot rank those masks without another, currently unspecified mapping. This is a definitional limitation, not a negative measured correlation.

Using candidate-specific updated encoders or prompted inputs makes the statistic action-dependent, but changes the measurement path. Then a decrease in discrepancy does not by itself establish anatomical correctness. Channelwise moments also discard spatial position: rearranging a fixed feature tensor's spatial entries preserves these moments while changing a spatial mask correspondence. This observation concerns the statistic, not a claim that arbitrary pixel permutations preserve network features.

[R52](../reports/r52-five-method-comparison/REPORT.md) illustrates the unresolved trade-off: VPTTA has macro Dice 74.402788 versus CW_LSO 78.786061, while its REFUGE_Valid OC order-0 cell is higher (74.312 versus 58.985). Those outcomes do not isolate BN discrepancy, memory, prompt or normalization as the cause. They do not justify combining the methods or selecting a domain-specific winner.

## Decision and bounded continuation

Do not register a frozen-source-discrepancy mask selector: its proposed score cannot distinguish its actions. Do not rerun native VPTTA or the source/current normalization contrast, or tune layer weights/warm-up/prompt strength under a new name. This does not rule out every source-statistic objective or prove that prompt adaptation cannot help.

A distinct follow-up must specify an action-dependent observation, a fixed comparison with the existing implementation, and what would falsify the link between the observation and the targeted error. The unresolved requirement is a candidate-specific correction signal, not another scalar computed solely from the unchanged input. Continuous research authorization remains active; no GPU job is implied by this note. Existing results, scientific gates and the prohibition on rescuing R61/R62 by post hoc sweeps remain intact.
