# Development campaign closeout after R31

The completed experiments do not support another automatic run on the current tested mechanisms. R31 is the last registered matrix; no R32 is started. Hourly monitoring is removed after public delivery because no campaign workers or justified next experiment remain. This decision is scientific, not a GPU-budget stop. All historical data and runtime artifacts are retained.

## Evidence used

- [R24](../r24-c-anchored-context/REPORT.md): the conditional adapter failed its advancement gate; ANCHOR was not better than matched VIEW, and the ANCHOR-C effect changed sign by order. A larger C learning rate was already tested.
- [R25](../r25-parameter-policy/REPORT.md) and [R26](../r26-residual-action/REPORT.md): parameter/residual exploration and policy mechanisms failed their frozen development gates. REPEAT5 was substantially worse, so simply adding updates has adverse evidence.
- [R27](../r27-nesting-selection/REPORT.md): nesting-based selection did not provide a supported improvement.
- [R28](../r28-common-state-diagnostic/REPORT.md): same-state residual-candidate oracle headroom was0.080122pp, below the0.3pp allocation gate. This finite candidate set did not justify a learned selector.
- [R29](../r29-update-strength-diagnostic/REPORT.md): immediate FULL/ZERO/HALF oracle headroom was0.010225pp. [R30](../r30-state-history/REPORT.md) extended the question to delayed effects: H64 oracle headroom remained0.009007pp and state-reset candidates failed. Persistent C learning did help relative to episodic C, so the finding is not that cross-image adaptation is useless.
- [R31](REPORT.md): allowing the66 existing final-classifier parameters to adapt produced only+0.003452pp versus C_CONT and-0.120867pp versus ANCHOR on SEARCH, with only3/6 positive trajectories and mixed order effects. The classifier demonstrably updated.

These are separate protocol-specific findings; values from different hardware/rounds are not pooled into a meta-effect. The evidence does not establish ANCHOR as a robust new method: its small average advantage over C also changes sign by order. C_CONT remains a useful simple reference.

## Why not continue automatically

The available evidence gives no positive mechanistic signal for increasing update effort, tuning the tested action/strength/reset families, or training an RL selector. A larger classifier LR or a different unfrozen backbone block would currently be an ungrounded extension of a negative fixed-setting result. Absence of support does not prove that those possibilities cannot work; it means this campaign has not supplied a defensible reason to allocate another automatic experiment to them.

All SEARCH and legacy REVIEW content is development-exposed. Additional seeds or relabeling REVIEW cannot create independent confirmation. Patient linkage and ROI provenance remain UNKNOWN. No new patient data, labels, source training or external model access is assumed.

A future restart should first supply a distinct mechanism with a concrete observable failure it addresses, or an authorized independent evaluation/data-provenance plan, and freeze its protocol before execution. No such new plan is silently substituted here. No broad impossibility, clinical utility or independent-generalization claim follows from this stop decision.
