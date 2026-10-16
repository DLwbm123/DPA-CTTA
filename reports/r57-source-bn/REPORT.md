# R57 source and current-image BN controls

{
  "incremental_over_source_supported": true,
  "mean_area_excess": 1.2911474788187363,
  "positive_pairs": 6,
  "independent_validation": false
}

SOURCE and BATCH are each one stateless stream reused across paired seeds/orders, not six independent runs. C retains its original update-before-predict convention. Bootstrap contrasts are descriptive and couple content across seeds/orders. No independent validation or algorithm improvement claim.

| Method | Primary macro Dice (%) | REFUGE_Valid OC pooled area/GT | ORIGA OC pooled area/GT |
|---|---:|---:|---:|
| SOURCE |68.203471|0.700080|0.338249|
| BATCH |75.079176|1.741451|0.636763|
| C |78.437246|1.991227|0.760854|

The fixed primary excess-over-source diagnostic passes6/6. Current-image BN already increases REFUGE_Valid OC area strongly; C adds further excess. Across all domains C improves macro Dice by10.233775pp over SOURCE and3.358069pp over BATCH, so returning to source statistics globally is not justified. The next fixed test will examine a50% source-statistics anchor against BATCH, with all negative domains retained. No mixing-weight search is authorized by that protocol.

Engineering: the initial CPU aggregate failed because new control records omitted containment/empty-foreground fields. Records were repaired and CPU-only scoring repeated after GPU exit0; original failure and first scoring records remain private. GPU inference was not retried. GPU process wall time including initialization109.859122s; total completion187.175833s. Stateless first-image replay passed;1951 images and35118 represented scoring rows complete. SOURCE/BATCH reused streams do not multiply independent evidence.
