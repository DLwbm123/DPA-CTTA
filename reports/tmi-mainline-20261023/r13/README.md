# r13 frozen diagnostic closeout

Only r13 was authorized and executed. The retained preregistration includes a prospective r14 design; r14 remains unauthorized and unstarted. The original r01–r12 window is closed.

See [Chinese report](REPORT_ZH.md), [aggregate results](REPORT.json), [authorization](AUTHORIZATION_ZH.md), and [original protocol](PROTOCOL_ZH.md). SOURCE/DIRECT/FULL/CONTROL comparisons use repeated source-development content, not independent patients or target validation. Boundary comparisons use paired finite coverage and report empty predictions separately.

## Reproduction dependencies

The published diagnostic worker depends on the previously published [r03 expert](../src/train_r03.py) and [r12 fusion](../src/train_r12.py), legally supplied source data/backbone code/weights, and private r12 run artifacts (split, canonical reference, checkpoints and historical readout rows). Exact data and checkpoint membership cannot be reconstructed from public aggregates. Patient identity and backbone training membership remain UNKNOWN.

The worker reads WORK_ROOT from the environment and config.private.json within it. Required dependencies are data_root, backbone_code, checkpoint, expert_module, expert_checkpoints keyed by seed, r12_root, fusion_module, and seeds. The supervisor also uses python and neutral_train. These are locally supplied paths; private configuration is excluded. The original runtime receipt must be retained for a retry; this completed experiment must not be restarted by reusing its expired budget. An independently authorized reproduction needs its own prospective authorization, budget and ledger.

The aggregator consumes functional.private.json, credit.private.json, metrics.private.json, actions.private.jsonl and FINISHED.json and emits only anonymous content-level summaries. Private inputs are excluded. No diagnostic or scoring rerun was performed during migration; the saved analysis and receipts were reviewed.

## Published scope

Only aggregate metrics, original protocol/authorization, deidentified data-use limitations, environment/completion/exit/reproduction receipts and source scripts are included. No images, masks, weights, cached probabilities, private rows/actions, identifiers, private configuration or host paths are published. Runtime is 541.337 seconds, allocated output 1,353,658,368 bytes, and peak allocated GPU memory 447,992,320 bytes. The 81,920 local action outputs are not independent cases; 1,920 additional equivalence-check reconstructions and shuffle/broadcast reconstructions are separate from model forwards.
