# Mixed-device qualification, declared before execution

User: 请你自己解决，之后直接开始实验. The failed all-GPU attempt is preserved; its tolerance is not relaxed. One bounded localization used 16 forwards / 2 backwards / 2 Adam: identical initial tensors and all eight inputs, seven pre-update outputs within original tolerance, final post-update output with 12 violating elements. This supports routing the historical adaptive baseline on CPU; it does not prove a particular CUDA kernel defective.

New route: C_BASE uses unchanged original CPU Host, all other arms use the source-qualified CUDA backbone and unchanged CPU methods/FP64 state. Source artifact/context bytes stay unchanged. One worker, physical GPU6 proposed. Fixed serial24 jobs and all scientific semantics/call budgets unchanged. GPU baseline is rejected explicitly. This prioritizes numerical preservation over maximum wall-clock speed.

New procedural qualification budget: C_BASE3 CPU visits plus one fresh CPU repeat =32F/4B/4Adam; require first-visit CPU repeat bitwise equality. C0 three GPU visits plus two repeats =5F. Six methods three GPU visits =36F; fresh STATIC comparisons =6F. Total79F(32CPU+47GPU),4B/4Adam, no VJP/AdamW;600s,64MiB output. No real images/checkpoints. Exercise fresh static equivalence, full persistent counters, six serialization/nine reloads, frozen backbone. Old GPU C_BASE numeric failure remains FAILED, never reclassified by this qualification. Source-side GPU bridge numeric result remains separate prior evidence.

Run only affected target synthetic acceptance (30 checks); the unchanged math/context/source code has121 prior checks from the preceding SHA. No whole166 suite, no source training or target run in qualification. Existing six real roundtrip results remain valid evidence for the unchanged loader/context/network modules; verify no diff there rather than rehash/reload assets repeatedly.

Formal launch still requires an externally reviewed exact binding. This new user request authorizes the repair and the requested experiment but is not itself an external PASS. Do not invent a reviewer or replace the review gate. Monitoring remains deleted.
