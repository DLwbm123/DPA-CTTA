# B4 frozen C transfer: launch report

B4_RUNNING — formal results are not yet complete or available.

Startup check: detached launcher alive, 271 formal Fundus C0 records persisted, no failure artifact, and full process/GPU command-name audit passed. This is a startup snapshot, not a completion claim.

Execution commit: `e3c8f5ff80e626ee7e77bcbf75ebd99c71ffd1cb`.

Five new CPU tests passed in the original server environment. A single registered-weight GPU smoke passed with exactly 76 forwards, 8 backwards and 8 Adam calls. It verified unchanged Fundus C parity; independent Polyp reference versus new implementation (logits, affine, Adam and RNG); C0 identity-based forward/reverse statelessness and first-original-C parity; and source-eval adapter parity. PraNet has 153 participating standard BN layers and 66,218 trainable affine scalars.

The detached formal task is bound to physical GPU 5, selected within the user's authorized GPU 3–7 range. It runs Fundus C0 canonical, Polyp C0 canonical, then both complete Polyp C orders sequentially. Formal budget: 7,357 scored records, 32,585 forwards and 3,604 backwards/Adam calls. C0 produces 3,753 canonical records reused at 11,408 CPU reference positions; these are not independent repetitions. No performance gate changes execution. Engineering failure stops without automatic retry. Active GPU work is capped at six hours and private output at 256 MiB.

The existing B1/B2/B3/P2 code, assets and results remain unchanged. Only the prescribed existing development groups and source checkpoints are used. The experiment has not yet established transfer efficacy or an advantage over C0. Full scalar validation, aggregates and conclusions are pending actual completion.

Source code, configuration, CPU/smoke evidence and this launch report are public. Private target identities, paths and individual scalar records remain on the server. No target images, masks, prediction maps or adapted checkpoints are saved or published.

No U/S/I/G expansion, DD training, radius/LR search or new domain order is executed. The B2/B3 interval series remains closed.
