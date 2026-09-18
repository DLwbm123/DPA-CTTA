# Isolation and failure evidence

`preflight` validates authority, clean code, frozen science, exact matrix/seed, resources, source release, registration/recurrence, target root and fresh output before dispatch. Target path checks inspect metadata/stat only: all image/mask paths must be under the approved target root, in frozen target domains, without traversal, symlink ancestors, nonregular files or hardlinks. Actual bytes are read only inside an authorized worker, hash-verified before decoding the same bytes. Native image geometry is capped. The target reader records inode/device/size/mtime/ctime/link-count and rechecks identity plus content after use, including decode/online failures. Same-content rewrites fail immutability too. Metadata, deployed artifacts and checkpoint are checked again at job termination.

`online(host, image_records, image_reader, ...)` receives only image path/digest/geometry, never mask paths, labels, domains, subsets or scores. It calls the unchanged image-only `step`, checks nominal physical counts, and saves packed exact sigmoid>=0.5 decisions. After all 1951 online visits, frozen-state checks and hook removal run and all host/model/optimizer references are dropped. Only then does `posthoc` instantiate label decoding and call the existing evaluator. Binary decisions reproduce the existing evaluator's thresholding exactly (including near-zero float logits); full logits are not needed. Masks are scored for all visits privately; the unchanged report excludes non-remaining_dev rows. No score feeds back into an online step or model selection.

The shell uses existing BudgetOutput byte reserves and owned-process cleanup, adding output-directory identity/ancestor checks. Each job's log, predictions, scalar file and evidence share its cap; aggregate caps include all job trees. Supervisor checks during execution and after exit/cleanup, including already-exited children. JOB_COMPLETE appears only after terminal evidence writes and another resource check. Matrix completion follows all jobs, descriptive report, terminal evidence and a final resource check. Failed pending files are not completion. Cap overshoot between bounded polling intervals fails the job; these are supervised caps, not filesystem quotas.

First execution exceptions are retained separately from cleanup, after-read and persistence errors. Evidence failures are best-effort recorded and printed without replacing the first error. No file or worker retries occur. Process ownership is recorded before fallible evidence writes. Signals are deferred across spawn until the process is owned, then all active owned groups are cleaned; unrelated processes are untouched. OS kill/power failure is not claimed recoverable, and interrupted runs cannot resume under this protocol.

## Acceptance mapping

The new `tests/r7/test_target_screen.py` covers:

- disabled scopes before reads; wrong authority/review/code/resource/matrix/seed; pending artifacts; registration/recurrence/inventory mismatch;
- actual six-artifact trusted loader, context/weight tampering and mismatched backbone rejection; production host factory creates independent models/methods/states;
- image-only arguments, online mask-decode trap, procedural complete-job ordering, FULL persistence and STATIC reset;
- frozen nominal counts, exact serial and three-lane scheduling, 24 distinct job directories/owners/processes, no extra scopes;
- traversal/symlink/hardlink rejection, output overlap/replacement/link rejection, target after-read and same-content rewrite rejection;
- wall/output limits and unqualified GPU rejection, already-exited terminal audits, actual owned sleeping-child termination;
- first execution versus cleanup/evidence failure, no retry, no completion on failure;
- threshold/evaluator equivalence and descriptive absolute/FULL-C/FULL-own-STATIC fields without nomination.

Existing R7 regression supplies exact tensor math, full-network procedural checks, frozen inference contexts and historical C preservation (8F/1B/1Adam), plus C0=1F. Existing SOURCE_PREP regression covers the reused verified-reader/output/terminal helper behavior. Test hashes pertain to required binding/immutability contracts, not unrelated archival verification. No target dataset is used. Production 1951-visit execution, real source artifacts, target scores and GPU numerics remain untested by design.
