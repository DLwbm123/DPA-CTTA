# R7 SOURCE_PREP GPU transition

User explicitly authorized stopping and preserving the CPU attempt, adapting and validating GPU execution, and a fresh complete SOURCE_PREP attempt on physical GPU 5/6/7. This supersedes the old CPU-only restriction for this source stage. No new external GPU review is claimed.

## Implementation and validation

Implementation: `ba04ca46a6bae5df4f4ebb272ef4045f5ee400e5`. This report's publication commit is not the executable commit. All tests ran on that clean implementation on Linux/Python 3.10.6/Torch 2.2.1+cu121. No local macOS test run is claimed.

The frozen network is placed on GPU 5. CPU preprocessing, local seeded source transformations, small method computations, FP64 latent algebra, source labels and loss remain on CPU. Tensor copies preserve autograd to CPU FiLM parameters. FP32 network kernels use no AMP or TF32; deterministic algorithms are enabled. CPU contexts retain their old structure, while GPU contexts explicitly distinguish their execution backend. The historical CPU review is preserved separately; the new receipt says USER_AUTHORIZED_GPU_QUALIFIED, not external PASS. Parent supervision, source same-byte readers, path isolation and complete phase budgets remain enabled. Legacy CPU-only prose in the original runner module header is superseded by its explicit optional device policy.

90 CPU tests passed: 34 execution-boundary checks and 56 math/contract/context checks. Zero failures, errors or skips; 2.196 s and 20.594 s respectively. The three old test_full_network methods and historical 166-test suite were not rerun; dedicated GPU full-network tests cover the changed route. Failure messages deliberately exercised in boundary tests are fixture evidence, not a failed real run.

15 GPU checks passed in 17.128 s on RTX 3090, including CPU/GPU logits, observer and FiLM gradient comparison; repeatability; context separation; a full 16-step synthetic oracle; one actual fit/calibration step and tensor package reload for each of six FULL/STATIC configurations; two VJPs and unchanged backbone. These microsteps are qualification, not completion of the 1000/256-step source training. Fixed comparison tolerances were rtol=0.003, atol=0.0003; maximum logit difference was 0.000198 and FiLM-gradient difference 5.26e-8. This supports the tested device bridge, not bitwise CPU/GPU or complete-training equivalence.

Qualification counted 133 backbone forwards (one CPU, 132 GPU), 22 source backwards, six calibration backwards, 16 source Adam, six AdamW, six calibration Adam, and two explicitly counted VJPs. In addition, the CPU/GPU comparison directly called autograd.grad twice; those probe calls are not included in the source_VJP counter. Peak GPU allocated/reserved memory was 1269121536/1392508928 bytes. No real image/mask/checkpoint was used in qualification. Logs and result JSONs are included.

## Preserved CPU attempt and GPU startup

The CPU attempt stopped by explicit user request after 4335.491 worker seconds, 4136 forwards and 2067 backward/Adam calls. Its native records retain FAILED/KeyboardInterrupt, parent exit 1 and child exit -2; the user-stop reason is a separate sidecar. Source after-check UNCHANGED; owned processes absent. This is an interrupted attempt, not a scientific failure or successful preparation, and its costs are not subtracted from the new run.

The new attempt started 2026-09-19T06:45:57Z. At 06:46:57Z, parent and owned worker were live; GPU 5 showed the owned worker using 1158 MiB. Decode/model_load/identity were complete with no error records. This is startup evidence, not completion. The same 111/23/25 split, science, seed, 135328-forward full budget and six independent FULL/STATIC artifacts are required. Patient/eye dependence and checkpoint pretraining exposure remain UNKNOWN. Source bindings were reused without another full asset pre-audit; original readers verify before decode/load and after execution.

The run is single-worker and has no automatic retry, resume or target dispatch. Terminal success still requires parent exit 0, mandatory exact phase counts, unchanged source checks, six artifacts and contexts, trusted loading, terminal resource checks and completion publication. Runtime data, original CPU evidence and absolute paths stay private. Existing hourly monitoring follows the new receipt.

## Review index

- DELIVERY.json: observed status and exact implementation/receipt identities.
- check_source_prep_cpu.py.json/.log: execution boundary acceptance.
- check_cpu.py.json/.log: math, lifecycle and context regression.
- check_source_gpu.py.json/.log: procedural full-network GPU qualification.
- implementation.patch: complete narrow code/test delta from the old reviewed implementation.

SOURCE_PREP=RUNNING; TARGET_SCREEN=NOT_RUN; TARGET_MECHANISM=NOT_RUN; TARGET_EXTENSION=NOT_RUN; external_gpu_review=NOT_RUN; next_scope_authorized=false.
