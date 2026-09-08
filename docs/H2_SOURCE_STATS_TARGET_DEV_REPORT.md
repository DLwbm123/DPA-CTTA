# H2 source-stats target-dev preflight report

**BLOCKED_TARGET_REGISTRATION — no H2 model execution.** The required development/reserved-evaluation role registration could not be established from the inspected metadata. This is an incomplete H2 experiment and a completed preflight report, not evidence for or against H.

## Provenance and delivered scope

Branch: `experiment/h2-source-stats-target-dev-v1`, created from H1 delivery `b42d38aef009de62a145b207ae30685a8c0ebf37`. The inspected H1 execution remains `daa95d25ad03e9240621c832d4e45d8ab188b1b2`; pinned dependency remains `dbff0d985c6c95345d9fb78f5b1daef57b392564`. H2 experiment execution commit: **none**. The delivery commit is the Git commit containing this report; it must not be described as an executed H2 revision.

Only this disabled H2 configuration, contract, preflight report and deidentified inventory/audit are added. H implementation, entry, new tests, registration receipt and GPU work are deferred at the prescribed registration hard stop. Existing source configuration, dependency, implementation and old branch history are unchanged.

## Evidence and registration gap

The original source registration matched its existing receipt, and the existing size/mtime checks passed for all 122 registered identities. This is reuse of recorded identity plus metadata checks, not freshly hashed equality of every asset. Original checkpoints, K=4 proxies and source queries were not replaced. Source queries are previous development material; checkpoint training membership remains UNKNOWN.

Seven target directories and CSVs were found. Historical and relocated whole-dataset manifests provide sample/domain/split, image/mask paths, recorded digests and image size. They do not provide target-development/reserved roles, target content-group IDs, or patient/video association declarations. A further bounded search of 175 small metadata files, alongside the inspected contracts and source registrations, found no applicable role registry. This does not prove no such registry exists elsewhere on NAS; no unrestricted NAS scan was performed.

| Task | Domain | Manifest rows / distinct recorded image-mask pairs | Original CSV split |
| --- | --- | ---: | --- |
| Fundus | REFUGE | 400 / 400 | train 320; test 80 |
| Fundus | ORIGA | 650 / 650 | train 500; test 150 |
| Fundus | REFUGE_Valid | 800 / 800 | combined 800 |
| Fundus | Drishti_GS | 101 / 101 | train 50; test 51 |
| Polyp | CVC-ClinicDB | 612 / 612 | train 478; test 134 |
| Polyp | ETIS-LaribPolypDB | 196 / 196 | train 152; test 44 |
| Polyp | Kvasir-SEG | 1000 / 1000 | train 800; test 200 |

These 3759 rows are **inventory, not eligible H2 groups or completed visits**. Recorded image and image-mask-pair overlap with each task's complete source manifest is zero, based only on existing digest strings. Selected target assets were not byte-revalidated, and group/association-level separation has not passed. Thus this check does not establish the required source/proxy/target-dev separation.

Existing audits and pinned readers document Fundus independent OD<255 / OC=0 masks and Polyp foreground>127 with nearest evaluation resizing. This is historical protocol evidence, not a new selected-asset audit; the alternative legacy Polyp interpolation is not adopted.

Historical full-target experiment records document prior evaluation of these domains. That history must be disclosed; it neither defines an eligible H2 development pool nor supports an untouched-final-test claim. Dataset train/test names cannot silently supply the missing roles, and REFUGE_Valid only has a combined split. Missing association fields mean unknown linkage, not independent patients/videos. The role gap alone invokes H2 prompt section 5.3; group identity/association registration also remains unresolved.

Required next input is an authoritative mapping of eligible development and reserved evaluation roles for all seven domains (including prior exposure), together with known patient/video links or an explicit protocol treatment when unavailable, and an existing target content-group identity rule. A revised protocol may resolve those gaps prospectively; this delivery makes no such amendment or selection.

## Actual stages, coverage and resource use

| Stage | Actual state | Command exit code |
| --- | --- | --- |
| Metadata inventory / source registration checks | Completed; source metadata checks passed | 0 |
| Target registration | BLOCKED_TARGET_REGISTRATION | Not applicable: preflight decision, no H2 runner invoked |
| CPU regression | NOT_RUN; 0 H2 tests | Not run |
| GPU smoke | NOT_RUN | Not run |
| Source N/H | NOT_RUN | Not run |
| Target N/A/B/H | NOT_RUN | Not run |
| Independent CPU metric reconstruction | NOT_RUN | Not run |

Selected groups: 0/224. Source records: 0/104. Target records: 0/896. GPU Adam calls: **0/830**, including smoke 0/106, source 0/52, target 0/672. H2 model/prompt/proxy forwards, proxy image processing, backward calls and memory pushes are all zero. GPU UUID and available memory were not queried because admission follows completed registration; this is not a GPU resource block.

No source N/H table or target scores exist. Per-domain/channel means, paired differences, empty/full counts, common ASSD cohorts, adverse tails, warnings and research screening are **not measured**, represented by JSON null rather than invented zero scores. H2 pipeline/host-step time and peak allocated memory are also not measured. A private 4764-byte metadata evidence file was written and read back in an owned 0700 directory with file mode 0600. No images, masks, weights or individual predictions were copied. Publication contains no private paths, asset IDs or per-asset digests.

For a future admitted run, N/A/H target adaptation uses the original source checkpoint without source sample rehearsal; B additionally retains the original four source images and labels and is a mechanism control under a different source-retention condition. The separate source check uses original source development images for evaluation.

## Research conclusion and stop

**NOT_EVALUATED.** H1 supports prioritizing this H candidate, not a claim of target benefit. No H2 result establishes net gain, reduced harm, mixed benefit or failure. The prescribed stages after registration were not entered. No background work, monitoring, automatic recovery or extra experiment was started.
