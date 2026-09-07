# Source pilot preparation report

**SOURCE_PILOT_PREPARED — AWAITING EXACT COMMIT/CONFIG REVIEW**

Branch: `experiment/source-pilot-io-v1`, from `c92b64b2284eb5da0631c00d15cc44af773f4b2d`. The commit containing this report is the candidate for review; the final delivery names its actual Git SHA. `main`, the previous review branch and the pinned CTTA checkout were preserved. No workflow was added; CI is **NOT_CONFIGURED**.

Draft configuration: [source_pilot_v0.json](../configs/source_pilot_v0.json), enabled=false, CPU. Exact file SHA256: `e1aeb2454899d024e00934e1f012b5a0fe8cb3bb656f5dd20f73468fdd7a3faf`. This hash is an identifier for future exact-config review, not permission to execute it.

## Implemented scope

Added strict in-memory source loading before clone, device placement before Adam, and first-live-forward state/storage/optimizer audit. Added source-only CSV/frozen-group registration, pixel and nearest-mask readers, installed-SciPy final-grid distances, correct No Adapt source BN, and sequential N/A/B/C assembly with post-step query evaluation and paired per-channel summaries. The real runner remains unconditionally blocked. No Atlas/PIPDE/core-loss redesign was made.

The supplied second-review report, probe code and result file were read as evidence about the baseline. Their PASS does not approve this new runner. No independent reviewer acceptance was generated here. The phase uses the existing Python 3.12.9, Torch 2.6.0, SciPy 1.16.0 CPU environment; no dependency was installed.

## Observed commands and results

Set `$REFERENCE` to the pinned local CTTA checkout and `$PYTHON` to the existing supported interpreter. Paths below are public-safe placeholders. Run from this repository; CLI commands additionally use `PYTHONPATH=src`.

| Command | Exit | Result |
|---|---:|---|
| `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_source_pilot_checks.py` | 0 | 166 passed; 315.71 s |
| `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_source_pilot_checks.py --follow-up` | 0 | 9 passed; 60.73 s |
| `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_source_pilot_checks.py --source-io-follow-up` | 0 | 1 passed; 0.45 s |
| `$PYTHON -P -m dpa_ctta.source_pilot prepare` | 0 | Config parsed; NOT_RUN |
| `$PYTHON -P -m dpa_ctta.source_pilot dry-run` | 0 | Config parsed; NOT_RUN |
| `$PYTHON -P -m dpa_ctta.source_pilot run` | 2 | PILOT_COMMIT_CONFIG_APPROVAL_REQUIRED |

The initial full suite ran **166 tests**. Later targeted runs covered newly added K=4/failure checks, changed runner summaries, final-device name resolution, and the positive registered proxy reader chain. The retained logs cover **168 distinct passing tests**; these counts come from observed runner results/log IDs. This is not a claim that a final 168-test suite ran again in one invocation. The unchanged 45-image native test was not repeated. Earlier local smoke runs (5 then 6 I/O checks, and an 8-test follow-up before final device normalization) also passed; the retained targeted reports supersede those preliminary checks.

Audit files: [full suite](../audit/source_pilot_results.json), [follow-up](../audit/source_pilot_followup_results.json), [positive source I/O](../audit/source_pilot_source_io_results.json), [CLI payloads](../audit/source_pilot_cli.json), and [combined machine audit with test IDs and file inventory](../audit/source_pilot_prepare_audit.json). Logs accompany each CPU report. `torch.load`, CUDA lazy initialization, downloads and in-process socket connects were denied in the audit; image opens accepted only encoded procedural BytesIO streams. No blocked API attempts occurred. Subprocess checks used existing inspected CPU CLI/local synthetic Git fixtures, not server commands.

## Important evidence

- Both full native model families loaded deliberately noninitial convolution weights and running statistics into equal, disjoint, frozen clones. Strict malformed state and load-only-live counterexamples were rejected. CPU optimizer parameters are the actual final prompt objects; CPU:0 resolves to CPU. No CUDA claim is made.
- N matches the pinned full source-only standard-BN reference and an independent running-statistic BN expression. Native count-zero AdaBN is not used for N.
- Each task ran 45 distinct native low-frequency keys at neighbor=16: first retrieval at visit 17, retrieval through 45, evictions at 42/43/44/45, final memory 41, Adam 45, expected warmup counts. Direct native and wrapper logits/state/RNG matched after every step; source state was unchanged.
- B/C K=4 tests observed one actual auxiliary batch of 4 and one native sample/Adam increment per update. Gradients were finite/nonzero. Procedural replay changed fake query labels while preserving order, all predictions, prompt and Adam; scores changed. Nonfinite output and wrong-label shape stopped before a second query.
- Independent brute EDT references used 3x5, 7x4, 9x11 geometries, small/nested objects and empty/full channels. Encoded procedural RGB/GT verified channel decoding, interpolation and dimensions. Temporary source registries exercised deterministic disjoint selection and content/role/path rejection. The positive registered-proxy chain read exactly four images and four masks per task, no query labels, and computed final-grid SDF; these were procedural streams, not registered patient images.

## Remaining real integration / NOT_RUN

No server login, real image or checkpoint read, CUDA initialization, source training, pilot, DD or target run occurred. Private root/CSV/source manifest bindings, real cohort counts and content freshness, registered checkpoint identity/membership, GPU/native memory behavior and actual costs remain unverified. Existing frozen metadata digests are checked; actual image/model bytes are not rehashed in this phase. Missing or inconsistent real registration must stop the later task; these mock manifests cannot replace it.

Four transformations repeat source development groups. They are not independent patients or unseen-generalization evidence. This raw-source proxy mechanism control is not DD or strict source-free deployment. No performance benefit or scientific pass is claimed.

Both CLI `run` and direct `run_registered` reject **PILOT_COMMIT_CONFIG_APPROVAL_REQUIRED** before real I/O. No environment/config override grants approval. Review the actual containing commit plus the above config hash before authorizing a future change that enables only the small pilot.

## Changed files

| Path | Change |
|---|---|
| `README.md` | Modified |
| `src/dpa_ctta/hosts/vptta.py` | Modified |
| `audit/run_source_pilot_checks.py` | Added |
| `audit/source_pilot_cli.json` | Added |
| `audit/source_pilot_cpu.log` | Added |
| `audit/source_pilot_followup_cpu.log` | Added |
| `audit/source_pilot_followup_results.json` | Added |
| `audit/source_pilot_prepare_audit.json` | Added |
| `audit/source_pilot_results.json` | Added |
| `audit/source_pilot_source_io_cpu.log` | Added |
| `audit/source_pilot_source_io_results.json` | Added |
| `configs/source_pilot_v0.json` | Added |
| `docs/SOURCE_PILOT_IO_CONTRACT.md` | Added |
| `docs/SOURCE_PILOT_PREPARE_REPORT.md` | Added |
| `src/dpa_ctta/source_io.py` | Added |
| `src/dpa_ctta/source_pilot.py` | Added |
| `tests/test_source_pilot.py` | Added |

The public safety scan covers these changed/new text files, checks repository/branch boundaries and excludes credentials, private paths, patient/raw data, model/image binaries, oversized payloads and workflows. Its observed result is recorded in the machine audit. Git push and anonymous commit/code/config/report access are verified after the commit exists, and reported in the final delivery receipt; this document does not fabricate a pre-commit push result.
