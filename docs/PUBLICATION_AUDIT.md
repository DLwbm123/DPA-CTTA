# Publication audit — 2026-09-07

Scientific status: **CODE_REVIEW_PENDING**. This is publication for independent
review, not permission to train. Publication/remote receipts are recorded separately
in DELIVERY_RECEIPT.md after a successful push; the current commit is identified
by Git, never by substituting the historical audit hash.

## Provenance and scope

- Repository: https://github.com/DLwbm123/DPA-CTTA (existing public repository).
- `original_audited_sha`: `617f67464df37d6e49972c4932a85f56d210ccab`.
- Original full Git tree: `83d099cd6093103e81cf5798f5a6a98bc6f51426`.
- `publication_source_tree_hash`:
  `7180103d06310cea32d70c05e24da3028047bffc87a7a086a474c0899adb2ca3`.
  This is a SHA-256 over the canonical scientific file manifest, not a commit SHA.
  Its precise encoding and all 34 file digests are in
  [publication_source_manifest.json](../audit/publication_source_manifest.json).
- Exported only tracked files using `git archive` from the explicit original commit.
  The original clean workspace and commit were not edited. The publication is
  initialized independently, with no parent containing the original workflow.
- All 34 existing files under src/configs/tests/scripts plus pyproject.toml match
  the original commit byte for byte and match their historical audit manifest.
  The current request explicitly required this per-file source comparison.
- `.github/workflows/unit-tests.yml` moved byte-for-byte to
  `ci_templates/unit-tests.yml.disabled`; no active workflow directory is published.
  **CI=NOT_CONFIGURED**. No green CI or external signoff is asserted.
- README receives a publication banner. Other inherited files remain unchanged.
  Additions are the research decision, supplied revised plan, review index, this
  audit, the new audit-only CPU runner, its log/JSON, provenance manifest, and
  publication receipts. No scientific algorithm or existing test was patched.
- Original `PRETRAINING_AUDIT.md`, `audit/pretraining_audit.json`, and
  `audit/file_manifest.json` are historical evidence, including the original
  workflow path and old README digest. They are not manifests of the new root.
  The historical pre-generated tree `749f57705a1c10bae300138ffdeb6c275f2df9bf`
  is not the original final tree or this publication tree.

## Executed CPU checks

Existing environment: Python 3.12.9, PyTorch 2.6.0. No dependencies installed.
Reference checkout: `DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`,
checked clean before import. Public model source was reused through the existing bridge.

The actual command, with machine-local paths replaced by environment placeholders:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
DPA_CTTA_BASE_ROOT="$REFERENCE" "$PYTHON" audit/run_publication_checks.py \
  > audit/publication_cpu_tests.log 2>&1
```

`$PYTHON` denotes the existing Python environment; `$REFERENCE` is the pinned
CTTA checkout. Run from the publication root. The audit runner imports this
snapshot's src explicitly, discovers the unchanged tests, and sets two CPU threads.
It does not install packages or initialize CUDA.

- Exit code: 0. Existing unittest suite: **143 tests, 0 failures, 0 errors, 0 skips**.
  Suite duration: 5.128 seconds, excluding the additional focused checks.
- Real ResUNet34 and PraNet initialization/forward/JVP checks ran on synthetic CPU
  tensors, without checkpoint or download. They are not toy substitutes.
- Complete descriptor maximum difference for `z_prev` in {0, 3, -2}: **0.0**.
- Fundus input-gradient norm: `2.2644035198027268e-05`;
  zero-state latent-gradient norm: `0.0007135490886867046`.
- Polyp input-gradient norm: `59737.0546875`;
  zero-state latent-gradient norm: `34.961273193359375`.
  These random-network magnitudes demonstrate finite nonzero connectivity only;
  they are not performance, stability, or comparable task-scale measures.
- Source parameters and registered buffers remained unchanged in both focused
  full-model checks. The upstream source-only loader replaces custom BN with
  standard BN; native custom state is outside this evidence.
- `training_executed=false`, `real_data_accessed=false`,
  `source_checkpoint_loaded=false`, `cuda_initialized=false`.
  CUDA was checked in the actual test process before and after execution.
  The runner blocks PIL image opens, Torch checkpoint loads, and standard Torch
  download entry points in-process. Existing CLI tests only exercise dry-run and
  rejection paths; their synthetic oracle checks are unit checks, not M0 training.

Evidence: [machine results](../audit/publication_checks.json),
[full test output](../audit/publication_cpu_tests.log),
[runnable audit](../audit/run_publication_checks.py).

## Repository command record

This table records repository-related preparation/checks. Absolute local paths
are replaced by `$ORIGINAL`, `$PUBLICATION`, `$REFERENCE`, `$PLAN`, or `$PYTHON`;
identity and authentication secrets are not reproduced. Local skill/memory reading
is operational context and is not part of the public payload. Later Git commands
and their actual exit codes are captured in the delivery receipt.

| Operation actually executed | Exit/result |
|---|---|
| `git status --short`; `git log -5 --oneline`; `git remote -v`; `git rev-parse HEAD` in original | 0 each; clean, one original commit, expected origin |
| `git config user.name`; `git config user.email` | 0 each; existing user identity retained, output omitted |
| `gh auth status` with Token line filtered from displayed output | 0; active DLwbm123, repo/read:org/gist scopes; no scope expansion |
| `git ls-remote --heads origin` in original, initial read | 0; empty |
| `git ls-files`; reads of README, METHOD, INJECTION_CONTRACT, PRETRAINING_AUDIT, implementation contract, audit JSON/manifest, source/config/test/script files | 0; inspected current implementation and audit provenance |
| `git -C "$REFERENCE" rev-parse HEAD`; `git -C "$REFERENCE" status --short` | 0 each; expected pin, clean |
| `rg` for weight/download calls in OPTIC and `VPTTA/POLYP/lib` | 2; latter directory does not exist; corrected to actual POLYP/networks below |
| `rg` for `pretrained`, `load_url`, `torch.load` in POLYP/networks/PraNet_Res2Net_TTA.py and Res2Net_v1b.py | 0; constructor uses pretrained=False |
| `rg` for private paths/credential terms in original tracked files | 0; only benign identifier `tokens` matched |
| `ls -l` existing Python executable; read upstream ctta_suite/models.py | 0; existing runtime and source-only construction inspected |
| Python export: `git rev-parse HEAD`; `git status --porcelain`; `git ls-remote --heads origin`; `git archive "$ORIGINAL_SHA"` | 0 each; gates passed, fresh directory, tracked-only archive |
| Python archive extraction, workflow template move, revised-plan path screen and copy | 0; no source workspace mutation |
| Add publication README banner, research decision, review index, audit-only CPU runner and this report | tool success; documentation/audit additions only |
| CPU publication runner command above | 0; 143 passed plus focused checks |
| Python comparison: `git ls-tree -r --name-only "$ORIGINAL_SHA"`; per-path `git show`; `git rev-parse "$ORIGINAL_SHA^{tree}"` | 0 each; 34 scientific files matched, inherited file exceptions declared |
| `gh api repos/DLwbm123/DPA-CTTA` with only full_name/private/default_branch/size/push fields | 0; existing public repository, size=0, push=true |
| Official PMLR Kang 2023 abstract read via web tool | success; prior-art statement supported, no shell exit code |

## Public safety and unrun work

Only the exported tracked source and declared documentation/audit additions are
eligible for publication. No real images, individual masks, weights, feature
tensors, private data, connection credentials, or third-party source were copied.
The supplied plan was screened for private machine paths and identifying sample
content; its dataset/domain names and prior aggregate metrics are supplied context.
The final payload safety scan and `git diff --check` results are recorded with
the publication commands. Git author/committer use the user's existing identity,
as requested; identity was not fabricated or replaced.

NOT_RUN: all experiments (M0, source pilot, coreset training, DD, target mini,
full target, multi-seed); any checkpoint-based integration; native VPTTA custom
counter/buffer replay semantics; end-to-end four-term DD training; revised modes
and medical/boundary-loss training; independent code review. Reasons: outside
this round's authorization, absent implementations, or tests of a different host
than the preserved Atlas. No validation of historical plan metrics was attempted.

Unresolved review points are mapped to actual code in
[CODE_REVIEW_INDEX.md](CODE_REVIEW_INDEX.md): differing anchor/target anatomy
inputs, free-chart confounding of DD benefit, incomplete training entry points,
native custom-state coverage, and historical audit-generator limitations.
The next action is independent review of the published commit, followed by
explicit selection and authorization of a commit/configuration before any experiment.
