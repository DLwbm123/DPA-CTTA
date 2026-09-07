# Publication delivery receipt — 2026-09-07

**PUBLISHED_FOR_REVIEW — NO TRAINING STARTED**

- Repository: https://github.com/DLwbm123/DPA-CTTA
- Branch: `main`.
- Original audited commit: `617f67464df37d6e49972c4932a85f56d210ccab` (preserved locally).
- Independent publication root: [`0dc9072d0f433753603dcf55362ae76e7f38880c`](https://github.com/DLwbm123/DPA-CTTA/commit/0dc9072d0f433753603dcf55362ae76e7f38880c).
- This receipt is a documentation/evidence follow-up to that root. The final
  publication commit is the Git commit containing this receipt; its own SHA is
  not embedded in its contents. No original workflow history is inherited.
- The root has no parent; root push succeeded without force and remote main was
  verified equal to the local root. The subsequent receipt commit adds evidence
  and updates the audit link only; no scientific file changes.
- Original workspace remains clean at the original SHA (both commands exited 0).
- CI: **NOT_CONFIGURED**, verified through anonymous GitHub API: 0 workflows.
- CPU: **143 tests passed, 0 failures/errors/skips**, plus focused numerical checks.
  No training, data access, checkpoint load, or CUDA initialization.
- Scientific differences: **0 across 34 files**, checked against the original
  commit and historical scientific manifest. Workflow content is an inactive
  `.disabled` template. No source checkpoint, data, or model artifact is published.

## Actual root Git commands

All commands below ran in the fresh publication directory. Detailed output is in
[publication_delivery.json](../audit/publication_delivery.json). Git initialization,
existing user identity, and the payload safety scan are recorded in
[publication_preflight.json](../audit/publication_preflight.json).

| Command | Exit code |
|---|---|
| `git add --all` | 0 |
| `git diff --check` | 0 |
| `git diff --cached --check` | 0 |
| `git diff --cached --name-only` | 0 |
| `git commit -m Publish preserved Atlas implementation and independent review evidence` | 0 |
| `git rev-list --parents -1 HEAD` | 0 |
| `git ls-remote origin` | 0 |
| `git push -u origin main` | 0 |
| `git ls-remote --heads origin main` | 0 |

The final pre-push `git ls-remote origin` was empty. The `git rev-list` output
contained only the root SHA, confirming no parents. Root push and subsequent
remote comparison both succeeded. No token refresh or security setting change.

## Anonymous access verified after root push

Requests used Python urllib with a User-Agent and no Authorization header.
All five required commit-pinned files returned nonempty HTTP 200 responses.

- HTTP 200: [README.md](https://raw.githubusercontent.com/DLwbm123/DPA-CTTA/0dc9072d0f433753603dcf55362ae76e7f38880c/README.md), no Authorization header.
- HTTP 200: [METHOD.md](https://raw.githubusercontent.com/DLwbm123/DPA-CTTA/0dc9072d0f433753603dcf55362ae76e7f38880c/docs/METHOD.md), no Authorization header.
- HTTP 200: [CODE_REVIEW_INDEX.md](https://raw.githubusercontent.com/DLwbm123/DPA-CTTA/0dc9072d0f433753603dcf55362ae76e7f38880c/docs/CODE_REVIEW_INDEX.md), no Authorization header.
- HTTP 200: [PUBLICATION_AUDIT.md](https://raw.githubusercontent.com/DLwbm123/DPA-CTTA/0dc9072d0f433753603dcf55362ae76e7f38880c/docs/PUBLICATION_AUDIT.md), no Authorization header.
- HTTP 200: [online.py](https://raw.githubusercontent.com/DLwbm123/DPA-CTTA/0dc9072d0f433753603dcf55362ae76e7f38880c/src/dpa_ctta/online.py), no Authorization header.
- HTTP 200: [DPA-CTTA](https://api.github.com/repos/DLwbm123/DPA-CTTA), no Authorization header.
- HTTP 200: [workflows](https://api.github.com/repos/DLwbm123/DPA-CTTA/actions/workflows), no Authorization header.

## Review entry points

- [Review index](CODE_REVIEW_INDEX.md)
- [Publication audit](PUBLICATION_AUDIT.md)
- [Research decision](RESEARCH_DECISION_20260907.md)
- [Supplied revised plan](REVISED_EXPERIMENT_PLAN.md)
- [Online adapter](../src/dpa_ctta/online.py)
- [Atlas](../src/dpa_ctta/atlas.py)
- [Proximal update](../src/dpa_ctta/proximal.py)

CODE_REVIEW_PENDING remains the scientific status. Independent review must select
this published commit before proposing a minimal patch and an explicitly authorized
experiment configuration. No M0/source pilot/coreset/DD/target/multi-seed run started.
