# CPU acceptance coverage

Final evidence applies to the fixed `IMPLEMENTATION_SHA`, not the later documentation publication commit. The final production suite has **47 unittest methods** (15 numerical,29 contract and3 full-network methods). The first full-network method independently exercises all three groups. The **22 independent NumPy reference checks are additional and do not replace implementation acceptance**. Actual per-environment results and costs are in `CPU_RESULTS.json` and `logs/`. Measured independent-reference errors from an additional matrix-only diagnostic are in `REFERENCE_ERRORS.json`; it has no model calls.

| Area | Runnable evidence |
|---|---|
| Independent A filter/covariance/subspace | `tests/r7/test_math.py`: precision reference, independent SVD covariance truncation, full-F offdiagonal P, SPD/high-R, rank failure, ambient1024 basis and projection |
| B calibrated finite correction | Same file: five-step NumPy agreement, kappa≠1 eta/energy, diagonal closed form, source gradients, spectral/column normalization and first descriptor difference |
| C robust estimates | Same file: independent IRLS, outlier weights, monotone fixed convex energy, ridge/zero residual, observation duplication mean scaling, variance bounds and geometric-mean constant |
| FiLM/observer | `test_contract.py`: bitwise zero and live derivative, hooks/current BN, dimensions/cache release, one-time fit-only scaler |
| RNG/state/GT | Independent same-seed FULL/STATIC, no global RNG use, image-only online signature, save/reload, changed future/GT, query-label and query-image isolation |
| Source fit/cal | Separate fold/role/oracle checks; sixteen actual combined oracle updates; independent STATIC source microfit; four-time unroll; task freeze during calibration; complete unperturbed cal observations for constant R; exact caps and first-error stop |
| Online/no fallback | Two forwards each A/B/C, C0 one; no online autograd; missing C constantR rejection; batch/nonfinite rejection; final-forward failure prevents commit/retry |
| Asset/IO/binding | Frozen inference digest/eta, disabled real scopes before any asset read, fresh owner, path escape, source/output overlap before writes, symlink/hardlink rejection, fixed ordinary aggregate with metadata-only publication aliases |
| Metadata/report |24/9/≤12 matrix costs, source PENDING, descriptive FULL−C and FULL−STATIC, paired tails, ASSD common-valid/undefined, no automated gate |
| Full random original ResUNet34 | `test_full_network.py`: each A/B/C one differentiable4-time source microepisode(12F), one four-observation calibration step(4F), two online visits(4F), required new-module gradients, frozen backbone unchanged |
| Full zero/old C | Original complete network bitwise FiLM zero, two nonzero readout VJPs through both sites, C0 single forward; two original-C paths bitwise identical, each8F/1B/1Adam |

The complete random backbone has22,554,570 parameters. New method parameters including observer/calibration: A23,633; B25,281; C48,696. Full model acceptance is at512×512 and is not a renamed toy model. The small fixture is explicitly labeled Small and separately counted.

Both final environments use two PyTorch CPU threads. CUDA discovery/init and asset loaders are disabled in the runner; legacy `is_available` calls receive a fixedFalse without driver discovery. No GPU state is queried. All real-source/target/smoke commands raise before loading. The guard is a procedural acceptance harness, not an OS security sandbox.

Per final implementation suite: R7 backbone448F plus historical-C16F = **464 physical backbone forwards**;36 actual Tensor.backward calls,2 autograd.grad calls,24 Adam steps and6 AdamW steps. Of these, full ResUNet79F (63 new/shared +16 old C), with the remaining385F on the explicit small procedural fixture. New source task microfits6, oracle updates16, calibration updates6 and old-C updates2 explain the30 model-related backward calls; the remaining6 are small-matrix/FiLM gradient checks. The2 VJPs are full-model zero-FiLM checks. Actual logs remain authoritative. No new model calls are reported as zero.

Development failures are retained, not rewritten as passing: initial NaN eta sentinel equality and two fixture path/API errors; subsequent dictionary norm precision test and snapshot-in-source fixture failure. B dictionary and C evidence row normalization were moved to float64 before small-matrix inference; the source/output fixture was separated. A new production wrapper now rejects source/output overlap before any write. An initial server Git alternates path failed because the donor was a worktree; only the fresh deployment's reference was fixed, then exact commit identity verified. No real experiment was retried.

No full166 historical model regression and no real source training, checkpoint run or target experiment occurred. Scientific effectiveness and external review remain untested.

Full-network wiring uses explicitly procedural basis/scaler/oracle tensors. A's predictive-basis construction is separately tested on synthetic Jacobians (including ambient1024), with two real random-network pooled-logit VJPs verifying live site derivatives. It does not claim to have built the true source basis or executed the future1024 source VJPs. Tokens use population std clamped below at1e-6; this epsilon placement is recorded as an implementation convention for the supplied normalization specification.

The first final-SHA server suite reached all three complete-network group checks but failed the old-C test because the existing batchgenerators site-packages directory was absent from sys.path. Its47-test/one-error log is retained. The final server rerun adds only that existing pinned0.25.2 dependency directory; no packages are installed and no scientific code changes.

`LATENT_COST.json` adds an actually measured procedural-feature update per group, with matrix API counts including SPD validation factorizations (not just solves) and module calls. It excludes backbone forwards, gradient/optimizer calls and initialization. Raw matrix-reference and latent-cost diagnostic logs are also retained. These small extra checks do not represent real source/target execution.
