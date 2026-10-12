# Joint sensitivity: real-model GPU profile only

Status: **COMPLETE_PROFILE_ONLY**. Formal experiment started: false.

Original source ResUNet checkpoint, batch1 RGB512; BN-affine CW_LSO versus CW_LSO_JS. GPU0 runs base then JS; GPU1 runs JS then base. Each fresh source initialization uses the same 12 prior authorized images and seed; 4 warmup,8 timed. No labels or predictions saved.

| GPU | Condition | Mean s/image | P95 s/image | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|
| 0 | CW_LSO | 0.4178 | 0.4380 | 588.0 | 624.0 |
| 0 | CW_LSO_JS | 0.6282 | 0.6461 | 816.9 | 896.0 |
| 1 | CW_LSO | 0.4465 | 0.4707 | 588.0 | 624.0 |
| 1 | CW_LSO_JS | 0.6439 | 0.6628 | 816.9 | 896.0 |

GPU0: JS/base wall time 1.504x; allocated 1.389x; reserved 1.436x.

GPU1: JS/base wall time 1.442x; allocated 1.389x; reserved 1.436x.

Charged GPU-worker seconds, including initialization and failures: 74.373.

R44 must fully retire before this queue initializes CUDA. Its configuration, artifacts, workers and budget are not modified. Other users may share these GPUs; contention and eight timed images limit transfer of the latency estimate. All profiles/failures remain reported. No efficacy gate, new seed matrix, promotion or automatic formal run.

## Engineering checks and interpretation

All four real-model profiles passed the frozen call-count checks: baseline8 forwards/1 backward/1 Adam step per image; joint module10 forwards/5 backwards (4 auxiliary VJPs plus the base backward)/1 Adam step. The real-model metric-norm contraction and non-BN frozen-parameter checks passed. Both joint profiles had a nonzero acquisition span of rank2 on all12 arrivals, so this measurement exercised the module rather than a rank-zero fallback.

Allocated memory increases by228.828MiB (+38.91%); reserved memory increases by272MiB (+43.59%). End-to-end latency increases by0.210419s/image on GPU0 and0.197393s/image on GPU1. Physical reverse-pass count is five times baseline, but measured total latency is1.44–1.50 times baseline because forward, base update and other work also contribute. This resource measurement does not test segmentation accuracy or anatomical protection.

Four bounded GPU tasks completed without failure; all workers and the queue retired. Total GPU-worker cost is74.372657s (0.020659h), including initialization and warmup. Server CPU qualification passed12 checks in8.815326s. Two earlier CPU packaging/import failures are retained: a missing existing R9 protocol dependency (elapsed CPU time uninstrumented/unknown), and a missing static runtime protocol asset (15.175003s CPU). Neither failure used GPU; dependencies were repaired before the original GPU queue launched. No GPU retries, parameter changes or formal trajectory launch occurred.
