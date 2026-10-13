# Repaired shared descent: completed real-model GPU profile

**PASS within the frozen profile. No formal experiment or efficacy scoring started.**

Original source ResUNet, batch1 RGB512,19,136 BN-affine parameters. Eight fresh-source jobs on physicalGPU0/1 only; two existing authorized order prefixes,64 arrivals per job,4 warmup+60 timed. GPU0 SHARED then repaired; GPU1 reversed.512 total arrivals/480 timed; repaired condition256 arrivals, all logged and checked. The repeated GPU runs are not independent subjects.

## Original tolerance and numerical repair

Tolerance remains1e-5*max(abs(native_proposal_dot),1e-10). Of256 repaired-condition updates,254 native proposals were negative and all254 passed the committed derivative inequality. Two native proposals were non-descent and were preserved exactly, as registered. All256 final derivatives were finite. **Zero final violations**, no failed/aborted jobs, no tolerance relaxation or automatic retry. The largest positive final gap/tolerance ratio was0.895653 (<1).

Closed-form constraint triggered220/256=85.9375%. Parameter-write repair restored native Adam on16/256=6.25%; the two pre-existing non-descent fallbacks add2/256=0.78125%, so total native fallback18/256=7.03125%. Repairs per64-arrival job were5 on order0 and3 on order1 on each GPU. The mechanism was exercised, not merely enabled.

On the16 repaired events, the returned-filter gap/tolerance maximum was0.000320373, while the parameter-write candidate gap/tolerance ranged1.001971–1.890740. Thus these observed violations arose after the returned displacement was added to parameters and remeasured. Exact native restoration reduced every repaired gap to zero. This is direct real-model evidence of the write-rounding failure class. The first repair on each order matched the old R47 failure visit16/42; historical missing R47 derivative values were not fabricated or recovered.

Native/returned-filter/pre-repair-committed/final derivatives, unconstrained gap, multiplier and tolerance are stored in private step traces. Public NUMERICAL_AGGREGATES.json and RESULTS.json contain anonymous distributions only. Final actual derivative mean-1.041605e-3, median-4.493191e-4, range[-9.359534e-3,+5.591408e-6]. The positive maximum is the preserved native non-descent proposal, not a guard violation. These are proxy-loss directions, not Dice or anatomy outcomes.

## Timing and memory against R46 SHARED

| GPU | Order | SHARED mean s/image | Repaired mean s/image | Wall overhead | Allocated MiB | Reserved SHARED→repaired MiB |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0.648792 | 0.663639 | +2.289% | 816.875 | 896→898 |
| 0 | 1 | 0.653812 | 0.661938 | +1.243% | 816.875 | 896→898 |
| 1 | 0 | 0.657965 | 0.658522 | +0.085% | 816.875 | 896→898 |
| 1 | 1 | 0.655122 | 0.655612 | +0.075% | 816.875 | 896→898 |

Equal-sized pooled timing: 0.653923→0.659928s/image, +0.918%. Matched ratios range+0.075% to+2.289%; CUDA-event ratios agree closely. Peak allocated816.874512MiB is identical; reserved896→898MiB (+2MiB,+0.2232%). Both conditions retain10F/5B/1Adam per image. No extra model pass, backward, optimizer step or model state. Shared-GPU load/dispatch noise prevents treating the small timing differences as exact overhead guarantees.

## Coverage, costs, delivery and limits

All8 receipts PASS, all8 private numerical traces have64 sequential visits, all online/profile/watch processes retired. Existing R47 remained INCOMPLETE with24 complete/2 failed/4 unrun jobs and0 scores; its full retirement was explicitly admitted. No old protocol, state, cost, partial output or GPU assignment was modified. No other task was terminated or restarted.

Independent profile T0:2026-10-13T08:49:35.223821+00:00. End2026-10-13T08:54:40.469241+00:00. GPU-worker cost470.969191s / 0.130824775h, including initialization and all jobs, within3600s total/450s per-job/2h wall/1GiB. Server41 CPU checks PASS22.228454s; local15 PASS3.586s, GPU0 for preparation. Previous cumulative GPU98.8653738331h plus this profile gives98.9961986083h. No formal experiment cost was incurred.

Own source/protocol and these anonymous aggregates/report are delivered on feature/shared-descent-repaired-profile. Source frozen before GPU execution:4073c12a684034137f1daec4d5888207bdc0f857. Step traces remain private; no labels, images, per-content predictions/scores, identifiers or hashes, model states, private paths/logs or third-party PDFs are public. There are no scientific positive/negative cells because no score was computed.

Passing these64-arrival prefixes does not prove that every future real-model update is safe, that6.25% is a population fallback rate, or that anatomical protection/Dice improve. The repair can reduce joint-filter activity by retaining native Adam. Future GPU qualification or a formal experiment needs separate authorization and freeze; no next round or monitor was started.
