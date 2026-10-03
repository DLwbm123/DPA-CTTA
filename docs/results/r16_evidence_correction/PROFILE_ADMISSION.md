# Source-only resource admission

Full matrix admitted before any target access. Safety factor1.3; at most2concurrent GPU workers from pool4/5/6/7; original T0 and cumulative costs retained.

|Tier|Source GPU-worker seconds|Target GPU-worker seconds|Total incl charged|Decision|
|---|---:|---:|---:|---|
|FULL|8339.018|20020.784|29303.184|PASS|
|SHORT|8339.018|11026.943|20309.344|PASS|

FULL is selected; SHORT remains solely the preregistered cost fallback. Earlier qualified implementations rejected both tiers, retaining495.031/378.210GPU-worker seconds. One mechanical hotspot profile cost28.869seconds/2F/0BP. Exactly equivalent topology code removes repeated flooding; v3 reuses sealed v2 native qualification and only measured8source images for cost,41.273seconds/16F/0BP. Prior unchanged Z/train/cache timings are retained conservatively. Cumulative qualification/profile cost943.382673seconds. All timings include CPU structural work and I/O. Projections are not actual runtime or efficacy. Private cache cap8GiB; measured preflight CUDA peak819986432bytes.
