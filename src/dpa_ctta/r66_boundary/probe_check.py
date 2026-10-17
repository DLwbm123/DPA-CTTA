"""Generated diagnostic-admission fixture, independent of experiment data."""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import binary_erosion
from .geometry import proposal


def main():
    start = time.monotonic()
    xx = np.indices((128, 128))[1]
    p = 1/(1+np.exp(np.clip(4*(xx-63.5), -60, 60)))
    feature = np.stack((p, 1-p))
    rows = []
    for shift in (-1, 0, 1):
        mask = xx < 64+shift
        _, field, audit = proposal(feature, np.stack((mask, np.zeros_like(mask))))
        edge = mask & ~binary_erosion(mask, border_value=1)
        values = field[0, 8:-8, 63+shift]
        # For this straight fixture every accepted point has the same displacement.
        rows.append(dict(candidate_shift=shift, candidate_boundary=63.5+shift,
            total_boundary=audit['channels'][0]['boundary'], accepted=audit['channels'][0]['accepted'],
            central_displacement_mean=float(values.mean()), central_displacement_nonzero=int(np.count_nonzero(values)),
            inferred_location_if_accepted=63.5+shift+float(values.mean()) if audit['channels'][0]['accepted'] else None))
        assert edge[8:-8, 63+shift].all()
    assert rows[1]['accepted'] > 0
    result = dict(status='COMPLETE', synthetic_only=True, real_images=0, target_labels=0,
        model_forwards=0, gpu_seconds=0, cpu_wall_seconds=time.monotonic()-start, fixture=rows,
        paired_measurement_admitted=all(r['accepted']>0 for r in rows))
    Path(os.environ['PROBE_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
