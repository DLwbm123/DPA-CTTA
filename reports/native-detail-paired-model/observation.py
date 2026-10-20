"""Frozen synthetic mapping checks reused from the CPU qualification."""

import json

from pathlib import Path

import torch

import torch.nn.functional as F

def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def resize(a, size):
    return F.interpolate(a, size=(size, size), mode='bilinear', align_corners=False)

def views(native, low, y, x):
    yy, xx = y * 25 // 16, x * 25 // 16
    n = resize(native[:, :, yy:yy+400, xx:xx+400], 512)
    r = resize(resize(low, 800)[:, :, yy:yy+400, xx:xx+400], 512)
    l = resize(low[:, :, y:y+256, x:x+256], 512)
    return n, r, l

def check(c):
    # Planes encode physical centers on the original 800-pixel grid.
    a = torch.arange(800, dtype=torch.float64)
    native = torch.stack((a[:, None].expand(800,800), a[None,:].expand(800,800)))[None]
    b = (torch.arange(512, dtype=torch.float64) + .5) * 25/16 - .5
    low = torch.stack((b[:, None].expand(512,512), b[None,:].expand(512,512)))[None]
    errors, border = [], []
    for y in (0, 16, 128, 240, 256):
        for x in (0, 16, 128, 240, 256):
            n,r,l = views(native,low,y,x)
            q = (torch.arange(512,dtype=torch.float64)+.5)*400/512-.5
            expected_y = y*25/16+q
            expected_x = x*25/16+q
            assert (n[0,0,2:-2,2:-2]-expected_y[2:-2,None]).abs().max() < 1e-9
            assert (n[0,1,2:-2,2:-2]-expected_x[None,2:-2]).abs().max() < 1e-9
            for v in (r,l):
                errors.append(float((n-v)[:,:,2:-2,2:-2].abs().max()))
                border.append(float((n-v).abs().max()))
            assert torch.equal(n, views(native,low,y,x)[0])
    const = views(torch.ones(1,3,800,800,dtype=torch.float64)*.37,
                  torch.ones(1,3,512,512,dtype=torch.float64)*.37,0,256)
    ce = max(float((v-.37).abs().max()) for v in const)
    assert max(errors) < 1e-9 and ce < 1e-12
    # align_corners=False samples beyond edge centers by clamping.
    assert float(views(native,low,0,0)[0][0,0,0,0]) == 0
    save(Path(c['root'])/'public/GEOMETRY.json', dict(status='PASS',windows=25,
         interior_border_excluded=2, interior_coordinate_error_max=max(errors),
         full_field_coordinate_difference_max=max(border), constant_error_max=ce,
         tolerance_coordinate_pixels=1e-9, tolerance_constant=1e-12,
         edge_clamp_checked=True, same_physical_field=True))
