"""One frozen physical field, three readouts; no labels, model, or score access."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def qualification(c):
    root, old = Path(c['root']), Path(c['sealed_root'])
    panel_path = old/'private/panel.json'
    panel = json.loads(panel_path.read_text())
    assert len(panel)==8 and all('mask_path' not in p for p in panel)
    rgb = {str(Path(p['image_path']).resolve()) for p in panel}
    bits = {str((old/'inputs'/f'{i}_0_SOURCE.bits').resolve()) for i in range(8)}
    allowed = rgb | bits | {str(panel_path.resolve())}
    opened, denied = set(), []
    def audit(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        # Private external reads are limited to the 8 RGBs and 8 source masks.
        if path.startswith('/remote-home/') and not path.startswith(str(root)+'/'):
            if path not in allowed:
                denied.append(path)
                raise PermissionError('outside frozen read set')
            opened.add(path)
    sys.addaudithook(audit)
    for forbidden in (old/'scorer/panel.json',old/'private/utility.json'):
        try: forbidden.read_bytes()
        except PermissionError: pass
        else: raise AssertionError('read gate')
    rows=[]
    for i,p in enumerate(panel):
        with Image.open(p['image_path']) as im:
            assert im.size==(800,800) and im.mode=='RGB' and im.format=='PNG'
            raw=np.asarray(im,dtype=np.float32).transpose(2,0,1).copy()/255
            legacy=np.asarray(im.resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1).copy()/255
        data=(old/'inputs'/f'{i}_0_SOURCE.bits').read_bytes()
        assert len(data)==65536
        od=np.unpackbits(np.frombuffer(data,np.uint8)).reshape(2,512,512)[0].astype(bool)
        yy,xx=np.where(od); assert len(yy)>0
        def start(v):
            t=int(np.floor((v.min()+v.max()+1)/2))-128
            return int(np.clip(16*np.floor((np.clip(t,0,256)+8)/16),0,256))
        y,x=start(yy),start(xx)
        native=torch.from_numpy(raw)[None];low=torch.from_numpy(legacy)[None]
        lo,hi=low.amin(),low.amax(); assert hi>lo
        build=lambda:tuple((v-lo)/(hi-lo) for v in views(native,low,y,x))
        first,second=build(),build()
        assert all(torch.equal(a,b) for a,b in zip(first,second))
        row=dict(crop_start_private=[y,x],lo=float(lo),span=float(hi-lo),diff={})
        for name,v in zip(('N','R','L'),first):
            row[name]=dict(min=float(v.min()),max=float(v.max()),
                          outside_count=int(((v<0)|(v>1)).sum()),elements=v.numel())
        for name,a,b in (('N_R',first[0],first[1]),('N_L',first[0],first[2]),('R_L',first[1],first[2])):
            row['diff'][name]={}
            for domain,d in (('full',a-b),('interior',(a-b)[:,:,2:-2,2:-2])):
                d=d.double()
                row['diff'][name][domain]=dict(mae=float(d.abs().mean()),mse=float(d.square().mean()),maximum=float(d.abs().max()))
        rows.append(row)
    assert len(opened & rgb)==8 and len(opened & bits)==8
    assert not torch.cuda.is_initialized()
    out={}
    for pair in ('N_R','N_L','R_L'):
        out[pair]={}
        for domain in ('full','interior'):
            items=[r['diff'][pair][domain] for r in rows]
            out[pair][domain]=dict(mean_absolute=float(np.mean([v['mae'] for v in items])),
                 rms=float(np.sqrt(np.mean([v['mse'] for v in items]))),
                 max_absolute=max(v['maximum'] for v in items))
    normalization={name:dict(min=min(r[name]['min'] for r in rows),max=max(r[name]['max'] for r in rows),
        outside_fraction=sum(r[name]['outside_count'] for r in rows)/sum(r[name]['elements'] for r in rows)) for name in ('N','R','L')}
    save(root/'private/PER_CONTENT.json',rows)
    save(root/'public/RESULTS.json',dict(status='PASS',registered_rgb_decoded=8,identity_source_od_reads=8,
         label_reads=0,utility_score_reads=0,model_weight_reads=0,model_calls=0,cuda_initialized=False,
         access_gate_negative_checks=len(denied),exact_replay_views=24,conditions=['identity'],
         roi_count=8,native_roi_size=400,legacy_roi_size=256,output_size=512,
         shared_normalization=True,clamp_applied=False,normalization=normalization,differences=out,
         task_benefit_established=False,sensor_native_resolution_established=False,source_sha=c['source_sha']))


def run(c):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    (check if os.environ['AUDIT_MODE']=='check' else qualification)(c)
