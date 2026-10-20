from pathlib import Path
from collections import Counter
from PIL import Image
from core import read, save


def run(c):
    root, old = Path(c['root']), Path(c['sealed_root'])
    panel = read(old / 'private/panel.json')
    assert len(panel) == 8
    details = []
    for item in panel:
        with Image.open(item['image_path']) as im:
            assert list(im.size) == item['image_size']
            details.append(dict(size=list(im.size), mode=im.mode, format=im.format))
    save(root / 'private/ASSET_HEADERS.json', details)
    counts = Counter((tuple(r['size']), r['mode'], r['format']) for r in details)
    save(root / 'public/ASSET_LINEAGE.json', dict(status='PASS', registered_rgb_files=8, image_header_reads=8,
         pixel_arrays_decoded=0, label_reads=0, model_weight_reads=0,
         header_groups=[dict(width=s[0], height=s[1], mode=m, format=f, count=n) for (s, m, f), n in counts.items()],
         input_pixels=512, resize='PIL RGB BICUBIC, before appearances and normalization',
         appearances=['identity', 'gamma .7', 'RGB(1.15,1,.85)', '7x7 Gaussian blur sigma1'],
         normalization='after appearance/clamp: per-image all-channel min-max; constant image rejected',
         e8_crop='normalized 512-view integer 256 crop, bilinear align_corners=False to512, same model; readout back256',
         spatial_qualification_only=True, inference_with_native_details_performed=False,
         original_details_usefulness_established=False, same_field_native_roi_width_if800=400,
         source_sha=c['source_sha'], source_files=['src/dpa_ctta/source_io.py:126',
             'src/dpa_ctta/r78_cross_patient/run.py:56', 'src/dpa_ctta/hosts/vptta.py:15',
             'src/dpa_ctta/srg_ctta/e8_worker.py:45']))
