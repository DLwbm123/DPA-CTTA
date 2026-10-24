"""VPTTA OD/OC data protocol. Does not launch training or select models."""
import csv
from pathlib import Path

DOMAINS = ('RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS')
COUNTS = (159, 400, 650, 800, 101)
IMAGE_SIZE = 512


def manifest(root):
    root = Path(root)
    result = {}
    seen = set()
    for domain, expected in zip(DOMAINS, COUNTS):
        files = ([domain + '.csv'] if domain == 'REFUGE_Valid'
                 else [domain + '_train.csv', domain + '_test.csv'])
        rows = []
        for filename in files:
            with (root / filename).open(newline='') as stream:
                for row in csv.DictReader(stream):
                    for key in ('image', 'mask'):
                        path = Path(row[key])
                        if path.is_absolute() or '..' in path.parts:
                            raise ValueError('Manifest paths must remain inside data root')
                    identity = str((root / row['image']).resolve())
                    if identity in seen:
                        raise ValueError('Repeated image path across data manifest')
                    seen.add(identity)
                    rows.append({'domain': domain, 'image': row['image'], 'mask': row['mask']})
        if len(rows) != expected:
            raise ValueError(f'{domain}: expected {expected}, got {len(rows)}')
        result[domain] = rows
    return result


def scenario(data, source):
    if source not in DOMAINS:
        raise ValueError(source)
    targets = [d for d in DOMAINS if d != source]
    return data[source], [row for d in targets for row in data[d]]


def read_image(root, row):
    import numpy as np
    from PIL import Image
    with Image.open(Path(root) / row['image']) as im:
        x = np.asarray(im.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    span = float(x.max() - x.min())
    if span == 0:
        raise ValueError('Constant image: official min-max normalization undefined')
    return ((x - x.min()) / span).transpose(2, 0, 1).copy()


def read_label(root, row):
    import numpy as np
    from PIL import Image
    path = row['mask']
    if path.endswith('.tif'):
        path = path.replace('.tif', '-1.tif')
    with Image.open(Path(root) / path) as im:
        mask = np.asarray(im.convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST))
    return np.stack([mask < 255, mask == 0]).astype(np.uint8)


def evaluate(root, rows, adapt_predict, calculate_metrics):
    """Callback receives only image, returns probabilities AFTER adaptation.

    Construct a fresh callback per source/checkpoint/method. Keep it alive across
    all four target domains. Pass official VPTTA calculate_metrics unchanged.
    No target masks, domain boundaries, or future images enter the callback.
    """
    import numpy as np
    scores = []
    for row in rows:
        prediction = np.asarray(adapt_predict(read_image(root, row)))
        if prediction.shape != (2, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError('Prediction must be OD/OC probability maps')
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise ValueError('Invalid probabilities')
        label = read_label(root, row)
        values = np.asarray(calculate_metrics(prediction[None].copy(), label[None]), dtype=float)
        if values.shape != (4, 1) or not np.isfinite(values).all():
            raise ValueError('Invalid official metrics; never silently omit a case')
        scores.append({'domain': row['domain'], 'od_dice': float(values[0, 0]),
                       'oc_dice': float(values[2, 0]), 'od_asd': float(values[1, 0]),
                       'oc_asd': float(values[3, 0])})
    if not scores:
        raise ValueError('Empty evaluation stream')
    keys = ('od_dice', 'oc_dice', 'od_asd', 'oc_asd')
    summary = {key: float(np.mean([s[key] for s in scores])) for key in keys}
    summary['mean_dice'] = (summary['od_dice'] + summary['oc_dice']) / 2
    summary['n'] = len(scores)
    return summary, scores


def aggregate(summaries):
    """Five source scenarios have equal weight, as in paper Table 1."""
    if set(summaries) != set(DOMAINS):
        raise ValueError('All five source scenarios are required for paper-level Average')
    for domain, count in zip(DOMAINS, COUNTS):
        if summaries[domain]['n'] != sum(COUNTS) - count:
            raise ValueError('Incomplete target stream')
    return sum(summaries[d]['mean_dice'] for d in DOMAINS) / len(DOMAINS)
