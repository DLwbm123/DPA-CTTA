"""Source-only metadata join, raw pixels, known masks and final-grid distances."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .integrations.ctta_suite import INPUT_SIZES
from .proxy_loss import FixedProxy, ProxyProvenance

SOURCES = {'fundus': 'RIM_ONE_r3', 'polyp': 'BKAI'}
SPLITS = ('basis_train', 'critic_train', 'critic_validation', 'proxy_test')


def registered_source(manifest_path, split_path, csv_specs, data_root, task, seed=20260907):
    """Consume frozen metadata only. Never rebuild a split or hash/read image files.

    csv_specs explicitly supplies name and split, as in the pinned source registry.
    Existing absolute manifest paths are provenance; CSV-relative paths bind the
    explicitly supplied private root, allowing an intentional root relocation.
    """
    source = SOURCES[task]
    root = Path(data_root).resolve()
    records = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    if (split.get('schema_version') != 'crisp-source-content-split-v1'
            or split.get('task') != task or split.get('source') != source):
        raise ValueError('frozen source split contract mismatch')
    payload = {k: v for k, v in split.items() if k != 'split_payload_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != split.get('split_payload_sha256'):
        raise ValueError('frozen split payload mismatch')

    def path(relative):
        p = Path(relative)
        if p.is_absolute() or not p.parts or p.parts[0] != source or '..' in p.parts:
            raise ValueError('CSV path outside source whitelist')
        resolved = (root / p).resolve()
        if not resolved.is_relative_to(root / source):
            raise ValueError('source path escapes domain/root')
        return str(resolved)

    csv_rows = {}
    if not csv_specs:
        raise ValueError('explicit source CSV registration required')
    for spec in csv_specs:
        if spec['split'] not in ('train', 'test') or spec['name'] != f"{source}_{spec['split']}.csv":
            raise ValueError('source CSV whitelist mismatch')
        csv_path = (root / spec['name']).resolve()
        if not csv_path.is_relative_to(root):
            raise ValueError('CSV escapes root')
        with csv_path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                sid = task + ':' + row['image']
                if sid in csv_rows:
                    raise ValueError('duplicate CSV sample')
                csv_rows[sid] = dict(sample_id=sid, source_split=spec['split'],
                                     image_path=path(row['image']), mask_path=path(row['mask']),
                                     image_relative=row['image'], mask_relative=row['mask'])
    by_id = {}
    for row in records:
        sid = row['sample_id']
        if sid in by_id or sid not in csv_rows or row['domain'] != source:
            raise ValueError('source manifest membership mismatch')
        current = csv_rows[sid]
        if row['split'] != current['source_split']:
            raise ValueError('CSV/manifest source split mismatch')
        for field in ('image', 'mask'):
            relative = Path(current[field + '_relative']).parts
            if Path(row[field + '_path']).parts[-len(relative):] != relative:
                raise ValueError('CSV/manifest path mismatch')
            value = row[field + '_sha256']
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('missing registered content digest')
        if len(row['image_size']) != 2 or any(type(v) is not int or v < 1 for v in row['image_size']):
            raise ValueError('invalid registered image geometry')
        by_id[sid] = dict(row, **{k: v for k, v in current.items() if k != 'sample_id'})
    if set(by_id) != set(csv_rows):
        raise ValueError('CSV/source manifest coverage mismatch')
    groups_by_split, used_groups, used_ids, image_roles = {}, set(), set(), {}
    for name in SPLITS:
        section = split['splits'][name]
        groups = {gid: [] for gid in section['group_ids']}
        if len(groups) != len(section['group_ids']) or used_groups.intersection(groups):
            raise ValueError('duplicate/overlapping content groups')
        for item in section['records']:
            sid = item['sample_id']
            if sid not in by_id or sid in used_ids:
                raise ValueError('split sample overlap/mismatch')
            row = by_id[sid]
            if any(item[k] != row[k] for k in ('source_split', 'image_sha256', 'mask_sha256')):
                raise ValueError('split/manifest content mismatch')
            gid = hashlib.sha256((row['image_sha256'] + row['mask_sha256']).encode()).hexdigest()
            if gid not in groups:
                raise ValueError('record content group mismatch')
            previous = image_roles.setdefault(row['image_sha256'], name)
            if previous != name:
                raise ValueError('same image content crosses split roles')
            groups[gid].append(dict(row, group_id=gid, split_role=name))
            used_ids.add(sid)
        counts = split['counts'][name]
        if (any(not rows for rows in groups.values()) or counts != dict(groups=len(groups), samples=len(section['records']))):
            raise ValueError('split coverage/count mismatch')
        used_groups.update(groups)
        groups_by_split[name] = groups
    if used_ids != set(by_id):
        raise ValueError('split/source manifest coverage mismatch')
    basis = groups_by_split['basis_train']
    query = groups_by_split['critic_validation']
    count = 20 if task == 'fundus' else 32
    if len(basis) < 4 or len(query) < count or (task == 'fundus' and len(query) != count):
        raise ValueError(f'INSUFFICIENT_OR_CHANGED_SOURCE_GROUPS: proxy={len(basis)}, query={len(query)}, required=4/{count}')
    ordered = sorted(basis, key=lambda gid: (hashlib.sha256(f'{seed}:{gid}'.encode()).hexdigest(), gid))[:4]
    def representatives(groups, keys):
        return [min(groups[gid], key=lambda row: row['sample_id']) for gid in keys]
    return {'proxy': representatives(basis, ordered),
            'query': representatives(query, list(query)[:count]),
            'split_payload_sha256': digest, 'checkpoint_training_membership': 'UNKNOWN'}


def read_pixels(path, task, expected_size=None):
    with Image.open(path) as image:
        if expected_size is not None and list(image.size) != list(expected_size):
            raise ValueError('registered image geometry mismatch')
        image = image.convert('RGB').resize((INPUT_SIZES[task],) * 2,
                  Image.Resampling.BICUBIC if task == 'fundus' else Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255
    return torch.from_numpy(array).unsqueeze(0)


def read_mask(path, task, expected_size=None):
    with Image.open(path) as image:
        if expected_size is not None and list(image.size) != list(expected_size):
            raise ValueError('registered label/image geometry mismatch')
        gray = np.asarray(image.convert('L').resize((INPUT_SIZES[task],) * 2, Image.Resampling.NEAREST))
    mask = np.stack((gray < 255, gray == 0)) if task == 'fundus' else (gray > 127)[None]
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def signed_distance(mask):
    """Final-grid opposite-class pixel-center EDT; undefined pairs retain region loss."""
    from scipy.ndimage import distance_transform_edt
    if (mask.ndim != 4 or mask.dtype != torch.float32 or mask.device.type != 'cpu'
            or mask.requires_grad or not mask.numel() or not ((mask == 0) | (mask == 1)).all()):
        raise ValueError('fixed CPU binary float32 BCHW mask required')
    distance = torch.zeros_like(mask)
    defined = torch.zeros(mask.shape[:2], dtype=torch.bool)
    for b in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            y = mask[b, c].numpy().astype(bool)
            if y.any() and not y.all():
                d = (distance_transform_edt(~y) - distance_transform_edt(y)) / math.hypot(*y.shape)
                distance[b, c] = torch.from_numpy(d.astype(np.float32))
                defined[b, c] = True
    return distance, defined


def source_proxy(rows, task):
    """Call only with the four registered basis_train representatives, never query rows."""
    if len(rows) != 4 or len({r['group_id'] for r in rows}) != 4 or any(r.get('split_role') != 'basis_train' for r in rows):
        raise ValueError('four distinct proxy content groups required')
    pixels = torch.cat([read_pixels(r['image_path'], task, r['image_size']) for r in rows])
    masks = torch.cat([read_mask(r['mask_path'], task, r['image_size']) for r in rows])
    distance, _defined = signed_distance(masks)
    return FixedProxy(pixels, masks, distance, ProxyProvenance.SOURCE)
