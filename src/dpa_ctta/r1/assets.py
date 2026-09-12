"""Read each authorized asset once; verify and decode the same bytes."""
import hashlib
import io
import re
import time
from pathlib import Path


class AssetMismatch(ValueError):
    """All trajectories sharing this registration are affected."""


def verified_bytes(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise AssetMismatch('missing or malformed registered content digest')
    try:data = Path(path).read_bytes()
    except OSError as error:raise AssetMismatch('registered asset unavailable') from error
    if hashlib.sha256(data).hexdigest() != expected:
        raise AssetMismatch('registered asset content digest mismatch')
    return data


def checkpoint(reg):
    import torch
    started = time.monotonic()
    entry = reg['checkpoint']
    data = verified_bytes(entry['path'], entry['sha256'])
    if len(data) != entry['bytes']:
        raise AssetMismatch('registered checkpoint size mismatch')
    # Neither unchanged mtime nor a second path-based load can replace this binding.
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    return state, dict(bytes=len(data), read_verify_load_seconds=time.monotonic()-started)


def target(row, kind):
    from ..source_io import read_pixels, read_mask
    if kind not in ('image', 'mask'):
        raise ValueError('current RGB or post-prediction mask only')
    started = time.monotonic()
    data = verified_bytes(row[kind+'_path'], row[kind+'_sha256'])
    reader = read_pixels if kind == 'image' else read_mask
    value = reader(io.BytesIO(data), 'fundus', row['image_size'])
    return value, dict(bytes=len(data), read_verify_decode_seconds=time.monotonic()-started)
