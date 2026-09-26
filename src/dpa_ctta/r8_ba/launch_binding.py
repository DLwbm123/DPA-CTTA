"""Verify the deployed source inventory and assigned physical device before work."""
import hashlib
import json
import subprocess
import platform
from importlib.metadata import distributions
import torch
from pathlib import Path


def environment():
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                packages=sorted((d.metadata["Name"], d.version) for d in distributions()))


def verify(config, root):
    if json.loads(json.dumps(environment())) != config['environment']:
        raise ValueError('R8 runtime environment lock changed')
    ref = config['code_inventory']
    raw = Path(ref['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('R8 deployed code inventory changed')
    packet = json.loads(raw)
    if packet['code_sha'] != config.get('runtime_code_sha', config['code_sha']) or not packet['files']:
        raise ValueError('R8 deployed code commit binding')
    root = Path(root).resolve()
    for relative, expected in packet['files'].items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('R8 deployed source bytes changed')
    for directory in ('src', 'scripts'):
        if any(str(path.relative_to(root)) not in packet['files'] for path in (root / directory).rglob('*.py')):
            raise ValueError('R8 unregistered Python source')
    gpu = config.get('physical_gpu')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = {int(parts[0]): (parts[1].strip(), int(parts[2])) for line in rows.splitlines()
               if (parts := line.split(','))}
    selected = (gpu,) if gpu is not None else (5, 6, 7)
    for index in selected:
        if index not in (5,6,7) or devices[index][0] != config['gpu_uuids'][str(index)]:
            raise ValueError('R8 physical GPU identity mismatch')
        if devices[index][1] * 1024**2 < config['required_gpu_bytes']:
            raise ValueError('R8 insufficient free GPU memory')
