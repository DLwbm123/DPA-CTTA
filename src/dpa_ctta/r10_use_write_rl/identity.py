"""R10 runtime SHA, scientific specification, assets and measured authorization."""
import hashlib,subprocess
from pathlib import Path
from .protocol import ROOT,SPEC_SHA,digest,CAPS,RECOVERY_POLICY


def inventory(root=ROOT):
    files=subprocess.check_output(['git','ls-files','src','scripts','configs'],cwd=root,text=True).splitlines()
    files.append('docs/review/r10_use_write_rl/input/R10_SPEC_AND_MATRIX.json')
    return {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in sorted(set(files))}


def verify(config):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if config['code_sha']!=head or config['inventory']!=inventory() or config['spec_sha256']!=SPEC_SHA:raise ValueError('R10 runtime identity')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src','scripts','configs'],cwd=ROOT,text=True):raise ValueError('runtime modified after binding')
    return dict(code_sha=head,spec_sha256=SPEC_SHA,inventory_sha256=digest(config['inventory']),bindings_sha256=digest(config['bindings']))


def require(config,profile=False):
    if config.get('schema')!='R10_LAUNCH_V1':raise ValueError('R10 launch schema')
    identity=verify(config);authorization=config.get('profile_authorization' if profile else 'authorization',{})
    fields=dict(**identity,gpu_assignments=config['gpu_assignments'],output_root=config['output_root'])
    if any(authorization.get(k)!=v for k,v in fields.items()) or not authorization.get('user_instruction'):raise PermissionError('bound R10 authorization required')
    if config['caps']!=CAPS or config['recovery_policy']!=RECOVERY_POLICY:raise ValueError('R10 resource protocol')
    if len(config['gpu_assignments'])!=1 or not Path(config['output_root']).is_absolute():raise ValueError('serial bound output/GPU')
    if profile:
        if config['execution_authorized'] is not False or authorization.get('scope')!='REAL_DATA_PROFILE_ONLY':raise PermissionError('profile scope')
    else:
        from .profile import projection
        if config['execution_authorized'] is not True or authorization.get('profile_sha256')!=digest(config['profile']):raise PermissionError('formal profile-bound authority')
        if config['profile'].get('identity')!=fields or any(r.get('binding')!=fields for r in config['profile']['measurements'].values()):raise ValueError('profile runtime/asset/GPU/root binding')
        proof=projection(config['profile'])
        if proof['status']!='PASS' or proof!=config['admission']:raise ValueError('real complete resource admission')
    return identity
