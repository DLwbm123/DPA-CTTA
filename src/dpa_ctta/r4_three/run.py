"""Metadata by default; future compute requires explicit exact-commit authorization."""
import argparse,json,os
from pathlib import Path
from .plan import ROOT,matrix,write_plan


def main():
    os.umask(0o077)
    if os.environ.get('RUN_MODE'):
        from .execution import worker
        return worker()
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--recompute',action='store_true')
    p.add_argument('--registration');p.add_argument('--out');p.add_argument('--assets')
    p.add_argument('--authorization',default=str(ROOT/'configs/r4t_execution.defaults.json'))
    args=p.parse_args()
    if args.run:
        auth=json.loads(Path(args.authorization).read_text())
        if auth.get('enabled') is not True:raise PermissionError('execution disabled; exact-commit external review and fresh user authorization required')
        if args.recompute or not args.assets or not args.out:raise ValueError('explicit assets and output required')
        from .execution import launch
        return launch(auth,json.loads(Path(args.assets).read_text()),args.out)
    if args.recompute:
        if not args.registration or not args.out:raise ValueError('explicit registration and output required')
        from .analyze import recompute
        return recompute(args.out,json.loads(Path(args.registration).read_text()))
    if bool(args.registration)!=bool(args.out):raise ValueError('registration and output must be supplied together')
    result=write_plan(args.out,json.loads(Path(args.registration).read_text())) if args.registration else matrix()
    print(json.dumps(dict(status=result['status'],jobs=len(result['jobs']),science_sha256=result['science_sha256'],GPU_requests_issued=0)))
