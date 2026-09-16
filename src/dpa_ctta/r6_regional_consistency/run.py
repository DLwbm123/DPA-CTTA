"""No arguments: metadata only. Real scopes require a new exact R6 authorization."""
import argparse,json,os
from pathlib import Path
from .plan import ROOT,dry_run


def main():
    os.umask(0o077)
    if os.environ.get('RUN_MODE'):
        from .execution import worker
        return worker()
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--registration');p.add_argument('--out');p.add_argument('--assets');p.add_argument('--authorization',default=str(ROOT/'configs/r6_execution.defaults.json'))
    args=p.parse_args()
    if args.run:
        auth=json.loads(Path(args.authorization).read_text())
        if auth.get('enabled') is not True:raise PermissionError('R6 disabled; earlier waivers are not R6 permission')
        if not args.assets or not args.out:raise ValueError('explicit assets/output')
        from .execution import launch
        return launch(auth,json.loads(Path(args.assets).read_text()),args.out)
    if bool(args.registration)!=bool(args.out):raise ValueError('metadata registration and output together')
    value=dry_run(json.loads(Path(args.registration).read_text()) if args.registration else None)
    if args.out:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
        (out/'DRY_RUN.json').write_text(json.dumps(value,indent=2)+'\n')
        for scope,rows in value['matrices'].items():(out/(scope+'_MATRIX.json')).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],jobs={k:len(v) for k,v in value['matrices'].items()},execution_started=False)))
