"""One fixed foreground CPU analysis; configure via R6D_CONFIG, never model paths in argv."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback

sys.dont_write_bytecode = True
from core import (ARMS, STREAMS, analysis, digest, disjoint, install_guard, public_safe,
                  read, regular, validators, write)


def allowed_sources(source, control):
    names=['receipt.json','R6_SCOPE.json','current_result.json','matrix.processes.json',
           'processes.started.json','public_aggregate.json','packet.private.json']
    names += [f'o{s}a{a}/{name}' for s in STREAMS for a in range(4)
              for name in ['completion.json','unlabeled.jsonl','evaluation.jsonl']]
    names += [f'device{i}/smoke.completion.json' for i in range(3)]
    return {name:Path(source)/name for name in names}


def source_inventory(source):
    # Metadata only for all output files, so original output-cap validation is retained.
    files={}; links={}
    for base,dirs,names in os.walk(source,followlinks=False):
        for name in list(dirs):
            p=Path(base)/name
            if p.is_symlink():links[str(p.relative_to(source))]=os.readlink(p);dirs.remove(name)
        for name in names:
            p=Path(base)/name
            if p.is_symlink():links[str(p.relative_to(source))]=os.readlink(p)
            else:files[str(p.relative_to(source))]=p.stat().st_size
    return dict(files=files,symlinks=links,total_bytes=sum(files.values()))


def source_record(p, target=None):
    regular(p);before=p.stat(); h=hashlib.sha256()
    flags=os.O_RDONLY | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_NOATIME',0)
    out=None
    try:
        if target is not None:
            target.parent.mkdir(parents=True,exist_ok=True);out=target.open('xb')
        with os.fdopen(os.open(p,flags),'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                h.update(b)
                if out:out.write(b)
    finally:
        if out:out.close()
    after=p.stat()
    state=lambda s:dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,mode=stat.S_IMODE(s.st_mode),inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink)
    if state(before)!=state(after):raise ValueError('source changed during read')
    return dict(state(after),sha256=h.hexdigest())


def check_snapshot(manifest, snapshot):
    for name,entry in manifest.items():
        p=snapshot/name;regular(p)
        if p.stat().st_ino==entry['inode'] and p.stat().st_dev==entry['device']:
            raise ValueError('snapshot shares source inode')
        if p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:
            raise ValueError('snapshot bytes mismatch')


def reconstruct(rows, v, original, spec):
    v['validate_cross_GT'](rows)
    values,domains,recurrence=v['gate_inputs'](rows)
    selection=v['gate'](values,domains,recurrence,'A')
    primary={}
    for arm in ARMS:
        primary[arm]={}
        for ch in range(2):
            primary[arm][('OD','OC')[ch]]=sum(sum(sum(100*r['metrics']['post'][ch]['dice'] for r in rows[s,arm] if r['subset']=='remaining_dev' and r['domain']==d)/n for d,n in spec['inputs']['remaining_dev_domain_counts'].items())/4 for s in (0,1))/2
        primary[arm]['macro']=sum(primary[arm].values())/2
    for c in values:
        for x,y in zip(values[c],original['primary_comparisons_pp'][c]):
            if not math.isclose(x,y,rel_tol=0,abs_tol=1e-10):raise ValueError('historical primary mismatch')
        if not math.isclose(recurrence[c],original['recurrence_comparisons_pp'][c],rel_tol=0,abs_tol=1e-10):raise ValueError('historical recurrence mismatch')
    for x,y in zip(domains,original['domain_comparisons_pp']):
        if not math.isclose(x,y,rel_tol=0,abs_tol=1e-10):raise ValueError('historical domain mismatch')
    for arm in ARMS:
        for ch in primary[arm]:
            if not math.isclose(primary[arm][ch],original['primary_order_equal']['remaining_dev'][arm][ch],rel_tol=0,abs_tol=1e-10):raise ValueError('full precision original primary')
    if selection!=original['selection'] or selection['passed'] or original['status']!='R6A_COMPLETE_NO_ADVANCE':raise ValueError('historical gate status')
    for label,c in [('C','C'),('SCALE','R_SCALE'),('SHUFFLE','R_SHUFFLE')]:
        for x,y in zip(values[c]+[recurrence[c]],spec['known_rounded_results_pp']['BAL_minus_'+label]):
            if abs(x-y)>spec['baseline_reconstruction']['display_rounding_check_atol_pp']:raise ValueError('display-only check')
    return dict(primary=primary,primary_comparisons_pp=values,domain_comparisons_pp=domains,recurrence_comparisons_pp=recurrence,selection=selection,status=original['status'])


def main(config):
    begin=time.monotonic();root=Path(config['code']);source=Path(config['source']);work=Path(config['work'])
    disjoint(source,work);disjoint(root,work)
    work.mkdir(exist_ok=False)
    sources=allowed_sources(source,config.get('control'))
    for p in sources.values():regular(p)
    inventory=source_inventory(source)
    if any(n.endswith('failure.json') or n=='dispatch.stopped.json' for n in inventory['files']):raise ValueError('historical failure record')
    required=sum(p.stat().st_size for p in sources.values())
    free=os.statvfs(work).f_bavail*os.statvfs(work).f_frsize
    if free<required+2147483648:raise OSError('snapshot plus output capacity unavailable')
    snapshot=work/'snapshot';snapshot.mkdir()
    # Probe only the new workspace on its actual filesystem.
    probe=work/'storage_probe';probe.write_bytes(b'r6d');assert probe.read_bytes()==b'r6d';probe.unlink()
    install_guard(list(sources.values()),[root],work)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800-second wall cap')))
    signal.alarm(1800)
    before={};after={};failure=None;stage='snapshot';fixed=dict(new_model_execution_started=False,new_model_forwards=0,
        new_model_backward_calls=0,new_adam_calls=0,new_vjp_calls=0,R6A='R6A_COMPLETE_NO_ADVANCE',R6B='NOT_RUN',next_gpu_execution_authorized=False)
    try:
        t=time.monotonic()
        for name,p in sources.items():before[name]=source_record(p,snapshot/name)
        write(work/'source-before.private.json',before);check_snapshot(before,snapshot)
        snapshot_seconds=time.monotonic()-t
        packet=read(snapshot/'packet.private.json');reg=packet['assets']['registration']
        specpath=root/'analysis/r6d_posthoc_v1/input/R6D_ANALYSIS_SPEC.json';spec=read(specpath)
        v=validators(root,inventory['total_bytes'])
        if v['fingerprint']()!=spec['historical_binding']['production_fingerprint']:raise ValueError('production source fingerprint')
        if digest(root/'configs/r6_science_v1.json')!=spec['historical_binding']['science_sha256']:raise ValueError('science bytes')
        if packet['binding']!=spec['historical_binding']:raise ValueError('packet binding')
        stage='fixed_scalar_analysis';analysis_start=time.monotonic()
        receipt,rows,_=v['completed'](snapshot,reg,'A')
        if receipt['binding']!=spec['historical_binding']:raise ValueError('historical binding')
        pointer=read(snapshot/'current_result.json')
        if pointer['binding']!=receipt['binding'] or not pointer['valid'] or pointer['status']!='R6A_COMPLETE_NO_ADVANCE':raise ValueError('original pointer')
        original=read(snapshot/'public_aggregate.json')
        baseline=reconstruct(rows,v,original,spec)
        summary=analysis(rows,v['stream_summary'](reg),spec,work/'public')
        summary.update(analysis_seconds=time.monotonic()-analysis_start,snapshot_seconds=snapshot_seconds)
        write(work/'public/BASELINE_RECONSTRUCTION.json',baseline)
        write(work/'public/ANALYSIS_BINDING.json',dict(inferential_status='POST_HOC_EXPLORATORY',historical_binding=spec['historical_binding'],
            historical_publication_sha=spec['historical_publication_sha'],analysis_sha=config['analysis_sha'],analysis_spec_sha256=digest(specpath),
            inputs=[dict(file_token=f'input_{i:03}',bytes=x['bytes'],sha256=x['sha256']) for i,x in enumerate(before.values())],
            source_file_mapping='PRIVATE_ONLY',validator_strategy='Audited function AST extraction without production module imports',**fixed))
        stage='source_after_check'
    except BaseException as exc:
        failure=exc
        (work/'failure.private.log').write_text(traceback.format_exc())
    finally:
        for name,p in sources.items():after[name]=source_record(p)
        inventory_after=source_inventory(source)
        write(work/'source-after.private.json',after)
        unchanged=bool(before) and before==after and inventory==inventory_after
        if not unchanged and failure is None:failure=ValueError('readonly audit mismatch')
        pub=work/'public';pub.mkdir(exist_ok=True)
        audit=dict(inferential_status='POST_HOC_EXPLORATORY',source_bytes_unchanged=before==after,source_metadata_unchanged=before==after,
            source_inventory_unchanged=inventory==inventory_after,result_pointer_unchanged=before.get('current_result.json')==after.get('current_result.json'),
            current_symlink_unchanged=inventory['symlinks']==inventory_after['symlinks'],ordinary_file_snapshot=True,
            snapshot_bytes=required,source_output_bytes=inventory['total_bytes'],free_bytes_before=free,output_source_disjoint=True,
            evidence=[dict(file_token=f'input_{i:03}',bytes=after[n]['bytes'],before_sha256=before.get(n,{}).get('sha256'),after_sha256=after[n]['sha256'],snapshot_sha256=digest(snapshot/n) if (snapshot/n).exists() else None) for i,n in enumerate(sources)],
            safeguards='Whitelist scalar files; O_NOFOLLOW/O_NOATIME source reads; audit hook denies external writes, imports and process/network/device calls; no production imports',
            limitations=['Python audit hook is not an OS sandbox; inspected code has no native/model dependency.',
                        'Stored evaluation GT counts used offline; no raw labels read.',
                        'mtime/ctime/mode/inode/link counts checked; filesystem-managed atime is not a semantic integrity claim.'],
            raw_asset_reads=0,model_imports=0,gpu_queries_or_initializations=0,**fixed)
        public_safe(audit);write(pub/'READONLY_AUDIT.json',audit)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result=dict(analysis_executed=stage!='snapshot',analysis_status='R6D_INCOMPLETE' if failure else 'ANALYSIS_TABLES_COMPLETE_INTERPRETATION_PENDING',
            stage=stage,failure_type=type(failure).__name__ if failure else None,wall_seconds=time.monotonic()-begin,
            cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,peak_rss_platform_units=usage.ru_maxrss,
            peak_rss_unit='KiB on Linux; bytes on macOS',max_threads=2,processes=1,**fixed)
        if not failure:result.update(summary)
        result['new_output_bytes_excluding_snapshot']=sum(p.stat().st_size for p in work.rglob('*') if p.is_file() and snapshot not in p.parents)
        if result['new_output_bytes_excluding_snapshot']>2147483648:
            result['analysis_status']='R6D_INCOMPLETE';failure=ValueError('output cap')
        public_safe(result);write(pub/'RUN_LOG.json',result)
        signal.alarm(0)
    print(json.dumps(result,allow_nan=False))
    if failure:raise RuntimeError('R6-D failed; preserve private failure log, no automatic retry') from failure


if __name__=='__main__':
    main(read(os.environ['R6D_CONFIG']))
