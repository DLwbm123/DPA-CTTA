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
from pinned_io import Reader, record_file, source_mapping, layout, check_layout


def allowed_sources(source, addendum):
    return {name:Path(source)/relative for name,relative in source_mapping(addendum,STREAMS).items()}


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
    return record_file(p,target)


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


def capture_error(errors, phase, work):
    """Called inside except; the first exception and each after-check stay separate."""
    kind=type(sys.exc_info()[1]).__name__
    token=f"failure_{len(errors):02}_{phase}"
    errors.append(dict(phase=phase,type=kind,log_token=token))
    if work is not None:
        with (work/(token+'.private.log')).open('x') as f:f.write(traceback.format_exc())


def main(config):
    begin=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_SELF)
    root=Path(config['code']).resolve();source=Path(config['source']).absolute();work=Path(config['work']).absolute()
    before={};after={};snapshots={};errors=[];stage='preflight';ready=False;analysis_started=False;tables_complete=False
    inventory=None;inventory_after=None;layout_before=None;layout_after=None;snapshot_checked=False
    required=None;free=None;sources={};mapping={};summary={};snapshot_seconds=None
    fixed=dict(new_model_execution_started=False,new_model_forwards=0,new_model_backward_calls=0,new_adam_calls=0,
        new_vjp_calls=0,R6A='R6A_COMPLETE_NO_ADVANCE',R6B='NOT_RUN',next_gpu_execution_authorized=False)
    previous_alarm=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800-second wall cap including preflight')))
    signal.alarm(1800)
    try:
        disjoint(source,work);disjoint(root,work);disjoint(source,root)
        for old in config.get('prior_work_directories',[]):disjoint(old,work)
        work.mkdir(exist_ok=False);ready=True
        specpath=root/'analysis/r6d_posthoc_v1/input/R6D_ANALYSIS_SPEC.json'
        addpath=root/'analysis/r6d_posthoc_v1/io_addendum/R6D_IO_ADDENDUM.json'
        spec=read(specpath);addendum=read(addpath)
        if digest(specpath)!=addendum['unchanged_analysis_spec_sha256'] or digest(addpath)!='250645247b145760e520214373b9930b8580b595165fbd1cc21ac7ec666e0b96':raise ValueError('immutable analysis/addendum bytes')
        if addendum['historical_binding']!=spec['historical_binding']:raise ValueError('addendum binding conflict')
        access={};reader=Reader(source,access);source=reader.root
        mapping=source_mapping(addendum,STREAMS);sources=allowed_sources(source,addendum)
        directories={source}
        for p in sources.values():
            while p.parent!=source:
                p=p.parent;directories.add(p)
        install_guard(sources.values(),[root],work,directories,access)
        layout_before,pointer_raw=layout(reader,addendum)
        write(work/'layout-before.private.json',layout_before)
        with (work/'pointer-before.raw.json').open('xb') as f:f.write(pointer_raw)
        write(work/'input-mapping.private.json',{n:dict(source_relative_path=r,snapshot_relative_path=n) for n,r in mapping.items()})
        sizes={}
        for n,r in mapping.items():
            with reader.opened(r) as (_,__,identity):sizes[n]=identity['bytes']
        required=sum(sizes.values());inventory=source_inventory(source)
        if any(n.endswith('failure.json') or n=='dispatch.stopped.json' for n in inventory['files']):raise ValueError('historical failure record')
        st=os.statvfs(work);free=st.f_bavail*st.f_frsize
        if free<required+2147483648:raise OSError('snapshot plus output capacity unavailable')
        probe=work/'storage_probe';probe.write_bytes(b'r6d');assert probe.read_bytes()==b'r6d';probe.unlink()
        snapshot=work/'snapshot';snapshot.mkdir();stage='snapshot';t=time.monotonic()
        for name,relative in mapping.items():before[name]=reader.record(relative,snapshot/name)
        if before['current_result.json']!=layout_before['pointer_record']:raise ValueError('pointer changed before snapshot')
        write(work/'source-before.private.json',before);check_snapshot(before,snapshot);snapshot_checked=True
        for name in before:snapshots[name]=digest(snapshot/name)
        check_layout(reader,addendum,layout_before)
        snapshot_seconds=time.monotonic()-t
        packet=read(snapshot/'packet.private.json');reg=packet['assets']['registration']
        v=validators(root,inventory['total_bytes'])
        if v['fingerprint']()!=spec['historical_binding']['production_fingerprint']:raise ValueError('production source fingerprint')
        if digest(root/'configs/r6_science_v1.json')!=spec['historical_binding']['science_sha256']:raise ValueError('science bytes')
        if packet['binding']!=spec['historical_binding']:raise ValueError('packet binding')
        stage='fixed_scalar_analysis';analysis_started=True;analysis_start=time.monotonic()
        receipt,rows,_=v['completed'](snapshot,reg,'A')
        if receipt['binding']!=spec['historical_binding']:raise ValueError('historical binding')
        pointer=read(snapshot/'current_result.json')
        if pointer['binding']!=receipt['binding'] or not pointer['valid'] or pointer['status']!='R6A_COMPLETE_NO_ADVANCE':raise ValueError('original pointer')
        original=read(snapshot/'public_aggregate.json')
        baseline=reconstruct(rows,v,original,spec)
        summary=analysis(rows,v['stream_summary'](reg),spec,work/'public')
        summary.update(analysis_seconds=time.monotonic()-analysis_start,snapshot_seconds=snapshot_seconds)
        write(work/'public/BASELINE_RECONSTRUCTION.json',baseline)
        tables_complete=True;stage='source_after_check'
    except BaseException:
        capture_error(errors,'primary',work if ready else None)
    finally:
        if ready:
            # Each after-check is independent; no exception replaces the first failure.
            if layout_before is not None:
                try:
                    layout_after=check_layout(reader,addendum,layout_before)
                    write(work/'layout-after.private.json',layout_after)
                except BaseException:capture_error(errors,'after_layout',work)
            for name in before:
                try:after[name]=reader.record(mapping[name])
                except BaseException:capture_error(errors,'after_payload',work)
            if before:
                try:
                    check_snapshot(before,work/'snapshot')
                    if len(before)!=len(mapping) or before!=after:raise ValueError('full source before/after mismatch')
                except BaseException:capture_error(errors,'after_snapshot',work)
            if inventory is not None:
                try:
                    inventory_after=source_inventory(source)
                    if inventory!=inventory_after:raise ValueError('source inventory mismatch')
                except BaseException:capture_error(errors,'after_inventory',work)
            write(work/'source-after.private.json',after)
            pub=work/'public';pub.mkdir(exist_ok=True)
            full=bool(before) and len(before)==len(mapping) and len(after)==len(mapping)
            evidence=[dict(file_token=f'input_{i:03}',bytes=before.get(n,{}).get('bytes'),before_sha256=before.get(n,{}).get('sha256'),after_sha256=after.get(n,{}).get('sha256'),snapshot_sha256=snapshots.get(n)) for i,n in enumerate(mapping)]
            audit=dict(inferential_status='POST_HOC_EXPLORATORY',source_bytes_unchanged=before==after if full else None,
                source_metadata_unchanged=before==after if full else None,source_inventory_unchanged=inventory==inventory_after if inventory is not None and inventory_after is not None else None,
                result_pointer_unchanged=layout_before['pointer_record']==layout_after['pointer_record'] if layout_before and layout_after else None,
                publication_link_metadata_unchanged=layout_before['links']==layout_after['links'] if layout_before and layout_after else None,
                ordinary_file_snapshot=snapshot_checked if full else None,snapshot_bytes=required if snapshot_checked else None,
                source_output_bytes=inventory['total_bytes'] if inventory else None,free_bytes_before=free,output_source_disjoint=True,
                evidence=evidence,missing_evidence_policy='null means NOT_CAPTURED/NOT_PERFORMED; never a successful check',
                raw_asset_reads=0,model_imports=0,gpu_queries_or_initializations=0,alias_content_reads=0,
                safeguards='Fixed-version pointer; lstat/readlink aliases only; directory-descriptor O_DIRECTORY/O_NOFOLLOW walk; fstat and path recheck; unchanged scalar validators and audit hook.',
                limitations=['Python audit hook is not an OS sandbox.','Byte checks cover allowlisted scalar payloads; other source files have metadata inventory only.',
                            'Protected mtime/ctime/mode/inode/device/nlink exclude filesystem-managed atime. No source timestamp restoration.',
                            'Concurrent hostile changes between checks cannot be ruled out absolutely; descriptor/path/layout checks detect observed replacement.'],
                prior_attempt='fad933d9c49dd1ae034a9368785381fb89c32cf4; old missing hashes/telemetry remain NOT_CAPTURED',**fixed)
            public_safe(audit);write(pub/'READONLY_AUDIT.json',audit)
            write(pub/'ANALYSIS_BINDING.json',dict(inferential_status='POST_HOC_EXPLORATORY',historical_binding=spec['historical_binding'] if 'spec' in locals() else None,
                historical_publication_sha='aa732b42b03d378149028a3770879b72ebc55db3',analysis_sha=config['analysis_sha'],
                analysis_spec_sha256=digest(specpath) if 'specpath' in locals() else None,input_adaptation_addendum_sha256=digest(addpath) if 'addpath' in locals() else None,
                inputs=evidence,source_file_mapping='PRIVATE_ONLY',prior_attempt='fad933d9c49dd1ae034a9368785381fb89c32cf4',**fixed))
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result=dict(analysis_executed=analysis_started,analysis_status='R6D_INCOMPLETE' if errors or not tables_complete else 'ANALYSIS_TABLES_COMPLETE_INTERPRETATION_PENDING',
            stage=stage,errors=errors,wall_seconds=time.monotonic()-begin,cpu_user_seconds=usage.ru_utime-cpu0.ru_utime,cpu_system_seconds=usage.ru_stime-cpu0.ru_stime,
            peak_rss_platform_units=usage.ru_maxrss,peak_rss_unit='KiB on Linux; bytes on macOS',max_threads=2,processes=1,**fixed)
        result.update(summary)
        if ready:
            result['new_output_bytes_excluding_snapshot']=sum(p.stat().st_size for p in work.rglob('*') if p.is_file() and work/'snapshot' not in p.parents)
            if result['new_output_bytes_excluding_snapshot']>2147483648:
                result['analysis_status']='R6D_INCOMPLETE';result['errors'].append(dict(phase='output_cap',type='ValueError'))
            public_safe(result);write(work/'public/RUN_LOG.json',result)
        else:result['telemetry_file']='NOT_WRITTEN_UNSAFE_OR_UNAVAILABLE_WORK_DIRECTORY'
        signal.alarm(0);signal.signal(signal.SIGALRM,previous_alarm)
    print(json.dumps(result,allow_nan=False),flush=True)
    if result['analysis_status']=='R6D_INCOMPLETE':raise RuntimeError('R6-D incomplete; preserve first failure and after-check errors; no automatic retry')


if __name__=='__main__':
    main(read(os.environ['R6D_CONFIG']))
