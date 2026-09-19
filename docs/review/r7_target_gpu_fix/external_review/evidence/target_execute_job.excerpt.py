def error(exc): return dict(type=type(exc).__name__, message=str(exc)[:3000])


def execute_job(approved, job, out):
    cfg = approved['config']; b = approved['binding']; started = time.monotonic()
    caps = dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes'])
    check = lambda: resource_check(out, caps, started)
    images = TargetReader(b['target_root'], cfg['max_asset_bytes'], 'image')
    masks = TargetReader(b['target_root'], cfg['max_asset_bytes'], 'mask')
    first = None; secondary = []; host = None
    try:
        rows = stream(approved['registration'], job['order'])
        if len(rows) != job['arrivals'] or sum(r['subset']=='remaining_dev' for r in rows) != job['scored_contents']:
            raise ValueError('job stream coverage')
        host, state = make_host(approved, job['arm'])
        counts = online(host, image_records(rows), images, job['arm'], out, check)
        expected = dict(forwards=job['network_forwards'], backwards=job['backwards'], Adam=job['Adam'])
        if counts != expected: raise ValueError('terminal physical counts')
        if job['arm'] == 'C_BASE': host.finish(state)
        else: host._check_frozen(boundary=True); host.segmenter.close()
        # Drop all model/optimizer/state references before the first target mask read.
        del host, state; host = None
        posthoc(rows, masks, job['arm'], job['order'], out, check)
        out.write('counts.json', counts)
    except BaseException as exc:
        first = exc
    finally:
        try:
            if host is not None:
                if hasattr(host, 'segmenter'): host.segmenter.close()
                else:
                    for h in host.handles: h.remove()
        except BaseException as exc: secondary.append(error(exc)); first = first or exc
        for reader in (images, masks):
            try: reader.after_check()
            except BaseException as exc: secondary.append(error(exc)); first = first or exc
        # After-read metadata and checkpoint/artifacts remain immutable too.
        try:
            for key in ('registration', 'inventory', 'checkpoint'):
                entry = b[key]; verified(entry['path'], entry['sha256'], cfg['max_asset_bytes'])
            for name, row in approved['inventory'].items():
                verified(Path(b['artifact_root'])/row['file'], row['training_asset_file_sha256'], cfg['max_asset_bytes'])
                verified(Path(b['artifact_root'])/(name+'.context.json'), row['context_file_sha256'], 1024**2)
            check()
        except BaseException as exc: secondary.append(error(exc)); first = first or exc
        records = [('worker.json', dict(status='FAILED' if first else 'JOB_PENDING_TERMINAL_AUDIT',
                   target_after_check='FAILED' if secondary else 'UNCHANGED', secondary=secondary,
                   image_IO=dict(images.counts), mask_IO=dict(masks.counts), wall_seconds=time.monotonic()-started, retry=False))]
        if first: records.insert(0, ('first_error.json', error(first)))
        for name, value in records:
            try: out.evidence(name, value)
            except BaseException as exc:
                first = first or exc
                try: print(json.dumps(dict(evidence_error=error(exc), first_error=error(first))), file=sys.stderr, flush=True)
                except BaseException: pass
    if first: raise first
