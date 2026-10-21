"""R94: existing content scores only; no model, image, mask, or label IO."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import io
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np

START = time.time()
METRICS = ('Dice', 'FP', 'FN', 'foreground', 'ASSD_px', 'HD95_px',
           'vCDR_abs_error', 'nesting_pixel_percent', 'nesting_present_percent',
           'pred_empty', 'gt_empty')
SEEDS = (20260907, 17011, 29009)
ARMS = ('SOURCE_ONLY', 'P0', 'Q0', 'NATIVE', 'K')


def usage(root):
    return sum(max(p.stat().st_size, p.stat().st_blocks * 512)
               for p in (root, *root.rglob('*')) if not p.is_symlink())


def finite_mean(a):
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def stats(a):
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(mean=None, median=None, p10=None, p90=None, minimum=None,
                    maximum=None, positive=0, negative=0, zero=0)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                minimum=float(a.min()), maximum=float(a.max()),
                positive=int((a > 0).sum()), negative=int((a < 0).sum()),
                zero=int((a == 0).sum()))


def weight_matrix(rng, n, draws=2000):
    return np.array([np.bincount(rng.integers(n, size=n), minlength=n)
                     for _ in range(draws)], dtype=np.float64) / n


def paired_draws(delta, weights):
    # delta: content x anchor x head x metric. Every draw retains all anchors.
    ok = np.isfinite(delta)
    num = np.where(ok, delta, 0).sum(axis=1)
    den = ok.sum(axis=1)
    n = np.einsum('bi,ihm->bhm', weights, num)
    d = np.einsum('bi,ihm->bhm', weights, den)
    return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)


def self_check():
    w = weight_matrix(np.random.default_rng(9), 5, 100)
    assert np.allclose(w.sum(1), 1)
    common = np.arange(5.)[:, None, None, None]
    base = np.broadcast_to(common, (5, 6, 2, 1)).copy()
    got = paired_draws((base + .4) - base, w)
    assert np.allclose(got, .4)
    cancel = np.broadcast_to(np.array([1., -1., 1., -1., 1., -1.])[None, :, None, None], (5, 6, 2, 1))
    assert np.allclose(paired_draws(cancel, w), 0)
    assert np.allclose(np.mean([np.full(100, d) for d in range(4)], axis=0), 1.5)
    x = np.full((5, 6, 2, 1), np.nan)
    assert np.isnan(paired_draws(x, w)).all()
    return dict(status='PASS',constant_paired_difference=True,
                coupled_anchor_cancellation=True,equal_domain_macro=True,
                missing_pairs_preserved=True)


def run(c):
    root = Path(c['root']).resolve()
    public = root / 'public'
    deadline = c['deadline']
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('R94 CPU alarm')))
    signal.alarm(max(1, int(min(840, deadline - time.time() - 20))))
    resource.setrlimit(resource.RLIMIT_CPU, (840, 845))
    assert time.time() < deadline and c['new_total_reservation_bytes'] < 20_000_000_000
    inputs = {k: Path(v).resolve() for k, v in c['inputs'].items()}
    input_paths = set(inputs.values())
    opened = set()
    # Imports finish before restricting project IO; scientific inputs are exact allowlisted files.
    project = root.parent
    prefix = Path(sys.prefix).resolve()
    def guard(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in ('torch', 'cupy', 'tensorflow', 'jax'):
            raise PermissionError('model/GPU import denied')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            flags = args[2] or 0
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not p.is_relative_to(root):
                raise PermissionError('write outside new audit root')
            if p.is_relative_to(project) and not p.is_relative_to(root) and not p.is_relative_to(prefix):
                if p not in input_paths or writing:
                    raise PermissionError('non-allowlisted project input')
            if p in input_paths:
                if writing:
                    raise PermissionError('scientific input is readonly')
                opened.add(p)
    sys.addaudithook(guard)

    def write(name, value, table=False):
        assert time.time() < deadline - 10
        if table:
            assert value
            f = io.StringIO()
            writer = csv.DictWriter(f, fieldnames=list(value[0]))
            writer.writeheader()
            for row in value:
                writer.writerow({k: 'NA' if v is None else v for k, v in row.items()})
            data = f.getvalue().encode()
        else:
            data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        assert usage(root) + math.ceil(len(data) / 4096) * 4096 + 8192 < 7 * 1024**2
        with (public / name).open('xb') as f:
            f.write(data)

    checks = self_check()
    with np.load(inputs['r91_values'], allow_pickle=False) as f:
        values = f['values']; domains = f['domains']
    with np.load(inputs['m0_values'], allow_pickle=False) as f:
        old = f['values']; old_domains = f['domains']
    assert values.shape == (192, 6, 6, 2, 2, 11)
    assert old.shape == (192, 6, 4, 2, 2, 2, 11)
    assert values.dtype == old.dtype == np.float64
    assert np.array_equal(domains, old_domains)
    assert np.bincount(domains).tolist() == [22, 57, 57, 56]
    for a in (0, 4):
        assert np.allclose(values[:, :, a], values[:, :1, a], atol=0, rtol=0, equal_nan=True)
    assert np.allclose(old[:, :, 0, 1], old[:, :1, 0, 1], atol=0, rtol=0, equal_nan=True)
    assert np.isfinite(values[..., 0]).all()
    delta = values[:, :, 2] - values[:, :, 4]
    identity = (values[:, :, 3] - values[:, :, 4]) - (values[:, :, 3] - values[:, :, 2])
    finite = np.isfinite(delta) & np.isfinite(identity)
    residual = float(np.max(np.abs(delta[finite] - identity[finite])))
    assert residual < 1e-10
    complete = json.loads(inputs['r91_completion'].read_text())
    assert complete['status'] == 'COMPLETE' and complete['PRE_K_reference_exact_all']
    assert complete['all_raw_post_sealed_before_labels'] and complete['Q0_P0_identity_exact']
    sealed = json.loads(inputs['r91_sealed'].read_text())
    assert sealed['status'] == 'ALL_RAW_AND_FIXED_POST_SEALED'

    refs = np.stack((old[:, :, 0, 1], values[:, :, 0], values[:, :, 4],
                     values[:, :, 5], values[:, :, 2]), axis=2)
    lookup = {}
    for key in ('r91_scores', 'm0_scores'):
        with inputs[key].open() as f:
            for row in csv.DictReader(f):
                if row['arm'] not in ('P0', 'Q0', 'NATIVE', 'K', 'SOURCE_ONLY'):
                    continue
                if key == 'm0_scores' and row['arm'] != 'SOURCE_ONLY':
                    continue
                k = (row['arm'],row['variant'],row['domain'],row['seed'],row['order'],row['channel'],row['metric'])
                assert k not in lookup
                lookup[k] = row
    absolute=[]; parity=0; max_error=0.
    for a, arm in enumerate(ARMS):
        for v, variant in enumerate(('raw', 'post')):
            for d in range(4):
                for j in range(6):
                    for h, head in enumerate(('OD', 'OC')):
                        for m, metric in enumerate(METRICS):
                            x=refs[domains==d,j,a,v,h,m]
                            row=lookup[(arm,variant,f'D{d}',str(SEEDS[j//2]),str(j%2),head,metric)]
                            mean=finite_mean(x); na=int((~np.isfinite(x)).sum())
                            assert na == int(row['n_NA']) and len(x)==int(row['n'])
                            if mean is not None:
                                error=abs(mean-float(row['mean']));max_error=max(max_error,error)
                                assert error < 1e-10
                            else:
                                assert row['mean'] in ('NA', '')
                            parity+=1
                            absolute.append(dict(arm=arm,variant=variant,domain=f'D{d}',seed=SEEDS[j//2],order=j%2,channel=head,metric=metric,mean=mean,n_contents=len(x),n_NA=na))
    write('STATIC_REFERENCE_CELLS.csv',absolute,True)
    rng=np.random.default_rng(20261021)
    weights=[weight_matrix(rng,int((domains==d).sum())) for d in range(4)]
    summary=[]; cells=[]; distribution=[]; anchors=[]; lodo=[]; decision_details={}
    for v, variant in enumerate(('raw','post')):
        x=delta[:,:,v]
        raw_a=values[:,:,2,v];raw_b=values[:,:,4,v]
        draws=[paired_draws(x[domains==d],weights[d]) for d in range(4)]
        for m,metric in enumerate(METRICS):
            heads=[0] if metric=='vCDR_abs_error' else [0,1]
            a=np.array([finite_mean(np.array([finite_mean(x[domains==d,j][:,heads,m]) for d in range(4)],dtype=float)) for j in range(6)])
            db=np.mean([z[:,:,m][:,heads].mean(1) for z in draws],axis=0)
            assert np.isfinite(db).all(), 'bootstrap nonfinite; retain failure rather than drop draws'
            q=np.quantile(db,[.025,.975,.0125,.9875]).tolist()
            summary.append(dict(variant=variant,metric=metric,macro_delta=float(a.mean()),ci95_low=q[0],ci95_high=q[1],ci97_5_low=q[2],ci97_5_high=q[3],positive_anchors=int((a>0).sum()),negative_anchors=int((a<0).sum()),NA_K=int((~np.isfinite(raw_a[:,:,:,m][:,:,heads])).sum()),NA_Q0=int((~np.isfinite(raw_b[:,:,:,m][:,:,heads])).sum()),scope='equal_domain_coupled_content_posthoc'))
            for j in range(6):
                anchors.append(dict(variant=variant,metric=metric,seed=SEEDS[j//2],order=j%2,macro_delta=float(a[j])))
            for d in range(4):
                for h,head in enumerate(('OD','OC')):
                    xx=x[domains==d,:,h,m]
                    n=np.isfinite(xx).sum(1)
                    xxmean=np.divide(np.nansum(xx,1),n,out=np.full(len(xx),np.nan),where=n>0)
                    qq=np.quantile(draws[d][:,h,m],[.025,.975,.0125,.9875]).tolist()
                    cells.append(dict(variant=variant,domain=f'D{d}',channel=head,metric=metric,paired_mean_delta=finite_mean(xx),ci95_low=qq[0],ci95_high=qq[1],ci97_5_low=qq[2],ci97_5_high=qq[3],n_contents=len(xx),paired_finite=int(np.isfinite(xx).sum()),K_NA=int((~np.isfinite(raw_a[domains==d,:,h,m])).sum()),Q0_NA=int((~np.isfinite(raw_b[domains==d,:,h,m])).sum())))
                    distribution.append(dict(variant=variant,domain=f'D{d}',channel=head,metric=metric,n_contents=len(xx),NA_content_means=int((~np.isfinite(xxmean)).sum()),**stats(xxmean)))
        for omit in range(4):
            lodo.append(dict(variant=variant,omitted_domain=f'D{omit}',macro_Dice_delta=float(np.mean([x[domains==d,:,:,0].mean() for d in range(4) if d!=omit])),status='POST_HOC_NOT_A_GATE'))
        decision_details[variant]=dict(orders=[float(np.mean([r['macro_delta'] for r in anchors if r['variant']==variant and r['metric']=='Dice' and r['order']==o])) for o in (0,1)],seeds=[float(np.mean([r['macro_delta'] for r in anchors if r['variant']==variant and r['metric']=='Dice' and r['seed']==s])) for s in SEEDS])
    for name,rows in [('SUMMARY.csv',summary),('DOMAIN_HEAD.csv',cells),('CONTENT_DISTRIBUTIONS.csv',distribution),('ANCHORS.csv',anchors),('LODO.csv',lodo)]:
        write(name,rows,True)
    checks.update(existing_aggregate_parity_checks=parity,existing_aggregate_max_error=max_error,per_content_identity_max_residual=residual)
    write('SELF_CHECK.json',checks)
    write('ASSET_AUDIT.json',dict(status='CONTENT_SCORES_AVAILABLE',r91_shape=list(values.shape),m0_shape=list(old.shape),dtype='float64',domain_quotas=np.bincount(domains).tolist(),canonical_alignment='same original order0 panel by preserved scorer code; domains equality and published aggregate parity verified',original_scorer_commit='b75b39a',canonical_identity_rederived=False,no_new_identity_claim=True,Q0_unique_contents=192,Q0_broadcast_anchors=6,content_identifiers_published=False,read_existing_score_files=True,raw_predictions_read=False,images_read=0,labels_read=0,models_loaded=0,asset_hashes_computed=0))
    write('EVIDENCE_ADMISSION.json',dict(status='RETAIN_VV_NO_ADMISSIBLE_GPU_SUCCESSOR',not_in_development_selection='NO_NEW_AUTHORIZED_UNEXPOSED_CONTENT_EVIDENCE',patient_eye_repeat_independence='UNKNOWN',source_pretraining_overlap='UNKNOWN',new_mechanism_evidence='NOT_ESTABLISHED',R87='NOT_AVAILABLE',R88='PRECONDITION_NOT_MET',audit_scope='new evidence supplied to this control session since R87/R88; no repeated dataset search; no new registry or identity roster supplied',new_GPU_stage_authorized=False,old_gates_unchanged=True,old_deadline_unchanged=c['old_campaign_deadline'],old_updates_unchanged=4253,old_forwards_unchanged=43700))
    write('ACCESS_AUDIT.json',dict(status='PASS',allowed_input_roles=sorted(inputs),opened_input_roles=sorted(k for k,p in inputs.items() if p in opened),new_model_calls=0,new_GPU_forwards=0,new_GPU_backwards=0,new_GPU_updates=0,new_image_reads=0,new_label_reads=0,new_prediction_reads=0,scientific_inputs_readonly=True,project_read_allowlist_enforced=True,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
    write('COMPLETION.json',dict(status='COMPLETE',CPU_body_seconds=time.time()-START,T0=c['T0'],deadline=deadline,bootstrap_draws=2000,bootstrap_seed=20261021,paired_contents=192,anchors=6,metrics=len(METRICS),absolute_reference_rows=len(absolute),domain_head_rows=len(cells),content_distribution_rows=len(distribution),posthoc_development_only=True,old_campaign_clock_not_reset=True,**decision_details))
    print(json.dumps(dict(status='COMPLETE',CPU_body_seconds=time.time()-START)),flush=True)


if __name__=='__main__':
    config=json.loads(Path(os.environ['CONFIG']).read_text())
    try:
        run(config)
    except BaseException as exc:
        # The wrapper retains a failure receipt even when the worker cannot write.
        print(json.dumps(dict(status='NOT_EVALUABLE',error=type(exc).__name__,reason=str(exc))),flush=True)
        raise
