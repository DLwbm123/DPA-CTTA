"""Reviewable source-pilot assembly. Real run is unconditionally approval-blocked."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .hosts.vptta import VPTTAHost, load_source_state, model_input_from_pixels
from .integrations.ctta_suite import build_reference_model
from .medical_losses import medical_loss
from .source_io import SOURCES, read_pixels, read_mask, registered_source, signed_distance, source_proxy

APPROVAL_REQUIRED = 'PILOT_COMMIT_CONFIG_APPROVAL_REQUIRED'
ARMS = {'N': None, 'A': (0., 0.), 'B': (.1, 0.), 'C': (.1, .1)}
SEGMENTS = ('clean', 'gamma_0.7', 'gamma_1.5', 'blur_5_sigma_1')
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/source_pilot_v0.json'


def validate_config(config):
    """Reject ignored/mutated scientific settings instead of silently using constants."""
    expected = json.loads(DEFAULT_CONFIG.read_text())
    if config != expected:
        raise ValueError('config differs from this commit draft; a new frozen review is required')
    if (config['enabled'] is not False or config['status'] != 'draft' or config['device'] != 'cpu'
            or config['seed'] != 20260907 or set(config['tasks']) != set(SOURCES)
            or config['native'] != dict(lr={'fundus':.05,'polyp':.01},betas=[.9,.99],weight_decay=0,
                                        iters=1,warm_n=5,neighbor=16,memory_size=40,prompt_alpha=.01)
            or tuple(config['query']['segments']) != SEGMENTS or config['query']['reset_between_segments'] is not False
            or config['proxy']['K'] != 4 or config['proxy']['input'] != 'clean' or config['proxy']['batch'] != 'full'):
        raise ValueError('unsupported pilot scientific/device settings')
    for task, source in SOURCES.items():
        spec = config['tasks'][task]
        if spec['source'] != source or spec['query_groups'] != (20 if task == 'fundus' else 32):
            raise ValueError('source/query registration settings mismatch')
    for arm in ARMS:
        if arm == 'N':
            if config['arms'][arm] != {'method':'source_only_standard_bn'}:
                raise ValueError('No Adapt settings mismatch')
        elif config['arms'][arm] != dict(method='native_vptta' if arm == 'A' else 'proxy_rehearsal',
                                         extra_weight=ARMS[arm][0], beta_boundary=ARMS[arm][1]):
            raise ValueError('arm settings mismatch')


def require_pilot_approval():
    # No CLI flag/environment variable can turn a core review into runner approval.
    raise RuntimeError(APPROVAL_REQUIRED)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SourceOnlyHost:
    def __init__(self, task, source_state, device='cpu'):
        self.task, self.device = task, torch.device(device)
        self.model, self.logits = build_reference_model(task)
        load_source_state(self.model, source_state)
        self.model.to(self.device)
        if any(isinstance(m, torch.nn.BatchNorm2d) and type(m) is not torch.nn.BatchNorm2d for m in self.model.modules()):
            raise RuntimeError('No Adapt requires standard source BN')

    def step(self, pixel_rgb):
        with torch.no_grad():
            return self.logits(self.model, model_input_from_pixels(pixel_rgb, self.task).to(self.device))


def assemble(task, arm, source_state, proxy=None, *, device='cpu', seed=20260907):
    seed_all(seed)
    if arm == 'N':
        return SourceOnlyHost(task, source_state, device)
    weight, beta = ARMS[arm]
    return VPTTAHost(task, source_state=source_state, device=device,
                     mode='proxy_rehearsal' if weight else 'base', extra_weight=weight,
                     beta_boundary=beta, proxy_factory=lambda: proxy)


def transform_pixels(pixels, segment):
    if segment == 'clean':
        return pixels
    if segment in ('gamma_0.7', 'gamma_1.5'):
        return pixels.pow(.7 if segment == 'gamma_0.7' else 1.5)
    if segment != 'blur_5_sigma_1':
        raise ValueError('unregistered query transform')
    axis = torch.arange(-2, 3, dtype=pixels.dtype, device=pixels.device)
    kernel = torch.exp(-axis.square() / 2)
    kernel = kernel / kernel.sum()
    weight = (kernel[:, None] * kernel[None, :]).expand(3, 1, 5, 5)
    return F.conv2d(F.pad(pixels, (2, 2, 2, 2), mode='reflect'), weight, groups=3).clamp(0, 1)


def evaluate_after_step(logits, mask, task):
    """Labels are obtained by the caller only after host.step returns."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    logits = logits.detach().cpu()
    distance, defined = signed_distance(mask)
    medical_loss(logits, mask, distance, .1)  # shape/dtype/value validation
    if len(logits) != 1 or logits.shape[1] != (2 if task == 'fundus' else 1):
        raise ValueError('query label/output channel mismatch')
    result = []
    for c, name in enumerate(('OD', 'OC') if task == 'fundus' else ('polyp',)):
        hard = logits[0, c].sigmoid().numpy() >= .5
        gt = mask[0, c].numpy().astype(bool)
        n = int(hard.sum()) + int(gt.sum())
        loss = medical_loss(logits[:, c:c+1], mask[:, c:c+1], distance[:, c:c+1], .1)
        assd = None
        if hard.any() and gt.any():
            # 4-connected pixel surfaces, including image border; pixel units.
            a = hard ^ binary_erosion(hard, border_value=0)
            b = gt ^ binary_erosion(gt, border_value=0)
            assd = float(np.concatenate((distance_transform_edt(~b)[a], distance_transform_edt(~a)[b])).mean())
        result.append(dict(channel=name, dice=1. if not n else float(2 * (hard & gt).sum() / n),
                           region=float(loss.region), boundary=float(loss.boundary),
                           boundary_defined=bool(defined[0, c]), assd=assd,
                           gt_empty=not bool(gt.any()), gt_full=bool(gt.all()),
                           pred_empty=not bool(hard.any()), pred_full=bool(hard.all())))
    return result


class ArmFailure(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('INCOMPLETE: ' + details['exception_type'])


def expected_visits(rows):
    identities = [(r['sample_id'], r['group_id']) for r in rows]
    if not rows or len({s for s,g in identities}) != len(rows) or len({g for s,g in identities}) != len(rows):
        raise ValueError('query registration requires distinct samples/groups')
    return [dict(visit=i+1, sample_id=r['sample_id'], group_id=r['group_id'], segment=segment)
            for i,(segment,r) in enumerate((s,r) for s in SEGMENTS for r in rows)]


def validate_arm(records, expected, task, arm):
    channels = ['OD','OC'] if task == 'fundus' else ['polyp']
    if arm not in ARMS or not expected or len(records) != len(expected):
        raise ValueError('arm coverage mismatch')
    seen = set()
    for row, wanted in zip(records, expected):
        if any(row.get(k) != v for k,v in wanted.items()):
            raise ValueError('visit identity/order mismatch')
        key = (row['sample_id'],row['group_id'],row['segment'])
        if key in seen:
            raise ValueError('duplicate visit')
        seen.add(key)
        metrics = row['metrics']
        if [m['channel'] for m in metrics] != channels:
            raise ValueError('missing/duplicate/wrong metric channel')
        for m in metrics:
            for field in ('dice','region','boundary'):
                if type(m[field]) not in (int,float) or not np.isfinite(m[field]):
                    raise ValueError('nonfinite metric')
            if not 0 <= m['dice'] <= 1 or (m['assd'] is not None and
                    (type(m['assd']) not in (int,float) or not np.isfinite(m['assd']) or m['assd'] < 0)):
                raise ValueError('metric range mismatch')
            for flag in ('boundary_defined','gt_empty','gt_full','pred_empty','pred_full'):
                if type(m[flag]) is not bool:
                    raise ValueError('metric flag must be explicit bool')
            if m['gt_empty'] and m['gt_full'] or m['pred_empty'] and m['pred_full']:
                raise ValueError('contradictory empty/full flags')
            if m['boundary_defined'] != (not m['gt_empty'] and not m['gt_full']):
                raise ValueError('boundary definition mismatch')
            if (m['assd'] is None) != (m['gt_empty'] or m['pred_empty']):
                raise ValueError('ASSD definition mismatch')
        for field in ('prompt_state_delta_norm','optimizer_update_norm','pipeline_elapsed_seconds','host_step_elapsed_seconds'):
            if not np.isfinite(row[field]) or row[field] < 0:
                raise ValueError('nonfinite/negative observation')
        if row['source_versions_unchanged'] is not True or row['optimizer_state_finite'] is not True:
            raise ValueError('invalid source/optimizer state')
        if arm == 'N':
            if row['adam_step'] != 0 or row['native_counts'] != [] or row['memory_size'] != 0 or row['optimizer_steps_this_visit'] != 0 or row['optimizer_update_norm'] != 0 or row['prompt_state_delta_norm'] != 0:
                raise ValueError('No Adapt unexpectedly adapted')
        elif (row['adam_step'] != wanted['visit'] or row['native_counts'] != [wanted['visit']]
              or row['optimizer_steps_this_visit'] != 1 or not 1 <= row['memory_size'] <= min(41,wanted['visit'])):
            raise ValueError('native lifecycle mismatch')
    return True


def _run_arm(host, rows, pixel_reader, evaluator, *, capture=None, record_sink=None, task=None, arm=None):
    """One sequential stream; a sink persists only completed scalar records."""
    records, handles = [], []
    observed = {'updates':0, 'norm':0.}
    source = {n:(t,t._version) for n,t in host.model.state_dict(keep_vars=True).items()}
    if hasattr(host,'optimizer'):
        def pre(opt,args,kwargs):
            observed['before'] = host.prompt.data_prompt.detach().clone()
        def post(opt,args,kwargs):
            observed['norm'] += float((host.prompt.data_prompt.detach()-observed.pop('before')).norm())
            observed['updates'] += 1
        handles = [host.optimizer.register_step_pre_hook(pre), host.optimizer.register_step_post_hook(post)]
    stage, visit, segment = 'initialization', 0, None
    if host.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(host.device)
    try:
        for segment in SEGMENTS:
            for row in rows:
                visit += 1
                stage = 'image_and_transform'
                if host.device.type == 'cuda': torch.cuda.synchronize(host.device)
                start = time.perf_counter()
                image = transform_pixels(pixel_reader(row), segment)
                before = host.prompt.data_prompt.detach().clone() if hasattr(host,'prompt') else None
                observed.update(updates=0,norm=0.)
                stage = 'host_step'
                if host.device.type == 'cuda': torch.cuda.synchronize(host.device)
                step_start = time.perf_counter()
                prediction = host.step(image)
                if host.device.type == 'cuda': torch.cuda.synchronize(host.device)
                step_elapsed = time.perf_counter()-step_start
                stage = 'state_validation'
                if not torch.isfinite(prediction).all(): raise ValueError('nonfinite prediction')
                current = host.model.state_dict(keep_vars=True)
                if current.keys() != source.keys() or any(current[n] is not t or t._version != version for n,(t,version) in source.items()):
                    raise ValueError('source parameter/buffer mutation')
                update, counts, adam, memory, proxy_loss = 0., [], 0, 0, None
                if before is not None:
                    p = host.prompt.data_prompt
                    if not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('nonfinite/missing prompt update')
                    if any(isinstance(t,torch.Tensor) and not torch.isfinite(t).all()
                           for state in host.optimizer.state.values() for t in state.values()):
                        raise ValueError('nonfinite Adam state')
                    update = float((p.detach()-before).norm())
                    counts = sorted({m.sample_num for m in host.model.modules() if isinstance(m,host.adabn)})
                    adam = int(host.optimizer.state.get(p,{}).get('step',0))
                    memory = host.memory_bank.get_size()
                    if host.last_proxy_loss is not None:
                        loss = host.last_proxy_loss
                        proxy_loss = dict(region=float(loss.region.detach()),boundary=float(loss.boundary.detach()),defined_boundary_pairs=loss.defined_boundary_pairs)
                        if not all(np.isfinite(v) for v in proxy_loss.values()): raise ValueError('nonfinite proxy loss')
                if capture is not None: capture(prediction.detach().cpu(),host)
                stage = 'query_evaluator'
                metrics = evaluator(prediction,row)  # First query label read after native step/push.
                if host.device.type == 'cuda': torch.cuda.synchronize(host.device)
                record = dict(visit=visit,group_id=row['group_id'],sample_id=row['sample_id'],segment=segment,
                    metrics=metrics,prompt_state_delta_norm=update,optimizer_update_norm=observed['norm'],
                    optimizer_steps_this_visit=observed['updates'],native_counts=counts,adam_step=adam,memory_size=memory,
                    proxy_loss=proxy_loss,source_versions_unchanged=True,optimizer_state_finite=True,
                    pipeline_elapsed_seconds=time.perf_counter()-start,host_step_elapsed_seconds=step_elapsed,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(host.device) if host.device.type=='cuda' else 'NOT_RUN')
                stage = 'record_validation'
                if task is not None:
                    validate_arm([record],[dict(visit=visit,sample_id=row['sample_id'],group_id=row['group_id'],segment=segment)],task,arm)
                stage = 'record_sink'
                if record_sink is not None: record_sink(record)
                records.append(record)
    except Exception as error:
        raise ArmFailure(dict(task=task,arm=arm,visit=visit,segment=segment,stage=stage,
                              exception_type=type(error).__name__,private_reason=str(error),completed_records=len(records))) from error
    finally:
        for handle in handles: handle.remove()
    return records


def summarize(results, expected=None, task=None):
    """Paired comparisons retain per-channel cohorts and every observed negative delta."""
    if expected is None or task not in SOURCES or set(results) != set(ARMS):
        raise ValueError('complete N/A/B/C and preregistered expected stream required')
    for arm, records in results.items():
        validate_arm(records, expected, task, arm)
    output = {}
    for segment in ('all', *SEGMENTS):
        output[segment] = {'arm_metrics': {}}
        for arm, rows in results.items():
            measurements = [(r, m) for r in rows if segment in ('all', r['segment']) for m in r['metrics']]
            output[segment]['arm_metrics'][arm] = {}
            for channel in sorted({m['channel'] for _, m in measurements}):
                pairs = [(r, m) for r, m in measurements if m['channel'] == channel]
                valid = [(r, m) for r, m in pairs if m['assd'] is not None]
                output[segment]['arm_metrics'][arm][channel] = dict(
                    **{key:float(np.mean([m[key] for _, m in pairs])) for key in ('dice','region','boundary')},
                    assd_conditional_mean=float(np.mean([m['assd'] for _, m in valid])) if valid else None,
                    assd_valid=len(valid), assd_undefined=len(pairs)-len(valid),
                    assd_valid_cohort=[[r['group_id'],r['segment']] for r, _ in valid],
                    **{flag+'_count':sum(m[flag] for _,m in pairs) for flag in ('gt_empty','gt_full','pred_empty','pred_full','boundary_defined')},
                    visits=len(pairs))
            output[segment]['arm_metrics'][arm]['macro'] = dict(dice_fraction=float(np.mean([m['dice'] for _,m in measurements])),
                dice_percent=100*float(np.mean([m['dice'] for _,m in measurements])))
        for left, right in (('B', 'A'), ('C', 'A'), ('C', 'B')):
            a = {(r['group_id'], r['segment'], m['channel']): m for r in results[left]
                 if segment in ('all', r['segment']) for m in r['metrics']}
            b = {(r['group_id'], r['segment'], m['channel']): m for r in results[right]
                 if segment in ('all', r['segment']) for m in r['metrics']}
            if not a or a.keys() != b.keys():
                raise ValueError('comparison cohort mismatch')
            compared = {}
            for channel in sorted({key[2] for key in a}):
                keys = [k for k in a if k[2] == channel]
                both = [k for k in keys if a[k]['assd'] is not None and b[k]['assd'] is not None]
                compared[channel] = dict(dice_delta=float(np.mean([a[k]['dice'] - b[k]['dice'] for k in keys])),
                    dice_delta_percentage_points=100*float(np.mean([a[k]['dice']-b[k]['dice'] for k in keys])),
                    visits=len(keys), independent_groups=len({k[0] for k in keys}),
                    assd_same_pair_delta=float(np.mean([a[k]['assd'] - b[k]['assd'] for k in both])) if both else None,
                    assd_same_pair_cohort=[list(k) for k in both],
                    assd_valid={left: sum(a[k]['assd'] is not None for k in keys), right: sum(b[k]['assd'] is not None for k in keys)},
                    assd_undefined={left: sum(a[k]['assd'] is None for k in keys), right: sum(b[k]['assd'] is None for k in keys)})
            output[segment][left + '-' + right] = compared
    output['units'] = dict(dice='fraction_0_1',dice_percent='percent',dice_delta='fraction',dice_delta_percentage_points='percentage_points',assd='final_grid_pixels',pipeline_elapsed_seconds='end_to_end_seconds',host_step_elapsed_seconds='adapt_predict_push_seconds')
    return output


def run_registered(config):
    """Future real entry: cannot read roots, metadata, images or weights before approval."""
    require_pilot_approval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('prepare', 'dry-run', 'run'), default='prepare')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    if args.action == 'run':
        print(json.dumps(dict(status='NOT_RUN', reason=APPROVAL_REQUIRED)))
        return 2
    raw = Path(args.config).read_bytes()
    config = json.loads(raw)
    validate_config(config)
    print(json.dumps(dict(status='NOT_RUN', reason=APPROVAL_REQUIRED, config=config,
                         config_sha256=hashlib.sha256(raw).hexdigest(), ci='NOT_CONFIGURED'), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
