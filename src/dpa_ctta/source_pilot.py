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


def _run_arm(host, rows, pixel_reader, evaluator, *, capture=None):
    """Shared sequential loop; failures end this arm and the enclosing pilot immediately."""
    records = []
    if host.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(host.device)
    for segment in SEGMENTS:
        for row in rows:
            if host.device.type == 'cuda':
                torch.cuda.synchronize(host.device)
            start = time.perf_counter()
            image = transform_pixels(pixel_reader(row), segment)
            before = host.prompt.data_prompt.detach().clone() if hasattr(host, 'prompt') else None
            prediction = host.step(image)
            if not torch.isfinite(prediction).all():
                raise ValueError('nonfinite prediction')
            update, counts, adam, memory, proxy_loss = 0., [], 0, 0, None
            if before is not None:
                p = host.prompt.data_prompt
                if not torch.isfinite(p).all() or p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError('nonfinite/missing prompt update')
                update = float((p.detach() - before).norm())
                counts = sorted({m.sample_num for m in host.model.modules() if isinstance(m, host.adabn)})
                adam = int(host.optimizer.state[p]['step'])
                memory = host.memory_bank.get_size()
                if host.last_proxy_loss is not None:
                    loss = host.last_proxy_loss
                    proxy_loss = dict(region=float(loss.region.detach()), boundary=float(loss.boundary.detach()))
                    if not all(np.isfinite(v) for v in proxy_loss.values()):
                        raise ValueError('nonfinite proxy loss')
            if capture is not None:  # tests only; production records contain no pixel predictions
                capture(prediction.detach().cpu(), host)
            metrics = evaluator(prediction, row)  # FIRST query label read, after complete native update/predict/push
            if host.device.type == 'cuda':
                torch.cuda.synchronize(host.device)
            elapsed = time.perf_counter() - start
            records.append(dict(group_id=row['group_id'], sample_id=row['sample_id'], segment=segment,
                                metrics=metrics, prompt_update_norm=update, native_counts=counts,
                                adam_step=adam, memory_size=memory, proxy_loss=proxy_loss,
                                elapsed_seconds=elapsed, peak_cuda_bytes=torch.cuda.max_memory_allocated(host.device) if host.device.type == 'cuda' else 'NOT_RUN'))
    return records


def summarize(results):
    """Paired comparisons retain per-channel cohorts and every observed negative delta."""
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
                    assd_valid_cohort=[[r['group_id'],r['segment']] for r, _ in valid])
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
                    visits=len(keys), independent_groups=len({k[0] for k in keys}),
                    assd_same_pair_delta=float(np.mean([a[k]['assd'] - b[k]['assd'] for k in both])) if both else None,
                    assd_same_pair_cohort=[list(k) for k in both],
                    assd_valid={left: sum(a[k]['assd'] is not None for k in keys), right: sum(b[k]['assd'] is not None for k in keys)},
                    assd_undefined={left: sum(a[k]['assd'] is None for k in keys), right: sum(b[k]['assd'] is None for k in keys)})
            output[segment][left + '-' + right] = compared
    return output


def run_registered(config):
    """Future real entry: cannot read roots, metadata, images or weights before approval."""
    require_pilot_approval()
    results = {}
    try:
        validate_config(config)
        for task, spec in config['tasks'].items():
            registration = registered_source(os.environ[spec['manifest_env']], os.environ[spec['split_env']],
                                             spec['csv_specs'], os.environ[spec['data_root_env']], task, config['seed'])
            state = torch.load(os.environ[spec['checkpoint_env']], weights_only=True, map_location='cpu')
            proxy = source_proxy(registration['proxy'], task)
            task_results = {}
            for arm in ARMS:
                host = assemble(task, arm, state, proxy, device=config['device'], seed=config['seed'])
                task_results[arm] = _run_arm(host, registration['query'],
                    lambda r: read_pixels(r['image_path'], task, r['image_size']),
                    lambda pred, r: evaluate_after_step(pred, read_mask(r['mask_path'], task, r['image_size']), task))
                if any(not torch.equal(t.cpu(), state[k].cpu()) for k, t in host.model.state_dict().items()):
                    raise ValueError('source weights/buffers changed')
                del host
            results[task] = dict(arms=task_results, comparisons=summarize(task_results))
    except Exception as error:
        # Discard mutable host state. Never continue with a polluted optimizer/memory.
        return dict(status='INCOMPLETE', reason=type(error).__name__, completed_tasks=results)
    return dict(status='COMPLETE', tasks=results)


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
