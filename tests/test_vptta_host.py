import copy
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dpa_ctta.hosts.vptta import VPTTAHost, model_input_from_pixels
from dpa_ctta.integrations.ctta_suite import INPUT_SIZES
from dpa_ctta.proxy_loss import FixedProxy, ProxyProvenance
from test_medical_losses import half_plane

EXTERNAL = os.environ.get('DPA_CTTA_BASE_ROOT')
HOST_EVIDENCE = {}


def pixels(task, index=0):
    size = INPUT_SIZES[task]
    y = torch.linspace(0, 1, size).view(1, 1, size, 1)
    x = torch.linspace(0, 1, size).view(1, 1, 1, size)
    return torch.cat([(x + y).expand(1, 1, size, size) / 2,
                      ((x * (index + 1.3) + y * .3) % 1).expand(1, 1, size, size),
                      ((y * (index + 1.7) + x * .2) % 1).expand(1, 1, size, size)], 1).float()


def proxy(task, invert=False):
    mask, distance = half_plane(INPUT_SIZES[task], 2 if task == 'fundus' else 1)
    if invert:
        mask, distance = 1 - mask, -distance
    return FixedProxy(pixels(task, 2), mask, distance, ProxyProvenance.FIXTURE)


def same(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        test.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            same(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            same(test, x, y)
    else:
        test.assertEqual(a, b)


def native_snapshot(host):
    return {
        'prompt': copy.deepcopy(host.prompt.state_dict()),
        'adam': copy.deepcopy(host.optimizer.state_dict()),
        'counters': [(name, module.sample_num, module.new_sample) for name, module in host.model.named_modules()
                     if isinstance(module, host.adabn)],
        'memory': copy.deepcopy(host.memory_bank.__dict__),
        'buffers': {name: value.clone() for name, value in host.model.named_buffers()},
        'bn_loss': [module.bn_loss.clone() for module in host.model.modules() if isinstance(module, host.adabn) and hasattr(module, 'bn_loss')],
    }


class InputContractTests(unittest.TestCase):
    def test_task_preprocessing_has_no_hidden_second_normalization(self):
        for task in INPUT_SIZES:
            x = pixels(task)
            before = x.clone()
            result = model_input_from_pixels(x, task)
            if task == 'fundus':
                expected = (x - x.min()) / (x.max() - x.min())
            else:
                expected = (x - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
            self.assertTrue(torch.equal(result, expected))
            self.assertTrue(torch.equal(before, x))
        with self.assertRaises(ValueError):
            model_input_from_pixels(result, 'polyp')
        self.assertEqual(list(inspect.signature(VPTTAHost.step).parameters), ['self', 'pixel_rgb'])
        with self.assertRaises(ValueError):
            model_input_from_pixels(torch.zeros(1, 3, 512, 512), 'fundus')


@unittest.skipUnless(EXTERNAL, 'pinned CTTA checkout required for native full-model checks')
class NativeHostTests(unittest.TestCase):
    def test_four_image_zero_weight_sequence_matches_direct_native_block(self):
        for task in INPUT_SIZES:
            with self.subTest(task=task):
                torch.manual_seed(113)
                reference = VPTTAHost(task, neighbor=2)
                torch.manual_seed(113)
                bridge = VPTTAHost(task, mode='proxy_rehearsal', extra_weight=0, neighbor=2,
                                   proxy_factory=lambda: self.fail('disabled factory was evaluated'))
                self.assertFalse(hasattr(bridge, '_proxy_model'))
                source_before = {name: value.clone() for name, value in bridge.model.state_dict().items()}
                maxima = []
                for index in range(4):
                    image = pixels(task, index)
                    rng = torch.get_rng_state().clone()
                    direct = reference.native_step(reference, model_input_from_pixels(image, task))
                    after_reference = torch.get_rng_state().clone()
                    torch.set_rng_state(rng)
                    actual = bridge.step(image)
                    self.assertTrue(torch.equal(after_reference, torch.get_rng_state()))
                    self.assertTrue(torch.equal(direct, actual))
                    maxima.append(float((direct - actual).abs().max()))
                    same(self, native_snapshot(reference), native_snapshot(bridge))
                    for a, b in zip(reference.model.parameters(), bridge.model.parameters()):
                        same(self, a.grad, b.grad)
                same(self, source_before, bridge.model.state_dict())
                counters = [m.sample_num for m in bridge.model.modules() if isinstance(m, bridge.adabn)]
                self.assertEqual(set(counters), {4})
                self.assertEqual(float(bridge.optimizer.state[bridge.prompt.data_prompt]['step']), 4)
                self.assertGreaterEqual(bridge.memory_bank.get_size(), 2)
                HOST_EVIDENCE[task + '_disabled'] = {'images': 4, 'max_abs_logit_differences': maxima,
                    'prompt_adam_counters_memory_buffers_rng_equal': True, 'native_sample_counts': sorted(set(counters)),
                    'memory_entries': bridge.memory_bank.get_size(), 'adam_steps': 4,
                    'source_weights_buffers_unchanged': True, 'native_source_gradient_tensors': sum(p.grad is not None for p in bridge.model.parameters()),
                    'neighbor': 2, 'default_neighbor': 16,
                    'reference': 'unchanged pinned VPTTA.run image-block AST, native model/AdaBN/Prompt/Memory'}
                del reference, bridge, source_before

    def test_proxy_gradient_label_sensitivity_and_native_state_isolation(self):
        for task in INPUT_SIZES:
            with self.subTest(task=task):
                torch.manual_seed(116)
                host = VPTTAHost(task, mode='proxy_rehearsal', extra_weight=.2,
                                 proxy_factory=lambda: proxy(task), neighbor=2)
                # One real native step makes AdaBN and hooks observable before the probe.
                host.native_step(host, model_input_from_pixels(pixels(task), task))
                before = native_snapshot(host)
                base_grads = [None if p.grad is None else p.grad.clone() for p in host.model.parameters()]
                hook_features = [hook.features for hook in getattr(host.model, 'feature_hooks', [])]
                source = {name: t.clone() for name, t in host.model.state_dict().items()}
                gradients = []
                for invert in (False, True):
                    host.proxy = proxy(task, invert)
                    loss = host._proxy_term()
                    gradient, = torch.autograd.grad(loss, host.prompt.data_prompt)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.norm()), 0)
                    gradients.append(gradient)
                    same(self, before, native_snapshot(host))
                    for old, p in zip(base_grads, host.model.parameters()):
                        same(self, old, p.grad)
                    for old, hook in zip(hook_features, getattr(host.model, 'feature_hooks', [])):
                        self.assertIs(old, hook.features)
                self.assertFalse(torch.allclose(*gradients))
                # Exercise actual combined backward through the shared prompt and one Adam step.
                host.proxy = proxy(task)
                result = host.step(pixels(task, 1))
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(float(host.optimizer.state[host.prompt.data_prompt]['step']), 2)
                self.assertEqual({m.sample_num for m in host.model.modules() if isinstance(m, host.adabn)}, {2})
                same(self, source, host.model.state_dict())
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in host._proxy_model.parameters()))
                HOST_EVIDENCE[task + '_proxy'] = {'mask_gradient_difference_norm': float((gradients[0] - gradients[1]).norm()),
                    'gradient_norms': [float(g.norm()) for g in gradients], 'native_visible_state_unchanged_by_auxiliary_forward': True,
                    'native_hooks_unchanged_by_auxiliary_forward': True, 'source_weights_buffers_unchanged': True,
                    'combined_update_adam_steps': 2, 'native_sample_count': 2,
                    'source_clone_parameters': sum(p.numel() for p in host._proxy_model.parameters())}
                del host, source, base_grads, hook_features
