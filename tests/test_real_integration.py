import json
import os
from pathlib import Path
import unittest

import torch

from dpa_ctta.hooks import directional_logits_jvp
from dpa_ctta.integrations.ctta_suite import (
    INPUT_SIZES,
    REFERENCE_COMMIT,
    build_reference_model,
    checkout_root,
    load_integration_config,
    load_reference_interfaces,
)
from dpa_ctta.manifold import MultiScaleLatentFiLM


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = os.environ.get("DPA_CTTA_BASE_ROOT")
SKIP_REASON = "optional real integration requires DPA_CTTA_BASE_ROOT at the pinned public commit"


class IntegrationConfigTests(unittest.TestCase):
    def test_fundus_config_has_three_scales(self):
        config = load_integration_config(ROOT / "configs" / "fundus_integration.json")
        self.assertEqual(len({tuple(point["shape"][-2:]) for point in config["injection_points"]}), 3)

    def test_polyp_config_has_three_scales(self):
        config = load_integration_config(ROOT / "configs" / "polyp_integration.json")
        self.assertEqual(len({tuple(point["shape"][-2:]) for point in config["injection_points"]}), 3)

    def test_configs_pin_reference_commit(self):
        for task in ("fundus", "polyp"):
            config = json.loads((ROOT / "configs" / f"{task}_integration.json").read_text())
            self.assertEqual(config["reference_commit"], REFERENCE_COMMIT)

    def test_configs_record_cpu_only_discovery(self):
        for task in ("fundus", "polyp"):
            config = json.loads((ROOT / "configs" / f"{task}_integration.json").read_text())
            self.assertFalse(config["cuda_initialized"])

    def test_configs_record_nonzero_jvps(self):
        for task in ("fundus", "polyp"):
            config = json.loads((ROOT / "configs" / f"{task}_integration.json").read_text())
            self.assertTrue(all(point["jvp_finite"] and point["jvp_norm"] > 0 for point in config["injection_points"]))


@unittest.skipUnless(EXTERNAL, SKIP_REASON)
class RealModelIntegrationMixin:
    task = None

    @classmethod
    def setUpClass(cls):
        cls.config = load_integration_config(ROOT / "configs" / f"{cls.task}_integration.json")
        cls.source, model_logits = build_reference_model(cls.task, EXTERNAL)
        cls.model_logits = staticmethod(model_logits)
        size = INPUT_SIZES[cls.task]
        cls.image = torch.linspace(-1, 1, 3 * size * size).reshape(1, 3, size, size)
        with torch.inference_mode():
            cls.baseline = cls.model_logits(cls.source, cls.image).clone()
        channels = {point["path"]: point["shape"][1] for point in cls.config["injection_points"]}
        torch.manual_seed(20260903)
        cls.wrapper = MultiScaleLatentFiLM(cls.source, channels)

    @classmethod
    def tearDownClass(cls):
        cls.wrapper.close()

    def test_full_cpu_forward_shape(self):
        self.assertEqual(list(self.baseline.shape[-2:]), self.config["input_shape"][-2:])

    def test_zero_state_bitwise_identity(self):
        with torch.inference_mode():
            result = self.model_logits(self.wrapper, self.image)
        self.assertTrue(torch.equal(result, self.baseline))

    def test_nonzero_state_changes_logits(self):
        with torch.inference_mode():
            output = self.wrapper(self.image, torch.ones(1, 16))
            output = output[0] if isinstance(output, tuple) else output
        self.assertFalse(torch.equal(output, self.baseline))

    def test_all_three_logits_jvps_nonzero(self):
        for point in self.config["injection_points"]:
            with self.subTest(path=point["path"]):
                jvp = directional_logits_jvp(
                    self.source, point["path"], self.image, self.model_logits
                )
                self.assertTrue(torch.isfinite(jvp).all())
                self.assertGreater(float(jvp.norm()), 0)

    def test_injection_hook_cleanup(self):
        modules = dict(self.source.named_modules())
        before = {path: len(modules[path]._forward_hooks) for path in self.wrapper.injection_paths}
        temporary = MultiScaleLatentFiLM(
            self.source,
            {
                point["path"]: point["shape"][1]
                for point in self.config["injection_points"]
            },
        )
        temporary.close()
        self.assertEqual(
            {path: len(modules[path]._forward_hooks) for path in self.wrapper.injection_paths}, before
        )

    def test_cuda_never_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


class TestFundusResUNet34(RealModelIntegrationMixin, unittest.TestCase):
    task = "fundus"


class TestPolypPraNet(RealModelIntegrationMixin, unittest.TestCase):
    task = "polyp"


@unittest.skipUnless(EXTERNAL, SKIP_REASON)
class ReferenceCheckoutTests(unittest.TestCase):
    def test_checkout_is_exact_pinned_commit(self):
        _root, commit = checkout_root(EXTERNAL)
        self.assertEqual(commit, REFERENCE_COMMIT)

    def test_reference_model_code_is_not_vendored(self):
        self.assertFalse((ROOT / "third_party").exists())

    def test_data_and_metrics_interfaces_import_without_access(self):
        models, data, metrics = load_reference_interfaces(EXTERNAL)
        self.assertTrue(callable(models.build_model))
        self.assertTrue(data.__name__.endswith(".data"))
        self.assertTrue(metrics.__name__.endswith(".metrics"))


if __name__ == "__main__":
    unittest.main()
