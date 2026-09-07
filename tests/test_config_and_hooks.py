import json
from pathlib import Path
import tempfile
import unittest

import torch

from dpa_ctta.config import DPAConfig
from dpa_ctta.hooks import capture_output, directional_logits_jvp, enumerate_4d_outputs
from common import TinySource


class ConfigTests(unittest.TestCase):
    def test_default_latent_dimension(self):
        self.assertEqual(DPAConfig().latent_dim, 16)

    def test_fundus_class_count(self):
        self.assertEqual(DPAConfig(task="fundus").num_classes, 2)

    def test_polyp_class_count(self):
        self.assertEqual(DPAConfig(task="polyp").num_classes, 1)

    def test_fundus_descriptor_dimension(self):
        self.assertEqual(DPAConfig(task="fundus").descriptor_dim, 33)

    def test_polyp_descriptor_dimension(self):
        self.assertEqual(DPAConfig(task="polyp").descriptor_dim, 25)

    def test_invalid_task_rejected(self):
        with self.assertRaises(ValueError):
            DPAConfig(task="brain")

    def test_non_v0_latent_dimension_rejected(self):
        with self.assertRaises(ValueError):
            DPAConfig(latent_dim=8)

    def test_nonpositive_scalars_rejected(self):
        for field in ("precision_floor", "temporal_lambda", "temperature"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                DPAConfig(**{field: 0})

    def test_unknown_json_key_rejected(self):
        with self.assertRaises(ValueError):
            DPAConfig.from_dict({"task": "polyp", "future_option": True})

    def test_json_round_trip(self):
        expected = DPAConfig(task="polyp", temperature=0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(expected.to_dict()))
            self.assertEqual(DPAConfig.from_json(path), expected)

    def test_repository_method_config_is_valid(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "method_v0.json"
        self.assertEqual(DPAConfig.from_json(path).latent_dim, 16)


class HookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = TinySource().eval().requires_grad_(False)
        self.image = torch.randn(1, 3, 16, 16)

    def test_capture_named_output(self):
        with capture_output(self.model, "early") as values:
            self.model(self.image)
        self.assertEqual(tuple(values[0].shape), (1, 4, 16, 16))

    def test_capture_hook_removed(self):
        module = dict(self.model.named_modules())["early"]
        before = len(module._forward_hooks)
        with capture_output(self.model, "early"):
            self.model(self.image)
        self.assertEqual(len(module._forward_hooks), before)

    def test_capture_hook_removed_after_exception(self):
        module = dict(self.model.named_modules())["early"]
        before = len(module._forward_hooks)
        with self.assertRaises(RuntimeError):
            with capture_output(self.model, "early"):
                raise RuntimeError("stop")
        self.assertEqual(len(module._forward_hooks), before)

    def test_unknown_capture_path_rejected(self):
        with self.assertRaises(ValueError):
            with capture_output(self.model, "missing"):
                pass

    def test_enumeration_records_only_4d_tensors(self):
        records = enumerate_4d_outputs(self.model, lambda: self.model(self.image))
        self.assertTrue(records)
        self.assertTrue(all(len(record["shape"]) == 4 for record in records))

    def test_enumeration_keeps_execution_order(self):
        records = enumerate_4d_outputs(self.model, lambda: self.model(self.image))
        self.assertEqual([r["order"] for r in records], list(range(len(records))))

    def test_directional_jvp_is_nonzero(self):
        jvp = directional_logits_jvp(self.model, "mid", self.image, lambda model, x: model(x))
        self.assertGreater(float(jvp.norm()), 0)

    def test_directional_jvp_is_finite(self):
        jvp = directional_logits_jvp(self.model, "tail", self.image, lambda model, x: model(x))
        self.assertTrue(torch.isfinite(jvp).all())


if __name__ == "__main__":
    unittest.main()
