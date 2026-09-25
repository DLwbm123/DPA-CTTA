import unittest

from dpa_ctta.r8_ba.schedule import CURRICULA
from dpa_ctta.r8_ba.source_select import SEEDS, select_grid
from dpa_ctta.r8_ba.trainer import SAVE_STEPS


class TestSourceSelect(unittest.TestCase):
    def test_full_only_family_and_independent_static_point(self):
        configs = ([dict(id=f"B{i}", route="B") for i in range(8)] +
                   [dict(id=f"A{i}", route="A") for i in range(4)])
        rows = {}
        for config in configs:
            for mode in ("FULL", "STATIC"):
                for seed in SEEDS:
                    records = {}
                    for step in SAVE_STEPS:
                        value = (0.8 if config["id"] == "B1" and mode == "FULL" else
                                 0.9 if config["id"] == "B2" and mode == "STATIC" else
                                 0.7 if config["id"] == "A1" and mode == "FULL" else 0.5)
                        value += 0.01 if step == (4000 if mode == "FULL" else 8000) else 0
                        records[step] = [dict(episode=i, curriculum=CURRICULA[i // 16],
                                              soft_Dice=value) for i in range(64)]
                    rows[config["id"], mode, seed] = records
        selected = select_grid(configs, rows, {row["id"]: 1.0 for row in configs})
        self.assertEqual(selected["selected_config"], {"B": "B1", "A": "A1"})
        self.assertEqual(selected["per_config_mode"]["B1"]["FULL"]["source_step"], 4000)
        self.assertEqual(selected["per_config_mode"]["B1"]["STATIC"]["source_step"], 8000)
        del rows["B1", "FULL", SEEDS[0]][1000]
        with self.assertRaisesRegex(ValueError, "five calibrated"):
            select_grid(configs, rows, {row["id"]: 1.0 for row in configs})


if __name__ == "__main__":
    unittest.main()
