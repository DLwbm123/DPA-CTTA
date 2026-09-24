import unittest

from dpa_ctta.r8_ba.schedule import anchors, episode_roles, episode_styles, oracle_roles


class TestSchedule(unittest.TestCase):
    def test_fixed_folds_and_group_isolation(self):
        self.assertEqual(tuple(anchors("fit").shape), (512, 9))
        self.assertEqual(tuple(anchors("cal").shape), (128, 9))
        self.assertEqual(tuple(anchors("val").shape), (128, 9))
        groups = [f"source_{i:03d}" for i in range(111)]
        for episode in range(64):
            styles, mode, severity = episode_styles("val", episode)
            self.assertEqual(len(styles), 32)
            self.assertIn(mode, ("abrupt_16_16", "gradual_32", "recurrence_8_8_8_8", "iid_style_32"))
            self.assertIn(severity, ("clean", "mild", "compound"))
        styles, _, _ = episode_styles("fit", 2)
        pairs = {i: oracle_roles(groups, "fit", i)[:2] for i in set(styles)}
        roles, reused = episode_roles(groups, "fit", 2, styles, pairs)
        self.assertFalse(reused)
        self.assertEqual(len({g for pair in roles for g in pair}), 64)
        self.assertTrue(all(q not in pairs[a] and q != s for (s, q), a in zip(roles, styles)))


if __name__ == "__main__":
    unittest.main()
