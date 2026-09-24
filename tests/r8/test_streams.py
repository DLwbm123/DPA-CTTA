import unittest

from dpa_ctta.r8_ba.streams import stress


class TestStreams(unittest.TestCase):
    def test_single_mixed_permutation_and_unbroken_long10(self):
        rows = [dict(group_id=f"g{i:04d}", subset="remaining_dev" if i < 1695 else "other")
                for i in range(1951)]
        mixed = stress(rows, "MIXED")
        long10 = stress(rows, "LONG10")
        self.assertEqual({row["group_id"] for row in mixed}, {row["group_id"] for row in rows})
        self.assertEqual(mixed, stress(rows, "MIXED"))
        self.assertNotEqual(mixed, rows)
        self.assertEqual(len(long10), 19510)
        self.assertTrue(all(long10[i * 1951:(i + 1) * 1951] == rows for i in range(10)))
        with self.assertRaisesRegex(ValueError, "identity"):
            stress(rows[:-1], "LONG10")


if __name__ == "__main__":
    unittest.main()
