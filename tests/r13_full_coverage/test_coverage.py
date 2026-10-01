import unittest
from dpa_ctta.r13_full_coverage.coverage import compose
class Coverage(unittest.TestCase):
    def test_disjoint_pairing_and_rejections(self):
        full=[dict(group_id=k,domain='D',subset='remaining_dev') for k in ('a','b','c')]
        row=lambda k:dict(content=k,domain='D',subset='remaining_dev',visit=9,metrics=[])
        self.assertEqual([r['content'] for r in compose(full,[[row('c'),row('a')],[row('b')]])],['a','b','c'])
        for parts in ([[row('a')],[row('a'),row('c')]],[[row('a')],[row('b')]],[[dict(row('a'),subset='other')],[row('b'),row('c')]]):
            with self.assertRaises(ValueError):compose(full,parts)
