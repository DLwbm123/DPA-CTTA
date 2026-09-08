import copy
import json
import random
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from dpa_ctta.m2_episodes import make_m2_episodes,validate_coverage,coverage_summary,TRANSFORMS


def old_fixture(task):
    rng=random.Random(20260907);s=list(range(32));rng.shuffle(s)
    q=list(range(40 if task=='fundus' else 64));rng.shuffle(q)
    return [dict(episode=i+1,state_index=s[i%32],query_index=q[i%len(q)],transform=TRANSFORMS[i%4]) for i in range(600)]


class EpisodeTests(unittest.TestCase):
    def test_exact_marginals_crossed_balance_unique_triples_determinism(self):
        for task in ('fundus','polyp'):
            old=old_fixture(task);new=make_m2_episodes(old,task)
            self.assertTrue(validate_coverage(old,new));self.assertEqual(new,make_m2_episodes(old,task))
            self.assertEqual(coverage_summary(old)['unique_triples'],160 if task=='fundus' else 64)
            self.assertEqual(coverage_summary(new)['unique_triples'],600)
            for field in ('query_index','state_index','transform'):
                self.assertEqual(Counter(e[field] for e in old),Counter(e[field] for e in new))

    def test_no_input_mutation_or_global_RNG_consumption(self):
        old=old_fixture('fundus');saved=copy.deepcopy(old)
        py=random.getstate();np_state=np.random.get_state();pt=torch.get_rng_state().clone()
        make_m2_episodes(old,'fundus')
        self.assertEqual(old,saved);self.assertEqual(random.getstate(),py)
        now=np.random.get_state();self.assertEqual(now[0],np_state[0]);np.testing.assert_array_equal(now[1],np_state[1]);self.assertEqual(now[2:],np_state[2:])
        self.assertTrue(torch.equal(pt,torch.get_rng_state()))

    def test_invalid_index_count_order_transform_and_duplicate_rejected(self):
        old=old_fixture('polyp')
        for field,value in [('episode',99),('state_index',-1),('query_index',64),('transform','unknown')]:
            bad=copy.deepcopy(old);bad[0][field]=value
            with self.assertRaises(ValueError):make_m2_episodes(bad,'polyp')
        with self.assertRaises(ValueError):make_m2_episodes(old[:-1],'polyp')
        new=make_m2_episodes(old,'polyp');new[1]=dict(new[0],episode=2)
        with self.assertRaises(ValueError):validate_coverage(old,new)
