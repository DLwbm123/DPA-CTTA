"""Ensure the reused runtime cannot dispatch as the stopped R47 round."""
import unittest
from unittest.mock import patch
from . import run
from ..r47_shared_descent.method import Host


class Tests(unittest.TestCase):
    def test_independent_dispatch_identity_and_guarded_host(self):
        with patch.object(run.reporting.retained, 'main') as dispatch:
            run.main()
            dispatch.assert_called_once_with()
        self.assertEqual(run.reporting.retained.ID, 'R48_REPAIRED_SHARED_DESCENT')
        self.assertIs(run.reporting.retained.Host, Host)
        self.assertIs(run.reporting.retained.report, run.report)


if __name__ == '__main__':
    unittest.main()
