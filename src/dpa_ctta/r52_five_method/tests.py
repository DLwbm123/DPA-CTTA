"""Coverage and paired-content safeguards for the new comparison."""
import unittest
from .run import validate_rows, METHODS

class CoverageTests(unittest.TestCase):
    def rows(self):
        return [dict(condition=m,seed=s,order=o,visit=i+1,content=str(i),domain='D',
            subset='remaining_dev' if i<1695 else 'context',role='SEARCH' if i<1017 else 'SEALED_REVIEW')
            for m in METHODS for s in (20260907,17011,29009) for o in (0,1) for i in range(1951)]

    def test_complete_and_missing(self):
        rows=self.rows()
        self.assertEqual(validate_rows(rows,[20260907,17011,29009]),30)
        with self.assertRaises(ValueError):validate_rows(rows[:-1],[20260907,17011,29009])

    def test_wrong_content_and_role(self):
        for key,value in [('content','foreign'),('role','CONTEXT')]:
            rows=self.rows();rows[-1][key]=value
            with self.assertRaises(ValueError):validate_rows(rows,[20260907,17011,29009])

if __name__=='__main__':unittest.main()
