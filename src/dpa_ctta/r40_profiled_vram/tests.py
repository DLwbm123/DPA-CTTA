import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ..r20_model_only_search.runtime import worker_peak
from ..r9_current_first.assets import available_memory


class Tests(unittest.TestCase):
    def test_measured_admission_preserves_margin_and_legacy_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'profiles').mkdir()
            (root/'profiles/W.json').write_text(json.dumps(dict(status='PASS',candidate='W',peak_reserved_bytes=704643072)))
            c=dict(output_root=folder,use_profiled_worker_memory=True);job=dict(candidate=dict(id='W'))
            assignment=dict(physical_id=2,uuid='TEST_UUID')
            peak=worker_peak(c,'online_W',job)
            with patch('dpa_ctta.r9_current_first.assets.subprocess.check_output',return_value='2, TEST_UUID, 5884\n'):
                self.assertEqual(available_memory(assignment,peak),5884*1024**2)
                with self.assertRaises(RuntimeError):available_memory(assignment,5*1024**3)
            with patch('dpa_ctta.r9_current_first.assets.subprocess.check_output',return_value='2, TEST_UUID, 1024\n'):
                with self.assertRaises(RuntimeError):available_memory(assignment,peak)
            self.assertEqual(worker_peak(c,'profile_W',job),5*1024**3)
            self.assertEqual(worker_peak(dict(c,use_profiled_worker_memory=False),'online_W',job),5*1024**3)

    def test_invalid_profile_cannot_lower_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'profiles').mkdir();c=dict(output_root=folder,use_profiled_worker_memory=True);job=dict(candidate=dict(id='W'))
            for bad in (dict(status='FAIL',candidate='W',peak_reserved_bytes=704643072),
                        dict(status='PASS',candidate='C',peak_reserved_bytes=704643072),
                        dict(status='PASS',candidate='W',peak_reserved_bytes=0)):
                (root/'profiles/W.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):worker_peak(c,'online_W',job)


if __name__ == '__main__': unittest.main()
