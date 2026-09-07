"""Observed CPU regression results for the second review; no experiment launcher."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT / 'audit')]
os.environ['PYTHONPATH'] = str(ROOT / 'src')

import torch
import torch.utils.model_zoo
from PIL import Image


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    calls = {}
    def deny(name):
        def forbidden(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            raise RuntimeError('forbidden during synthetic review: ' + name)
        return forbidden
    log = io.StringIO()
    started = time.perf_counter()
    error = None
    focused = None
    result = None
    with ExitStack() as stack, redirect_stdout(log), redirect_stderr(log):
        for owner, name in ((torch, 'load'), (Image, 'open'), (torch.hub, 'download_url_to_file'),
                            (torch.utils.model_zoo, 'load_url'), (torch.cuda, '_lazy_init'),
                            (socket.socket, 'connect')):
            stack.enter_context(patch.object(owner, name, deny(owner.__name__ + '.' + name)))
        try:
            if torch.cuda.is_initialized():
                raise RuntimeError('CUDA was already initialized')
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            if result.wasSuccessful() and os.environ.get('DPA_CTTA_BASE_ROOT'):
                from run_publication_checks import focused_checks
                focused = focused_checks()
        except Exception:
            import traceback
            error = traceback.format_exc()
            log.write(error)
    elapsed = time.perf_counter() - started
    def public(text):
        for value, label in ((str(ROOT), '$REVIEW_ROOT'), (os.environ.get('DPA_CTTA_BASE_ROOT', ''), '$REFERENCE'),
                             (sys.prefix, '$PYTHON_PREFIX')):
            if value:
                text = text.replace(value, label)
        return text
    success = result is not None and result.wasSuccessful() and error is None and not calls and not torch.cuda.is_initialized()
    report = {
        'status': 'PASS_FOR_SECOND_REVIEW' if success else 'CHECK_FAILED',
        'baseline_commit': '54911f9e1aff4cc0338dfac0264e67456024d503',
        'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
        'tests_run': result.testsRun if result else None,
        'failures': len(result.failures) if result else None, 'errors': len(result.errors) if result else None,
        'skipped': len(result.skipped) if result else None,
        'skip_reasons': [reason for _, reason in result.skipped] if result else [],
        'runner_exception': public(error) if error else None,
        'exit_code': 0 if success else 1, 'elapsed_seconds': elapsed,
        'cuda_initialized': torch.cuda.is_initialized(), 'blocked_api_attempts': calls,
        'access_evidence': 'Image.open, torch.load, Torch downloads, CUDA lazy init and in-process socket connects were blocked; subprocess tests only exercise inspected CLI guards/local synthetic Git fixtures.',
        'real_data_training': 'NOT_RUN', 'server_access': 'NOT_RUN', 'checkpoint_integration': 'NOT_RUN',
        'real_data_io': 'NOT_OBSERVED_IN_INSTRUMENTED_APIS', 'independent_signoff': 'PENDING',
        'ci': 'NOT_CONFIGURED',
        'host_checks': getattr(sys.modules.get('test_vptta_host'), 'HOST_EVIDENCE', {}),
        'full_model_focused_checks': focused if focused is not None else 'NOT_RUN',
        'not_implemented': ['DD trainer', 'real source/target loader', 'style transfer', 'real distance-map generator'],
        'command': "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_review_fix_checks.py",
    }
    (ROOT / 'audit/review_fix_cpu.log').write_text(public(log.getvalue()))
    (ROOT / 'audit/review_fix_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
