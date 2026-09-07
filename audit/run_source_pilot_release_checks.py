"""CPU/procedural preparation checks; writes only this phase's new evidence."""
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
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests')]
os.environ['PYTHONPATH'] = str(ROOT/'src')
import torch
import torch.utils.model_zoo
from PIL import Image
import scipy


def main():
    prefix='source_pilot_release'
    torch.set_num_threads(2)
    calls={}
    image_stream_reads=[]
    original_open=Image.open
    def fixture_image_open(fp,*args,**kwargs):
        if not isinstance(fp,io.BytesIO):
            calls['nonfixture_Image.open']=calls.get('nonfixture_Image.open',0)+1
            raise RuntimeError('only in-memory encoded procedural images allowed')
        image_stream_reads.append('BytesIO')
        return original_open(fp,*args,**kwargs)
    def deny(name):
        def blocked(*args,**kwargs):
            calls[name]=calls.get(name,0)+1
            raise RuntimeError('forbidden in CPU prepare: '+name)
        return blocked
    started=time.perf_counter()
    error=None; result=None
    with (ROOT/f'audit/{prefix}_cpu.log').open('w') as log, ExitStack() as stack, redirect_stdout(log),redirect_stderr(log):
        stack.enter_context(patch.object(Image,'open',fixture_image_open))
        for owner,name in ((torch,'load'),(torch.cuda,'_lazy_init'),(torch.hub,'download_url_to_file'),
                           (torch.utils.model_zoo,'load_url'),(socket.socket,'connect')):
            stack.enter_context(patch.object(owner,name,deny(owner.__name__+'.'+name)))
        try:
            if torch.cuda.is_initialized(): raise RuntimeError('CUDA already initialized')
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        except Exception:
            import traceback
            error=traceback.format_exc(); log.write(error)
    def public(text):
        for value,label in ((str(ROOT),'$PILOT_ROOT'),(os.environ.get('DPA_CTTA_BASE_ROOT',''),'$REFERENCE'),(sys.prefix,'$PYTHON_PREFIX')):
            if value: text=text.replace(value,label)
        return text
    path=ROOT/f'audit/{prefix}_cpu.log'; path.write_text(public(path.read_text()))
    success=bool(result and result.wasSuccessful() and not result.skipped and not calls and not error and not torch.cuda.is_initialized())
    report=dict(status='CPU_PROCEDURAL_CHECKS_PASS' if success else 'CHECK_FAILED',
        baseline_commit='8d417ca864288136b3b44a4245a241e80d072ea6',
        python=platform.python_version(),torch=torch.__version__,scipy=scipy.__version__,device='cpu',
        tests_run=result.testsRun if result else 0,failures=len(result.failures) if result else None,
        errors=len(result.errors) if result else None,skipped=len(result.skipped) if result else None,
        runner_exception=public(error) if error else None,exit_code=0 if success else 1,
        elapsed_seconds=time.perf_counter()-started,blocked_api_attempts=calls,
        procedural_image_stream_reads=len(image_stream_reads),cuda_initialized=torch.cuda.is_initialized(),
        evidence=getattr(sys.modules.get('test_source_pilot_release'),'RELEASE_EVIDENCE',{}),
        prior_host_regressions=getattr(sys.modules.get('test_vptta_host'),'HOST_EVIDENCE',{}),
        real_checkpoint='NOT_RUN',real_source_registration='NOT_RUN',real_image_io='NOT_RUN',
        cuda_device_and_memory_semantics='NOT_RUN',source_pilot='NOT_RUN',server='NOT_RUN',DD='NOT_RUN',target='NOT_RUN',
        CI='NOT_CONFIGURED',runner_independent_review='PENDING_EXACT_COMMIT_CONFIG_REVIEW',
        command="PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_source_pilot_release_checks.py")
    (ROOT/f'audit/{prefix}_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return report['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
