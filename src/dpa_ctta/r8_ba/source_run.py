"""One bounded R8 source fit job; scheduling and failure classification stay outside."""
from torch import nn

from ..r7_shared.numerics import COUNTS
from .calibration_journal import CalibrationJournal
from .journal import SourceJournal
from .trainer import MAX_STEPS


def _drive(worker, journal, limit, step, guard):
    hook = worker.segmenter.register_forward_pre_hook(lambda *_: guard())
    try:
        while worker.steps < limit:
            guard()
            before = COUNTS.copy()
            next_step = worker.steps + 1
            try:
                trace = step()
            except BaseException as exc:
                try:
                    journal.record_failed_call(next_step, dict(COUNTS - before), exc)
                except BaseException:
                    pass
                raise
            previous_log_bytes = journal.physical.stat().st_size
            try:
                journal.append(trace)
            except BaseException as exc:
                try:
                    if journal.physical.stat().st_size == previous_log_bytes:
                        journal.record_failed_call(next_step, dict(COUNTS - before), exc)
                except BaseException:
                    pass
                raise
            guard()
        return journal.complete()
    finally:
        hook.remove()


def run_fit(trainer, job_root, guard, resume_failure=None):
    if (not callable(guard) or not isinstance(trainer.segmenter, nn.Module) or
            trainer.steps != 0):
        raise ValueError("R8 source fit runner binding")
    journal = SourceJournal(job_root, trainer)
    if resume_failure is None:
        journal.create()
    else:
        journal.recover_once(resume_failure)
    return _drive(trainer, journal, MAX_STEPS, trainer.fit_step, guard)


def run_calibration(calibrator, job_root, source_step, guard, resume_failure=None):
    if (not callable(guard) or not isinstance(calibrator.segmenter, nn.Module) or
            calibrator.steps != 0):
        raise ValueError("R8 calibration runner binding")
    journal = CalibrationJournal(job_root, source_step, calibrator)
    if resume_failure is None:
        journal.create()
    else:
        journal.recover_once(resume_failure)
    return _drive(calibrator, journal, 1024, calibrator.step, guard)
