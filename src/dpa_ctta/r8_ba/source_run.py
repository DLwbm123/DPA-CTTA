"""One bounded R8 source fit job; scheduling and failure classification stay outside."""
from torch import nn

from ..r7_shared.numerics import COUNTS
from .journal import SourceJournal
from .trainer import MAX_STEPS


def run_fit(trainer, job_root, guard, resume_failure=None):
    if (not callable(guard) or not isinstance(trainer.segmenter, nn.Module) or
            trainer.steps != 0):
        raise ValueError("R8 source fit runner binding")
    journal = SourceJournal(job_root, trainer)
    if resume_failure is None:
        journal.create()
    else:
        journal.recover_once(resume_failure)
    hook = trainer.segmenter.register_forward_pre_hook(lambda *_: guard())
    try:
        while trainer.steps < MAX_STEPS:
            guard()
            before = COUNTS.copy()
            next_step = trainer.steps + 1
            try:
                trace = trainer.fit_step()
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
