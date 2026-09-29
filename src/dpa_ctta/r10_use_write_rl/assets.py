"""Historical Screen24 asset identity is separate from the R10 runtime."""
import contextlib
from ..r9_current_first.assets import open_source as verified_source,validate_metadata,gpu_policy,available_memory
from ..r8_ba.target_factory import load_deployed
from .controller import Controller


@contextlib.contextmanager
def open_source(binding,assignment,guard):
    with verified_source(binding,assignment,guard) as (data,segmenter,oracles,bases,scaler):
        row=binding['screen24_index']['source_jobs']['B_FULL_20260924']
        carrier=load_deployed(row,binding['configs']['B'],20260924,'FULL')
        controller=Controller(carrier,binding['gradient_scale'])
        yield data,segmenter,oracles,controller
