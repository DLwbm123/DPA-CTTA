"""Neutral runpy entry; separate scope, no default source execution."""
import os
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
from dpa_ctta.r7_source_prep.runner import main,child_entry
if __name__=='__main__':
    child_entry() if os.environ.get('SOURCE_PREP_CHILD')=='1' else main()
