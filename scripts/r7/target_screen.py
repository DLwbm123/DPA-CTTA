"""Neutral environment-selected entry. No CLI auto-scope or resume switches."""
import os
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
from dpa_ctta.r7_target_screen.runner import main, child_entry, readiness_entry
if __name__ == '__main__':
    (readiness_entry if os.environ.get('SCREEN_READY') == '1' else child_entry if os.environ.get('SCREEN_CHILD') == '1' else main)()
