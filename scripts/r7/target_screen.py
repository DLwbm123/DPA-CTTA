"""Neutral environment-selected entry. No CLI auto-scope or resume switches."""
import os
from dpa_ctta.b3_runtime import neutral_subprocesses
neutral_subprocesses()
from dpa_ctta.r7_target_screen.runner import main, child_entry
if __name__ == '__main__':
    (child_entry if os.environ.get('SCREEN_CHILD') == '1' else main)()
