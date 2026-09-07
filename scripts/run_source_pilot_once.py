"""Narrow receipt-bound source pilot entry; no resume or background worker."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dpa_ctta.source_pilot_release import main
if __name__=='__main__':
    raise SystemExit(main())
