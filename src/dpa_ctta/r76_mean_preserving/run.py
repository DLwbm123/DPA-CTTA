"""Reuse R75's guarded execution, admission, retirement, scoring and cost handling."""
from .method import ARMS,Readouts,self_check
from ..r75_dense_wcw import run as retained

retained.ARMS=ARMS
retained.Readouts=Readouts
retained.self_check=self_check

if __name__=='__main__':raise SystemExit(retained.main())
