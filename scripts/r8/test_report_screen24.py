"""Run: python3 scripts/r8/test_report_screen24.py."""
from report_screen24 import describe, macro

# Large domains must not dominate the frozen equal-domain metric.
rows=[(d,c,1.0 if d==0 else 0.0) for d in range(4) for c in ('OD','OC')
      for _ in range(100 if d==0 else 1)]
assert macro(rows)==0.25
s=describe([-1.0,0.0,1.0])
assert s['mean']==s['median']==0 and s['negative']==s['zero']==s['positive']==1
assert s['q05']==-.9 and s['worst_decile_mean']==-1
print('Screen24 report weighting and paired summaries: PASS')
