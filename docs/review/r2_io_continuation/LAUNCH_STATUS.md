# R2 IO continuation: verified running

Snapshot: 2026-09-13 18:30:20 Asia/Shanghai. This is a startup observation, not
current progress or an experiment-completion claim.

- Runtime commit: `66eea7e880e16d4d25efa4edabc9d8ad59ff175d`, deployed as a clean,
  separate checkout. The original execution checkout and output were preserved.
- Continuation run: `a8693fe3eab44ddb802c6de6c0caafda`.
- Source run: `8ce180e3db2249c19bba9d103ede9cb2` at original implementation
  `0d515328a6cc42d8e0c6a458265b41e41c454fa6`.
- Science: `882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353`.
- Registration: `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`.
- The two authorized devices passed their fresh mechanical smoke checks.
- Formal `o2a4` and `o3a2` each had 65 scoring records, past the first scheduled
  capacity scan. No failure artifact or launcher error was present.
- The detached launcher and its two formal workers were alive; their full
  command lines and GPU process names used the neutral entry convention.
- Pending finite work: `o2a4`, `o3a1`, `o3a2`, `o3a3`, `o3a4`, in their original
  two-slot rotation. The other 15 completed trajectories are carried unchanged.

The existing runner performs CPU recomputation after all five trajectories finish.
The final comparison and public result closeout remain pending. No monitoring
automation or automatic retry was created. Private host paths, process IDs,
device UUIDs, raw per-image records and asset mappings are not published here.

See [repair and accounting](README.md) and [19 passing CPU checks](CPU_TEST_LOG.txt).
