# R6-D pinned input adapter

POST_HOC_EXPLORATORY. This is an explicitly authorized continuation, not an automatic retry or a new model experiment.

Implementation: `adf8fdaa17fe68cfb2b420ee8f381f27ced6a89e`. Prior failed implementation/publication: `ad75672e2931c5dd09ffb19f3249d6b65ec6aa0d` / `fad933d9c49dd1ae034a9368785381fb89c32cf4`. Historical source execution: `c94fff7c05cec38d541c441b62a9381ee46ba076`.

The old adapter treated a publication alias as a regular payload. The new adapter first preserves and validates raw ordinary current_result bytes, full binding, status and the pinned version. It audits the two known aliases with lstat/readlink and opens only the exact version payload. The logical snapshot aggregate name remains compatible with unchanged pure validators. All real source-to-snapshot mappings and link identities are private; public evidence uses file tokens, byte lengths and separate source/snapshot digests. Link-text hashes are never used as content hashes.

`Reader` opens the bound source root and each directory component with O_DIRECTORY/O_NOFOLLOW, then a single-hardlink regular leaf with O_RDONLY/O_NOFOLLOW/O_NONBLOCK and O_NOATIME where supported. The same descriptor supplies copied bytes and SHA256. fstat, path identity and directory-chain checks detect observed replacements; pointer/link checks occur before copying, after copying and after analysis. No alias content reads, recursive dereferencing, source chmod/utime, old recompute/invalidate/publish or model dependency imports are introduced.

The only core.py change is its I/O audit helper: exact lexical file paths plus explicitly permitted directory-descriptor opens. Python's audit event omits dir_fd, so the inspected synchronous helper supplies the anchored path for that one os.open call. This is defense in depth, not an OS sandbox. Filesystem-managed atime remains excluded; protected metadata includes mtime/ctime/mode/inode/device/link count. Byte checks cover only the allowlisted scalar inputs; other original output files receive metadata inventory, not a claim of full-byte auditing.

Preflight now lies inside the alarm, timer and exception boundary. The original exception and every after-check error have distinct exclusive-created logs. Missing evidence is null/NOT_CAPTURED. Old attempt telemetry and missing hashes remain untouched and unknown. No repeated real analysis is authorized by a successful test.

Original analysis-spec SHA256 remains `ffd3c648931e27130c2c275da7ca83f7bac62906b9e9e3d51ec4b3fd62d07368`; I/O-addendum SHA256 is independently `250645247b145760e520214373b9930b8580b595165fbd1cc21ac7ec666e0b96`. The original 25 tests, formulas, pair keys, windows, strata, thresholds, production source/configuration and old result directory are unchanged. The supplement's five manifest entries were byte-verified; this is not a test PASS.

[Narrow implementation patch](NARROW_IO.patch) excludes bundled specification text. Final test/run evidence and actual diagnostic completion are reported separately in TEST_SUMMARY, RUN_LOG, READONLY_AUDIT and REPORT.
