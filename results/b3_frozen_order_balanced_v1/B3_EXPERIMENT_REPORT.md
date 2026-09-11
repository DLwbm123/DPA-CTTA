# B3 frozen balanced order comparison — launch audit

B3_RUNNING. No completed B3 scientific result is claimed.

Execution commit: `0fac9b2b1e762ed57b624953edf95058cb1f0380`. Snapshot UTC: 2026-09-11T04:37:03Z.

Existing remote Python 3.10.6 / Torch 2.2.1+cu121: five relevant CPU checks passed, no skips. Single paired GPU smoke passed for A/C/U/S/I: 20 Adam/backwards, 136 segmentation forwards and 8 observed A Prompt calls (one fft2/ifft2 each). State, logits and exact RNG/counters match between frozen direct hosts and the thin entry.

Four order identities and twelve directed adjacent pairs validated; only order2/3 are new. Ten trajectories are scheduled sequentially on GPU7, with 19,510 new scored updates/backwards and 132,668 formal forwards. Old order0/1 are reused; no new N/G/O2/D4 inference. No algorithm, seed or hyperparameter change.

The detached parent and GPU worker use neutral argv; full ps and nvidia-smi names were checked. 149 formal records were complete at this snapshot, with no immediate failure. The run survives session closure and will execute CPU reconstruction once after all trajectories. Eight active hours / 1 GiB private output caps; no automatic retry or periodic monitor.

Code/config/tests/contract and this deidentified startup audit are public. Private asset mappings, receipts, process audit paths and per-image logs remain on NAS. Final performance, order sensitivity and candidate decision are pending. B2 historical results and conclusions are unchanged.
