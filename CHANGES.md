# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): posterior refresh no longer blocks the tick loop -- 2026-08-11

The PARAM_POSTERIOR wing-sigma2 refresh (pricer_adapter, minutes-long GARCH
posterior fit) ran synchronously inside build_snapshot on cache expiry,
freezing the quote loop ~2.5 min per refresh (observed live 2026-08-10
14:03-14:12, slowing the skew-incident recovery). Refreshes now run in a
single-flight daemon thread per cache key while the STALE bands keep being
served; cold start (first price of a ladder, warmup) stays synchronous; a
failed background refresh re-caches the stale value with a 300s retry TTL
(no thread-per-tick spam). SimClock-safe (expiry stamps precomputed from
the caller's clock). Module seam _POSTERIOR_ASYNC + _join_posterior_refresh
test helper. 3 new tests (serve-stale-then-upgrade, single-flight,
background-failure retry); TTL test updated to join the worker.
Files: market_maker/pricer_adapter.py, tests/test_mm_pricer_adapter.py.

## fix(tests): run_control flake root-caused and fixed -- 2026-08-11

The "flaky" real-subprocess tests were a real Windows-only defect in the
test dummy, not timing noise: os.replace onto heartbeat.json raises
PermissionError when a reader (engine_status or the test) holds the file
open, the unhandled error KILLED the dummy mid-loop, and every observed
symptom (dead pid, leftover pid/stop files) was downstream. Linux never
conflicts, hence the platform-dependent flake rate. Fixes, all in the test
file's dummy script + assertions: retried os.replace heartbeat writes;
retried cleanup unlinks (same sharing-violation class); poll-based
stop/pid-file assertions instead of instant point-asserts; liveness proven
by heartbeat tick advance instead of sleep+pid_alive; dummy self-deadline
30s -> 120s. 12/12 consecutive local runs green (was ~1-in-2 failing).
Files: tests/test_mm_run_control.py.

## chore(mm): weekly k_arrival refit #3 -- HELD at 12.8 -- 2026-08-11

scripts/mm_calibrate_k.py on the VPS state db suggested k=9.4 (7d window,
1366 market-hours) but the fit is UNSTABLE across sub-windows (5d=8.0,
3d=5.7, monotone drift; window spans the 2026-08-10 skew incident and the
post-restart quiet period). Per the adoption discipline from refit #2
(stable sub-windows required), k_arrival stays 12.8; re-fit next cycle
(~2026-08-17). Arrival term negligible at either value (0.03-0.08c ATM).
No code change.
