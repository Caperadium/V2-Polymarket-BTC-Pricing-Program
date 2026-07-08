# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Add daily heartbeat message to `scripts/mm_alert_check.py`: once per UTC day, at the first timer check at/after 08:00 UTC (override via `$MM_HEARTBEAT_HOUR_UTC`, disable via `$MM_HEARTBEAT_DISABLE=1`), the script sends a one-line status summary (engine state, exit_reason when stopped, tick, feed_healthy, fills, BTC data age, feed restarts, free disk) through the same `$MM_ALERT_WEBHOOK`, regardless of engine state. Tracked separately from the 6h fault de-dupe as `heartbeat_last_date` in `alert_state.json`, so webhook silence is now distinguishable from a dead alert pipeline. Adds 8 tests to `tests/test_mm_alert_check.py` (41 total pass). Docs updated: `CLAUDE.md` deploy-kit paragraph, `deploy/README.md` file list, `DOCS/guides/market-maker-deployment.md` kit table. Deployed to the VPS by direct copy (VPS clone's `scripts/mm_alert_check.py` is ahead of git until next push+pull).
