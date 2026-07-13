# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Set `max_expiries` to 3 in `market_maker/paper_run_config.json` (VPS unattended config), ending the multi-expiry burn-in at 1 ladder. The sizing bankroll now splits statically 1000/3 per ladder. Applied to the VPS in place on 2026-07-13 (config edited + `mm-paper` restarted) after the first in-process rollover completed cleanly; this commit makes the flip durable against the deploy script's `git reset --hard`.
