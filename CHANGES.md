# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Adopt the fitted Dalen arrival decay: `MMConfig.k_arrival` 10.0 (interim judgment value) -> 18.2, fitted by `scripts/mm_calibrate_k.py` on the VPS state db (2457 joined trade prints / 289 market-hours, k=18.21, A=3.58/market-hour, implied arrival half-spread ~0.01c ATM). Caveats recorded in the config comment: only ~3 days of prints, lumpy histogram head near the touch; refit planned ~2026-07-21 after a full weekly volume cycle. Also add a GATE-BEFORE-REAL-CAPITAL note to CLAUDE.md's risk_controller bullet: the deferred option-C breach-metric redesign (remaining-loss notional / hedge-adjusted q instead of raw shares vs the S'(x)-shrinking q_max) must be implemented before any live-money sizing. Files: `market_maker/config.py`, `CLAUDE.md`.
