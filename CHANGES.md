# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Weekly k_arrival refit #2 (2026-08-01): MMConfig.k_arrival 18.3 -> 12.8, fitted by scripts/mm_calibrate_k.py on the VPS state db (6974 joined prints / 2023 market-hours over 7d, k=12.75, A=1.05/market-hour). The 30% drop vs the 2026-07-21 fit is stable across sub-windows (5d=11.6, 3d=12.3, both excluding the 2026-07-26 burst weekend) so it reads as a genuine flow-regime shift, not a burst artifact; implied arrival half-spread stays negligible (~0.015c ATM). Affects market_maker/config.py only.
