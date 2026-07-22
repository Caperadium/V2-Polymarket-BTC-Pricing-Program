# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Weekly k_arrival refit (scheduled 2026-07-21): rerun scripts/mm_calibrate_k.py on the VPS live state-db over the first full 7-day volume cycle (5639 joined prints / 3065 market-hours, vs 2457 prints / ~3 days at the 2026-07-14 first fit). Fitted k=18.33, A=0.75/market-hour, implied arrival half-spread ~0.007c ATM. Update MMConfig.k_arrival 18.2 -> 18.3 in market_maker/config.py (fit stable; arrival term stays negligible). No other params changed; tests pin their own k so no test impact.
