Implement unified backtesting module (core/backtesting/) — per approved 20-file plan.

**New files:**
- core/backtesting/__init__.py — public API exports (12 symbols)
- core/backtesting/contract_store.py — CSV-backed store for historical Polymarket prices (7-col schema, dedup on clobTokenId+date)
- core/backtesting/polymarket_fetcher.py — Gamma /events slug-based discovery + CLOB /prices-history with rate-limit handling, 429 retry, stale-data refresh
- core/backtesting/batch_loader.py — canonical batch CSV normalization (extracted from duplicated dashboard copies)
- core/backtesting/backrunner.py — time-travel MC pricing engine (disk-native streaming, idempotent skip)
- core/backtesting/backtest_engine.py — chronological backtest simulation (moved from scripts/backtesting/)
- core/backtesting/diagnostics.py — Spearman/AUC/DTE signal diagnostics (absorbed from core/strategy/signal_diagnostics.py)
- core/backtesting/orchestrator.py — full pipeline entry point: fetch → backrun → fit → backtest → diagnostics
- tests/test_unified_backtester.py — 49 unit tests across 9 test classes

**API discovery fix:**
- Replaced broken /markets?tag= approach with date-slug iteration via /events?slug=bitcoin-above-on-{month}-{day} (matches provider_polymarket.py:619)
- Slugs returned as list (no pagination needed); Gamma filters correctly
- Strike parser handles 3 formats: k-suffix (bitcoin-above-94k), numeric (bitcoin-above-94000), full (will-the-price-of-bitcoin-be-above-78000-on-…)

**Rate-limit & error handling:**
- CLOB delay: 200→50ms; timeout: 30→10s; 429 detection with 2s backoff + 1 retry
- fetch_incremental_prices() returns (count, errors_list) for Streamlit display
- Errors surfaced: "no markets found", HTTP status codes, network failures, rate-limit summary

**Dashboard backtesting page (app/pages/backtesting.py):**
- Two-mode interface: Existing Batch Files (preserved) + Live Fetch from Polymarket (new)
- Live Fetch uses BacktestingOrchestrator with st.status() progress
- Signal Diagnostics display: Spearman ρ, p-value, AUC, mean edge winners/losers, DTE breakdown, moneyness breakdown
- Error display: fetcher warnings shown as st.warning captions

**Deprecation shims:**
- scripts/backtesting/prob_backrunner_engine.py → core.backtesting.backrunner
- scripts/backtesting/backtest_engine.py → core.backtesting.backtest_engine
- core/strategy/signal_diagnostics.py → core.backtesting.diagnostics

**Documentation:**
- CLAUDE.md updated (new commands, directory structure, architecture section)
- 5 DOCS files updated with new import paths
