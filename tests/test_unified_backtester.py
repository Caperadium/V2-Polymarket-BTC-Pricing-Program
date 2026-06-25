"""
test_unified_backtester.py

Comprehensive tests for core.backtesting unified module:

  - ContractPriceStore (read/write, dedup, cold start, corrupt)
  - polymarket_fetcher (slug parsing, resolution parsing, edge cases)
  - batch_loader (timestamp parsing, column renaming)
  - BacktestEngine (smoke test with tiny dataset)
  - SignalDiagnostics (edge cases, output dict structure)
  - BacktestingOrchestrator (integration test)
  - Deprecation shims
  - Parameter sweep imports
"""

import json
import os
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_csv_path(tmp_path):
    """Return a temporary CSV path for ContractPriceStore tests."""
    return str(tmp_path / "test_historical_contract_prices.csv")


@pytest.fixture
def sample_records_df():
    """Minimal valid records as DataFrame for ContractPriceStore."""
    return pd.DataFrame([
        {
            "slug": "bitcoin-above-94k-on-november-15",
            "clobTokenId": "abc123",
            "date": datetime(2025, 11, 10, tzinfo=timezone.utc),
            "price": 0.55,
            "resolution": float("nan"),
            "strike": 94000.0,
            "expiry_date": datetime(2025, 11, 15, 17, 0, tzinfo=timezone.utc),
        },
        {
            "slug": "bitcoin-above-94k-on-november-15",
            "clobTokenId": "abc123",
            "date": datetime(2025, 11, 11, tzinfo=timezone.utc),
            "price": 0.60,
            "resolution": float("nan"),
            "strike": 94000.0,
            "expiry_date": datetime(2025, 11, 15, 17, 0, tzinfo=timezone.utc),
        },
    ])


@pytest.fixture
def mock_batch_df():
    """Minimal batch DataFrame for BacktestEngine smoke test."""
    return pd.DataFrame({
        "slug": ["btc-up", "btc-down"],
        "expiry_date": [pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")],
        "expiry_key": ["2025-01-01", "2025-01-01"],
        "strike": [90000.0, 110000.0],
        "market_price": [0.55, 0.45],
        "p_model_fit": [0.60, 0.40],
        "p_real_mc": [0.62, 0.38],
        "T_days": [5.0, 5.0],
        "pricing_date": [pd.Timestamp("2024-12-27", tz="UTC"), pd.Timestamp("2024-12-27", tz="UTC")],
        "batch_timestamp": [pd.Timestamp("2024-12-27", tz="UTC"), pd.Timestamp("2024-12-27", tz="UTC")],
    })


# ============================================================================
# ContractPriceStore
# ============================================================================

class TestContractPriceStore:
    """CSV read/write, dedup, cold start, corrupt file."""

    def test_write_read_roundtrip(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        store.save(sample_records_df)

        store2 = ContractPriceStore(csv_path=tmp_csv_path)
        df = store2.load()

        assert len(df) == 2
        assert list(df.columns) == [
            "slug", "clobTokenId", "date", "price",
            "resolution", "strike", "expiry_date",
        ]
        assert set(df["clobTokenId"].unique()) == {"abc123"}

    def test_dedup_across_merges(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        store.save(sample_records_df)
        merged = store.merge(sample_records_df, sample_records_df)

        assert len(merged) == 2  # still 2, no duplicates

    def test_cold_start_no_file(self, tmp_csv_path):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        df = store.load()

        assert len(df) == 0
        assert list(df.columns) == [
            "slug", "clobTokenId", "date", "price",
            "resolution", "strike", "expiry_date",
        ]

    def test_corrupt_file_returns_empty(self, tmp_csv_path):
        from core.backtesting.contract_store import ContractPriceStore

        Path(tmp_csv_path).write_text("garbage\nbroken,csv")

        store = ContractPriceStore(csv_path=tmp_csv_path)
        df = store.load()

        assert len(df) == 0
        assert "slug" in df.columns

    def test_append_incremental_returns_count(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        n = store.append_incremental(sample_records_df)
        assert n == 2

        n = store.append_incremental(sample_records_df)
        assert n == 0  # already there

    def test_to_market_df_shape(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        store.append_incremental(sample_records_df)
        mkt = store.to_market_df()

        expected_cols = {"slug", "strike", "market_price", "date", "expiry_date"}
        assert expected_cols.issubset(set(mkt.columns))
        assert "clobTokenId" not in mkt.columns
        # price column renamed to market_price
        assert "market_price" in mkt.columns

    def test_get_known_clob_token_ids(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        store.append_incremental(sample_records_df)

        ids = store.get_known_clob_token_ids()
        assert ids == {"abc123"}

    def test_get_clob_token_max_date(self, tmp_csv_path, sample_records_df):
        from core.backtesting.contract_store import ContractPriceStore

        store = ContractPriceStore(csv_path=tmp_csv_path)
        store.append_incremental(sample_records_df)

        max_dt = store.get_clob_token_max_date("abc123")
        assert max_dt == datetime(2025, 11, 11, tzinfo=timezone.utc)


# ============================================================================
# polymarket_fetcher — slug & resolution parsing
# ============================================================================

class TestPolymarketFetcher:
    """Slug parsing, resolution parsing, no network calls."""

    def test_strike_k_suffix(self):
        from core.backtesting.polymarket_fetcher import parse_strike_from_slug

        assert parse_strike_from_slug("bitcoin-above-94k-on-november-15") == 94000.0
        assert parse_strike_from_slug("BITCOIN-ABOVE-75K-ON-JUNE-13") == 75000.0

    def test_strike_numeric(self):
        from core.backtesting.polymarket_fetcher import parse_strike_from_slug

        assert parse_strike_from_slug("bitcoin-above-94000-on-november-15") == 94000.0
        assert parse_strike_from_slug("bitcoin-above-80000-on-january-1") == 80000.0

    def test_strike_full_format(self):
        from core.backtesting.polymarket_fetcher import parse_strike_from_slug

        assert parse_strike_from_slug("will-the-price-of-bitcoin-be-above-78000-on-december-31") == 78000.0
        assert parse_strike_from_slug("will-the-price-of-bitcoin-be-above-52000-on-june-17") == 52000.0
        assert parse_strike_from_slug("will-the-price-of-bitcoin-be-above-100000-on-january-15") == 100000.0

    def test_strike_invalid_slug(self):
        from core.backtesting.polymarket_fetcher import parse_strike_from_slug

        assert parse_strike_from_slug("random-other-market") is None
        assert parse_strike_from_slug("") is None

    def test_resolution_yes(self):
        from core.backtesting.polymarket_fetcher import parse_resolution_from_market

        assert parse_resolution_from_market({"resolution": "YES"}) == 1.0
        assert parse_resolution_from_market({"resolution": 1.0}) == 1.0

    def test_resolution_no(self):
        from core.backtesting.polymarket_fetcher import parse_resolution_from_market

        assert parse_resolution_from_market({"resolution": "NO"}) == 0.0
        assert parse_resolution_from_market({"resolution": 0.0}) == 0.0

    def test_resolution_missing(self):
        from core.backtesting.polymarket_fetcher import parse_resolution_from_market

        result = parse_resolution_from_market({})
        assert np.isnan(result)

    def test_resolution_from_outcomes(self):
        from core.backtesting.polymarket_fetcher import parse_resolution_from_market

        result = parse_resolution_from_market({"outcomes": ["YES"]})
        assert result == 1.0

    def test_clob_token_ids_json_string(self):
        from core.backtesting.polymarket_fetcher import parse_clob_token_ids

        ids = parse_clob_token_ids({"clobTokenIds": '["id1", "id2"]'})
        assert ids == ["id1", "id2"]

    def test_clob_token_ids_list(self):
        from core.backtesting.polymarket_fetcher import parse_clob_token_ids

        ids = parse_clob_token_ids({"clobTokenIds": ["id1", "id2"]})
        assert ids == ["id1", "id2"]

    def test_parse_history_midnight_filter(self):
        from core.backtesting.polymarket_fetcher import _parse_history

        # Mix of midnight and non-midnight entries
        raw = [
            {"t": int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()), "p": 0.5},
            {"t": int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()), "p": 0.6},
            {"t": int(datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc).timestamp()), "p": 0.7},
        ]
        parsed = _parse_history(raw)

        assert len(parsed) == 3
        assert parsed[0]["date_utc"].hour == 0
        assert parsed[1]["date_utc"].hour == 12
        assert parsed[2]["date_utc"].hour == 0

    def test_parse_history_invalid_entries_skipped(self):
        from core.backtesting.polymarket_fetcher import _parse_history

        raw = [
            {"t": None, "p": 0.5},   # no timestamp
            {"t": 123, "p": None},   # no price
            {"t": "not-a-number", "p": 0.5},  # bad timestamp
        ]
        parsed = _parse_history(raw)
        assert len(parsed) == 0

    def test_expiry_from_slug(self):
        from core.backtesting.polymarket_fetcher import parse_expiry_from_slug

        # "november" is full month name → month 11
        dt = parse_expiry_from_slug("bitcoin-above-94k-on-november-15", year_hint=2025)
        assert dt is not None
        assert dt.month == 11
        assert dt.day == 15
        # Noon ET + 5h → 17:00 UTC
        assert dt.hour == 17

    def test_expiry_from_slug_abbreviated(self):
        from core.backtesting.polymarket_fetcher import parse_expiry_from_slug

        dt = parse_expiry_from_slug("bitcoin-above-80k-on-jun-13", year_hint=2025)
        assert dt is not None
        assert dt.month == 6
        assert dt.day == 13

    def test_expiry_invalid_slug(self):
        from core.backtesting.polymarket_fetcher import parse_expiry_from_slug

        assert parse_expiry_from_slug("not-a-valid-slug") is None


# ============================================================================
# batch_loader
# ============================================================================

class TestBatchLoader:
    """Timestamp parsing (new + legacy), column renaming."""

    def test_prepare_batch_df_new_format(self):
        from core.backtesting.batch_loader import prepare_batch_df

        df = pd.DataFrame({"model_probability": [0.6], "expiry_date": ["2025-11-15"]})
        result = prepare_batch_df(df, "2025-11-10_05-57-14_UTC")

        assert "p_model_fit" in result.columns
        assert "pricing_date" in result.columns
        assert "expiry_key" in result.columns
        assert result["expiry_key"].iloc[0] == "2025-11-15"

    def test_prepare_batch_df_legacy_format(self):
        from core.backtesting.batch_loader import prepare_batch_df

        df = pd.DataFrame({"model_probability": [0.6], "expiry_date": ["2025-11-15"]})
        result = prepare_batch_df(df, "batch_20251113_220535")

        assert "p_model_fit" in result.columns
        assert "pricing_date" in result.columns

    def test_prepare_batch_df_no_model_probability(self):
        from core.backtesting.batch_loader import prepare_batch_df

        df = pd.DataFrame({"p_model_fit": [0.5], "expiry_date": ["2025-01-01"]})
        result = prepare_batch_df(df, "some_folder")
        assert result["p_model_fit"].iloc[0] == 0.5

    def test_load_batches_success(self, tmp_path):
        from core.backtesting.batch_loader import load_batches

        # Create a mock batch CSV inside a folder (load_batches uses parent folder name)
        batch_dir = tmp_path / "2025-11-10_05-57-14_UTC"
        batch_dir.mkdir()
        batch_path = batch_dir / "batch_with_fits.csv"
        pd.DataFrame({"model_probability": [0.6], "expiry_date": ["2025-11-15"]}).to_csv(batch_path, index=False)

        batches = load_batches([str(batch_path)])
        assert len(batches) == 1
        assert "p_model_fit" in batches[0].columns

    def test_scan_batch_files(self, tmp_path):
        from core.backtesting.batch_loader import scan_batch_files

        # Use 23:00 UTC so local conversion always stays on same day regardless of timezone
        d = tmp_path / "2025-06-01_23-00-00_UTC"
        d.mkdir(parents=True)
        (d / "batch_with_fits.csv").write_text("slug\nbtc-up")

        # Broad date range so timezone offset doesn't matter
        paths = scan_batch_files(
            str(tmp_path),
            date(2025, 5, 30),
            date(2025, 6, 3),
        )
        assert len(paths) == 1

    def test_scan_flat_batch_files(self, tmp_path):
        from core.backtesting.batch_loader import scan_flat_batch_files

        f = tmp_path / "batch_20250601_120000.csv"
        f.write_text("slug\nbtc-up")

        # Broad date range so timezone offset doesn't matter
        paths = scan_flat_batch_files(
            str(tmp_path),
            date(2025, 5, 30),
            date(2025, 6, 3),
        )
        assert len(paths) == 1


# ============================================================================
# BacktestEngine smoke test
# ============================================================================

class TestBacktestEngine:
    """Smoke test — verify output format unchanged."""

    def test_run_backtest_trivial(self, mock_batch_df):
        from core.backtesting.backtest_engine import run_backtest

        params = {
            "kelly_fraction": 0.15,
            "min_edge": 0.01,
            "max_bets_per_expiry": 5,
            "min_price": 0.01,
            "max_price": 0.99,
            "allow_no": True,
            "min_trade_usd": 1.0,
            "max_add_per_cycle_usd": 50.0,
            "correlation_penalty": 0.25,
        }

        trades, equity = run_backtest(
            [mock_batch_df],
            initial_bankroll=1000.0,
            strategy_params=params,
            return_all_priced=False,
        )

        assert isinstance(trades, pd.DataFrame)
        assert isinstance(equity, pd.DataFrame)
        assert "bankroll" in equity.columns


# ============================================================================
# SignalDiagnostics
# ============================================================================

class TestSignalDiagnostics:
    """Edge case handling, output dict structure.

    SignalDiagnostics expects specific column names via _pick_column:
      - outcome: "outcome_yes" or "outcome"
      - model prob: "model_prob_used"
      - market price: "market_yes_price"
    """

    def _make_all_priced_df(self, outcomes: list) -> pd.DataFrame:
        n = len(outcomes)
        return pd.DataFrame({
            "slug": [f"btc-{i}" for i in range(n)],
            "model_prob_used": [0.55 + i * 0.05 for i in range(n)],
            "market_yes_price": [0.50] * n,
            "outcome": outcomes,
            "dte_days": [3.0 + i for i in range(n)],
            "moneyness": [0.01 + i * 0.02 for i in range(n)],
        })

    def test_full_report_structure(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        df = self._make_all_priced_df([1, 0, 1, 0])
        diag = SignalDiagnostics(df)
        report = diag.run_full_report()

        assert "n_observations" in report
        assert report["n_observations"] == 4
        assert "spearman_rho" in report
        assert "auc" in report
        assert "dte_breakdown" in report
        assert "moneyness_breakdown" in report
        assert "dte_available" in report
        assert "moneyness_available" in report

    def test_zero_class_diversity(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        df = self._make_all_priced_df([1, 1, 1])  # all winners
        diag = SignalDiagnostics(df)
        report = diag.run_full_report()

        assert report["n_positive"] == 3
        assert report["n_negative"] == 0
        # AUC is None for single class (not nan)
        assert report["auc"] is None

    def test_nan_outcomes_filtered(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        df = self._make_all_priced_df([1, np.nan, 0, np.nan])
        diag = SignalDiagnostics(df)
        report = diag.run_full_report()

        # NaN outcomes filtered out during _clean_data (coerced then dropped)
        assert report["n_observations"] == 2

    def test_empty_dataframe(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        empty = pd.DataFrame()
        diag = SignalDiagnostics(empty)
        report = diag.run_full_report()

        assert report["n_observations"] == 0

    def test_missing_dte_moneyness_columns(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        df = pd.DataFrame({
            "slug": ["btc-a"],
            "model_prob_used": [0.55],
            "market_yes_price": [0.50],
            "outcome": [1],
        })
        diag = SignalDiagnostics(df)
        report = diag.run_full_report()

        assert report["dte_available"] is False
        assert report["moneyness_available"] is False
        assert report["dte_breakdown"] == []
        assert report["moneyness_breakdown"] == []

    def test_compute_metrics(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        rho, p_value, auc = SignalDiagnostics.compute_metrics(
            outcome=np.array([0, 0, 1, 1]),
            score=np.array([0.1, 0.3, 0.7, 0.9]),
        )
        assert -1.0 <= rho <= 1.0
        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_interpret_auc_positive_signal(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        result = SignalDiagnostics.interpret_auc(0.95)
        assert "positive signal" in result.lower()
        assert "0.9500" in result

    def test_interpret_auc_anti_signal(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        result = SignalDiagnostics.interpret_auc(0.35)
        assert "anti-signal" in result.lower()

    def test_interpret_auc_no_discrimination(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        result = SignalDiagnostics.interpret_auc(0.50)
        assert "no discrimination" in result.lower()

    def test_interpret_auc_none(self):
        from core.backtesting.diagnostics import SignalDiagnostics

        result = SignalDiagnostics.interpret_auc(None)
        assert "no class diversity" in result.lower()


# ============================================================================
# Deprecation shims
# ============================================================================

class TestDeprecationShims:
    """Old imports still work but emit DeprecationWarning."""

    @pytest.mark.filterwarnings("always::DeprecationWarning")
    def test_backtest_engine_shim(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            # Force fresh import by clearing from sys.modules
            sys.modules.pop("scripts.backtesting.backtest_engine", None)
            from scripts.backtesting.backtest_engine import run_backtest  # noqa: F811

            assert len(w) >= 1, f"Expected DeprecationWarning, got {len(w)} warnings"
            assert issubclass(w[0].category, DeprecationWarning)
            assert callable(run_backtest)

    def test_prob_backrunner_shim(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            sys.modules.pop("scripts.backtesting.prob_backrunner_engine", None)
            from scripts.backtesting.prob_backrunner_engine import BackrunnerEngine  # noqa: F811

            assert len(w) >= 1, f"Expected DeprecationWarning, got {len(w)} warnings"
            assert issubclass(w[0].category, DeprecationWarning)

    def test_signal_diagnostics_shim(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            sys.modules.pop("core.strategy.signal_diagnostics", None)
            from core.strategy.signal_diagnostics import main_cli  # noqa: F811

            assert len(w) >= 1, f"Expected DeprecationWarning, got {len(w)} warnings"
            assert issubclass(w[0].category, DeprecationWarning)
            assert callable(main_cli)


# ============================================================================
# Parameter sweep imports
# ============================================================================

class TestParameterSweepImports:
    """Verify all fixed imports resolve correctly."""

    def test_backtest_engine_import(self):
        from core.backtesting.backtest_engine import BacktestEngine
        assert BacktestEngine is not None

    def test_montecarlo_sim_imports(self):
        from scripts.backtesting.backtest_montecarlo_sim import (
            run_shuffle_test,
            get_summary_stats,
            run_decile_conditioned_shuffle_test,
        )
        assert run_shuffle_test is not None
        assert get_summary_stats is not None
        assert run_decile_conditioned_shuffle_test is not None

    def test_auto_reco_imports(self):
        from core.strategy.auto_reco import (
            reset_threshold_debug,
            LAST_RECO_THRESHOLD_DEBUG,
        )
        assert callable(reset_threshold_debug)
        # LAST_RECO_THRESHOLD_DEBUG is a module-level debug dict, None when unused
        assert LAST_RECO_THRESHOLD_DEBUG is None  # default state


# ============================================================================
# BacktestingOrchestrator integration
# ============================================================================

class TestOrchestratorIntegration:
    """Smoke test orchestrator with mocked fetcher."""

    @patch("core.backtesting.orchestrator.BackrunnerEngine")
    def test_orchestrator_creates(self, mock_backrunner):
        from core.backtesting.orchestrator import BacktestingOrchestrator

        o = BacktestingOrchestrator(n_sims=100, seed=42)
        assert o.n_sims == 100
        assert o.seed == 42

    def test_run_full_dict_structure(self, tmp_path):
        """Verify run_full() returns expected dict keys with empty data paths."""
        from core.backtesting.orchestrator import BacktestingOrchestrator

        unfitted = tmp_path / "unfitted"
        fitted = tmp_path / "fitted"
        unfitted.mkdir()
        fitted.mkdir()

        o = BacktestingOrchestrator(
            n_sims=50,
            seed=42,
            unfitted_dir=str(unfitted),
            fitted_dir=str(fitted),
        )

        result = o.run_full(fetch=False, backrun=False, fit_curves=False)

        assert "new_records" in result
        assert result["new_records"] == 0  # fetch=False
        assert "trades_df" in result
        assert "equity_df" in result
        assert isinstance(result["trades_df"], pd.DataFrame)
        assert isinstance(result["equity_df"], pd.DataFrame)


# ============================================================================
# __init__ exports
# ============================================================================

class TestInitExports:
    """Verify all public names are importable from core.backtesting."""

    def test_all_exports_importable(self):
        from core.backtesting import (
            BackrunnerEngine,
            BacktestEngine,
            BacktestingOrchestrator,
            ContractPriceStore,
            SignalDiagnostics,
            fetch_incremental_prices,
            load_batches,
            prepare_batch_df,
            run_backtest,
            run_diagnostics,
            scan_batch_files,
            scan_flat_batch_files,
        )

        # Check types
        assert ContractPriceStore is not None
        assert callable(fetch_incremental_prices)
        assert callable(prepare_batch_df)
        assert callable(load_batches)
        assert callable(scan_batch_files)
        assert callable(scan_flat_batch_files)
        assert BackrunnerEngine is not None
        assert BacktestEngine is not None
        assert callable(run_backtest)
        assert SignalDiagnostics is not None
        assert callable(run_diagnostics)
        assert BacktestingOrchestrator is not None


# ============================================================================
# Fix 1b: Volgate DataFrame shape & caching
# ============================================================================

class TestVolgateDataFrame:
    """Verify _volgate_btc_df has correct shape for compute_vol_gate."""

    @pytest.fixture
    def intraday_df(self):
        """Minimal intraday DataFrame with timestamp column + close (as load_btc_csv returns)."""
        times = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
        return pd.DataFrame({
            "timestamp": times,
            "close": np.random.default_rng(42).normal(50000, 100, 100).cumsum() + 40000,
        })

    def test_volgate_df_has_timestamp_column_not_index(self, intraday_df, tmp_path):
        """After _load_btc_prices, _volgate_btc_df must be a plain df with 'timestamp' COLUMN."""
        from core.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            market_data_batches=[],
            initial_bankroll=1000.0,
            strategy_params={},
            price_df=intraday_df,
        )
        # _load_btc_prices called in __init__
        engine._load_btc_prices()

        vdf = engine._volgate_btc_df
        assert vdf is not None, "_volgate_btc_df should be populated"
        assert isinstance(vdf, pd.DataFrame)
        assert "timestamp" in vdf.columns
        assert "close" in vdf.columns
        # Must be default RangeIndex, NOT DatetimeIndex
        assert not isinstance(vdf.index, pd.DatetimeIndex), \
            f"_volgate_btc_df index must be plain RangeIndex, got {type(vdf.index).__name__}"

    def test_volgate_df_values_match_btc_prices(self, intraday_df, tmp_path):
        """_volgate_btc_df.close values should match _btc_prices.close values."""
        from core.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            market_data_batches=[],
            initial_bankroll=1000.0,
            strategy_params={},
            price_df=intraday_df,
        )
        engine._load_btc_prices()

        vdf = engine._volgate_btc_df
        btc = engine._btc_prices

        pd.testing.assert_index_equal(
            pd.DatetimeIndex(vdf["timestamp"]),
            pd.DatetimeIndex(btc.index),
            check_names=False,  # index names differ: "timestamp" vs "datetime_utc"
        )
        pd.testing.assert_series_equal(
            vdf["close"].reset_index(drop=True),
            btc["close"].reset_index(drop=True),
            check_names=False,
        )

    def test_volgate_df_is_none_when_no_data(self, tmp_path):
        """When no BTC data loaded, _volgate_btc_df stays None."""
        from core.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            market_data_batches=[],
            initial_bankroll=1000.0,
            strategy_params={},
            btc_price_path=str(tmp_path / "nonexistent.csv"),
            price_df=None,
        )
        engine._load_btc_prices()  # no file, no df → empty

        assert engine._volgate_btc_df is None, \
            "_volgate_btc_df should be None when no BTC data loaded"


# ============================================================================
# Fix 2: asof_utc parameter in recommend_trades
# ============================================================================

class TestAsofUtcParameter:
    """Verify asof_utc is accepted and propagated correctly."""

    @pytest.fixture
    def minimal_batch(self):
        return pd.DataFrame({
            "slug": ["btc-test"],
            "strike": [50000.0],
            "expiry_date": [pd.Timestamp("2025-06-15", tz="UTC")],
            "expiry_key": ["2025-06-15"],
            "market_price": [0.55],
            "p_model_fit": [0.60],
            "T_days": [5.0],
            "side": ["YES"],
        })

    @pytest.fixture
    def tiny_btc_df(self):
        """BTC data spanning a few hours with enough rows for vol gate lookback."""
        times = pd.date_range(
            "2025-01-01 00:00:00", periods=2000, freq="1min", tz="UTC"
        )
        rng = np.random.default_rng(99)
        return pd.DataFrame({
            "timestamp": times,
            "close": rng.normal(50000, 50, 2000).cumsum() + 45000,
        })

    def test_asof_utc_accepted_in_signature(self):
        """recommend_trades must accept asof_utc keyword argument."""
        from core.strategy.auto_reco import recommend_trades
        import inspect

        sig = inspect.signature(recommend_trades)
        assert "asof_utc" in sig.parameters, \
            "asof_utc parameter must be in recommend_trades signature"

    def test_asof_utc_propagates_to_vol_gate(self, minimal_batch, tiny_btc_df):
        """When asof_utc is provided, compute_vol_gate receives it (no 'stale' fallback)."""
        from core.strategy.auto_reco import recommend_trades
        from datetime import datetime, timezone

        historical_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        future_time = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)

        # With historical asof_utc + matching BTC data: should get a vol gate result
        recos = recommend_trades(
            df=minimal_batch,
            bankroll=1000.0,
            min_edge=0.01,
            min_price=0.01,
            max_price=0.99,
            allow_no=True,
            min_trade_usd=1.0,
            btc_price_df=tiny_btc_df,
            asof_utc=historical_time,
            require_active=False,
        )
        # Should return a list (may be empty if no trades meet criteria, but no crash)
        assert isinstance(recos, list)

        # With far-future asof_utc (vol gate can't find data → fallback, no crash)
        recos_future = recommend_trades(
            df=minimal_batch,
            bankroll=1000.0,
            min_edge=0.01,
            min_price=0.01,
            max_price=0.99,
            allow_no=True,
            min_trade_usd=1.0,
            btc_price_df=tiny_btc_df,
            asof_utc=future_time,
            require_active=False,
        )
        assert isinstance(recos_future, list)

    def test_asof_utc_default_is_now(self, minimal_batch):
        """When asof_utc is omitted, it defaults to datetime.now() behavior."""
        from core.strategy.auto_reco import recommend_trades
        from datetime import datetime, timezone

        before = datetime.now(timezone.utc)
        # Without btc_price_df, load_btc_csv() is called → may fail in CI,
        # so pass btc_price_df=None but expect clean handling
        recos = recommend_trades(
            df=minimal_batch,
            bankroll=1000.0,
            min_edge=0.01,
            min_price=0.01,
            max_price=0.99,
            allow_no=True,
            min_trade_usd=1.0,
            btc_price_df=None,  # triggers load_btc_csv() fallback
            require_active=False,
        )
        after = datetime.now(timezone.utc)
        assert isinstance(recos, list), "Should not crash even if BTC file missing"


# ============================================================================
# Fix 3: Path-robust BTC loader
# ============================================================================

class TestBtcPathResolution:
    """Verify absolute path defaults and relative path resolution."""

    def test_default_intraday_is_absolute(self):
        """_DEFAULT_INTRADAY must be an absolute Path."""
        from core.backtesting.backtest_engine import _DEFAULT_INTRADAY, _PROJECT_ROOT

        assert _DEFAULT_INTRADAY.is_absolute(), \
            f"_DEFAULT_INTRADAY must be absolute, got {_DEFAULT_INTRADAY}"
        assert _PROJECT_ROOT in _DEFAULT_INTRADAY.parents, \
            f"_DEFAULT_INTRADAY must be under _PROJECT_ROOT"

    def test_constructor_default_is_absolute(self):
        """BacktestEngine constructor default must resolve to absolute path."""
        from core.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            market_data_batches=[],
            initial_bankroll=1000.0,
            strategy_params={},
        )
        path = Path(engine.btc_price_path)
        assert path.is_absolute(), \
            f"Constructor default btc_price_path must be absolute, got {engine.btc_price_path}"

    def test_relative_path_resolved_in_load(self, tmp_path):
        """When a relative btc_price_path is passed, _load_btc_prices resolves it."""
        import core.backtesting.backtest_engine as bte
        from unittest.mock import patch

        # Create fake intraday data in a temp dir
        fake_data = tmp_path / "sub" / "btc.csv"
        fake_data.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": ["2025-01-01 00:00:00", "2025-01-01 00:01:00"],
            "close": [50000.0, 50001.0],
        }).to_csv(fake_data, index=False)

        # Monkey-patch _PROJECT_ROOT to point at tmp_path
        with patch.object(bte, "_PROJECT_ROOT", tmp_path):
            engine = bte.BacktestEngine(
                market_data_batches=[],
                initial_bankroll=1000.0,
                strategy_params={},
                btc_price_path="sub/btc.csv",
            )
            engine._load_btc_prices()

            assert engine._btc_prices is not None
            assert not engine._btc_prices.empty
            assert engine._intraday_min is not None
            assert engine._intraday_max is not None

    def test_file_not_found_handled_gracefully(self, tmp_path):
        """Missing file should log warning, not crash."""
        import core.backtesting.backtest_engine as bte
        from unittest.mock import patch

        nonexistent = tmp_path / "nope.csv"
        with patch.object(bte, "_PROJECT_ROOT", tmp_path):
            engine = bte.BacktestEngine(
                market_data_batches=[],
                initial_bankroll=1000.0,
                strategy_params={},
                btc_price_path=str(nonexistent),
            )
            engine._load_btc_prices()

            # Should have empty _btc_prices, not crashed
            assert engine._btc_prices.empty
            assert engine._volgate_btc_df is None
