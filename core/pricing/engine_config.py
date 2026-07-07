"""
engine_config.py

Single source of truth for the v2 pricing-engine flag bundle passed to
``calculate_probabilities()``.

Wave 2 / T5 (H2 remediation): the backtest path (core/backtesting/backrunner.py)
and the live pipelines (scripts/pipelines/run_full_pipeline.py,
scripts/pipelines/batch_pricing_runner.py) used to construct DIFFERENT engine
configurations -- backtest called ``calculate_probabilities`` with regime
switching, calibrated jumps, and horizon gating; live called
``simulate_paths`` directly with none of that. Any backtest-derived quantity
(edge thresholds, the M2 logit-shift calibration, sweep-selected strategy
parameters) was therefore validated against a different model than the one
serving live prices.

``build_engine_kwargs`` is the fix: both call sites build their
``calculate_probabilities`` kwargs through this one function so they cannot
silently drift apart again.

NOT covered by this function (each call site supplies these itself, by
design -- they are legitimately different between live and backtest, not an
accidental drift):
  - strikes, hours_to_expiry
  - hourly_df / intraday_df (or the _csv path variants)
  - disable_staleness_check (True in backtest -- time-travel snapshots are
    always "stale" relative to wall clock; False/default live)
  - garch_cache, s0_override (per-snapshot dedup caches; backtest shares one
    per snapshot across expiry groups, live shares one per run)
  - as_of IS covered (see below) but its VALUE legitimately differs: None
    live (wall-clock regime refit gating), a fixed snapshot timestamp in
    backtest (leak-free, deterministic refit gating).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def build_engine_kwargs(
    advanced_features: bool,
    detector,
    regime_params: Optional[dict],
    jump_params: Optional[dict],
    n_sims: int,
    seed: Optional[int] = None,
    as_of: Optional[datetime] = None,
    use_xgb: bool = False,
    xgb_model=None,
    xgb_tilt_lambda: Optional[float] = None,
    macro_df=None,
) -> Dict[str, Any]:
    """Build the v2 engine flag bundle for ``calculate_probabilities()``.

    Used by BOTH ``core/backtesting/backrunner.py`` AND the live pipelines
    (``scripts/pipelines/run_full_pipeline.py``,
    ``scripts/pipelines/batch_pricing_runner.py``) so they cannot drift apart
    again (H2).

    Args:
        advanced_features: Master switch for SVCJ + skewed-t + FIGARCH +
            regime switching (mirrors the existing CLI ``--advanced-features``
            flag in both live pipelines and the backrunner).
        detector: A ``RegimeDetector`` instance, or None to disable regime
            switching regardless of ``advanced_features``.
        regime_params: Per-regime jump-parameter overrides, typically from
            ``build_regime_jump_params()``. May be None.
        jump_params: Base (non-regime) calibrated jump parameters dict, keyed
            'lambda'/'crash_prob'/'eta_up'/'eta_down'/'mu_v'/'rho_J'/
            'rho_j_slope' (the keys ``simulate_paths`` expects -- NOT the raw
            'lam'/'p_crash' dict returned by ``load_calibrated_jumps``).
        n_sims: Number of Monte Carlo paths.
        seed: RNG seed, or None for nondeterministic.
        as_of: Snapshot timestamp for leak-free regime refit gating. None
            (default) reproduces live wall-clock behavior; backtest passes
            the snapshot's timestamp.
        use_xgb: Master switch for the XGBoost directional drift shift.
        xgb_model: A trained ``DirectionalXGB`` instance for this DTE bucket,
            or None. ``use_xgb_direction`` is only True when both this is
            True AND ``xgb_model`` is not None (mirrors the existing
            backrunner ``_process_one`` logic).
        xgb_tilt_lambda: Tilt strength override; None uses the engine module
            default (``XGB_TILT_LAMBDA``).
        macro_df: Leak-free macro DataFrame slice for the XGB directional
            model (None runs BTC-only features).

    Returns:
        Dict suitable for ``**kwargs`` into ``calculate_probabilities()``.
        Does NOT include strikes/hours_to_expiry/hourly_df/intraday_df/
        disable_staleness_check/garch_cache/s0_override -- see module
        docstring.
    """
    return {
        "n_sims": n_sims,
        "seed": seed,
        "jump_params": jump_params,
        "use_svcj": advanced_features,
        "use_skewed_t": advanced_features,
        "use_figarch": advanced_features,
        "use_regime_switching": (advanced_features and detector is not None),
        "regime_detector": detector,
        "regime_params": regime_params,
        "as_of": as_of,
        "use_xgb_direction": (use_xgb and xgb_model is not None),
        "xgb_model": xgb_model,
        "xgb_tilt_lambda": xgb_tilt_lambda,
        "macro_df": macro_df,
    }
