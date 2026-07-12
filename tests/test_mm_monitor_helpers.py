"""Tests for app/mm_monitor_helpers.py -- the pure (streamlit-free) helpers
behind mm_monitor.py's per-expiry tabs: tab ordering, expiry attribution via
the markets registry, split-by-expiry, and the single-expiry degenerate case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import mm_monitor_helpers as mmh  # noqa: E402


def _registry_df():
    return pd.DataFrame([
        {"market_id": "a-98k", "expiry_key": "2026-07-06", "strike": 98000.0},
        {"market_id": "a-102k", "expiry_key": "2026-07-06", "strike": 102000.0},
        {"market_id": "b-98k", "expiry_key": "2026-07-07", "strike": 98000.0},
    ])


def test_run_meta_expiries_multi_events():
    run_meta = {
        "expiry_key": "2026-07-06",
        "events": [
            {"expiry_key": "2026-07-06", "event_slug": "ev-a"},
            {"expiry_key": "2026-07-07", "event_slug": "ev-b"},
        ],
    }
    assert mmh.run_meta_expiries(run_meta) == ["2026-07-06", "2026-07-07"]


def test_run_meta_expiries_legacy_singular():
    assert mmh.run_meta_expiries({"expiry_key": "2026-07-06"}) == ["2026-07-06"]
    assert mmh.run_meta_expiries(None) == []
    assert mmh.run_meta_expiries({}) == []


def test_registry_maps():
    reg = _registry_df()
    emap = mmh.registry_expiry_map(reg)
    smap = mmh.registry_strike_map(reg)
    assert emap["a-98k"] == "2026-07-06"
    assert emap["b-98k"] == "2026-07-07"
    assert smap["a-102k"] == 102000.0
    assert mmh.registry_expiry_map(None) == {}
    assert mmh.registry_expiry_map(pd.DataFrame()) == {}


def test_attach_expiry_maps_and_flags_unknown():
    fills = pd.DataFrame([
        {"market": "a-98k", "size": 5.0},
        {"market": "b-98k", "size": 2.0},
        {"market": "zzz", "size": 1.0},
    ])
    tagged = mmh.attach_expiry(fills, mmh.registry_expiry_map(_registry_df()), "market")
    assert tagged["expiry"].tolist() == ["2026-07-06", "2026-07-07", mmh.UNKNOWN_EXPIRY]


def test_expiry_tabs_order_union_sorted_unknown_last():
    tabs = mmh.expiry_tabs_order(
        ["2026-07-07", "2026-07-06"],
        ["2026-07-08", mmh.UNKNOWN_EXPIRY],
    )
    assert tabs == ["2026-07-06", "2026-07-07", "2026-07-08", mmh.UNKNOWN_EXPIRY]


def test_expiry_tabs_order_single_expiry_degenerate():
    # legacy single-expiry run: exactly one tab, no unknown
    assert mmh.expiry_tabs_order(["2026-07-06"], ["2026-07-06"]) == ["2026-07-06"]
    assert mmh.expiry_tabs_order([], []) == []


def test_split_by_expiry():
    df = pd.DataFrame([
        {"market": "a-98k", "expiry": "2026-07-06"},
        {"market": "a-102k", "expiry": "2026-07-06"},
        {"market": "b-98k", "expiry": "2026-07-07"},
    ])
    parts = mmh.split_by_expiry(df)
    assert set(parts.keys()) == {"2026-07-06", "2026-07-07"}
    assert len(parts["2026-07-06"]) == 2
    assert mmh.split_by_expiry(pd.DataFrame()) == {}


def test_event_meta_by_expiry_multi_and_legacy():
    multi = {
        "events": [
            {"expiry_key": "2026-07-06", "event_slug": "ev-a", "strikes": [98000.0]},
            {"expiry_key": "2026-07-07", "event_slug": "ev-b", "strikes": [99000.0]},
        ],
    }
    by_ek = mmh.event_meta_by_expiry(multi)
    assert by_ek["2026-07-07"]["event_slug"] == "ev-b"

    legacy = {"expiry_key": "2026-07-06", "event_slug": "ev-a", "strikes": [98000.0]}
    by_ek2 = mmh.event_meta_by_expiry(legacy)
    assert by_ek2["2026-07-06"]["event_slug"] == "ev-a"
    assert mmh.event_meta_by_expiry(None) == {}
