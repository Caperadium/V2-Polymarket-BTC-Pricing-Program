#!/usr/bin/env python
"""
Determine what UTC time the Polymarket CLOB /prices-history endpoint
snaps its daily timestamps to (e.g., midnight UTC, noon UTC, etc.).

Usage:
    python scripts/utilities/check_price_history_timestamps.py
"""

import json
import sys
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"

# Try these slug patterns in order to find an active contract
SLUG_CANDIDATES = [
    "bitcoin-above-on-june-{}",
    "bitcoin-above-on-july-{}",
    "bitcoin-above-75k-on-june-{}",
    "bitcoin-above-75k-on-july-{}",
    "bitcoin-above-70k-on-june-{}",
    "bitcoin-above-80k-on-june-{}",
]

# Fallback: known high-volume resolved BTC contract clobTokenId
# (from Polymarket Gamma for a past bitcoin-above market)
FALLBACK_CLOB_TOKEN_IDS = [
    # We'll populate this from the Gamma search below
]


def fetch_btc_events(slug_pattern: str, day: int) -> list:
    """Fetch events from Gamma API for given slug pattern and day."""
    slug = slug_pattern.format(day)
    resp = requests.get(GAMMA_API_URL, params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


def find_active_clob_token_id() -> str | None:
    """
    Search recent/future days across slug patterns for an active BTC
    contract that has a clobTokenId.  We try days from today to +14 days.
    """
    today = datetime.now(timezone.utc)
    for offset in range(-2, 15):  # look back 2 days, forward 14
        day = today.day + offset
        # Simple day rollover (handles month boundaries roughly)
        for pattern in SLUG_CANDIDATES:
            try:
                events = fetch_btc_events(pattern, day)
                for event in events:
                    for market in event.get("markets", []):
                        clob_ids_raw = market.get("clobTokenIds", "[]")
                        clob_ids = json.loads(clob_ids_raw)
                        if clob_ids and len(clob_ids) > 0:
                            question = market.get("question", event.get("title", "?"))
                            volume = market.get("volume", "0")
                            print(f"[+] Found active contract: {question}")
                            print(f"    clobTokenId: {clob_ids[0]}")
                            print(f"    volume: {volume}")
                            return clob_ids[0]
            except Exception:
                continue
    return None


def search_fallback_clob_token_id() -> str | None:
    """
    Broad Gamma search for any 'bitcoin-above' or 'btc' market
    that has a clobTokenId (resolved or active).
    """
    search_terms = ["bitcoin-above", "btc-above", "bitcoin"]
    for term in search_terms:
        print(f"[*] Searching Gamma for slug containing '{term}'...")
        try:
            resp = requests.get(
                GAMMA_API_URL,
                params={"slug": term, "limit": 10},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                continue
            for event in data:
                for market in event.get("markets", []):
                    clob_ids_raw = market.get("clobTokenIds", "[]")
                    clob_ids = json.loads(clob_ids_raw)
                    if clob_ids and len(clob_ids) > 0:
                        question = market.get("question", event.get("title", "?"))
                        closed = market.get("closed", event.get("closed", False))
                        volume = market.get("volume", "0")
                        print(f"[+] Found contract: {question}")
                        print(f"    clobTokenId: {clob_ids[0]}")
                        print(f"    volume: {volume}")
                        print(f"    closed: {closed}")
                        return clob_ids[0]
        except Exception as e:
            print(f"[-] Search failed for '{term}': {e}")
            continue
    return None


def query_prices_history(
    clob_token_id: str, interval: str, fidelity: int | None = None
) -> list[dict]:
    """
    Query /prices-history for the given token.
    Returns list of {t, p} dicts.
    """
    params = {
        "market": clob_token_id,
        "interval": interval,
    }
    if fidelity is not None:
        params["fidelity"] = fidelity

    print(f"\n[*] Querying /prices-history with params: {params}")
    resp = requests.get(CLOB_PRICES_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("history", [])


def print_history_table(label: str, history: list[dict]):
    """Print a formatted table of (t, datetime_utc, p)."""
    print(f"\n--- {label} ---")
    if not history:
        print("  (empty)")
        return
    print(f"{'unixtime':>12}  {'datetime_utc':<26}  {'price':>10}")
    print("-" * 52)
    for entry in history:
        t = entry["t"]
        p = entry["p"]
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"{t:>12}  {dt_str:<26}  {p:>10.4f}")


def analyze_timestamps(history: list[dict], label: str) -> dict:
    """Extract hour/minute/second patterns from timestamps."""
    hours = set()
    minutes = set()
    seconds = set()
    dows = set()
    for entry in history:
        dt = datetime.fromtimestamp(entry["t"], tz=timezone.utc)
        hours.add(dt.hour)
        minutes.add(dt.minute)
        seconds.add(dt.second)
        dows.add(dt.strftime("%A"))
    return {
        "label": label,
        "count": len(history),
        "hours": sorted(hours),
        "minutes": sorted(minutes),
        "seconds": sorted(seconds),
        "days_of_week": sorted(dows),
    }


def main():
    print("=" * 60)
    print("Polymarket /prices-history Timestamp Alignment Check")
    print("=" * 60)

    # Step 0: Find a clobTokenId
    clob_token_id = find_active_clob_token_id()
    if clob_token_id is None:
        print("[!] No active contract found via slug search. Trying fallback...")
        clob_token_id = search_fallback_clob_token_id()

    if clob_token_id is None:
        print("[FATAL] Could not find any clobTokenId. Exiting.")
        sys.exit(1)

    print(f"\n[*] Using clobTokenId: {clob_token_id}")

    # Step 1: Daily granularity
    print("\n" + "=" * 40)
    print("STEP 1: Daily granularity (interval=1d)")
    print("=" * 40)
    try:
        daily = query_prices_history(clob_token_id, interval="1d")
        print_history_table("Daily candles", daily)
    except Exception as e:
        print(f"[-] interval=1d failed: {e}")
        daily = []

    # If daily returned empty, try fidelity=1440 as fallback
    if not daily:
        print("\n[*] interval=1d returned empty, trying fidelity=1440 + interval=max...")
        try:
            daily = query_prices_history(
                clob_token_id, interval="max", fidelity=1440
            )
            print_history_table("Daily candles (fidelity=1440, interval=max)", daily)
        except Exception as e:
            print(f"[-] fidelity=1440 fallback also failed: {e}")

    # Step 2: 12h granularity
    print("\n" + "=" * 40)
    print("STEP 2: 12-hour granularity (interval=max, fidelity=720)")
    print("=" * 40)
    try:
        halfday = query_prices_history(
            clob_token_id, interval="max", fidelity=720
        )
        print_history_table("12h candles", halfday)
    except Exception as e:
        print(f"[-] 12h query failed: {e}")
        halfday = []

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    daily_analysis = analyze_timestamps(daily, "Daily") if daily else None
    h12_analysis = analyze_timestamps(halfday, "12h") if halfday else None

    print("\nTIMESTAMP ALIGNMENT")
    if daily_analysis and daily_analysis["hours"]:
        hours_str = ", ".join(f"{h:02d}:00 UTC" for h in daily_analysis["hours"])
        print(f"  Daily candles snap to: {hours_str}")
        print(f"  Minutes observed: {daily_analysis['minutes']}")
        print(f"  Seconds observed: {daily_analysis['seconds']}")
        print(f"  Days of week: {daily_analysis['days_of_week']}")
    else:
        print("  Daily: NO DATA (contract may be resolved or too new)")

    if h12_analysis and h12_analysis["hours"]:
        hours_str = ", ".join(f"{h:02d}:00 UTC" for h in h12_analysis["hours"])
        print(f"  12h candles snap to: {hours_str}")
        print(f"  Minutes observed: {h12_analysis['minutes']}")
        print(f"  Seconds observed: {h12_analysis['seconds']}")
    else:
        print("  12h: NO DATA")

    print("\nRAW DATA RECAP")
    if daily:
        print("  Daily candles: see table above")
    if halfday:
        print("  12h candles: see table above")

    # Interpretation
    print("\nINTERPRETATION")
    if daily_analysis and daily_analysis["hours"]:
        if daily_analysis["hours"] == [0]:
            print("  -> Daily candles align to midnight UTC (00:00).")
        elif daily_analysis["hours"] == [12]:
            print("  -> Daily candles align to noon UTC (12:00).")
        else:
            print(f"  -> Daily candles align to unusual hour(s): {daily_analysis['hours']}")

    if h12_analysis and h12_analysis["hours"]:
        if set(h12_analysis["hours"]) == {0, 12}:
            print("  -> 12h candles split at 00:00 and 12:00 UTC.")
        else:
            print(f"  -> 12h candles snap to: {h12_analysis['hours']}")


if __name__ == "__main__":
    main()
