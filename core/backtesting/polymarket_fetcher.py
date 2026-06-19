#!/usr/bin/env python3
"""
polymarket_fetcher.py

Fetches historical contract prices from Polymarket APIs:

  - Gamma API  (``/events`` for slug search, ``/markets`` for closed-market browse)
  - CLOB API   (``/prices-history`` for daily price candles)

Consumed by :class:`BacktestingOrchestrator` and usable standalone.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

from core.backtesting.contract_store import ContractPriceStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

CALL_DELAY_S = 0.05      # inter-call delay for CLOB requests (was 0.2)
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0     # exponential backoff base for 429 responses

# ---------------------------------------------------------------------------
# Slug parsing
# ---------------------------------------------------------------------------

# k-suffix (Gamma event slug):   bitcoin-above-94k-on-november-15  → 94000
RE_STRIKE_K = re.compile(r"(?:bitcoin-above|be-above)-(\d+)k", re.IGNORECASE)
# numeric (old CSV slug):         bitcoin-above-94000-on-november-15 → 94000
RE_STRIKE_NUMERIC = re.compile(r"bitcoin-above-(\d+)-on", re.IGNORECASE)
# full-format (actual Polymarket): will-the-price-of-bitcoin-be-above-78000-on-june-17 → 78000
RE_STRIKE_FULL = re.compile(r"be-above-(\d+)-on", re.IGNORECASE)
# month-day: …-on-<month>-<day>  (handles abbreviated and full month names)
RE_EXPIRY = re.compile(
    r"on-([a-z]+)-(\d{1,2})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)
# event discovery slug pattern (used to query Gamma /events)
RE_EVENT_SLUG = re.compile(r"bitcoin-above-on-([a-z]+)-(\d{1,2})", re.IGNORECASE)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    # Abbreviated
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ET noon → UTC offset (UTC-5 EST / UTC-4 EDT, treat as UTC-5 for simplicity
# since exact ET → UTC mapping varies by date; resolution stored as UTC datetime)
ET_OFFSET = timedelta(hours=5)  # EST; EDT would be 4
EXPIRY_HOUR_ET = 12  # noon ET


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_strike_from_slug(slug: str) -> Optional[float]:
    """Extract BTC strike price from a Polymarket slug.

    Handles three formats:

    * k-suffix:     ``bitcoin-above-94k-on-november-15``             → 94000.0
    * numeric:      ``bitcoin-above-94000-on-november-15``           → 94000.0
    * full-format:  ``will-the-price-of-bitcoin-be-above-78000-on-…`` → 78000.0

    Returns *None* if no strike can be parsed.
    """
    m = RE_STRIKE_K.search(slug)
    if m:
        return float(m.group(1)) * 1000.0

    m = RE_STRIKE_NUMERIC.search(slug)
    if m:
        return float(m.group(1))

    m = RE_STRIKE_FULL.search(slug)
    if m:
        return float(m.group(1))

    return None


def parse_expiry_from_slug(slug: str, year_hint: Optional[int] = None) -> Optional[datetime]:
    """Extract expiry datetime from a Polymarket slug.

    Parses ``…on-<month>-<day>…`` and constructs a UTC datetime
    representing **12:00 PM ET** on that date.

    *year_hint* is used when the slug does not contain a year; defaults to
    the current UTC year.
    """
    m = RE_EXPIRY.search(slug)
    if not m:
        return None

    month_name = m.group(1).lower()
    day = int(m.group(2))

    month = MONTH_MAP.get(month_name)
    if month is None:
        return None

    if year_hint is None:
        year_hint = datetime.now(timezone.utc).year

    # Construct noon ET then add ET→UTC offset
    try:
        et_noon = datetime(year_hint, month, day, EXPIRY_HOUR_ET, 0, 0)
        expiry_utc = et_noon + ET_OFFSET
        return expiry_utc.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_resolution_from_market(market: dict) -> float:
    """Extract resolution status from a Gamma market dict.

    Returns
    -------
    float
        1.0 = YES, 0.0 = NO, NaN = unresolved.
    """
    # Some markets have a top-level 'resolution' key
    resolution = market.get("resolution")
    if resolution is None:
        # Check the outcome / outcomes array
        outcomes = market.get("outcomes")
        if isinstance(outcomes, list):
            resolution = outcomes[0] if len(outcomes) > 0 else None

    if resolution is None:
        return float("nan")

    # Resolution can be a string "YES"/"NO" or numeric 1.0/0.0
    if isinstance(resolution, str):
        return 1.0 if resolution.upper() == "YES" else 0.0
    try:
        return float(resolution)
    except (TypeError, ValueError):
        return float("nan")


def parse_clob_token_ids(market: dict) -> List[str]:
    """Parse clobTokenIds from a market dict (may be a JSON string or list)."""
    raw = market.get("clobTokenIds", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    return []


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "btc-prediction-market/2.0"})
    return s


def _handle_rate_limit(resp: requests.Response, attempt: int) -> bool:
    """Return True if we should retry after a 429 / 5xx."""
    if resp.status_code == 429:
        wait = BACKOFF_BASE_S * (2 ** attempt)
        logger.warning("HTTP 429 – backing off %.1fs (attempt %d)", wait, attempt + 1)
        time.sleep(wait)
        return True

    if resp.status_code >= 500:
        wait = BACKOFF_BASE_S * (2 ** attempt)
        logger.warning("HTTP %d – backing off %.1fs (attempt %d)", resp.status_code, wait, attempt + 1)
        time.sleep(wait)
        return True

    return False


def _get_json(url: str, params: dict, session: requests.Session, timeout: int = 10) -> dict:
    """GET *url* with retry + backoff, returning parsed JSON dict.

    *timeout* is connection+read timeout in seconds (reduced from 30 — Gamma/CLOB
    respond in <1s normally; 10s handles slow-but-not-dead connections).
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if _handle_rate_limit(resp, attempt):
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                logger.debug("Request failed (attempt %d): %s – retrying in %.1fs", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]  # only reached if all retries hit 429/5xx


# ---------------------------------------------------------------------------
# Gamma – closed bitcoin-above markets  (date-slug event lookup)
# ---------------------------------------------------------------------------

def _generate_date_slugs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[str]:
    """Generate ``bitcoin-above-on-{month}-{day}`` slugs for a date range.

    Defaults to scanning from 2024-01-01 through yesterday.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc) - timedelta(days=1)
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    slugs: List[str] = []
    current = start_date
    while current <= end_date:
        month_name = current.strftime("%B").lower()
        day = current.day
        slugs.append(f"bitcoin-above-on-{month_name}-{day}")
        current += timedelta(days=1)

    return slugs


def fetch_closed_bitcoin_above_markets(
    session: Optional[requests.Session] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    progress_callback: Optional[callable] = None,
) -> List[dict]:
    """Discover ``bitcoin-above`` contracts by iterating date-slugs via
    Gamma ``/events?slug=bitcoin-above-on-{month}-{day}``.

    The Gamma ``/markets`` endpoint does NOT reliably filter by title/tag/slug
    for these contracts.  Instead we query the ``/events`` endpoint with the
    discovery slug format used by Polymarket's own provider layer
    (cf. ``provider_polymarket.py`` line 619).

    Returns a list of market dicts with keys:
        slug, clobTokenId, question, closeTime, volume, outcomes, resolution

    Notes
    -----
    Each event contains a ``markets`` array with strike-level entries.
    We deduplicate by clobTokenId (YES token) across all events.
    """
    if session is None:
        session = _build_session()

    date_slugs = _generate_date_slugs(start_date, end_date)
    total_slugs = len(date_slugs)
    seen_clob_ids: Set[str] = set()
    all_markets: List[dict] = []

    logger.info("Scanning %d date-slugs for bitcoin-above events...", total_slugs)

    for idx, ds in enumerate(date_slugs):
        try:
            data = _get_json(GAMMA_EVENTS_URL, {"slug": ds}, session)
        except Exception:
            logger.debug("Gamma /events failed for slug '%s'", ds)
            data = []

        # Progress callback every 100 slugs during discovery
        if progress_callback and idx % 100 == 0:
            progress_callback("discovering", idx + 1, total_slugs)

        # Response is a list of event dicts
        events: List[dict] = data if isinstance(data, list) else []

        for evt in events:
            evt_slug = evt.get("slug", "")
            # Parse year from the event slug if present, else from event metadata
            year_hint = _extract_year_from_slug(evt_slug) or _extract_year_from_event(evt)

            for mkt in evt.get("markets", []):
                mkt_slug = mkt.get("slug", "")
                if not mkt_slug or "bitcoin-above" not in mkt_slug.lower():
                    continue

                clob_ids = parse_clob_token_ids(mkt)
                if not clob_ids:
                    continue

                yes_token = clob_ids[0]
                if yes_token in seen_clob_ids:
                    continue  # dedup across events
                seen_clob_ids.add(yes_token)

                strike = parse_strike_from_slug(mkt_slug)
                if strike is None:
                    logger.debug("Could not parse strike from '%s', skipping", mkt_slug)
                    continue

                expiry = parse_expiry_from_slug(mkt_slug, year_hint=year_hint)

                all_markets.append({
                    "slug": mkt_slug,
                    "clobTokenId": yes_token,
                    "question": mkt.get("question", evt.get("title", "")),
                    "closeTime": mkt.get("closeTime") or mkt.get("endDate") or evt.get("closeTime"),
                    "volume": mkt.get("volume", 0),
                    "outcomes": mkt.get("outcomes"),
                    "resolution": mkt.get("resolution"),
                    "strike": strike,
                    "expiry": expiry,
                })

    logger.info("Gamma: fetched %d bitcoin-above markets from %d date-slugs",
                len(all_markets), total_slugs)
    return all_markets


def _extract_year_from_slug(slug: str) -> Optional[int]:
    """Try to extract a 4-digit year from a slug string."""
    m = re.search(r"(\d{4})", slug)
    if m:
        return int(m.group(1))
    return None


def _extract_year_from_event(evt: dict) -> Optional[int]:
    """Extract year from event metadata (closeTime, endDate, etc.)."""
    for key in ("closeTime", "endDate", "createdAt"):
        raw = evt.get(key)
        if raw:
            parsed = _parse_close_time(raw)
            if parsed:
                return parsed.year
    return None


# ---------------------------------------------------------------------------
# CLOB – daily price history
# ---------------------------------------------------------------------------

def fetch_price_history(
    clob_token_id: str,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """Fetch daily price candles for a single CLOB token.

    Strategy:

    1. Try ``interval=1d`` (native daily candles).
    2. If empty or error, fall back to ``interval=max&fidelity=720``
       (12‑hour candles) and filter to midnight UTC rows.
    3. If still empty, return all fidelity=720 points as best‑effort.

    Returns a list of ``{"date_utc": datetime, "price": float}`` dicts.
    """
    if session is None:
        session = _build_session()

    # --- attempt 1: interval=1d ---
    history = _fetch_price_history_raw(clob_token_id, interval="1d", session=session)
    parsed = _parse_history(history)

    if parsed:
        return parsed

    # --- attempt 2: fidelity=720 (12h), filter midnight ---
    logger.debug("clobTokenId %s: 1d returned empty, trying fidelity=720", clob_token_id)
    history = _fetch_price_history_raw(
        clob_token_id, interval="max", fidelity=720, session=session
    )
    midnight = [r for r in _parse_history(history) if r["date_utc"].hour == 0]

    if midnight:
        return midnight

    # --- attempt 3: best effort – return ALL fidelity=720 points ---
    if _parse_history(history):
        logger.debug("clobTokenId %s: no midnight points in fidelity=720; returning all", clob_token_id)
        return _parse_history(history)

    return []


def _fetch_price_history_raw(
    clob_token_id: str,
    interval: str = "1d",
    fidelity: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> List[dict]:
    """Raw CLOB /prices-history call returning the ``history`` list."""
    if session is None:
        session = _build_session()

    params: Dict[str, Any] = {"market": clob_token_id, "interval": interval}
    if fidelity is not None:
        params["fidelity"] = fidelity

    try:
        data = _get_json(CLOB_PRICES_URL, params, session, timeout=30)
    except Exception:
        logger.debug("CLOB prices-history failed for %s (interval=%s)", clob_token_id, interval)
        return []

    history = data.get("history")
    if not isinstance(history, list):
        return []
    return history


def _parse_history(history: List[dict]) -> List[Dict[str, Any]]:
    """Convert raw ``{t, p}`` entries to ``{date_utc, price}`` dicts."""
    results: List[Dict[str, Any]] = []
    for entry in history:
        try:
            t = entry.get("t")
            p = entry.get("p")
            if t is None or p is None:
                continue
            ts = int(t)
            price = float(p)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            results.append({"date_utc": dt, "price": price})
        except (TypeError, ValueError, OverflowError):
            continue
    return results


# ---------------------------------------------------------------------------
# Incremental fetch (main entry point for orchestrator)
# ---------------------------------------------------------------------------

# Staleness threshold: if a contract is unresolved and its last stored
# price date is MORE than this many days ago, refetch.
STALENESS_DAYS = 1


def fetch_incremental_prices(
    store: ContractPriceStore,
    session: Optional[requests.Session] = None,
    progress_callback: Optional[callable] = None,
) -> int:
    """Main orchestrator: fetch new + stale contract prices into *store*.

    1. Load known contracts from *store*.
    2. Discover all closed ``bitcoin-above`` markets via Gamma.
    3. For each **new** contract: fetch full price history.
    4. For each **known** unresolved contract: refetch if stale (>1 day gap).
    5. Append new records to *store*.

    Parameters
    ----------
    store : ContractPriceStore
        CSV-backed store for historical contract prices.
    session : requests.Session or None
        Reusable HTTP session.
    progress_callback : callable or None
        ``progress_callback(stage: str, n_done: int, n_total: int)``

    Returns
    -------
    int
        Total number of NEW records added to the store.
    """
    if session is None:
        session = _build_session()

    store.load()
    known_ids, token_max_dates = store.build_token_index()
    now = datetime.now(timezone.utc)

    # --- discover all relevant markets ---
    markets = fetch_closed_bitcoin_above_markets(session, progress_callback=progress_callback)
    total_markets = len(markets)
    all_new_records: List[dict] = []
    errors: List[str] = []
    skipped_empty: int = 0
    skipped_error: int = 0
    rate_limited: int = 0

    if total_markets == 0:
        logger.warning("No bitcoin-above markets found via Gamma /events")
        return 0, ["No bitcoin-above markets found — API may be down or no data available"]

    if progress_callback:
        progress_callback("fetching", 0, total_markets)

    for idx, market in enumerate(markets):
        clob_id = market["clobTokenId"]
        slug = market["slug"]
        is_new = clob_id not in known_ids

        # Determine staleness for known contracts
        should_fetch = False
        if is_new:
            should_fetch = True
        else:
            max_stored = token_max_dates.get(clob_id)
            close_time = _parse_close_time(market.get("closeTime"))
            if max_stored is None:
                should_fetch = True  # known ID but no stored rows — refetch
            elif close_time is not None:
                # Normal path: only refetch if contract is still open and data stale
                gap_days = (now - max_stored).days
                if close_time > now and max_stored < close_time and gap_days > STALENESS_DAYS:
                    should_fetch = True
            else:
                # closeTime missing from API metadata — use stored recency only
                gap_days = (now - max_stored).days
                if gap_days > STALENESS_DAYS * 2:
                    should_fetch = True  # data is clearly stale, refetch as safety measure

        if not should_fetch:
            if progress_callback:
                progress_callback("fetching", idx + 1, total_markets)
            continue

        # --- fetch price history ---
        time.sleep(CALL_DELAY_S)
        try:
            prices = fetch_price_history(clob_id, session)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') and e.response is not None else 0
            if status == 429:
                rate_limited += 1
                logger.warning("Rate limited on %s, pausing 2s...", slug)
                time.sleep(2)
                # Retry once after backoff
                try:
                    prices = fetch_price_history(clob_id, session)
                except Exception:
                    skipped_error += 1
                    errors.append(f"429 rate-limit on {slug}")
                    if progress_callback:
                        progress_callback("fetching", idx + 1, total_markets)
                    continue
            else:
                skipped_error += 1
                errors.append(f"HTTP {status} on {slug}")
                if progress_callback:
                    progress_callback("fetching", idx + 1, total_markets)
                continue
        except Exception:
            skipped_error += 1
            errors.append(f"Network error on {slug}")
            if progress_callback:
                progress_callback("fetching", idx + 1, total_markets)
            continue

        if not prices:
            skipped_empty += 1
            if progress_callback:
                progress_callback("fetching", idx + 1, total_markets)
            continue

        # --- parse metadata (pre-parsed from market discovery) ---
        strike = market.get("strike")
        expiry = market.get("expiry")
        resolution = parse_resolution_from_market(market)

        if strike is None:
            if progress_callback:
                progress_callback("fetching", idx + 1, total_markets)
            continue

        # --- build records ---
        records = _build_records(slug, clob_id, prices, resolution, strike, expiry, is_new)
        all_new_records.extend(records)

        if progress_callback:
            progress_callback("fetching", idx + 1, total_markets)

    # --- commit to store ---
    if not all_new_records:
        logger.info("fetch_incremental_prices: no new records to add")
        return 0, errors if errors else []

    new_df = pd.DataFrame(all_new_records)
    n_added = store.append_incremental(new_df)
    logger.info(
        "fetch_incremental_prices: %d new records from %d markets"
        " (skipped: %d errors, %d empty, %d rate-limited)",
        n_added, total_markets, skipped_error, skipped_empty, rate_limited,
    )

    if rate_limited:
        errors.append(f"Warning: hit rate limit {rate_limited} time(s) — some price data may be missing")

    return n_added, errors if errors else []


def _build_records(
    slug: str,
    clob_token_id: str,
    prices: List[Dict[str, Any]],
    resolution: float,
    strike: float,
    expiry: Optional[datetime],
    is_new: bool,
) -> List[dict]:
    """Convert price-history entries to store-compatible record dicts."""
    records = []
    for entry in prices:
        date_utc = entry["date_utc"]
        price = entry["price"]

        # For known contracts, we typically only refetch new dates,
        # but dedup in the store via (clobTokenId, date) handles overlaps.
        records.append({
            "slug": slug,
            "clobTokenId": clob_token_id,
            "date": date_utc,
            "price": price,
            "resolution": resolution,
            "strike": strike,
            "expiry_date": expiry,
        })
    return records


def _parse_close_time(raw: Optional[str]) -> Optional[datetime]:
    """Parse closeTime from Gamma API into a UTC datetime, or None."""
    if not raw:
        return None
    try:
        # Try ISO-8601
        if "T" in str(raw):
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        # Unix timestamp (seconds)
        ts = int(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError):
        return None
