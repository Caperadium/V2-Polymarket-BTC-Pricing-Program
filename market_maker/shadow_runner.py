"""Stage-A SHADOW runner (plan Section 7 Stage A) — live data, no orders sent.

Drives the existing PaperTradingLoop against LIVE Polymarket books via REST
polling. Strictly read-only against the venue: books are fetched over REST,
quotes are computed and journaled, and the only "orders" go to the in-process
paper fill simulator. REST polling carries no trade prints, so the simulator
can never fill — the run is fill-free (shadow) by construction.

The real pricing engine is wrapped in a re-price cache: `calculate_
probabilities` (FIGARCH fit, minutes) runs at most every --reprice-s seconds;
between re-prices the cached ladder is returned instantly so the book/quote
loop can tick at --tick-s.

Usage (from repo root; BTC data must be fresh — run core/data/data_fetcher.py
first):

    python -m market_maker.shadow_runner --event-slug bitcoin-above-on-july-10-2026 \
        --minutes 40 --tick-s 30 --reprice-s 300

Outputs under --out (default temp/shadow_run/<UTC ts>/):
    quotes.csv   one row per (tick, market): market touch vs our quote, mode,
                 terms, credibility, snapshot age, no-arb status
    ticks.csv    one row per tick: wall latency, reprice latency when it ran
    summary.md   end-of-run Stage-A report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.settlement_handler import settlement_instant_utc
from market_maker.state_store import MMStateStore

logger = logging.getLogger("mm.shadow")

UA = {"User-Agent": "btc-prediction-market/2.0"}
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

_RETRY_ATTEMPTS = 5
_RETRY_BACKOFF0_S = 2.0
_RETRY_BACKOFF_CAP_S = 30.0


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_retry(url: str) -> Any:
    """`_get(url)` with exponential backoff (2s -> 30s cap, 5 attempts) on
    transient failures (M4). A 404 (HTTPError code 404) means "no such
    event/slug" -- it is NOT retried and propagates immediately so callers
    (notably `resolve_next_event`'s per-candidate probing) can tell "not
    found yet" apart from "venue/network unreachable". After the final
    attempt of a non-404 failure this raises SystemExit -- `_get` itself
    stays the sole patchable network seam for tests."""
    backoff = _RETRY_BACKOFF0_S
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return _get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 - deliberately broad, retried below
            last_exc = exc
        if attempt < _RETRY_ATTEMPTS:
            logger.warning(
                "fetch attempt %d/%d failed for %s: %s; retrying in %.0fs",
                attempt, _RETRY_ATTEMPTS, url, last_exc, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, _RETRY_BACKOFF_CAP_S)
    raise SystemExit(f"fetch failed after {_RETRY_ATTEMPTS} attempts: {url}: {last_exc}")


def resolve_event(event_slug: str) -> Tuple[str, List[Tuple[str, float, str]]]:
    """Return (expiry_key, [(market_slug, strike, clob_token_id)]) for a
    bitcoin-above event. Strike parsed from the question text."""
    try:
        evs = _get_retry(f"{GAMMA_API}/events?slug={event_slug}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(f"event not found: {event_slug}")
        raise
    if not evs:
        raise SystemExit(f"event not found: {event_slug}")
    e = evs[0]
    end = e.get("endDate") or ""
    expiry_key = end[:10]
    out: List[Tuple[str, float, str]] = []
    for m in e.get("markets") or []:
        q = m.get("question") or ""
        toks = m.get("clobTokenIds")
        tok = json.loads(toks)[0] if isinstance(toks, str) else (toks or [None])[0]
        if tok is None:
            continue
        digits = "".join(c for c in q.split("$")[-1].split(" on ")[0] if c.isdigit() or c == ".")
        if not digits:
            continue
        out.append((m.get("slug"), float(digits), tok))
    if len(out) < 2:
        raise SystemExit(f"could not build a ladder from event {event_slug}")
    out.sort(key=lambda t: t[1])
    return expiry_key, out


def _candidate_event_slugs(day: datetime) -> List[str]:
    """Both padded and unpadded day forms for a `bitcoin-above-on-<month>-
    <day>-<year>` slug -- the venue's zero-padding convention is unverified
    (plan 2.2 / reviewer suggestion 7), so both are probed. De-duplicated for
    days >= 10 where both forms are identical."""
    month = _MONTH_NAMES[day.month - 1]
    unpadded = f"bitcoin-above-on-{month}-{day.day}-{day.year}"
    padded = f"bitcoin-above-on-{month}-{day.day:02d}-{day.year}"
    return list(dict.fromkeys([unpadded, padded]))


def resolve_next_event(
    now: datetime, lead_days: int, config: Optional[MMConfig] = None
) -> Tuple[str, List[Tuple[str, float, str]]]:
    """Auto-select the next bitcoin-above event with a real quoting window
    (plan 2.2). Probes Gamma for candidate slugs on `now+1 .. now+lead_days+4`
    (both padded and unpadded day forms) and picks the first whose
    settlement instant clears `near_resolution_pull_hours + 12h` past `now`
    -- i.e. there is at least half a day of quoting left after the
    near-resolution pull window closes. Reuses `resolve_event` (which does
    its own retrying fetch + full ladder parse) once a candidate is chosen."""
    cfg = config if config is not None else MMConfig()
    near_h = cfg.near_resolution_pull_hours
    min_lead = timedelta(hours=near_h + 12.0)
    horizon_days = int(lead_days) + 4

    for offset in range(1, horizon_days + 1):
        day = now + timedelta(days=offset)
        for slug in _candidate_event_slugs(day):
            try:
                evs = _get_retry(f"{GAMMA_API}/events?slug={slug}")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue  # no event published yet for this date/form
                raise
            if not evs:
                continue
            end = (evs[0].get("endDate") or "")
            expiry_key = end[:10]
            if not expiry_key:
                continue
            try:
                settle_at = settlement_instant_utc(expiry_key)
            except Exception:
                continue
            if settle_at > now + min_lead:
                logger.info("auto-selected event %s (expiry %s, settles %s)",
                            slug, expiry_key, settle_at.isoformat())
                return resolve_event(slug)
    raise SystemExit(
        f"resolve_next_event: no suitable bitcoin-above event found in the next {horizon_days} days"
    )


def resolve_events_multi(
    now: datetime,
    lead_days: int,
    max_n: int,
    exclude_expiries: Optional[set] = None,
    config: Optional[MMConfig] = None,
) -> List[Tuple[str, str, List[Tuple[str, float, str]]]]:
    """Multi-expiry event resolution: return up to `max_n` quotable
    bitcoin-above events as `[(event_slug, expiry_key, ladder), ...]`,
    probing the same `now+1 .. now+lead_days+4` window and the same
    `near_resolution_pull_hours + 12h` min-lead rule as `resolve_next_event`.

    Differences from `resolve_next_event` (multi-expiry orchestrator
    contract):
      - returns a possibly-EMPTY list instead of raising SystemExit when
        nothing is found -- a running process with live ladders must not die
        because acquisition came up empty;
      - `exclude_expiries` skips expiries already active or already
        completed this process (teardown exclusion set);
      - intra-call dedup by expiry_key: for day<10 both the padded and
        unpadded slug forms can resolve to the SAME event -- without dedup
        that would build two slots for one expiry (duplicate markets, double
        bankroll share);
      - each candidate's full `resolve_event` is wrapped so its SystemExit
        (404/empty event, thin ladder, or venue unreachable after retries)
        skips THAT candidate instead of killing the process; a venue outage
        during the cheap probe fetch aborts the remaining probe and returns
        whatever was already resolved.
    """
    cfg = config if config is not None else MMConfig()
    near_h = cfg.near_resolution_pull_hours
    min_lead = timedelta(hours=near_h + 12.0)
    horizon_days = int(lead_days) + 4
    seen: set = set(exclude_expiries or ())

    out: List[Tuple[str, str, List[Tuple[str, float, str]]]] = []
    for offset in range(1, horizon_days + 1):
        if len(out) >= max_n:
            break
        day = now + timedelta(days=offset)
        for slug in _candidate_event_slugs(day):
            if len(out) >= max_n:
                break
            try:
                evs = _get_retry(f"{GAMMA_API}/events?slug={slug}")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue  # no event published yet for this date/form
                logger.warning("probe failed for %s: %s", slug, exc)
                continue
            except SystemExit as exc:
                # Venue/network unreachable after retries: stop probing, keep
                # whatever we already have (never kill a live process here).
                logger.warning("venue unreachable during multi-event probe (%s); "
                               "returning %d event(s) resolved so far", exc, len(out))
                return out
            if not evs:
                continue
            end = (evs[0].get("endDate") or "")
            probe_ek = end[:10]
            if not probe_ek or probe_ek in seen:
                continue
            try:
                settle_at = settlement_instant_utc(probe_ek)
            except Exception:
                continue
            if settle_at <= now + min_lead:
                continue
            try:
                expiry_key, ladder = resolve_event(slug)
            except SystemExit as exc:
                logger.warning("resolve_events_multi: skipping %s: %s", slug, exc)
                continue
            if expiry_key in seen:
                continue
            seen.add(expiry_key)
            seen.add(probe_ek)
            out.append((slug, expiry_key, ladder))
            logger.info("multi-resolved event %s (expiry %s, settles %s)",
                        slug, expiry_key, settle_at.isoformat())
    return out


def load_jump_params_for_engine() -> Optional[Dict[str, Any]]:
    """Load bipower-calibrated Kou jump params for calculate_probabilities.

    Module-level so both `CachedEngine` (Stage-A/B single-expiry) and
    `multi_runner.SharedPricingEngine` (multi-expiry) share one loader.
    Mirrors run_full_pipeline's key mapping: load_calibrated_jumps returns
    'lam'/'p_crash' keys, simulate_paths expects 'lambda'/'crash_prob' --
    passing the raw dict through would silently drop the calibrated
    lambda/crash back to module defaults. Returns None (engine defaults) if
    calibration is unavailable or unconverged -- quoting must not die on a
    calibration failure.
    """
    try:
        from core.pricing.btc_pricing_engine import load_calibrated_jumps

        cal = load_calibrated_jumps()
        if not cal.get("fit_converged"):
            logger.warning(
                "jump calibration not converged; using engine default jumps"
            )
            return None
        return {
            "lambda": cal["lam"], "crash_prob": cal["p_crash"],
            "eta_up": cal["eta_up"], "eta_down": cal["eta_down"],
            "mu_v": cal["mu_v"], "rho_J": cal["rho_J"],
            "rho_j_slope": cal.get("rho_j_slope", 0.0),
        }
    except Exception:
        logger.warning(
            "jump calibration load failed; using engine default jumps",
            exc_info=True,
        )
        return None


class CachedEngine:
    """Re-price at most every reprice_s; return the cached raw ladder between.

    Wraps the REAL calculate_probabilities with the production live feature
    set (SVCJ + skewed-t + FIGARCH, naive prior; regime/XGB off) and a
    per-run garch_cache. Records per-call latency.

    `garch_refit_s` (H2, default 6h): the GARCH/FIGARCH cache is cleared once
    it has lived at least this long, so the MLE fit from tick 1 does not
    silently price the whole run -- S0 already reloads every reprice
    (s0_override=None), only the fit itself was frozen.

    Jump params (2026-07-10 fix): bipower-calibrated via load_calibrated_jumps,
    refreshed on the same cadence as the GARCH refit. Previously no jump_params
    were passed, so simulate_paths fell back to the hardcoded module defaults,
    which sit ~+1.2-1.5c above the calibrated ladder near ATM at 1-7 DTE (see
    temp/VR.md section 8). Falls back to engine defaults (None) if calibration
    is unavailable or unconverged -- quoting must not die on a calibration
    failure.
    """

    def __init__(self, reprice_s: float, seed: int = 42, garch_refit_s: float = 21_600.0) -> None:
        self.reprice_s = reprice_s
        self.seed = seed
        self.garch_refit_s = garch_refit_s
        self._cache: Optional[Dict[Any, Any]] = None
        self._cached_at: float = 0.0
        self._garch_cache: Dict[Any, Any] = {}
        self._garch_fitted_at: Optional[float] = None
        self._jump_params: Optional[Dict[str, Any]] = None
        self.latencies: List[float] = []

    def _load_jump_params(self) -> Optional[Dict[str, Any]]:
        """Delegates to the module-level `load_jump_params_for_engine` (kept
        as a method for existing callers/tests that patch it here)."""
        return load_jump_params_for_engine()

    def __call__(self, strikes, hours_to_expiry, **kwargs):
        now = time.time()
        if self._cache is not None and (now - self._cached_at) < self.reprice_s:
            return dict(self._cache)
        from core.pricing.btc_pricing_engine import calculate_probabilities

        if (self._garch_fitted_at is not None
                and (now - self._garch_fitted_at) >= self.garch_refit_s):
            self._garch_cache.clear()
            self._garch_fitted_at = None
            logger.info("GARCH cache age >= %.0fs; cleared for refit", self.garch_refit_s)

        cache_was_empty = not self._garch_cache
        if cache_was_empty:
            # Same cadence as the GARCH refit; load_calibrated_jumps has its
            # own 30d CSV cache so this is cheap between recalibrations.
            self._jump_params = self._load_jump_params()
        t0 = time.time()
        res = calculate_probabilities(
            list(strikes),
            hours_to_expiry,
            n_sims=15000,
            seed=self.seed,
            use_svcj=True,
            use_skewed_t=True,
            use_figarch=True,
            jump_params=self._jump_params,
            garch_cache=self._garch_cache,
        )
        self.latencies.append(time.time() - t0)
        if cache_was_empty and self._garch_cache:
            self._garch_fitted_at = time.time()
        self._cache = dict(res)
        self._cached_at = time.time()
        logger.info("re-priced ladder in %.1fs", self.latencies[-1])
        return dict(res)


def fetch_book_message(token_id: str, seq: int, now: datetime) -> Optional[Dict[str, Any]]:
    """One BookMirror snapshot message from a REST /book call; None on failure."""
    try:
        book = _get(f"{CLOB_API}/book?token_id={token_id}")
        bids = [(float(b["price"]), float(b["size"])) for b in (book.get("bids") or [])]
        asks = [(float(a["price"]), float(a["size"])) for a in (book.get("asks") or [])]
        bids.sort(key=lambda t: -t[0])
        asks.sort(key=lambda t: t[0])
        return {"type": "snapshot", "bids": bids, "asks": asks, "ts": now, "seq": seq}
    except Exception:
        logger.warning("book fetch failed for %s", token_id[:16], exc_info=True)
        return None


def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-A shadow runner (read-only)")
    ap.add_argument("--event-slug", required=True)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--tick-s", type=float, default=30.0)
    ap.add_argument("--reprice-s", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    start = datetime.now(timezone.utc)
    out_dir = Path(args.out) if args.out else Path("temp/shadow_run") / start.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    expiry_key, ladder = resolve_event(args.event_slug)
    markets = [(slug, strike) for slug, strike, _tok in ladder]
    tokens = {slug: tok for slug, _strike, tok in ladder}
    logger.info("event %s expiry %s: %d strikes %s", args.event_slug, expiry_key,
                len(markets), [k for _, k in markets])

    engine = CachedEngine(reprice_s=args.reprice_s)
    store = MMStateStore(str(out_dir / "shadow_state.db"))
    clock = SimClock(start - timedelta(seconds=args.tick_s))

    # Zero-arg vol-gate closure over the fresh intraday data, evaluated at
    # real wall time each call (the RiskController expects a no-arg callable).
    import pandas as pd
    from core.strategy.vol_gate import compute_vol_gate

    btc_df = pd.read_csv("DATA/btc_intraday_1m.csv").tail(100_000)

    def live_vol_gate():
        return compute_vol_gate(btc_df, datetime.now(timezone.utc))
    loop = PaperTradingLoop(
        store=store,
        expiry_key=expiry_key,
        markets=markets,
        engine_fn=engine,
        config=MMConfig(),
        clock=clock,
        vol_gate_fn=live_vol_gate,
        tick_dt_s=args.tick_s,
    )

    quotes_csv = (out_dir / "quotes.csv").open("w", newline="", encoding="ascii")
    ticks_csv = (out_dir / "ticks.csv").open("w", newline="", encoding="ascii")
    qw = csv.writer(quotes_csv)
    tw = csv.writer(ticks_csv)
    qw.writerow([
        "ts", "market", "strike", "mkt_bid", "mkt_ask", "mkt_spread",
        "our_bid", "our_ask", "our_spread", "bid_size", "ask_size",
        "mode", "credibility", "p_hat", "consensus_p", "noarb_ok",
    ])
    tw.writerow(["ts", "tick", "wall_s", "reprice_s", "snapshot_failed", "books_failed"])

    end_time = time.time() + args.minutes * 60.0
    seq = 0
    tick_n = 0
    noarb_violations = 0
    pulled_ticks = 0

    while time.time() < end_time:
        loop_start = time.time()
        now = datetime.now(timezone.utc)
        seq += 1
        tick_n += 1

        messages: Dict[str, List[Dict[str, Any]]] = {}
        books_failed = 0
        for slug, _strike in markets:
            msg = fetch_book_message(tokens[slug], seq, now)
            if msg is None:
                books_failed += 1
                messages[slug] = []
            else:
                messages[slug] = [msg]
            time.sleep(0.15)  # venue rate-limit courtesy

        clock.set(now - timedelta(seconds=args.tick_s))
        n_lat_before = len(engine.latencies)
        try:
            loop.tick(messages, feed_healthy=(books_failed < len(markets)))
        except Exception:
            logger.error("tick %d failed", tick_n, exc_info=True)
            tw.writerow([now.isoformat(), tick_n, time.time() - loop_start, "", "TICK_ERROR", books_failed])
            time.sleep(max(0.0, args.tick_s - (time.time() - loop_start)))
            continue

        repriced = engine.latencies[n_lat_before:] if len(engine.latencies) > n_lat_before else []
        snap = loop.last_snapshot
        fv = loop.last_fair_value
        directives = loop.last_directives or {}
        quote_sets = loop.last_quote_sets or {}
        # None = the composed ladder failed the no-arb check and was rejected
        # (repair mode re-emits a repaired list, so None only on hard reject).
        noarb_ok = loop.last_checked_quote_sets is not None
        if not noarb_ok:
            noarb_violations += 1

        for slug, strike in markets:
            msgs = messages.get(slug) or []
            book = msgs[0] if msgs else None
            qs = quote_sets.get(slug)
            d = directives.get(slug)
            mode = d.mode.name if d is not None else ""
            if mode == "PULLED":
                pulled_ticks += 1
            mkt_bid = book["bids"][0][0] if book and book["bids"] else float("nan")
            mkt_ask = book["asks"][0][0] if book and book["asks"] else float("nan")
            cred = fv.credibility if fv is not None else float("nan")
            p_hat = snap.p_hat.get(strike, float("nan")) if snap is not None else float("nan")
            cons = fv.consensus_p.get(strike, float("nan")) if fv is not None else float("nan")
            qw.writerow([
                now.isoformat(), slug, strike,
                f"{mkt_bid:.4f}", f"{mkt_ask:.4f}",
                f"{(mkt_ask - mkt_bid):.4f}" if book else "",
                f"{qs.bid_price:.4f}" if qs else "", f"{qs.ask_price:.4f}" if qs else "",
                f"{(qs.ask_price - qs.bid_price):.4f}" if qs else "",
                f"{qs.bid_size:.2f}" if qs else "", f"{qs.ask_size:.2f}" if qs else "",
                mode, f"{cred:.4f}", f"{p_hat:.4f}", f"{cons:.4f}", int(noarb_ok),
            ])

        tw.writerow([
            now.isoformat(), tick_n, f"{time.time() - loop_start:.2f}",
            f"{repriced[0]:.1f}" if repriced else "",
            int(bool(getattr(loop, "snapshot_failed", False))), books_failed,
        ])
        quotes_csv.flush()
        ticks_csv.flush()

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, args.tick_s - elapsed))

    quotes_csv.close()
    ticks_csv.close()

    summary = out_dir / "summary.md"
    summary.write_text(
        "# Stage-A shadow run summary\n\n"
        f"- event: {args.event_slug} (expiry {expiry_key}, {len(markets)} strikes)\n"
        f"- start: {start.isoformat()}  duration: {args.minutes} min, "
        f"tick {args.tick_s}s, reprice {args.reprice_s}s\n"
        f"- ticks: {tick_n}\n"
        f"- self-inflicted no-arb violations (post-repair): {noarb_violations}\n"
        f"- PULLED (market,tick) rows: {pulled_ticks}\n"
        f"- re-price latencies (s): {['%.1f' % v for v in engine.latencies]}\n"
        f"- data: quotes.csv / ticks.csv in this directory\n",
        encoding="ascii",
    )
    logger.info("shadow run complete: %s", out_dir)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(run())
