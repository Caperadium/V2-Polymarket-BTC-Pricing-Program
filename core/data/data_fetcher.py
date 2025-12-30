#!/usr/bin/env python3
"""
data_fetcher.py

Download BTC historical data for the pricing engine:
- 5 years of daily closes from CoinGecko
- ~3 months of 1-minute candles from Binance

Outputs are stored under DATA/ and overwrite previous files.
"""

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, parse, request

DATA_DIR = Path(__file__).resolve().parent / "DATA"
DAILY_PATH = DATA_DIR / "btc_daily.csv"
INTRADAY_PATH = DATA_DIR / "btc_intraday_1m.csv"

BINANCE_URL = "https://api.binance.com/api/v3/klines"

USER_AGENT = "btc_data_fetcher/1.1"


def _read_json(url, retries=5, backoff=1.0):
    """Fetch JSON with retries/backoff; raise RuntimeError on failure."""
    last_err = None
    for attempt in range(retries):
        try:
            req = request.Request(url, headers={"User-Agent": USER_AGENT})
            with request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict) and data.keys() <= {"code", "msg"}:
                raise RuntimeError(f"API error: {data}")
            return data
        except (error.HTTPError, error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            last_err = exc
            sleep = backoff * (2 ** attempt)
            time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_daily(days=1825, symbol="BTCUSDT", interval="1d", throttle=0.5):
    """Fetch daily closes using Binance klines only."""
    limit = 1000
    interval_ms = 24 * 60 * 60 * 1000
    end = int(time.time() * 1000)
    start = end - days * interval_ms
    rows = []
    fetches = 0
    while start < end:
        window_end = min(start + limit * interval_ms, end)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start,
            "endTime": window_end - 1,
        }
        url = f"{BINANCE_URL}?{parse.urlencode(params)}"
        klines = _read_json(url)
        if not isinstance(klines, list) or not klines:
            break
        progressed = False
        for k in klines:
            open_time = int(k[0])
            close_price = float(k[4])
            if not math.isfinite(close_price) or close_price <= 0:
                continue
            dt = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
            rows.append((dt.strftime("%Y-%m-%d"), f"{close_price:.8f}"))
            progressed = True
        fetches += 1
        start = int(klines[-1][0]) + interval_ms
        if not progressed or start >= end:
            break
        time.sleep(throttle)
    if not rows:
        raise RuntimeError("Binance daily fetch returned no data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DAILY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} daily rows to {DAILY_PATH} (Binance, {fetches} calls)")


def fetch_intraday(days=90, symbol="BTCUSDT", interval="1m", throttle=0.5):
    """Download high-frequency BTC candles from Binance.
    
    Uses incremental fetching: if data already exists, only fetches new data
    since the last timestamp and appends it. Falls back to full fetch if
    no existing data or file is too old.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    limit = 1000
    interval_ms = 60_000  # 1 minute
    end = int(time.time() * 1000)
    start_cutoff = end - days * 24 * 60 * 60 * 1000
    
    # Check for existing data to enable incremental fetch
    existing_rows = []
    last_timestamp_ms = None
    
    if INTRADAY_PATH.exists():
        try:
            with INTRADAY_PATH.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        existing_rows.append(tuple(row))
                    
                    if existing_rows:
                        # Parse the last timestamp to know where to resume
                        last_ts_str = existing_rows[-1][0]
                        # Handle both ISO format and other formats
                        try:
                            last_dt = datetime.fromisoformat(last_ts_str.replace('Z', '+00:00'))
                            last_timestamp_ms = int(last_dt.timestamp() * 1000)
                        except ValueError:
                            # If parsing fails, do full fetch
                            existing_rows = []
                            last_timestamp_ms = None
        except Exception as e:
            print(f"Could not read existing intraday data: {e}, doing full fetch")
            existing_rows = []
            last_timestamp_ms = None
    
    # Decide fetch strategy
    if last_timestamp_ms and last_timestamp_ms > start_cutoff:
        # Incremental fetch: start from last timestamp + 1 minute
        fetch_start = last_timestamp_ms + interval_ms
        print(f"Incremental fetch: appending data from {datetime.fromtimestamp(fetch_start/1000, tz=timezone.utc).isoformat()}")
        is_incremental = True
    else:
        # Full fetch
        fetch_start = None
        existing_rows = []  # Discard old data if too old
        print(f"Full fetch: downloading {days} days of intraday data")
        is_incremental = False
    
    # Fetch new data
    new_rows = []
    fetches = 0
    
    if is_incremental:
        # Fetch forward from last timestamp to now
        cursor = fetch_start
        while cursor < end:
            window_end = min(cursor + limit * interval_ms, end)
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": cursor,
                "endTime": window_end - 1,
            }
            url = f"{BINANCE_URL}?{parse.urlencode(params)}"
            klines = _read_json(url)
            if not isinstance(klines, list) or not klines:
                break
            chunk = []
            for k in klines:
                open_time = int(k[0])
                open_p, high_p, low_p, close_p, vol = k[1:6]
                try:
                    open_f = float(open_p)
                    high_f = float(high_p)
                    low_f = float(low_p)
                    close_f = float(close_p)
                    vol_f = float(vol)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(x) for x in (open_f, high_f, low_f, close_f, vol_f)):
                    continue
                chunk.append(
                    (
                        datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat(),
                        f"{open_f:.8f}",
                        f"{high_f:.8f}",
                        f"{low_f:.8f}",
                        f"{close_f:.8f}",
                        f"{vol_f:.8f}",
                    )
                )
            new_rows.extend(chunk)
            fetches += 1
            if not chunk:
                break
            cursor = int(klines[-1][0]) + interval_ms
            time.sleep(throttle)
    else:
        # Full fetch (original logic, backward from end)
        cursor = end
        while cursor > start_cutoff:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "endTime": cursor - 1,
            }
            url = f"{BINANCE_URL}?{parse.urlencode(params)}"
            klines = _read_json(url)
            if not isinstance(klines, list) or not klines:
                break
            chunk = []
            for k in klines:
                open_time = int(k[0])
                if open_time < start_cutoff:
                    continue
                open_p, high_p, low_p, close_p, vol = k[1:6]
                try:
                    open_f = float(open_p)
                    high_f = float(high_p)
                    low_f = float(low_p)
                    close_f = float(close_p)
                    vol_f = float(vol)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(x) for x in (open_f, high_f, low_f, close_f, vol_f)):
                    continue
                chunk.append(
                    (
                        datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat(),
                        f"{open_f:.8f}",
                        f"{high_f:.8f}",
                        f"{low_f:.8f}",
                        f"{close_f:.8f}",
                        f"{vol_f:.8f}",
                    )
                )
            if not chunk:
                break
            new_rows = chunk + new_rows
            fetches += 1
            cursor = int(klines[0][0]) - interval_ms
            time.sleep(throttle)
    
    # Combine existing and new rows
    all_rows = list(existing_rows) + new_rows
    
    if not all_rows:
        raise RuntimeError("Binance returned no intraday data")
    
    # Write combined data
    with INTRADAY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        writer.writerows(all_rows)
    
    if is_incremental:
        print(f"Appended {len(new_rows)} new rows ({fetches} API calls). Total: {len(all_rows)} rows")
    else:
        print(f"Saved {len(all_rows)} intraday rows ({fetches} API calls) to {INTRADAY_PATH}")


def main():
    """CLI entry point to refresh local data cache."""
    try:
        skip_daily = False
        if DAILY_PATH.exists():
            last_modified = datetime.fromtimestamp(DAILY_PATH.stat().st_mtime, tz=timezone.utc)
            if last_modified.date() == datetime.now(timezone.utc).date():
                print(f"Daily data already up-to-date (last modified {last_modified.isoformat()}); skipping daily fetch.")
                skip_daily = True
        if not skip_daily:
            fetch_daily()
        fetch_intraday()
        print("Data refresh complete.")
    except Exception as exc:
        print("Data fetch failed:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

