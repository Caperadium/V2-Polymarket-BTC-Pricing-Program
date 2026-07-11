"""Tests for core/data/data_fetcher.py's atomic CSV write helper (plan
temp/plan_settlement_cache_fix.md, Task 2).

Covers `_write_csv_atomic`: correct content on a fresh path, correct
replacement of an existing file's content, and no leftover `.tmp` file
after a successful write.
"""
from __future__ import annotations

import csv

from core.data.data_fetcher import _write_csv_atomic


def test_write_csv_atomic_writes_header_and_rows(tmp_path):
    path = tmp_path / "fresh.csv"
    header = ["date", "close"]
    rows = [("2026-01-01", "100.0"), ("2026-01-02", "101.5")]

    _write_csv_atomic(path, header, rows)

    assert path.exists()
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        written = list(reader)

    assert written[0] == header
    assert written[1:] == [list(r) for r in rows]


def test_write_csv_atomic_replaces_existing_content(tmp_path):
    path = tmp_path / "existing.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close"])
        writer.writerow(["2020-01-01", "1.0"])

    new_header = ["date", "close"]
    new_rows = [("2026-01-01", "999.0")]
    _write_csv_atomic(path, new_header, new_rows)

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        written = list(reader)

    assert written == [new_header, list(new_rows[0])]


def test_write_csv_atomic_leaves_no_tmp_residue(tmp_path):
    path = tmp_path / "clean.csv"
    _write_csv_atomic(path, ["date", "close"], [("2026-01-01", "100.0")])

    tmp_sibling = path.with_name(path.name + ".tmp")
    assert not tmp_sibling.exists()
    assert list(tmp_path.iterdir()) == [path]
