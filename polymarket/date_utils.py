"""
polymarket/date_utils.py

Date utilities for the Run Engine feature.
Provides ET timezone handling and expiry date calculations.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional

import pytz


def get_now_et() -> datetime:
    """Get current datetime in US/Eastern timezone."""
    et = pytz.timezone('US/Eastern')
    return datetime.now(et)


def get_today_et() -> date:
    """Get current date in US/Eastern timezone."""
    return get_now_et().date()


def is_before_noon_et() -> bool:
    """Check if current ET time is before noon (12:00 PM)."""
    now_et = get_now_et()
    return now_et.hour < 12


def compute_expiry_dates(today: Optional[date] = None, force_before_noon: Optional[bool] = None) -> List[date]:
    """
    Compute expiry dates based on ET time.
    
    Logic:
    - Before noon ET: today to today+6 (contracts for today still tradeable)
    - After noon ET: today+1 to today+7 (today's contracts have expired)
    
    Args:
        today: Override for current date (for testing). Defaults to ET today.
        force_before_noon: Override for noon check (for testing). 
                          None = use actual ET time.
        
    Returns:
        List of 7 date objects
    """
    if today is None:
        today = get_today_et()
    
    # Determine if before noon
    if force_before_noon is not None:
        before_noon = force_before_noon
    else:
        before_noon = is_before_noon_et()
    
    if before_noon:
        # Before noon: today to today+6
        return [today + timedelta(days=i) for i in range(0, 7)]
    else:
        # After noon: today+1 to today+7
        return [today + timedelta(days=i) for i in range(1, 8)]


def last_day_of_month(d: date) -> date:
    """
    Get last day of month (safe implementation).
    
    Uses first-of-next-month minus 1 day approach to handle
    varying month lengths including leap years.
    """
    if d.month == 12:
        first_next = date(d.year + 1, 1, 1)
    else:
        first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def group_dates_by_month(dates: List[date]) -> List[Dict]:
    """
    Group dates by month for pipeline runs.
    
    Args:
        dates: List of date objects to group
        
    Returns:
        List of dicts with keys: month, year, start_date, end_date, slug_pattern
        
    Example:
        Given dates spanning Dec 30 - Jan 5:
        [
            {"month": "december", "year": 2024, "start_date": date(2024,12,30), "end_date": date(2024,12,31), "slug_pattern": "bitcoin-above-on-december-{}"},
            {"month": "january", "year": 2025, "start_date": date(2025,1,1), "end_date": date(2025,1,5), "slug_pattern": "bitcoin-above-on-january-{}"},
        ]
    """
    if not dates:
        return []
    
    # Sort dates
    sorted_dates = sorted(dates)
    
    groups = []
    current_group_dates = [sorted_dates[0]]
    
    for d in sorted_dates[1:]:
        # Check if same month/year as current group
        if d.month == current_group_dates[0].month and d.year == current_group_dates[0].year:
            current_group_dates.append(d)
        else:
            # Finalize current group and start new one
            groups.append(_create_group(current_group_dates))
            current_group_dates = [d]
    
    # Add final group
    if current_group_dates:
        groups.append(_create_group(current_group_dates))
    
    return groups


def _create_group(dates: List[date]) -> Dict:
    """Create a group dict from a list of dates in the same month."""
    month_name = dates[0].strftime("%B").lower()  # "december", "january"
    return {
        "month": month_name,
        "year": dates[0].year,
        "start_date": min(dates),
        "end_date": max(dates),
        "slug_pattern": f"bitcoin-above-on-{month_name}-{{}}",
    }


def format_date_range_summary(groups: List[Dict]) -> str:
    """
    Format date groups for display.
    
    Returns string like "Dec 30-31, Jan 1-5"
    """
    parts = []
    for g in groups:
        month_abbr = g["start_date"].strftime("%b")
        if g["start_date"] == g["end_date"]:
            parts.append(f"{month_abbr} {g['start_date'].day}")
        else:
            parts.append(f"{month_abbr} {g['start_date'].day}-{g['end_date'].day}")
    return ", ".join(parts)
