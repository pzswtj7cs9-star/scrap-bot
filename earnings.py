"""تجنب الأسهم القريبة من إعلان الأرباح."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import yfinance as yf

log = logging.getLogger("halal-bot.earnings")
NY = ZoneInfo("America/New_York")


def next_earnings_date(symbol: str) -> Optional[datetime]:
    try:
        t = yf.Ticker(symbol)
        # calendar قد يكون DataFrame أو dict حسب إصدار yfinance
        cal = getattr(t, "calendar", None)
        if cal is None:
            return None
        if hasattr(cal, "empty") and cal.empty:
            return None
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            if raw is None:
                return None
            if isinstance(raw, datetime):
                return raw if raw.tzinfo else raw.replace(tzinfo=NY)
            return datetime.fromisoformat(str(raw)[:10]).replace(tzinfo=NY)
        # DataFrame
        if "Earnings Date" in getattr(cal, "index", []):
            val = cal.loc["Earnings Date"]
            if hasattr(val, "iloc"):
                val = val.iloc[0]
            if isinstance(val, datetime):
                return val if val.tzinfo else val.replace(tzinfo=NY)
            return datetime.fromisoformat(str(val)[:10]).replace(tzinfo=NY)
    except Exception as exc:
        log.debug("earnings %s: %s", symbol, exc)
    return None


def is_near_earnings(symbol: str, within_days: int = 2) -> tuple[bool, Optional[str]]:
    """True إذا الإعلان خلال within_days أيام (قبل أو يوم الإعلان)."""
    dt = next_earnings_date(symbol)
    if not dt:
        return False, None
    now = datetime.now(NY)
    # نافذة: من قبل يومين إلى يوم بعد الإعلان
    start = now - timedelta(days=0)
    end = now + timedelta(days=within_days)
    # إذا الإعلان في الماضي القريب جداً (نفس اليوم)
    if abs((dt.date() - now.date()).days) <= within_days and dt.date() >= (now.date() - timedelta(days=1)):
        return True, dt.strftime("%Y-%m-%d")
    if start.date() <= dt.date() <= end.date():
        return True, dt.strftime("%Y-%m-%d")
    return False, dt.strftime("%Y-%m-%d")
