"""أوقات جلسة السوق الأمريكي + فلتر نظام السوق (Market Regime)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
RIYADH = ZoneInfo("Asia/Riyadh")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


def now_ny() -> datetime:
    return datetime.now(NY)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))

def _last_weekday(year: int, month: int, weekday: int) -> date:
    first_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = first_next - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)

def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d

def us_market_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    # Gregorian Easter Sunday, then Good Friday.
    a = year % 19; b = year // 100; c = year % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30; i = c // 4; k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7; m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    holidays.add(date(year, month, day) - timedelta(days=2))
    return holidays

def us_market_early_closes(year: int) -> dict[date, time]:
    # Standard NYSE 1:00 PM ET early closes relevant to this bot.
    early = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    july3 = date(year, 7, 3)
    july4 = date(year, 7, 4)
    if july3.weekday() < 5 and july4.weekday() in (5, 6):
        early.add(july3)
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5:
        early.add(christmas_eve)
    full = us_market_holidays(year)
    return {d: time(13, 0) for d in early if d.weekday() < 5 and d not in full}

def is_weekday(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    return dt.weekday() < 5

def is_us_market_holiday(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    return dt.date() in us_market_holidays(dt.year)

def regular_close_for_date(dt: datetime | None = None) -> time:
    dt = dt or now_ny()
    return us_market_early_closes(dt.year).get(dt.date(), REGULAR_CLOSE)

def is_us_regular_session(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    if not is_weekday(dt) or is_us_market_holiday(dt):
        return False
    return REGULAR_OPEN <= dt.time() <= regular_close_for_date(dt)


def is_post_close_window(
    dt: datetime | None = None,
    start_h: int = 16,
    start_m: int = 10,
    end_h: int = 16,
    end_m: int = 50,
) -> bool:
    """نافذة بعد الإغلاق لإرسال الملخص اليومي."""
    dt = dt or now_ny()
    if not is_weekday(dt) or is_us_market_holiday(dt):
        return False
    if dt.time() > regular_close_for_date(dt):
        return False
    t = dt.time()
    return time(start_h, start_m) <= t <= time(end_h, end_m)


def is_friday_post_close(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    return dt.weekday() == 4 and is_post_close_window(dt)


def session_label(dt: datetime | None = None) -> str:
    dt = dt or now_ny()
    riyadh = dt.astimezone(RIYADH)
    if not is_weekday(dt):
        return (
            f"السوق مغلق (عطلة نهاية الأسبوع) — "
            f"نيويورك {dt.strftime('%H:%M')} | الرياض {riyadh.strftime('%H:%M')}"
        )
    if is_us_market_holiday(dt):
        return (
            f"السوق مغلق (عطلة رسمية أمريكية) — "
            f"نيويورك {dt.strftime('%H:%M')} | الرياض {riyadh.strftime('%H:%M')}"
        )
    t = dt.time()
    if t < time(4, 0):
        status = "خارج الجلسة"
    elif t < REGULAR_OPEN:
        status = "ما قبل الافتتاح"
    elif t <= regular_close_for_date(dt):
        status = "الجلسة النظامية مفتوحة الآن"
    elif t <= time(20, 0):
        status = "بعد الإغلاق / تداول ممتد"
    else:
        status = "السوق مغلق"
    return f"{status} — نيويورك {dt.strftime('%H:%M')} | الرياض {riyadh.strftime('%H:%M')}"


# ---------------------------------------------------------------------------
# Market Regime Filter
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def get_market_regime(symbol: str = "SPY") -> dict:
    """
    فلتر قوة السوق العام.

    يرجع:
      regime: "strong_bull" | "bull" | "neutral" | "weak" | "bear"
      min_score_adj: الحد الأدنى المقترح للدرجة
      allow_auto: هل يُسمح بالتنبيهات التلقائية
      detail: نص عربي مختصر
    """
    try:
        from market_data import fetch_history

        df = fetch_history(symbol, period="1y")
        if df is None or len(df) < 210:
            return {
                "regime": "neutral",
                "min_score_adj": 84,
                "allow_auto": True,
                "detail": "بيانات السوق غير كافية — وضع محايد",
                "spy_price": None,
                "sma50": None,
                "sma200": None,
            }

        close = df["Close"]
        price = float(close.iloc[-1])
        sma50 = float(_sma(close, 50).iloc[-1])
        sma200 = float(_sma(close, 200).iloc[-1])

        # ميل متوسط 50 خلال آخر 10 أيام
        sma50_series = _sma(close, 50)
        slope = float(sma50_series.iloc[-1] - sma50_series.iloc[-10]) / max(abs(sma50_series.iloc[-10]), 1e-9)

        above_50 = price > sma50
        above_200 = price > sma200
        stacked = sma50 > sma200
        rising_50 = slope > 0.001  # ميل إيجابي واضح

        if above_50 and above_200 and stacked and rising_50:
            regime = "strong_bull"
            min_adj = 84
            allow = True
            detail = f"{symbol} قوي جداً — فوق 50 و200 ومتوسط 50 صاعد"
        elif above_50 and above_200 and stacked:
            regime = "bull"
            min_adj = 84
            allow = True
            detail = f"{symbol} صاعد — فوق 50 و200"
        elif above_50 and above_200:
            regime = "neutral"
            min_adj = 88
            allow = True
            detail = f"{symbol} محايد إيجابي — فوق المتوسطات لكن الترتيب غير مثالي"
        elif above_200 and not above_50:
            regime = "weak"
            min_adj = 92
            allow = True
            detail = f"{symbol} ضعيف — تحت متوسط 50 وفوق 200"
        else:
            regime = "bear"
            min_adj = 95
            allow = True  # يبقى مسموحاً لكن بحد أعلى جداً
            detail = f"{symbol} هابط — تحت متوسط 200 (حد الدرجة 95)"

        return {
            "regime": regime,
            "min_score_adj": min_adj,
            "allow_auto": allow,
            "detail": detail,
            "spy_price": round(price, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
        }
    except Exception as exc:
        return {
            "regime": "neutral",
            "min_score_adj": 85,
            "allow_auto": True,
            "detail": f"تعذر فحص النظام: {str(exc)[:60]}",
            "spy_price": None,
            "sma50": None,
            "sma200": None,
        }


def regime_label() -> str:
    """نص قصير يُعرض في الحالة والصحة."""
    r = get_market_regime("SPY")
    emoji = {
        "strong_bull": "🟢",
        "bull": "🟢",
        "neutral": "🟡",
        "weak": "🟠",
        "bear": "🔴",
    }.get(r["regime"], "⚪")
    return f"{emoji} نظام السوق: {r['detail']} | حد الدرجة المعدّل: {r['min_score_adj']}"
