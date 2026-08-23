"""أوقات جلسة السوق الأمريكي بتوقيت نيويورك."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
RIYADH = ZoneInfo("Asia/Riyadh")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


def now_ny() -> datetime:
    return datetime.now(NY)


def is_weekday(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    return dt.weekday() < 5


def is_us_regular_session(dt: datetime | None = None) -> bool:
    dt = dt or now_ny()
    if not is_weekday(dt):
        return False
    t = dt.timetz().replace(tzinfo=None) if False else dt.time()
    return REGULAR_OPEN <= t <= REGULAR_CLOSE


def is_post_close_window(dt: datetime | None = None, start_h: int = 16, start_m: int = 10, end_h: int = 16, end_m: int = 50) -> bool:
    """نافذة بعد الإغلاق لإرسال الملخص اليومي."""
    dt = dt or now_ny()
    if not is_weekday(dt):
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
        return f"السوق مغلق (عطلة نهاية الأسبوع) — نيويورك {dt.strftime('%H:%M')} | الرياض {riyadh.strftime('%H:%M')}"
    t = dt.time()
    if t < time(4, 0):
        status = "خارج الجلسة"
    elif t < REGULAR_OPEN:
        status = "ما قبل الافتتاح"
    elif t <= REGULAR_CLOSE:
        status = "الجلسة النظامية مفتوحة الآن"
    elif t <= time(20, 0):
        status = "بعد الإغلاق / تداول ممتد"
    else:
        status = "السوق مغلق"
    return f"{status} — نيويورك {dt.strftime('%H:%M')} | الرياض {riyadh.strftime('%H:%M')}"
