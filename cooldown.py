"""منع تكرار نفس السهم لمدة 5 أيام تداول."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def _today() -> date:
    return datetime.now(NY).date()


def trading_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    days = 0
    cur = start
    while cur < end:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


class CooldownBook:
    def __init__(self, path: Path, days: int = 5):
        self.path = path
        self.days = days
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict[str, str]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def mark(self, symbol: str, when: Optional[date] = None) -> None:
        data = self._load()
        data[symbol.upper()] = (when or _today()).isoformat()
        self._save(data)

    def last_date(self, symbol: str) -> Optional[date]:
        raw = self._load().get(symbol.upper())
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except Exception:
            return None

    def is_blocked(self, symbol: str) -> bool:
        last = self.last_date(symbol)
        if not last:
            return False
        return trading_days_between(last, _today()) < self.days

    def remaining_days(self, symbol: str) -> int:
        last = self.last_date(symbol)
        if not last:
            return 0
        used = trading_days_between(last, _today())
        return max(0, self.days - used)

    def blocked_list(self) -> list[tuple[str, int, str]]:
        out = []
        for sym, raw in sorted(self._load().items()):
            if self.is_blocked(sym):
                out.append((sym, self.remaining_days(sym), raw[:10]))
        return out
