"""سجل أداء الإشارات: حفظ + متابعة الوصول للهدف أو الوقف."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import yfinance as yf

from analyzer import SignalResult

log = logging.getLogger("halal-bot.perf")
NY = ZoneInfo("America/New_York")


def _now() -> datetime:
    return datetime.now(NY)


class PerformanceLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    def add_signal(self, sig: SignalResult, source: str = "auto") -> dict[str, Any]:
        rows = self._load()
        row = {
            "id": f"{sig.symbol}-{_now().strftime('%Y%m%d%H%M%S')}",
            "symbol": sig.symbol,
            "name": sig.name,
            "source": source,
            "opened_at": _now().isoformat(),
            "entry": round(sig.price, 4),
            "buy_low": round(sig.buy_low, 4),
            "buy_high": round(sig.buy_high, 4),
            "stop_loss": round(sig.stop_loss, 4),
            "tp1": round(sig.tp1, 4),
            "tp2": round(sig.tp2, 4),
            "tp3": round(sig.tp3, 4),
            "score": sig.score,
            "status": "open",
            "exit_price": None,
            "exit_at": None,
            "result": None,
            "pnl_pct": None,
            "notes": "",
        }
        rows.append(row)
        self._save(rows)
        return row

    def open_signals(self) -> list[dict[str, Any]]:
        return [r for r in self._load() if r.get("status") == "open"]

    def all_signals(self) -> list[dict[str, Any]]:
        return self._load()

    def update_open_outcomes(self, prefer_intraday: bool = True) -> list[dict[str, Any]]:
        """يفحص الإشارات المفتوحة. يعيد قائمة الأحداث الجديدة للتنبيه."""
        rows = self._load()
        events: list[dict[str, Any]] = []
        dirty = False
        for row in rows:
            if row.get("status") != "open":
                continue
            row.setdefault("hit_tp1", False)
            row.setdefault("hit_tp2", False)
            row.setdefault("hit_tp3", False)
            new_events = self._evaluate_row(row, prefer_intraday=prefer_intraday)
            if new_events:
                dirty = True
                for ev in new_events:
                    events.append(ev)
        if dirty:
            self._save(rows)
        return events

    def _fetch_bars(self, symbol: str, opened: datetime, prefer_intraday: bool):
        t = yf.Ticker(symbol)
        age_hours = (_now() - opened).total_seconds() / 3600
        # للإشارات الحديثة: شموع 5 دقائق أدق لمتابعة الوقف/الأهداف
        if prefer_intraday and age_hours <= 72:
            try:
                df5 = t.history(period="5d", interval="5m", auto_adjust=True)
                if df5 is not None and not df5.empty:
                    return df5, "5m"
            except Exception:
                pass
        start = (opened - timedelta(days=1)).strftime("%Y-%m-%d")
        df = t.history(start=start, interval="1d", auto_adjust=True)
        return df, "1d"

    def _evaluate_row(self, row: dict[str, Any], prefer_intraday: bool = True) -> list[dict[str, Any]]:
        symbol = row["symbol"]
        events: list[dict[str, Any]] = []
        try:
            opened = datetime.fromisoformat(row["opened_at"])
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=NY)
            df, tf = self._fetch_bars(symbol, opened, prefer_intraday)
            if df is None or df.empty:
                return []

            stop = float(row["stop_loss"])
            tp1 = float(row["tp1"])
            tp2 = float(row["tp2"])
            tp3 = float(row["tp3"])
            entry = float(row["entry"])

            # أعلى/أدنى منذ الفتح (تقريبي)
            try:
                # صفوف بعد وقت الفتح قدر الإمكان
                highs = df["High"].astype(float)
                lows = df["Low"].astype(float)
                last_close = float(df["Close"].iloc[-1])
                max_high = float(highs.max())
                min_low = float(lows.min())
            except Exception:
                return []

            def _evt(kind: str, price: float, close_trade: bool) -> dict[str, Any]:
                pnl = round((price - entry) / entry * 100, 2)
                if close_trade:
                    row["status"] = "closed"
                    row["result"] = kind
                    row["exit_price"] = round(price, 4)
                    row["exit_at"] = _now().isoformat()
                    row["pnl_pct"] = pnl
                return {
                    "symbol": symbol,
                    "name": row.get("name") or symbol,
                    "kind": kind,
                    "price": round(price, 4),
                    "entry": entry,
                    "pnl_pct": pnl,
                    "closed": close_trade,
                    "timeframe": tf,
                }

            # الوقف أولاً (تحفظي)
            if min_low <= stop:
                events.append(_evt("stop", stop, True))
                return events

            # أهداف تدريجية: ينبّه عند كل هدف، ويغلق عند الثالث
            if max_high >= tp1 and not row.get("hit_tp1"):
                row["hit_tp1"] = True
                events.append(_evt("tp1", tp1, False))
            if max_high >= tp2 and not row.get("hit_tp2"):
                row["hit_tp2"] = True
                events.append(_evt("tp2", tp2, False))
            if max_high >= tp3 and not row.get("hit_tp3"):
                row["hit_tp3"] = True
                events.append(_evt("tp3", tp3, True))
                return events

            # انتهاء زمني
            age_days = (_now() - opened).days
            if age_days >= 15 and row.get("status") == "open":
                events.append(_evt("timeout", last_close, True))
                return events

            # حفظ أعلام الأهداف حتى لو ما في إغلاق
            if events:
                row["last_price"] = round(last_close, 4)
                return events
            row["last_price"] = round(last_close, 4)
        except Exception as exc:
            log.warning("تقييم %s فشل: %s", symbol, exc)
        return events

    @staticmethod
    def format_event_ar(ev: dict[str, Any]) -> str:
        kind = ev.get("kind")
        sym = ev.get("symbol")
        name = ev.get("name") or sym
        price = ev.get("price")
        entry = ev.get("entry")
        pnl = ev.get("pnl_pct") or 0
        if kind == "stop":
            title = f"🛑 ضرب الوقف — {sym}"
            body = f"تم لمس وقف الخسارة عند {price:.2f} $"
        elif kind == "tp1":
            title = f"🎯 الهدف 1 — {sym}"
            body = f"وصل الهدف الأول عند {price:.2f} $"
        elif kind == "tp2":
            title = f"🎯 الهدف 2 — {sym}"
            body = f"وصل الهدف الثاني عند {price:.2f} $"
        elif kind == "tp3":
            title = f"🏁 الهدف 3 (إغلاق) — {sym}"
            body = f"وصل الهدف الثالث عند {price:.2f} $"
        elif kind == "timeout":
            title = f"⏰ انتهاء مدة الصفقة — {sym}"
            body = f"أُغلقت بعد 15 يوماً عند {price:.2f} $"
        else:
            title = f"📋 تحديث — {sym}"
            body = f"نتيجة: {kind}"
        lines = [
            title,
            name,
            body,
            f"الدخول: {entry:.2f} $ | النتيجة: {pnl:+.2f}%",
        ]
        if not ev.get("closed") and kind in {"tp1", "tp2"}:
            lines.append("الصفقة ما زالت مفتوحة لمتابعة الأهداف التالية / الوقف.")
        elif ev.get("closed"):
            lines.append("تم إغلاق الصفقة في السجل.")
        lines.append("متابعة تقريبية — ليست تنفيذاً آلياً للأوامر.")
        return "\n".join(lines)

    def stats_text(self) -> str:
        self.update_open_outcomes()
        rows = self._load()
        if not rows:
            return "لا توجد إشارات مسجّلة بعد."

        closed = [r for r in rows if r.get("status") == "closed"]
        open_n = len([r for r in rows if r.get("status") == "open"])
        wins = [r for r in closed if (r.get("pnl_pct") or 0) > 0]
        losses = [r for r in closed if (r.get("pnl_pct") or 0) <= 0]
        by_result: dict[str, int] = {}
        for r in closed:
            by_result[r.get("result") or "?"] = by_result.get(r.get("result") or "?", 0) + 1

        avg_pnl = sum(r.get("pnl_pct") or 0 for r in closed) / len(closed) if closed else 0
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        lines = [
            "📈 سجل أداء الإشارات",
            f"الإجمالي: {len(rows)} | مفتوحة: {open_n} | مغلقة: {len(closed)}",
        ]
        if closed:
            lines += [
                f"نسبة الربح (إشارات مغلقة): {win_rate:.1f}%",
                f"متوسط العائد/الخسارة: {avg_pnl:+.2f}%",
                "التفصيل:",
            ]
            for k, v in sorted(by_result.items()):
                lines.append(f"  • {k}: {v}")
            lines.append("")
            lines.append("آخر 5 مغلقة:")
            for r in closed[-5:][::-1]:
                lines.append(
                    f"  {r['symbol']} | {r.get('result')} | {r.get('pnl_pct'):+.2f}% | دخول {r['entry']}"
                )
        if open_n:
            lines.append("")
            lines.append("مفتوحة الآن:")
            for r in [x for x in rows if x.get("status") == "open"][-5:]:
                lines.append(f"  {r['symbol']} | دخول {r['entry']} | وقف {r['stop_loss']} | TP1 {r['tp1']}")
        lines.append("")
        lines.append("ملاحظة: التقييم تقريبي على بيانات يومية وليس تنفيذاً حقيقياً.")
        return "\n".join(lines)

    def weekly_report(self) -> str:
        self.update_open_outcomes()
        rows = self._load()
        now = _now()
        week_ago = now - timedelta(days=7)
        week = []
        for r in rows:
            try:
                opened = datetime.fromisoformat(r["opened_at"])
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=NY)
                if opened >= week_ago:
                    week.append(r)
            except Exception:
                continue
        if not week:
            return "📊 التقرير الأسبوعي\nلا توجد إشارات خلال آخر 7 أيام."

        closed = [r for r in week if r.get("status") == "closed"]
        opened_n = len(week)
        wins = [r for r in closed if (r.get("pnl_pct") or 0) > 0]
        losses = [r for r in closed if (r.get("pnl_pct") or 0) <= 0]
        avg = sum((r.get("pnl_pct") or 0) for r in closed) / len(closed) if closed else 0
        win_rate = len(wins) / len(closed) * 100 if closed else 0
        best = max(closed, key=lambda x: x.get("pnl_pct") or -999) if closed else None
        worst = min(closed, key=lambda x: x.get("pnl_pct") or 999) if closed else None

        lines = [
            "📊 التقرير الأسبوعي (7 أيام)",
            f"إشارات جديدة: {opened_n} | أُغلقت: {len(closed)} | ما زالت مفتوحة: {opened_n - len(closed)}",
        ]
        if closed:
            lines += [
                f"نسبة الربح: {win_rate:.1f}%",
                f"متوسط النتيجة: {avg:+.2f}%",
            ]
            if best:
                lines.append(f"أفضل: {best['symbol']} {best.get('pnl_pct'):+.2f}% ({best.get('result')})")
            if worst:
                lines.append(f"أضعف: {worst['symbol']} {worst.get('pnl_pct'):+.2f}% ({worst.get('result')})")
        lines.append("")
        lines.append("تفصيل الأسبوع:")
        for r in week[-10:]:
            st = r.get("result") or r.get("status")
            pnl = r.get("pnl_pct")
            pnl_s = f"{pnl:+.2f}%" if pnl is not None else "—"
            lines.append(f"• {r['symbol']} | {st} | {pnl_s} | {r.get('score')}/100")
        lines.append("")
        lines.append("تقييم تقريبي — ليست نتائج حساب حقيقي.")
        return "\n".join(lines)

    def today_closed_and_open(self, day: str) -> tuple[list, list]:
        rows = self._load()
        today_new = [r for r in rows if str(r.get("opened_at", "")).startswith(day)]
        opened = [r for r in rows if r.get("status") == "open"]
        return today_new, opened
