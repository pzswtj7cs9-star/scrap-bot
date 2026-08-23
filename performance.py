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

    def update_open_outcomes(self) -> list[dict[str, Any]]:
        """يفحص الإشارات المفتوحة مقابل الأسعار الحالية/التاريخية."""
        rows = self._load()
        changed: list[dict[str, Any]] = []
        for row in rows:
            if row.get("status") != "open":
                continue
            updated = self._evaluate_row(row)
            if updated:
                changed.append(row)
        if changed:
            self._save(rows)
        return changed

    def _evaluate_row(self, row: dict[str, Any]) -> bool:
        symbol = row["symbol"]
        try:
            opened = datetime.fromisoformat(row["opened_at"])
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=NY)
            # اجلب من يوم الفتح إلى الآن
            start = (opened - timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
            if df is None or df.empty:
                return False
            # فلتر الأيام بعد الفتح
            df = df[df.index.tz_localize(None) >= opened.replace(tzinfo=None).date() if False else df.index]
            stop = float(row["stop_loss"])
            tp1 = float(row["tp1"])
            tp2 = float(row["tp2"])
            tp3 = float(row["tp3"])
            entry = float(row["entry"])

            for idx, bar in df.iterrows():
                low = float(bar["Low"])
                high = float(bar["High"])
                close = float(bar["Close"])
                # الوقف أولاً (تحفظي)
                if low <= stop:
                    row["status"] = "closed"
                    row["result"] = "stop"
                    row["exit_price"] = round(stop, 4)
                    row["exit_at"] = str(idx)
                    row["pnl_pct"] = round((stop - entry) / entry * 100, 2)
                    return True
                if high >= tp3:
                    row["status"] = "closed"
                    row["result"] = "tp3"
                    row["exit_price"] = round(tp3, 4)
                    row["exit_at"] = str(idx)
                    row["pnl_pct"] = round((tp3 - entry) / entry * 100, 2)
                    return True
                if high >= tp2:
                    row["status"] = "closed"
                    row["result"] = "tp2"
                    row["exit_price"] = round(tp2, 4)
                    row["exit_at"] = str(idx)
                    row["pnl_pct"] = round((tp2 - entry) / entry * 100, 2)
                    return True
                if high >= tp1:
                    row["status"] = "closed"
                    row["result"] = "tp1"
                    row["exit_price"] = round(tp1, 4)
                    row["exit_at"] = str(idx)
                    row["pnl_pct"] = round((tp1 - entry) / entry * 100, 2)
                    return True

            # إذا مر أكثر من 15 يوم تداول وما وصل شيء — إغلاق عند آخر سعر
            age_days = (_now() - opened).days
            if age_days >= 15 and len(df) > 0:
                last = float(df["Close"].iloc[-1])
                row["status"] = "closed"
                row["result"] = "timeout"
                row["exit_price"] = round(last, 4)
                row["exit_at"] = str(df.index[-1])
                row["pnl_pct"] = round((last - entry) / entry * 100, 2)
                return True
        except Exception as exc:
            log.warning("تقييم %s فشل: %s", symbol, exc)
        return False

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
