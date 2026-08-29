"""سجل أداء الإشارات: حفظ + متابعة الوصول للهدف أو الوقف."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
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
        # factors: لربط الإشارة بنظام الأوزان التكيفية (إن وُجد)
        factors = list(getattr(sig, "factor_keys", None) or [])
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
            "factors": factors,
            "regime": getattr(sig, "regime", "") or "",
            "volume_ratio": round(float(getattr(sig, "volume_ratio", 0) or 0), 3),
            "ext_sma20": round(float(getattr(sig, "ext_sma20", 0) or 0), 2),
            "atr_pct": round(float(getattr(sig, "atr_pct", 0) or 0), 2),
            "structure_zone": getattr(sig, "structure_zone", "") or "",
            "quality_ok": bool(getattr(sig, "quality_ok", True)),
            "vwap_day_note": getattr(sig, "vwap_day_note", "") or "",
            "vol_at_level": bool(getattr(sig, "vol_at_level", False)),
            "dumped_then_reclaimed": bool(getattr(sig, "dumped_then_reclaimed", False)),
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

    def reopen_symbol(self, symbol: str) -> dict[str, Any] | None:
        """يعيد فتح آخر صفقة مغلقة لنفس الرمز (مثلاً بعد إغلاق وقف خاطئ)."""
        symbol = symbol.upper().strip()
        rows = self._load()
        for row in reversed(rows):
            if row.get("symbol") != symbol:
                continue
            if row.get("status") == "open":
                return {"ok": False, "reason": "already_open", "row": row}
            row["status"] = "open"
            row["exit_price"] = None
            row["exit_at"] = None
            row["result"] = None
            row["pnl_pct"] = None
            row["hit_tp1"] = False
            row["hit_tp2"] = False
            row["hit_tp3"] = False
            note = (row.get("notes") or "").strip()
            tag = "reopened_after_false_stop"
            row["notes"] = f"{note} | {tag}".strip(" |")
            self._save(rows)
            return {"ok": True, "row": row}
        return None

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
            try:
                from auto_tune import AUTO
                AUTO.recompute_from_rows(self._load())
            except Exception:
                pass
        return events

    def _fetch_bars(self, symbol: str, opened: datetime, prefer_intraday: bool):
        age_hours = (_now() - opened).total_seconds() / 3600
        if prefer_intraday and age_hours <= 72:
            try:
                from market_data import fetch_intraday

                df5 = fetch_intraday(symbol, period="5d", interval="5m")
                if df5 is not None and not df5.empty:
                    return df5, "5m"
            except Exception:
                pass
        try:
            from market_data import fetch_history

            df = fetch_history(symbol, period="3mo")
            return df, "1d"
        except Exception:
            t = yf.Ticker(symbol)
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

            # أعلى/أدنى منذ الفتح فقط (لا نستخدم قيعان ما قبل الإشارة)
            try:
                work = df.copy()
                # توحيد الفهرس الزمني وفلترة ما بعد وقت الدخول
                if not isinstance(work.index, pd.DatetimeIndex):
                    for col in ("Datetime", "Date", "index"):
                        if col in work.columns:
                            work = work.set_index(col)
                            break
                if isinstance(work.index, pd.DatetimeIndex):
                    idx = work.index
                    if idx.tz is None:
                        idx = idx.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
                    else:
                        idx = idx.tz_convert(NY)
                    work = work.copy()
                    work.index = idx
                    work = work[~work.index.isna()]
                    # ابدأ من شمعة الدخول تقريباً (هامش دقيقتين)
                    work = work[work.index >= (opened - timedelta(minutes=2))]
                if work.empty:
                    # لا نغلق الصفقة إذا ما قدرنا نعزل فترة ما بعد الدخول
                    return []

                highs = work["High"].astype(float)
                lows = work["Low"].astype(float)
                closes = work["Close"].astype(float)
                last_close = float(closes.iloc[-1])
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
                    factors = row.get("factors") or []
                    if factors:
                        try:
                            from weights import WEIGHTS
                            won = pnl > 0
                            WEIGHTS.record_outcome(factors, won=won)
                        except Exception:
                            pass
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

            # الوقف: نطلب تأكيداً أقوى من لمسة وهمية واحدة
            # 1) إغلاق شمعة تحت/عند الوقف، أو
            # 2) قاعان متتاليان تحت الوقف على 5م
            stop_hit = False
            stop_px = stop
            try:
                below = (closes <= stop) | (lows <= stop)
                if bool((closes <= stop).iloc[-3:].any()):
                    stop_hit = True
                    stop_px = float(min(stop, float(closes.iloc[-3:].min())))
                elif tf == "5m" and len(lows) >= 2:
                    if float(lows.iloc[-1]) <= stop and float(lows.iloc[-2]) <= stop:
                        stop_hit = True
                        stop_px = stop
                elif tf == "1d" and float(lows.iloc[-1]) <= stop and float(closes.iloc[-1]) <= stop * 1.002:
                    # يومي: قاع اليوم + إغلاق قريب من الوقف
                    stop_hit = True
                    stop_px = stop
            except Exception:
                stop_hit = min_low <= stop and last_close <= stop

            if stop_hit:
                events.append(_evt("stop", stop_px, True))
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

            age_days = (_now() - opened).days
            if age_days >= 15 and row.get("status") == "open":
                events.append(_evt("timeout", last_close, True))
                return events

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

        # متوسط R المتحقق: pnl / المخاطرة الأولية لكل صفقة
        r_multiples = []
        equity = [0.0]
        for r in closed:
            entry = float(r.get("entry") or 0)
            stop = float(r.get("stop_loss") or 0)
            pnl = float(r.get("pnl_pct") or 0)
            risk_pct = ((entry - stop) / entry * 100) if entry and stop and entry > stop else 0
            if risk_pct > 0:
                r_multiples.append(pnl / risk_pct)
            equity.append(equity[-1] + pnl)
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
        # أقصى انخفاض (drawdown) على منحنى مجموع العوائد %
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd

        # نسبة النجاح حسب شريحة الدرجة
        def _bucket_stats(lo: int, hi: int) -> str:
            subset = [r for r in closed if lo <= int(r.get("score") or 0) <= hi]
            if not subset:
                return f"{lo}-{hi}: لا بيانات"
            w = [r for r in subset if (r.get("pnl_pct") or 0) > 0]
            return f"{lo}-{hi}: {len(w)}/{len(subset)} ({len(w)/len(subset)*100:.0f}%)"

        def _r_of(r: dict[str, Any]) -> float | None:
            entry = float(r.get("entry") or 0)
            stop = float(r.get("stop_loss") or 0)
            pnl = float(r.get("pnl_pct") or 0)
            risk_pct = ((entry - stop) / entry * 100) if entry and stop and entry > stop else 0
            if risk_pct <= 0:
                return None
            return pnl / risk_pct

        def _regime_line(label: str, keys: set[str]) -> str:
            subset = [r for r in closed if str(r.get("regime") or "") in keys]
            if not subset:
                return f"{label}: لا بيانات"
            w = [r for r in subset if (r.get("pnl_pct") or 0) > 0]
            return f"{label}: {len(w)}/{len(subset)} نجاح ({len(w)/len(subset)*100:.0f}%)"

        fail_counts: dict[str, int] = {}
        for r in losses:
            reasons = []
            vol = float(r.get("volume_ratio") or 0)
            ext = float(r.get("ext_sma20") or 0)
            zone = str(r.get("structure_zone") or "")
            regime = str(r.get("regime") or "")
            if vol and vol < 1.05:
                reasons.append("حجم ضعيف")
            if ext > 7:
                reasons.append("امتداد عن متوسط 20")
            if "توزيع" in zone:
                reasons.append("قرب توزيع")
            if regime in {"weak", "bear"}:
                reasons.append("سوق ضعيف")
            note = str(r.get("vwap_day_note") or "")
            if "تحت" in note:
                reasons.append("تحت VWAP اليوم")
            if r.get("vol_at_level") is False and "vwap_day_note" in r:
                reasons.append("بدون حجم عند المستوى")
            if r.get("result") == "stop":
                reasons.append("ضرب وقف")
            if r.get("result") == "timeout":
                reasons.append("انتهاء مهلة")
            if not reasons:
                reasons.append("غير مصنّف")
            for reason in reasons:
                fail_counts[reason] = fail_counts.get(reason, 0) + 1

        lines = [
            "📈 سجل أداء الإشارات",
            f"الإجمالي: {len(rows)} | مفتوحة: {open_n} | مغلقة: {len(closed)}",
        ]
        if closed:
            lines += [
                f"نسبة الربح (إشارات مغلقة): {win_rate:.1f}%",
                f"متوسط العائد/الخسارة: {avg_pnl:+.2f}%",
                f"متوسط R المتحقق: {avg_r:+.2f}R",
                f"أقصى انخفاض تراكمي: {max_dd:.2f}%",
                "",
                "تفصيل النتائج:",
                f"  • وقف: {by_result.get('stop', 0)}",
                f"  • لمس هدف 1: {sum(1 for r in rows if r.get('hit_tp1'))}",
                f"  • لمس هدف 2: {sum(1 for r in rows if r.get('hit_tp2'))}",
                f"  • لمس هدف 3: {sum(1 for r in rows if r.get('hit_tp3')) or by_result.get('tp3', 0)}",
                f"  • انتهاء مهلة: {by_result.get('timeout', 0)}",
                "",
                "حسب الدرجة:",
                f"  • {_bucket_stats(84, 89)}",
                f"  • {_bucket_stats(90, 94)}",
                f"  • {_bucket_stats(95, 100)}",
                "",
                "حسب نظام السوق:",
                f"  • {_regime_line('قوي/صاعد', {'strong_bull', 'bull'})}",
                f"  • {_regime_line('محايد', {'neutral'})}",
                f"  • {_regime_line('ضعيف/هابط', {'weak', 'bear'})}",
            ]
            def _flag_line(title: str, pred) -> str:
                subset = [r for r in closed if pred(r)]
                if not subset:
                    return f"{title}: لا بيانات"
                w = [r for r in subset if (r.get("pnl_pct") or 0) > 0]
                return f"{title}: {len(w)}/{len(subset)} نجاح ({len(w)/len(subset)*100:.0f}%)"

            lines += [
                "",
                "حسب VWAP والحجم (معلومة للتعلم):",
                f"  • {_flag_line('فوق VWAP', lambda r: 'فوق' in str(r.get('vwap_day_note') or ''))}",
                f"  • {_flag_line('تحت VWAP', lambda r: 'تحت' in str(r.get('vwap_day_note') or ''))}",
                f"  • {_flag_line('حجم عند المستوى', lambda r: bool(r.get('vol_at_level')))}",
                f"  • {_flag_line('بدون حجم مستوى', lambda r: r.get('vol_at_level') is False)}",
            ]
            if fail_counts:
                lines.append("")
                lines.append("أكثر أسباب الخسارة المتكررة:")
                for k, v in sorted(fail_counts.items(), key=lambda x: -x[1])[:6]:
                    lines.append(f"  • {k}: {v}")
            extra = [k for k in sorted(by_result) if k not in {"stop", "tp1", "tp2", "tp3", "timeout"}]
            if extra:
                lines.append("")
                lines.append("نتائج أخرى:")
                for k in extra:
                    lines.append(f"  • {k}: {by_result[k]}")
            lines.append("")
            lines.append("آخر 5 صفقات:")
            for r in closed[-5:][::-1]:
                rr = _r_of(r)
                rtxt = f"{rr:+.1f}R" if rr is not None else "—"
                lines.append(
                    f"  {r.get('symbol')} | {r.get('score')} | {r.get('result') or '?'} | {rtxt}"
                )
        if open_n:
            lines.append("")
            lines.append("مفتوحة الآن:")
            for r in [x for x in rows if x.get("status") == "open"][-5:]:
                lines.append(f"  {r['symbol']} | دخول {r['entry']} | وقف {r['stop_loss']} | TP1 {r['tp1']}")
        try:
            from weights import WEIGHTS
            lines.append("")
            lines.append(WEIGHTS.summary_text())
        except Exception:
            pass
        try:
            from auto_tune import AUTO
            lines.append("")
            lines.append(AUTO.summary_text())
        except Exception:
            pass
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
