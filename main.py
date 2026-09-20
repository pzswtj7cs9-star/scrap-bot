#!/usr/bin/env python3
"""
بوت تليجرام لأسهم السوق الأمريكي الحلال — النسخة المحسّنة

التحسينات:
• تسجيل الإشارة + تحديث تلقائي للأوزان (Adaptive Weights)
• وقف خسارة محسّن (هيكل + ATR ديناميكي + قاع 15د)
• إطار 15 دقيقة للتأكيد
• فلتر قوة السوق العام (SPY regime) — يرفع الحد أو يوقف التنبيهات عند الهبوط
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from analyzer import (
    analyze,
    format_signal_ar,
    rank_all,
    scan_symbols,
    ADAPTIVE_POLICY_FILE as DAILY_ADAPTIVE_POLICY_FILE,
    get_daily_market_context,
)
from analyzer_intraday import (
    INTRADAY_MIN_SCORE,
    analyze_intraday,
    format_intraday_ar,
    scan_intraday,
    session_window_ok,
    monthly_self_optimization,
    get_live_entry_price,
    ADAPTIVE_POLICY_FILE as INTRADAY_ADAPTIVE_POLICY_FILE,
    get_intraday_market_context,
)
from backtest import run_backtest
from charting import build_signal_chart
from cooldown import CooldownBook
from earnings import is_near_earnings
from market import (
    get_market_regime,
    is_friday_post_close,
    is_post_close_window,
    is_us_regular_session,
    is_us_trading_day,
    now_ny,
    regime_label,
    session_label,
)
from performance import PerformanceLog
from auto_tune import AUTO
from position import calc_position, format_position_ar
from stocks import CORE_WATCHLIST, HALAL_STOCKS, display_name, is_known_halal
from weights import WEIGHTS

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("halal-bot")

# Runtime diagnostic: prove exactly which analyzer_intraday.py Render loads.
# /scani display cleanup is session-only; it does not alter scan or strategy logic.
import analyzer_intraday as _runtime_analyzer_intraday
log.info(
    "RUNTIME ANALYZER | file=%s | version=%s",
    getattr(_runtime_analyzer_intraday, "__file__", "MISSING"),
    getattr(_runtime_analyzer_intraday, "INTRADAY_ANALYZER_VERSION", "MISSING"),
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "").strip()
MIN_SCORE = int(os.getenv("MIN_SCORE", "84"))
DAILY_MAX = int(os.getenv("DAILY_MAX_ALERTS", "5"))
ALERT_EVERY_MINUTES = int(os.getenv("ALERT_EVERY_MINUTES", "90"))
INTRADAY_MAX = int(os.getenv("INTRADAY_MAX_ALERTS", "3"))
INTRADAY_EVERY_MINUTES = int(os.getenv("INTRADAY_EVERY_MINUTES", "120"))
LIVE_SCAN_SECONDS = int(os.getenv("LIVE_SCAN_SECONDS", "60"))
EARNINGS_DAYS = int(os.getenv("EARNINGS_DAYS", "2"))
COOLDOWN_DAYS = int(os.getenv("COOLDOWN_DAYS", "5"))
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
SUBS_FILE = DATA_DIR / "subscribers.json"
STATE_FILE = DATA_DIR / "daily_state.json"
PERF_FILE = DATA_DIR / "signals_log.json"
PERF_INTRA_FILE = DATA_DIR / "signals_log_intraday.json"
COOL_FILE = DATA_DIR / "cooldown.json"
REPORTS_FILE = DATA_DIR / "reports_state.json"
WEIGHTS_FILE = DATA_DIR / "weights_state.json"
CHART_DIR = DATA_DIR / "charts"

# ربط مسار الأوزان
WEIGHTS.path = WEIGHTS_FILE
WEIGHTS._load()
AUTO.path = DATA_DIR / "auto_tune.json"
AUTO._load()

PERF = PerformanceLog(PERF_FILE)
PERF_INTRA = PerformanceLog(PERF_INTRA_FILE)
COOL = CooldownBook(COOL_FILE, days=COOLDOWN_DAYS)
BOT_STARTED = now_ny()
LAST_SCAN_AT: datetime | None = None
LAST_YF_OK: bool | None = None
LAST_YF_NOTE = ""


def load_subs() -> set[int]:
    if SUBS_FILE.exists():
        try:
            return {int(x) for x in json.loads(SUBS_FILE.read_text())}
        except Exception:
            pass
    subs: set[int] = set()
    if OWNER_CHAT_ID.isdigit():
        subs.add(int(OWNER_CHAT_ID))
    return subs


def save_subs(subs: set[int]) -> None:
    try:
        SUBS_FILE.write_text(json.dumps(sorted(subs)))
    except Exception as exc:
        log.warning("تعذر حفظ المشتركين: %s", exc)


def _empty_state(day: str) -> dict:
    return {
        "date": day,
        "sent": [],
        "scores": {},
        "last_sent_at": None,
        "sent_intraday": [],
        "scores_intraday": {},
        "last_sent_intraday_at": None,
    }


def load_state() -> dict:
    day = now_ny().strftime("%Y-%m-%d")
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get("date") == day:
                data.setdefault("sent", [])
                data.setdefault("scores", {})
                data.setdefault("last_sent_at", None)
                data.setdefault("sent_intraday", [])
                data.setdefault("scores_intraday", {})
                data.setdefault("last_sent_intraday_at", None)
                return data
        except Exception:
            pass
    state = _empty_state(day)
    save_state(state)
    return state


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.warning("تعذر حفظ الحالة اليومية: %s", exc)


SUBSCRIBERS = load_subs()
_scan_lock = asyncio.Lock()
_intra_scan_lock = asyncio.Lock()
LAST_DAILY_SCAN_ATTEMPT: datetime | None = None
LAST_INTRADAY_SCAN_ATTEMPT: datetime | None = None
SCAN_RETRY_MINUTES = int(os.getenv("SCAN_RETRY_MINUTES", "5"))


def load_reports() -> dict:
    if REPORTS_FILE.exists():
        try:
            return json.loads(REPORTS_FILE.read_text())
        except Exception:
            pass
    return {
        "daily_sent_on": "",
        "weekly_sent_on": "",
        "daily_swing_performance_sent_on": "",
        "daily_intraday_performance_sent_on": "",
        "weekly_swing_performance_sent_on": "",
        "weekly_intraday_performance_sent_on": "",
        "monthly_swing_performance_sent_on": "",
        "monthly_intraday_performance_sent_on": "",
        "monthly_learning_sent_on": "",
        "adaptive_daily_alert_key": "",
        "adaptive_intraday_alert_key": "",
    }


def save_reports(data: dict) -> None:
    try:
        REPORTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.warning("reports state: %s", exc)


def ping_yahoo() -> tuple[bool, str]:
    global LAST_YF_OK, LAST_YF_NOTE
    try:
        from market_data import ping_sources

        ok, note = ping_sources()
        LAST_YF_OK = ok
        LAST_YF_NOTE = note
        return ok, note
    except Exception as exc:
        LAST_YF_OK = False
        LAST_YF_NOTE = str(exc)[:80]
        return False, LAST_YF_NOTE


def spy_day_change() -> str:
    try:
        import yfinance as yf

        df = yf.Ticker("SPY").history(period="5d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 2:
            return "SPY: غير متاح"
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        chg = (last - prev) / prev * 100
        return f"SPY: {last:.2f} ({chg:+.2f}%)"
    except Exception:
        return "SPY: غير متاح"


def start_health_server() -> None:
    port = int(os.getenv("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"status":"ok","service":"halal-us-stock-bot"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Health server on port %s", port)


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🔍 أقوى التأكيدات", callback_data="scan"),
                InlineKeyboardButton("📡 حالة السوق", callback_data="status"),
            ],
            [
                InlineKeyboardButton("📈 سجل الأداء", callback_data="perf"),
                InlineKeyboardButton("🩺 صحة النظام", callback_data="health"),
            ],
            [
                InlineKeyboardButton("📋 قائمة الحلال", callback_data="list"),
                InlineKeyboardButton("🔔 تفعيل التنبيه", callback_data="sub"),
            ],
            [InlineKeyboardButton("❓ المساعدة", callback_data="help")],
        ]
    )


WELCOME = """بسم الله الرحمن الرحيم

بوت الأسهم الأمريكية الحلال — النسخة المحسّنة.

التنبيه التلقائي:
• حد أساسي 84/100 + تأكيد 5د + تأكيد 15د + فلتر سعر ≤200$
• فلتر نظام السوق (SPY): يرفع الحد أو يوقف التنبيهات عند الهبوط
• سهم واحد كل 90 دقيقة (الأقوى)
• سقف 5 أسهم يومياً
• يتجنب الأسهم قرب إعلان الأرباح
• وقف خسارة محسّن + أوزان تكيفية من نتائج سابقة

أدوات:
/size 10000 1 NVDA — حاسبة حجم
/chart NVDA — رسم
/perf — سجل أداء + الأوزان
/backtest — باكتست
/analyze NVDA | /scan | /today | /status | /health | /weekly

تحليل تعليمي وليس توصية. تحقق من الحكم الشرعي قبل الشراء."""


HELP = """الأوامر

تحليل:
/analyze NVDA
/scan — أقوى التأكيدات الآن
/chart NVDA — رسم بياني

إدارة رأس المال:
/size رأس_المال نسبة_المخاطرة الرمز
مثال: /size 10000 1 NVDA

الأداء:
/perf — نتائج الإشارات + الأوزان التكيفية
/backtest — اختبار تاريخي مبسّط

التنبيه:
/subscribe | /unsubscribe
/today | /status | /health | /weekly

الإعدادات الحالية:
حد التنبيه الأساسي: {min_score}
سقف اليوم: {daily_max}
المباعدة: {interval} دقيقة
تهدئة السهم: {cool} أيام تداول
تجنب الإعلانات: خلال يومين من تاريخ الأرباح
فلتر السوق: SPY فوق/تحت متوسط 50 و200"""


def _watch(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return context.user_data.setdefault("watch", list(CORE_WATCHLIST))


def remaining_slots() -> int:
    state = load_state()
    return max(0, DAILY_MAX - len(state["sent"]))


def minutes_since_last_alert(state: dict) -> float | None:
    raw = state.get("last_sent_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now_ny().tzinfo)
        return (now_ny() - ts.astimezone(now_ny().tzinfo)).total_seconds() / 60
    except Exception:
        return None


def next_alert_text(state: dict) -> str:
    if len(state.get("sent", [])) >= DAILY_MAX:
        return "اكتملت حصة اليوم — التالي غداً"
    elapsed = minutes_since_last_alert(state)
    if elapsed is None:
        return "أول سهم عند أول تأكيد قوي بعد فتح السوق"
    wait = max(0, ALERT_EVERY_MINUTES - elapsed)
    if wait <= 0.2:
        return "النافذة مفتوحة: ينتظر أقوى تأكيد"
    nxt = now_ny() + timedelta(minutes=wait)
    return f"التالي بعد نحو {wait:.0f} دقيقة (نيويورك {nxt.strftime('%H:%M')})"




def _daily_market_display_context() -> tuple[bool, str, str, int, str]:
    """Return the Daily analyzer market state and its exact policy floor for display."""
    ok, state, condition = get_daily_market_context()
    floors = {
        "قوي": 82,
        "إيجابي_تحت_VWAP": 85,
        "مختلط": 86,
        "ضعيف": 92,
        "غير مؤكد": 92,
    }
    effective_min = max(MIN_SCORE, floors.get(condition, 92))
    labels = {
        "قوي": "🟢 قوي",
        "إيجابي_تحت_VWAP": "🟠 إيجابي",
        "مختلط": "🟡 مختلط",
        "ضعيف": "🔴 ضعيف",
        "غير مؤكد": "⚪️ غير مؤكد",
    }
    return bool(ok), str(state), str(condition), int(effective_min), labels.get(condition, "⚪️ غير مؤكد")

def today_summary() -> str:
    state = load_state()
    sent = state["sent"]
    scores = state.get("scores", {})
    _, _, _, effective_min, market_label = _daily_market_display_context()
    lines = [
        "📡 " + session_label(),
        f"{market_label} | نظام السوق",
        f"🎯 حد التنبيه الأساسي: {MIN_SCORE}/100 → المعدّل الآن: {effective_min}",
        f"⏱ إرسال: سهم واحد كل {ALERT_EVERY_MINUTES} دقيقة",
        f"📦 حصة اليوم: {len(sent)}/{DAILY_MAX}",
        f"⏳ المتبقي: {remaining_slots()}",
        f"🕒 {next_alert_text(state)}",
        f"🚫 يتجنب الإعلانات خلال {EARNINGS_DAYS} يوم",
        f"🔁 لا يكرر السهم قبل {COOLDOWN_DAYS} أيام تداول",
        "🤖 تنبيهات تلقائية: مفعّلة",
        "",
    ]
    if sent:
        lines.append("المرسل اليوم:")
        for i, sym in enumerate(sent, 1):
            lines.append(f"  {i}. {sym} ({scores.get(sym, '—')}/100) — {display_name(sym)}")
    else:
        lines.append("لم يُرسل أي سهم بعد اليوم.")
    return "\n".join(lines)


async def send_signal_with_chart(chat_id: int, bot, sig, header: str, source: str = "manual") -> None:
    PERF.add_signal(sig, source=source)
    text = header + "\n\n" + format_signal_ar(sig, MIN_SCORE)
    try:
        near, edt = await asyncio.to_thread(is_near_earnings, sig.symbol, EARNINGS_DAYS)
        if near:
            text += f"\n\n⚠️ قرب إعلان أرباح: {edt}"
    except Exception:
        pass
    await bot.send_message(chat_id=chat_id, text=text)
    try:
        path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
        if path and path.exists():
            with open(path, "rb") as f:
                await bot.send_photo(chat_id=chat_id, photo=InputFile(f, filename=path.name))
    except Exception as exc:
        log.warning("chart failed: %s", exc)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, reply_markup=main_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        HELP.format(
            min_score=MIN_SCORE,
            daily_max=DAILY_MAX,
            interval=ALERT_EVERY_MINUTES,
            cool=COOLDOWN_DAYS,
        ),
        reply_markup=main_keyboard(),
    )


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["📋 الأسهم المعتمدة حالياً:\n"]
    for i, (sym, name) in enumerate(HALAL_STOCKS.items(), 1):
        lines.append(f"{i:02d}. {sym} — {name}")
    await update.message.reply_text("\n".join(lines))


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("اكتب: /watch NVDA")
        return
    sym = context.args[0].upper()
    watch = _watch(context)
    if sym not in watch:
        watch.append(sym)
    await update.message.reply_text(f"تمت إضافة {sym}.")


async def cmd_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("اكتب: /unwatch NVDA")
        return
    sym = context.args[0].upper()
    watch = _watch(context)
    if sym in watch:
        watch.remove(sym)
        await update.message.reply_text(f"تم حذف {sym}")
    else:
        await update.message.reply_text("غير موجود")


async def cmd_mywatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("المراقبة:\n" + " • ".join(_watch(context)))


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    SUBSCRIBERS.add(update.effective_chat.id)
    save_subs(SUBSCRIBERS)
    await update.message.reply_text(
        f"تم التفعيل. سهم واحد كل {ALERT_EVERY_MINUTES} د، حد {MIN_SCORE}، سقف {DAILY_MAX}."
    )


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    SUBSCRIBERS.discard(update.effective_chat.id)
    save_subs(SUBSCRIBERS)
    await update.message.reply_text("تم إيقاف التنبيهات.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(today_summary())


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(today_summary())


def health_text() -> str:
    global LAST_SCAN_AT
    ok, note = ping_yahoo()
    up = now_ny() - BOT_STARTED
    hours = int(up.total_seconds() // 3600)
    mins = int((up.total_seconds() % 3600) // 60)
    last_scan = LAST_SCAN_AT.strftime("%H:%M:%S") if LAST_SCAN_AT else "لا يوجد بعد"
    blocked = COOL.blocked_list()
    state = load_state()
    regime = get_market_regime("SPY")
    lines = [
        "🩺 صحة النظام",
        session_label(),
        regime_label(),
        f"التشغيل: {hours}س {mins}د",
        f"المشتركون: {len(SUBSCRIBERS)}",
        f"مصدر البيانات: {'يعمل' if ok else 'تعثر'} — {note}",
        f"آخر مسح: {last_scan}",
        f"آخر تنبيه تلقائي: {state.get('last_sent_at') or '—'}",
        f"حصة السوينغ: {len(state.get('sent', []))}/{DAILY_MAX}",
        f"حصة اللحظي: {len(state.get('sent_intraday', []))}/{INTRADAY_MAX}",
        f"تهدئة الأسهم: {COOLDOWN_DAYS} أيام تداول",
        f"تنبيهات تلقائية: {'نعم' if regime['allow_auto'] else 'موقوفة (سوق هابط)'}",
    ]
    if blocked:
        lines.append("في فترة تهدئة:")
        for sym, rem, dt in blocked[:12]:
            lines.append(f"• {sym} — متبقي {rem} يوم (آخر مرة {dt})")
    else:
        lines.append("لا توجد أسهم في التهدئة.")
    return "\n".join(lines)


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("جاري فحص النظام...")
    text = await asyncio.to_thread(health_text)
    await msg.edit_text(text)


async def cmd_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("جاري إعداد التقرير الأسبوعي...")
    text = await asyncio.to_thread(PERF.weekly_report)
    await msg.edit_text(text)


async def cmd_perf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("جاري تحديث سجل الأداء...")

    def _both():
        swing = PERF.stats_text()
        intra = PERF_INTRA.stats_text()
        return (
            "===== سوينغ (يومي) =====\n"
            + swing
            + "\n\n===== لحظي (ساعة) =====\n"
            + intra
        )

    text = await asyncio.to_thread(_both)
    await msg.edit_text(text)


async def cmd_reopen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إعادة فتح صفقة أُغلقت بالخطأ في السجل (مثلاً وقف وهمي)."""
    if not context.args:
        await update.message.reply_text("الاستخدام:\n/reopen SBUX")
        return
    sym = context.args[0].upper().strip()
    msg = await update.message.reply_text(f"جاري إعادة فتح {sym} في السجل...")

    def _do():
        return PERF.reopen_symbol(sym)

    res = await asyncio.to_thread(_do)
    if not res:
        await msg.edit_text(f"ما لقيت صفقة مغلقة لـ {sym} في السجل.")
        return
    if not res.get("ok"):
        row = res.get("row") or {}
        await msg.edit_text(
            f"{sym} مفتوحة أصلاً في السجل.\nدخول: {row.get('entry')} | وقف: {row.get('stop_loss')}"
        )
        return
    row = res["row"]
    await msg.edit_text(
        f"تمت إعادة فتح {sym} في السجل ✅\n"
        f"دخول: {row.get('entry')} | وقف: {row.get('stop_loss')}\n"
        f"TP1: {row.get('tp1')} | TP2: {row.get('tp2')} | TP3: {row.get('tp3')}\n"
        f"المتابعة تشتغل بالمنطق الجديد (بعد الإصلاح)."
    )


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text(
        "جاري الباكتست على آخر سنة (قد يستغرق دقيقة)..."
    )
    text = await asyncio.to_thread(run_backtest, CORE_WATCHLIST[:12], "1y", MIN_SCORE)
    await msg.edit_text(text)


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text(
            "الاستخدام:\n/size رأس_المال نسبة_المخاطرة الرمز\nمثال:\n/size 10000 1 NVDA"
        )
        return
    try:
        capital = float(context.args[0].replace(",", ""))
        risk_pct = float(context.args[1])
    except ValueError:
        await update.message.reply_text("أدخل أرقاماً صحيحة. مثال: /size 10000 1 NVDA")
        return

    symbol = context.args[2].upper() if len(context.args) >= 3 else None
    if not symbol:
        await update.message.reply_text("حدد الرمز. مثال: /size 10000 1 NVDA")
        return

    msg = await update.message.reply_text(f"جاري حساب الحجم لـ {symbol}...")
    try:
        sig = await asyncio.to_thread(analyze, symbol, display_name(symbol), True)
        pos = calc_position(capital, risk_pct, sig.price, sig.stop_loss)
        text = (
            f"{format_position_ar(pos)}\n\n"
            f"مرجع الإشارة: {sig.symbol} @ {sig.price:.2f} | درجة {sig.score}/100\n"
            f"منطقة الشراء {sig.buy_low:.2f}-{sig.buy_high:.2f} | وقف {sig.stop_loss:.2f} ({sig.sl_method})"
        )
        await msg.edit_text(text)
    except Exception as exc:
        await msg.edit_text(f"تعذر الحساب: {exc}")


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("مثال: /chart NVDA")
        return
    symbol = context.args[0].upper()
    msg = await update.message.reply_text(f"جاري تجهيز الرسم لـ {symbol}...")
    try:
        sig = await asyncio.to_thread(analyze, symbol, display_name(symbol), True)
        path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
        await msg.edit_text(format_signal_ar(sig, MIN_SCORE))
        if path and path.exists():
            with open(path, "rb") as f:
                await update.message.reply_photo(photo=InputFile(f, filename=path.name))
        else:
            await update.message.reply_text("تعذر إنشاء الرسم.")
    except Exception as exc:
        await msg.edit_text(f"تعذر الرسم: {exc}")


async def analyze_and_reply(update: Update, symbol: str) -> None:
    symbol = symbol.upper().replace("$", "").strip()
    if not symbol.isalnum() or len(symbol) > 6:
        await update.message.reply_text("رمز غير صالح. مثال: NVDA")
        return
    msg = await update.message.reply_text(f"جاري التحليل اليومي لـ {symbol}...")
    try:
        extra = ""
        if not is_known_halal(symbol):
            extra = "\n\n⚠️ ليس ضمن القائمة الحلال الافتراضية."
        near, edt = await asyncio.to_thread(is_near_earnings, symbol, EARNINGS_DAYS)
        if near:
            extra += f"\n\n🚫 قرب إعلان أرباح ({edt}) — تجنّب الدخول التلقائي."
        _, _, _, effective_min, market_label = await asyncio.to_thread(_daily_market_display_context)
        extra += f"\n\n{market_label} | نظام السوق"
        sig = await asyncio.to_thread(analyze, symbol, display_name(symbol), True)
        signal_min = max(MIN_SCORE, {"قوي": 82, "إيجابي_تحت_VWAP": 85, "مختلط": 86, "ضعيف": 92, "غير مؤكد": 92}.get(str(getattr(sig, "market_condition", "غير مؤكد") or "غير مؤكد"), effective_min))
        await msg.edit_text(format_signal_ar(sig, signal_min) + extra)
        path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
        if path and path.exists():
            with open(path, "rb") as f:
                await update.message.reply_photo(photo=InputFile(f, filename=path.name))
    except Exception as exc:
        log.exception("analyze failed")
        await msg.edit_text(f"تعذر التحليل: {exc}")


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("مثال: /analyze NVDA")
        return
    await analyze_and_reply(update, context.args[0])


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip().upper()
    if text in HALAL_STOCKS or (text.isalpha() and 1 <= len(text) <= 5):
        await analyze_and_reply(update, text)


async def run_scan_message(target_message, symbols: list[str]) -> None:
    if not is_us_trading_day():
        await target_message.reply_text(session_label())
        return
    if _scan_lock.locked():
        await target_message.reply_text("⏳ يوجد مسح يومي جارٍ الآن؛ انتظر نتيجته بدل بدء مسح موازٍ.")
        return
    async with _scan_lock:
        await _run_scan_message_locked(target_message, symbols)


async def _run_scan_message_locked(target_message, symbols: list[str]) -> None:
    global LAST_DAILY_SCAN_ATTEMPT
    LAST_DAILY_SCAN_ATTEMPT = now_ny()
    _, _, _, effective_min, market_label = await asyncio.to_thread(_daily_market_display_context)
    status = await target_message.reply_text(
        f"{session_label()}\n{market_label} | نظام السوق\n"
        f"جاري الترتيب (حد {MIN_SCORE} + بدون إعلانات قريبة)..."
    )
    # Analyzer owns the Daily market policy and adjusts the threshold itself.
    hits = await asyncio.to_thread(
        scan_symbols,
        symbols,
        HALAL_STOCKS,
        MIN_SCORE,
        DAILY_MAX,
    )
    if not hits:
        await status.edit_text(
            f"لا يوجد تأكيد {effective_min}+ حاليًا.\n\n{today_summary()}"
        )
        return
    skip_txt = ""
    await status.edit_text(f"أقوى {len(hits)} تأكيد:{skip_txt}")
    for i, sig in enumerate(hits, 1):
        signal_min = max(MIN_SCORE, {"قوي": 82, "إيجابي_تحت_VWAP": 85, "مختلط": 86, "ضعيف": 92, "غير مؤكد": 92}.get(str(getattr(sig, "market_condition", "غير مؤكد") or "غير مؤكد"), effective_min))
        await target_message.reply_text(format_signal_ar(sig, signal_min))
        path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
        if path and path.exists():
            with open(path, "rb") as f:
                await target_message.reply_photo(photo=InputFile(f, filename=path.name))
        await asyncio.sleep(0.3)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_scan_message(update.message, _watch(context))


async def cmd_scan_intra(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("جاري المسح اللحظي (ساعة + 5د)...")

    # Display-only session handling: never run an intraday scan while the
    # regular US session is closed, and do not show in-session VWAP/window
    # labels when the market is actually closed.
    if not is_us_regular_session():
        await msg.edit_text(
            f"🔴 السوق مغلق — لا يوجد مسح لحظي ولا تنبيهات\n"
            f"{session_label()}"
        )
        return

    ok, reason = session_window_ok()
    if not ok:
        await msg.edit_text(
            f"🟠 السوق مفتوح — {session_label()}\n"
            f"🔴 نافذة اللحظي غير مسموحة الآن — {reason}"
        )
        return

    if _intra_scan_lock.locked():
        await msg.edit_text("⏳ يوجد مسح لحظي جارٍ الآن؛ انتظر نتيجته بدل بدء مسح موازٍ.")
        return
    async with _intra_scan_lock:
        def _run():
            return scan_intraday(CORE_WATCHLIST, HALAL_STOCKS, INTRADAY_MIN_SCORE, 5)
        global LAST_INTRADAY_SCAN_ATTEMPT
        LAST_INTRADAY_SCAN_ATTEMPT = now_ny()
        hits = await asyncio.to_thread(_run)
    if not hits:
        _, _, market_condition = await asyncio.to_thread(get_intraday_market_context)
        market_labels = {
            "قوي": "🟢 قوي",
            "إيجابي_تحت_VWAP": "🟠 إيجابي",
            "مختلط": "🟡 مختلط",
            "ضعيف": "🔴 ضعيف",
            "غير مؤكد": "⚪️ غير مؤكد",
        }
        market_label = market_labels.get(market_condition, "⚪️ غير مؤكد")
        # No candidates is still an open-market result, so show the actual
        # market regime and the permitted intraday window.
        await msg.edit_text(
            f"لا مرشحين لحظيين الآن.\n{session_label()}\n"
            f"{market_label} | نظام السوق\n"
            f"حد {INTRADAY_MIN_SCORE} | نافذة اللحظي: 🟢 بعد الافتتاح وقبل الإغلاق"
        )
        return
    parts = [f"⚡ مسح لحظي — {session_label()}", f"السقف اليومي للحظي: {INTRADAY_MAX}"]
    for i, sig in enumerate(hits, 1):
        parts.append(format_intraday_ar(sig, INTRADAY_MIN_SCORE))
        if i == 1:
            parts[1] = parts[1]  # keep header
    await msg.edit_text("\n\n".join(parts[:4]))


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    data = q.data
    if data == "help":
        await q.message.reply_text(
            HELP.format(
                min_score=MIN_SCORE,
                daily_max=DAILY_MAX,
                interval=ALERT_EVERY_MINUTES,
                cool=COOLDOWN_DAYS,
            )
        )
    elif data == "list":
        lines = ["📋 الأسهم المعتمدة:\n"] + [f"• {s} — {n}" for s, n in HALAL_STOCKS.items()]
        await q.message.reply_text("\n".join(lines))
    elif data == "status":
        await q.message.reply_text(today_summary())
    elif data == "perf":
        text = await asyncio.to_thread(PERF.stats_text)
        await q.message.reply_text(text)
    elif data == "backtest":
        msg = await q.message.reply_text("جاري الباكتست...")
        text = await asyncio.to_thread(run_backtest, CORE_WATCHLIST[:12], "1y", MIN_SCORE)
        await msg.edit_text(text)
    elif data == "sub":
        SUBSCRIBERS.add(q.message.chat_id)
        save_subs(SUBSCRIBERS)
        await q.message.reply_text("تم الاشتراك.")
    elif data == "health":
        text = await asyncio.to_thread(health_text)
        await q.message.reply_text(text)
    elif data == "scan":
        await run_scan_message(q.message, CORE_WATCHLIST)


async def broadcast(bot, text: str) -> None:
    for chat_id in list(SUBSCRIBERS):
        try:
            await bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:
            log.warning("فشل الإرسال إلى %s: %s", chat_id, exc)


async def live_scan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """المسح التلقائي مع فلتر نظام السوق."""
    global LAST_SCAN_AT
    LAST_SCAN_AT = now_ny()
    if not SUBSCRIBERS or not is_us_regular_session() or remaining_slots() <= 0:
        return
    if _scan_lock.locked():
        return
    global LAST_DAILY_SCAN_ATTEMPT
    if LAST_DAILY_SCAN_ATTEMPT is not None:
        elapsed_attempt = (now_ny() - LAST_DAILY_SCAN_ATTEMPT).total_seconds() / 60.0
        if elapsed_attempt < SCAN_RETRY_MINUTES:
            return

    async with _scan_lock:
        LAST_DAILY_SCAN_ATTEMPT = now_ny()
        # Daily analyzer owns the market state/policy. Main does not override it
        # with the separate generic market.py regime.
        effective_min = AUTO.effective_floor(MIN_SCORE)

        state = load_state()
        already = set(state["sent"])
        if len(already) >= DAILY_MAX:
            return
        elapsed = minutes_since_last_alert(state)
        if elapsed is not None and elapsed < ALERT_EVERY_MINUTES:
            return

        # Daily V2 API: (symbols, names, min_score, limit).
        # Keep the automatic path aligned with the manual /scan command.
        hits = await asyncio.to_thread(
            scan_symbols,
            CORE_WATCHLIST,
            HALAL_STOCKS,
            effective_min,
            DAILY_MAX,
        )
        fresh = [
            s
            for s in hits
            if s.symbol not in already and not COOL.is_blocked(s.symbol)
        ]
        if not fresh:
            return

        sig = fresh[0]
        # لا نسجل الإشارة كـ"مُرسلة" قبل نجاح Telegram فعلياً.
        # هذا يمنع ضياع التنبيه إذا فشل الإرسال.
        slot = len(state["sent"]) + 1
        market_labels = {
            "قوي": "🟢 قوي",
            "إيجابي_تحت_VWAP": "🟠 إيجابي",
            "مختلط": "🟡 مختلط",
            "ضعيف": "🔴 ضعيف",
            "غير مؤكد": "⚪️ غير مؤكد",
        }
        signal_market_condition = str(getattr(sig, "market_condition", "") or "غير مؤكد")
        market_label = market_labels.get(signal_market_condition, "⚪️ غير مؤكد")
        header = (
            f"🔔 سوينغ/يومي — الدفعة {slot}/{DAILY_MAX}\n"
            f"{session_label()}\n"
            f"{market_label} | نظام السوق\n"
            f"النوع: يومي V2 (أسبوعي + يومي + 4س) | التالي بعد {ALERT_EVERY_MINUTES} د"
        )
        body = format_signal_ar(sig, effective_min)
        delivered = False
        for chat_id in list(SUBSCRIBERS):
            try:
                await context.bot.send_message(chat_id=chat_id, text=header + "\n\n" + body)
                delivered = True
                path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
                if path and path.exists():
                    with open(path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id, photo=InputFile(f, filename=path.name)
                        )
            except Exception as exc:
                log.warning("إرسال %s فشل: %s", chat_id, exc)

        if not delivered:
            log.warning("لم يُرسل التنبيه اليومي %s لأي مشترك؛ لن يُحسب ضمن الحصة", sig.symbol)
            return

        state["sent"].append(sig.symbol)
        state["scores"][sig.symbol] = sig.score
        state["last_sent_at"] = now_ny().isoformat()
        save_state(state)
        COOL.mark(sig.symbol)
        PERF.add_signal(sig, source="auto")
        if len(state["sent"]) >= DAILY_MAX:
            await broadcast(
                context.bot,
                f"✅ اكتملت حصة اليوم ({DAILY_MAX})\n" + " | ".join(state["sent"]),
            )


def minutes_since_last_intraday(state: dict) -> float | None:
    ts = state.get("last_sent_intraday_at")
    if not ts:
        return None
    try:
        last = datetime.fromisoformat(ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=now_ny().tzinfo)
        return (now_ny() - last).total_seconds() / 60.0
    except Exception:
        return None


async def live_scan_intraday_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """مسح لحظي معزول: ساعة + 5د | سقف 3 | خارج أول/آخر الجلسة."""
    if not SUBSCRIBERS or not is_us_regular_session():
        return
    ok, reason = session_window_ok()
    if not ok:
        return
    if _intra_scan_lock.locked():
        return
    global LAST_INTRADAY_SCAN_ATTEMPT
    if LAST_INTRADAY_SCAN_ATTEMPT is not None:
        elapsed_attempt = (now_ny() - LAST_INTRADAY_SCAN_ATTEMPT).total_seconds() / 60.0
        if elapsed_attempt < SCAN_RETRY_MINUTES:
            return

    async with _intra_scan_lock:
        LAST_INTRADAY_SCAN_ATTEMPT = now_ny()
        state = load_state()
        sent_i = list(state.get("sent_intraday") or [])
        if len(sent_i) >= INTRADAY_MAX:
            return
        elapsed = minutes_since_last_intraday(state)
        if elapsed is not None and elapsed < INTRADAY_EVERY_MINUTES:
            return

        hits = await asyncio.to_thread(
            scan_intraday,
            CORE_WATCHLIST,
            HALAL_STOCKS,
            INTRADAY_MIN_SCORE,
            10,
        )
        # يستبعد فقط ما أُرسل لحظياً اليوم (لا يمنع بسبب السوينغ)
        already = set(sent_i)
        fresh = [
            s
            for s in hits
            if s.symbol not in already and getattr(s, "entry_type", "") != "اختراق فاشل"
        ]
        if not fresh:
            return

        # scan_intraday رتّب جميع الاستراتيجيات الـ19 بالفعل؛ نحافظ على ترتيبه
        # ولا نفرض أولوية يدوية على 3 استراتيجيات فقط هنا.
        sig = fresh[0]
        # لا نسجل الإشارة كـ"مُرسلة" قبل نجاح Telegram فعلياً.
        # نأخذ سعرًا لحظيًا واحدًا وقت الإرسال لسعر الدخول والحساب والتعلّم،
        # مع إبقاء sig.price محفوظًا كسعر التحليل الأصلي.
        live_entry = await asyncio.to_thread(get_live_entry_price, sig.symbol)
        if live_entry > 0:
            sig.alert_entry_price = live_entry
        slot = len(state.get("sent_intraday") or []) + 1
        market_labels = {
            "قوي": "🟢 قوي",
            "إيجابي_تحت_VWAP": "🟠 إيجابي",
            "مختلط": "🟡 مختلط",
            "ضعيف": "🔴 ضعيف",
            "غير مؤكد": "⚪️ غير مؤكد",
        }
        signal_market_condition = str(getattr(sig, "market_condition", "") or "غير مؤكد")
        market_label = market_labels.get(signal_market_condition, "⚪️ غير مؤكد")
        header = (
            f"⚡ لحظي — الدفعة {slot}/{INTRADAY_MAX}\n"
            f"{session_label()}\n"
            f"{market_label} | نظام السوق\n"
            f"النوع: لحظي (ساعة + 5د)\n"
            f"{getattr(sig, 'entry_emoji', '🟢')} {getattr(sig, 'entry_type', 'دخول')}\n"
            f"فاصل {INTRADAY_EVERY_MINUTES} د | يفضّل الخروج قبل الإغلاق"
        )
        body = format_intraday_ar(sig, INTRADAY_MIN_SCORE)
        delivered = False
        for chat_id in list(SUBSCRIBERS):
            try:
                await context.bot.send_message(chat_id=chat_id, text=header + "\n\n" + body)
                delivered = True
            except Exception as exc:
                log.warning("إرسال لحظي %s فشل: %s", chat_id, exc)

        if not delivered:
            log.warning("لم يُرسل التنبيه اللحظي %s لأي مشترك؛ لن يُحسب ضمن الحصة", sig.symbol)
            return

        state.setdefault("sent_intraday", []).append(sig.symbol)
        state.setdefault("scores_intraday", {})[sig.symbol] = sig.score
        state["last_sent_intraday_at"] = now_ny().isoformat()
        save_state(state)
        PERF_INTRA.add_signal(sig, source="auto_intraday")


def build_daily_close_text() -> str:
    PERF.update_open_outcomes()
    state = load_state()
    sent = state.get("sent", [])
    scores = state.get("scores", {})
    day = now_ny().strftime("%Y-%m-%d")
    today_new, opened = PERF.today_closed_and_open(day)
    lines = [
        "🌆 ملخص ما بعد الإغلاق",
        session_label(),
        regime_label(),
        spy_day_change(),
        "",
        f"تنبيهات اليوم: {len(sent)}/{DAILY_MAX}",
    ]
    if sent:
        for i, sym in enumerate(sent, 1):
            lines.append(f"{i}. {sym}  {scores.get(sym, '—')}/100  — {display_name(sym)}")
    else:
        lines.append("لا تنبيهات تلقائية اليوم.")
    if opened:
        lines.append("")
        lines.append(f"إشارات ما زالت مفتوحة: {len(opened)}")
        for r in opened[-5:]:
            lines.append(f"• {r['symbol']} دخول {r['entry']} وقف {r['stop_loss']}")
    blocked = COOL.blocked_list()
    if blocked:
        lines.append("")
        lines.append("في تهدئة 5 أيام:")
        lines.append(" • ".join(s for s, _, __ in blocked[:8]))
    lines.append("")
    lines.append("تحليل تعليمي — ليست توصية.")
    return "\n".join(lines)


async def daily_close_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال ملخصي اليومي واللحظي بعد الإغلاق؛ العرض فقط ولا يغيّر التعلم."""
    if not SUBSCRIBERS or not is_post_close_window():
        return
    reports = load_reports()
    day = now_ny().strftime("%Y-%m-%d")

    if reports.get("daily_swing_performance_sent_on") != day:
        text = await asyncio.to_thread(PERF.daily_report, day)
        await broadcast(context.bot, text)
        reports["daily_swing_performance_sent_on"] = day

    if reports.get("daily_intraday_performance_sent_on") != day:
        text = await asyncio.to_thread(PERF_INTRA.daily_intraday_report, day)
        await broadcast(context.bot, text)
        reports["daily_intraday_performance_sent_on"] = day

    save_reports(reports)


async def monthly_performance_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال ملخص الشهر السابق لليومي واللحظي يوم 1 بعد الإغلاق."""
    if not SUBSCRIBERS or not is_post_close_window():
        return
    now = now_ny()
    if now.day != 1:
        return
    reports = load_reports()
    month_key = now.strftime("%Y-%m")

    if reports.get("monthly_swing_performance_sent_on") != month_key:
        text = await asyncio.to_thread(PERF.monthly_swing_report)
        await broadcast(context.bot, text)
        reports["monthly_swing_performance_sent_on"] = month_key

    if reports.get("monthly_intraday_performance_sent_on") != month_key:
        text = await asyncio.to_thread(PERF_INTRA.monthly_report)
        await broadcast(context.bot, text)
        reports["monthly_intraday_performance_sent_on"] = month_key

    save_reports(reports)



def _load_adaptive_policy_snapshot(path: str) -> dict:
    """قراءة Policy الحالية للعرض فقط؛ لا تعدّل أي ملف."""
    try:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("adaptive policy snapshot failed: %s", exc)
    return {}


def _policy_live_view(policy: dict) -> dict:
    """يستبعد بيانات السجل/الإحصاءات المتغيرة ويُبقي إعدادات الـPolicy الفعلية."""
    if not isinstance(policy, dict):
        return {}
    volatile = {
        "history", "strategy_stats", "sample_count", "monthly_cycle_total",
        "monthly_cycle_base_samples", "last_monthly_period",
        "last_monthly_sample_count", "generation", "created_at", "updated_at",
        "timestamp", "last_train_at", "last_optimization_at",
    }
    return {k: v for k, v in policy.items() if k not in volatile}


def _flatten_policy_changes(old: object, new: object, prefix: str = "") -> list[tuple[str, object, object]]:
    """فرق مختصر بين Policy القديمة والجديدة."""
    changes: list[tuple[str, object, object]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(set(old) | set(new))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                changes.append((path, "غير موجود", new[key]))
            elif key not in new:
                changes.append((path, old[key], "أزيل"))
            else:
                changes.extend(_flatten_policy_changes(old[key], new[key], path))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if old != new:
            changes.append((prefix or "قائمة", old, new))
        return changes
    if old != new:
        changes.append((prefix or "قيمة", old, new))
    return changes


def _arabic_policy_label(path: str) -> str:
    labels = {
        "weights": "الأوزان",
        "interaction_weights": "أوزان التفاعلات",
        "entry_limits": "حدود الدخول",
        "min_volume_ratio": "حد نسبة الحجم",
        "regime_weights": "أوزان حالة السوق",
        "exit_policy": "سياسة الخروج",
        "reason_policy": "سياسة الأسباب",
        "legacy_factor_bias": "انحياز العوامل القديم",
        "legacy_strategy_bias": "انحياز الاستراتيجية القديم",
        "regime_active": "تعلم حالة السوق",
        "exit_active": "تعلم الخروج",
        "approved": "حالة الاعتماد",
        "kill_switch": "مفتاح الإيقاف التكيفي",
    }
    tail = path.split(".")[-1]
    return labels.get(tail, path)


def _fmt_policy_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return raw[:180] + ("…" if len(raw) > 180 else "")
    return str(value)


def _policy_change_lines(old_policy: dict, new_policy: dict, limit: int = 12) -> list[str]:
    old_view = _policy_live_view(old_policy)
    new_view = _policy_live_view(new_policy)
    changes = _flatten_policy_changes(old_view, new_view)
    if not changes:
        return ["• لم يتغير أي إعداد في السياسة ظاهر."]
    lines = []
    for path, old, new in changes[:limit]:
        lines.append(
            f"• {_arabic_policy_label(path)}: "
            f"{_fmt_policy_value(old)} → {_fmt_policy_value(new)}"
        )
    if len(changes) > limit:
        lines.append(f"• ... وهناك {len(changes) - limit} تغييرات إضافية")
    return lines


def _adaptive_result_text(label: str, result: dict, old_policy: dict, new_policy: dict, cycle_samples: int = 0) -> str | None:
    status = str(result.get("status", "unknown"))
    samples = int(result.get("samples", 0) or 0)
    generation = int(result.get("generation", 0) or 0)
    old_rate = float(result.get("current_shadow_rate", 0) or 0) * 100
    new_rate = float(result.get("candidate_shadow_rate", 0) or 0) * 100
    coverage = float(result.get("coverage", 0) or 0) * 100
    prefix = f"🧠 التعلم التكيفي — {label}"

    if status == "approved":
        lines = [
            prefix,
            "",
            "🟢 تم اعتماد تعديل جديد وتطبيقه فعليًا",
            "",
            f"الجيل: #{generation}",
            f"الصفقات التراكمية: {cycle_samples or samples} / 100",
            "",
            "📊 اختبار خارج العينة",
            f"قبل: {old_rate:.1f}%",
            f"بعد: {new_rate:.1f}%",
            f"التغطية: {coverage:.1f}%",
            "",
            "🔧 ما الذي تم تعديله؟",
            *_policy_change_lines(old_policy, new_policy),
            "",
            "🧪 التحقق:",
            "✅ اجتاز اختبار خارج العينة",
            "✅ تم اعتماد النسخة الجديدة",
            "",
            "📌 الحالة:",
            "نشط — التعديل مطبق فعليًا",
        ]
        return "\n".join(lines)

    if status == "rejected":
        lines = [
            prefix,
            "",
            "❌ لم يتم تطبيق التعديل",
            "",
            f"الجيل المرشح: #{generation}",
            f"الصفقات التراكمية: {cycle_samples or samples} / 100",
            "",
            "📊 اختبار خارج العينة",
            f"الحالي: {old_rate:.1f}%",
            f"المرشح: {new_rate:.1f}%",
            f"التغطية: {coverage:.1f}%",
            "",
            "🔧 التغيير المقترح:",
            *_policy_change_lines(old_policy, new_policy),
            "",
            "🔒 السياسة الحالية مستمرة بدون تغيير",
            f"السبب: {result.get('message', 'لم تحقق النسخة شروط الاعتماد')}",
        ]
        return "\n".join(lines)

    if status == "rollback":
        from_gen = int(result.get("from_generation", 0) or 0)
        to_gen = int(result.get("to_generation", 0) or 0)
        lines = [
            prefix,
            "",
            "🔄 تم تنفيذ الاسترجاع",
            "",
            f"الجيل الذي تم التراجع عنه: #{from_gen}",
            f"الجيل المستعاد: #{to_gen}",
            "",
            "🔧 ما الذي تغير بعد الاسترجاع؟",
            *_policy_change_lines(old_policy, new_policy),
            "",
            "📌 الحالة:",
            "تم استعادة النسخة السابقة/الأفضل المحفوظة",
            "⚠️ تأثرت طبقات التعلم التكيفي فقط — الاستراتيجية الأساسية مستمرة",
        ]
        return "\n".join(lines)

    if status == "kill_switch":
        return "\n".join([
            prefix,
            "",
            "🚨 تم تفعيل مفتاح الإيقاف التكيفي",
            "",
            f"السبب: {result.get('reason', 'تراجع الأداء')}",
            "🔒 تم تعطيل طبقات التعلم التكيفي فقط",
            "✅ الاستراتيجية الأساسية مستمرة",
        ])

    return None


async def monthly_learning_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """تشغيل التعلم الجديد فقط، ثم إرسال نتيجة القرار الفعلي بوضوح."""
    reports = load_reports()
    try:
        # نأخذ snapshot قبل القرار للعرض فقط.
        intra_path = str(INTRADAY_ADAPTIVE_POLICY_FILE)
        daily_path = str(DAILY_ADAPTIVE_POLICY_FILE)
        intra_old = _load_adaptive_policy_snapshot(intra_path)
        daily_old = _load_adaptive_policy_snapshot(daily_path)
        intra_cycle_before = int(intra_old.get("adaptive_cycle_total", 0) or 0)
        daily_cycle_before = int(daily_old.get("adaptive_cycle_total", 0) or 0)

        # الـAnalyzer هو صاحب قرار التعلم والاعتماد؛ Main لا يغيّر Policy.
        try:
            intra_result = await asyncio.to_thread(monthly_self_optimization)
        except Exception as exc:
            log.exception("intraday monthly optimization failed: %s", exc)
            intra_result = {"status": "error", "samples": 0, "generation": 0}

        try:
            from analyzer import monthly_self_optimization as daily_monthly_self_optimization
            daily_result = await asyncio.to_thread(daily_monthly_self_optimization)
        except Exception as exc:
            log.exception("daily V2 monthly optimization failed: %s", exc)
            daily_result = {"status": "error", "samples": 0, "generation": 0}

        intra_new = _load_adaptive_policy_snapshot(intra_path)
        daily_new = _load_adaptive_policy_snapshot(daily_path)

        def _event_key(r: dict) -> str:
            return (
                f"{r.get('status','unknown')}|"
                f"{int(r.get('generation',0) or 0)}|"
                f"{int(r.get('from_generation',-1) or -1)}|"
                f"{int(r.get('to_generation',-1) or -1)}"
            )

        events = []

        intra_text = _adaptive_result_text(
            "اللحظي", intra_result, intra_old, intra_new, intra_cycle_before
        )
        intra_key = _event_key(intra_result)
        if intra_text and reports.get("adaptive_intraday_alert_key") != intra_key:
            events.append(intra_text)
            reports["adaptive_intraday_alert_key"] = intra_key

        daily_text = _adaptive_result_text(
            "اليومي", daily_result, daily_old, daily_new, daily_cycle_before
        )
        daily_key = _event_key(daily_result)
        if daily_text and reports.get("adaptive_daily_alert_key") != daily_key:
            events.append(daily_text)
            reports["adaptive_daily_alert_key"] = daily_key

        if events:
            await broadcast(context.bot, "\n\n" + "\n\n".join(events))
            log.info("adaptive learning event alert sent: %s", events)

        save_reports(reports)
    except Exception as exc:
        log.exception("monthly learning job: %s", exc)


async def weekly_report_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال ملخص الأسبوع لليومي واللحظي بعد إغلاق الجمعة."""
    if not SUBSCRIBERS or not is_friday_post_close():
        return
    reports = load_reports()
    week_key = now_ny().strftime("%Y-%W")

    if reports.get("weekly_swing_performance_sent_on") != week_key:
        text = await asyncio.to_thread(PERF.weekly_swing_report)
        await broadcast(context.bot, text)
        reports["weekly_swing_performance_sent_on"] = week_key

    if reports.get("weekly_intraday_performance_sent_on") != week_key:
        text = await asyncio.to_thread(PERF_INTRA.weekly_report)
        await broadcast(context.bot, text)
        reports["weekly_intraday_performance_sent_on"] = week_key

    save_reports(reports)


async def perf_update_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """متابعة الصفقات المفتوحة (سوينغ + لحظي): وقف / أهداف."""
    try:
        prefer_intraday = is_us_regular_session()

        def _all_events():
            evs = []
            for log_obj in (PERF, PERF_INTRA):
                try:
                    evs.extend(log_obj.update_open_outcomes(prefer_intraday))
                except Exception as exc:
                    log.warning("perf update one log: %s", exc)
            return evs

        events = await asyncio.to_thread(_all_events)
        if not events or not SUBSCRIBERS:
            return
        for ev in events:
            text = PerformanceLog.format_event_ar(ev)
            await broadcast(context.bot, text)
            await asyncio.sleep(0.4)
    except Exception as exc:
        log.warning("perf job: %s", exc)


def build_app() -> Application:
    if not BOT_TOKEN:
        raise SystemExit("ضع BOT_TOKEN في .env أو Render")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("scan_intra", cmd_scan_intra))
    app.add_handler(CommandHandler("scani", cmd_scan_intra))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("mywatch", cmd_mywatch))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("perf", cmd_perf))
    app.add_handler(CommandHandler("reopen", cmd_reopen))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("size", cmd_size))
    app.add_handler(CommandHandler("chart", cmd_chart))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("weekly", cmd_weekly))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    if app.job_queue:
        app.job_queue.run_repeating(live_scan_job, interval=LIVE_SCAN_SECONDS, first=25, name="live")
        app.job_queue.run_repeating(
            live_scan_intraday_job, interval=max(45, LIVE_SCAN_SECONDS), first=40, name="live-intra"
        )
        app.job_queue.run_repeating(perf_update_job, interval=180, first=90, name="perf")
        app.job_queue.run_repeating(daily_close_job, interval=300, first=40, name="daily-close")
        app.job_queue.run_repeating(weekly_report_job, interval=300, first=50, name="weekly")
        app.job_queue.run_repeating(monthly_performance_job, interval=300, first=60, name="monthly-performance")
        app.job_queue.run_daily(
            monthly_learning_job,
            time=dt_time(17, 15, tzinfo=ZoneInfo("America/New_York")),
            name="monthly-learning",
        )
    return app


def main() -> None:
    start_health_server()
    application = build_app()
    log.info(
        "جاهز | حد=%s | سقف=%s | كل %s د | إعلانات±%s | أوزان تكيفية + فلتر نظام السوق",
        MIN_SCORE,
        DAILY_MAX,
        ALERT_EVERY_MINUTES,
        EARNINGS_DAYS,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
