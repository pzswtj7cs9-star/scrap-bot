#!/usr/bin/env python3
"""
بوت تليجرام لأسهم السوق الأمريكي الحلال
- حد 85+ وتأكيد لحظي
- سهم واحد كل 90 دقيقة، بحد 5 يومياً
- سجل أداء + حاسبة حجم + رسم + تجنب إعلانات + باكتست
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timedelta
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

from analyzer import analyze, format_signal_ar, rank_all, scan_symbols
from backtest import run_backtest
from charting import build_signal_chart
from cooldown import CooldownBook
from earnings import is_near_earnings
from market import (
    is_friday_post_close,
    is_post_close_window,
    is_us_regular_session,
    now_ny,
    session_label,
)
from performance import PerformanceLog
from position import calc_position, format_position_ar
from stocks import CORE_WATCHLIST, HALAL_STOCKS, display_name, is_known_halal

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("halal-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "").strip()
MIN_SCORE = int(os.getenv("MIN_SCORE", "85"))
DAILY_MAX = int(os.getenv("DAILY_MAX_ALERTS", "5"))
ALERT_EVERY_MINUTES = int(os.getenv("ALERT_EVERY_MINUTES", "90"))
LIVE_SCAN_SECONDS = int(os.getenv("LIVE_SCAN_SECONDS", "60"))
EARNINGS_DAYS = int(os.getenv("EARNINGS_DAYS", "2"))
COOLDOWN_DAYS = int(os.getenv("COOLDOWN_DAYS", "5"))
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
SUBS_FILE = DATA_DIR / "subscribers.json"
STATE_FILE = DATA_DIR / "daily_state.json"
PERF_FILE = DATA_DIR / "signals_log.json"
COOL_FILE = DATA_DIR / "cooldown.json"
REPORTS_FILE = DATA_DIR / "reports_state.json"
CHART_DIR = DATA_DIR / "charts"

PERF = PerformanceLog(PERF_FILE)
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
    return {"date": day, "sent": [], "scores": {}, "last_sent_at": None}


def load_state() -> dict:
    day = now_ny().strftime("%Y-%m-%d")
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get("date") == day:
                data.setdefault("sent", [])
                data.setdefault("scores", {})
                data.setdefault("last_sent_at", None)
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


def load_reports() -> dict:
    if REPORTS_FILE.exists():
        try:
            return json.loads(REPORTS_FILE.read_text())
        except Exception:
            pass
    return {"daily_sent_on": "", "weekly_sent_on": ""}


def save_reports(data: dict) -> None:
    try:
        REPORTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.warning("reports state: %s", exc)


def ping_yahoo() -> tuple[bool, str]:
    global LAST_YF_OK, LAST_YF_NOTE
    try:
        import yfinance as yf

        info = yf.Ticker("SPY").fast_info
        last = getattr(info, "last_price", None)
        LAST_YF_OK = last is not None
        LAST_YF_NOTE = f"SPY ≈ {float(last):.2f}" if last else "بدون سعر"
        return LAST_YF_OK, LAST_YF_NOTE
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

بوت الأسهم الأمريكية الحلال — النسخة المتكاملة.

التنبيه التلقائي:
• حد 85/100 + تأكيد لحظي
• سهم واحد كل 90 دقيقة (الأقوى)
• سقف 5 أسهم يومياً
• يتجنب الأسهم قرب إعلان الأرباح

أدوات إضافية:
/size 10000 1 NVDA — حاسبة حجم الصفقة
/chart NVDA — رسم مع مناطق الشراء/الوقف/الأهداف
/perf — سجل أداء الإشارات
/backtest — باكتست مبسّط لآخر سنة
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
/perf — نتائج الإشارات المسجّلة
/backtest — اختبار تاريخي مبسّط

التنبيه:
/subscribe | /unsubscribe
/today | /status | /health | /weekly

الإعدادات الحالية:
حد التنبيه: {min_score}
سقف اليوم: {daily_max}
المباعدة: {interval} دقيقة
تهدئة السهم: {cool} أيام تداول
تجنب الإعلانات: خلال يومين من تاريخ الأرباح"""


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


def today_summary() -> str:
    state = load_state()
    sent = state["sent"]
    scores = state.get("scores", {})
    lines = [
        "📡 " + session_label(),
        f"🎯 حد التنبيه: {MIN_SCORE}/100",
        f"⏱ إرسال: سهم واحد كل {ALERT_EVERY_MINUTES} دقيقة",
        f"📦 حصة اليوم: {len(sent)}/{DAILY_MAX}",
        f"⏳ المتبقي: {remaining_slots()}",
        f"🕒 {next_alert_text(state)}",
        f"🚫 يتجنب الإعلانات خلال {EARNINGS_DAYS} يوم",
        f"🔁 لا يكرر السهم قبل {COOLDOWN_DAYS} أيام تداول",
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
    lines = [
        "🩺 صحة النظام",
        session_label(),
        f"التشغيل: {hours}س {mins}د",
        f"المشتركون: {len(SUBSCRIBERS)}",
        f"Yahoo Finance: {'يعمل' if ok else 'تعثر'} — {note}",
        f"آخر مسح: {last_scan}",
        f"آخر تنبيه تلقائي: {state.get('last_sent_at') or '—'}",
        f"حصة اليوم: {len(state.get('sent', []))}/{DAILY_MAX}",
        f"تهدئة الأسهم: {COOLDOWN_DAYS} أيام تداول",
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
    text = await asyncio.to_thread(PERF.stats_text)
    await msg.edit_text(text)


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text(
        "جاري الباكتست على آخر سنة (قد يستغرق دقيقة)..."
    )
    text = await asyncio.to_thread(run_backtest, CORE_WATCHLIST[:12], "1y", MIN_SCORE)
    await msg.edit_text(text)


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/size 10000 1 NVDA  أو  /size 10000 1  مع تحليل مسبق"""
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
            f"منطقة الشراء {sig.buy_low:.2f}-{sig.buy_high:.2f} | وقف {sig.stop_loss:.2f}"
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
    msg = await update.message.reply_text(f"جاري التحليل اللحظي لـ {symbol}...")
    try:
        extra = ""
        if not is_known_halal(symbol):
            extra = "\n\n⚠️ ليس ضمن القائمة الحلال الافتراضية."
        near, edt = await asyncio.to_thread(is_near_earnings, symbol, EARNINGS_DAYS)
        if near:
            extra += f"\n\n🚫 قرب إعلان أرباح ({edt}) — تجنّب الدخول التلقائي."
        sig = await asyncio.to_thread(analyze, symbol, display_name(symbol), True)
        await msg.edit_text(format_signal_ar(sig, MIN_SCORE) + extra)
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
    status = await target_message.reply_text(
        f"{session_label()}\nجاري الترتيب (حد {MIN_SCORE} + بدون إعلانات قريبة)..."
    )
    hits = await asyncio.to_thread(
        scan_symbols, symbols, HALAL_STOCKS, MIN_SCORE, True, DAILY_MAX, True, EARNINGS_DAYS
    )
    skipped = getattr(scan_symbols, "last_skipped_earnings", [])
    if not hits:
        ranked = await asyncio.to_thread(rank_all, symbols, HALAL_STOCKS)
        top = ranked[:5]
        extra = ""
        if top:
            extra = "\n\nأقرب المرشحين:"
            for i, s in enumerate(top, 1):
                extra += f"\n{i}. {s.symbol} {s.score}/100"
        skip_txt = f"\nتم استبعاد قرب أرباح: {', '.join(skipped)}" if skipped else ""
        await status.edit_text(
            f"لا يوجد تأكيد 85+ مع لحظة حالياً.{skip_txt}{extra}\n\n{today_summary()}"
        )
        return
    skip_txt = f"\n(استُبعد قرب أرباح: {', '.join(skipped)})" if skipped else ""
    await status.edit_text(f"أقوى {len(hits)} تأكيد:{skip_txt}")
    for i, sig in enumerate(hits, 1):
        await target_message.reply_text(format_signal_ar(sig, MIN_SCORE, rank=i))
        path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
        if path and path.exists():
            with open(path, "rb") as f:
                await target_message.reply_photo(photo=InputFile(f, filename=path.name))
        await asyncio.sleep(0.3)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await run_scan_message(update.message, _watch(context))


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
    global LAST_SCAN_AT
    LAST_SCAN_AT = now_ny()
    if not SUBSCRIBERS or not is_us_regular_session() or remaining_slots() <= 0:
        return
    if _scan_lock.locked():
        return

    async with _scan_lock:
        state = load_state()
        already = set(state["sent"])
        if len(already) >= DAILY_MAX:
            return
        elapsed = minutes_since_last_alert(state)
        if elapsed is not None and elapsed < ALERT_EVERY_MINUTES:
            return

        hits = await asyncio.to_thread(
            scan_symbols,
            CORE_WATCHLIST,
            HALAL_STOCKS,
            MIN_SCORE,
            True,
            12,
            True,
            EARNINGS_DAYS,
        )
        fresh = [
            s
            for s in hits
            if s.symbol not in already and not COOL.is_blocked(s.symbol)
        ]
        if not fresh:
            return

        sig = fresh[0]
        state["sent"].append(sig.symbol)
        state["scores"][sig.symbol] = sig.score
        state["last_sent_at"] = now_ny().isoformat()
        save_state(state)
        COOL.mark(sig.symbol)
        PERF.add_signal(sig, source="auto")

        slot = len(state["sent"])
        header = (
            f"🔔 سهم واحد — الدفعة {slot}/{DAILY_MAX}\n"
            f"{session_label()}\n"
            f"التالي بعد {ALERT_EVERY_MINUTES} دقيقة إن وُجد تأكيد جديد."
        )
        body = format_signal_ar(sig, MIN_SCORE, rank=slot)
        for chat_id in list(SUBSCRIBERS):
            try:
                await context.bot.send_message(chat_id=chat_id, text=header + "\n\n" + body)
                path = await asyncio.to_thread(build_signal_chart, sig, CHART_DIR)
                if path and path.exists():
                    with open(path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id, photo=InputFile(f, filename=path.name)
                        )
            except Exception as exc:
                log.warning("إرسال %s فشل: %s", chat_id, exc)

        if len(state["sent"]) >= DAILY_MAX:
            await broadcast(
                context.bot,
                f"✅ اكتملت حصة اليوم ({DAILY_MAX})\n" + " | ".join(state["sent"]),
            )


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
    if not SUBSCRIBERS or not is_post_close_window():
        return
    reports = load_reports()
    day = now_ny().strftime("%Y-%m-%d")
    if reports.get("daily_sent_on") == day:
        return
    text = await asyncio.to_thread(build_daily_close_text)
    await broadcast(context.bot, text)
    reports["daily_sent_on"] = day
    save_reports(reports)


async def weekly_report_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not SUBSCRIBERS or not is_friday_post_close():
        return
    reports = load_reports()
    week_key = now_ny().strftime("%Y-%W")
    if reports.get("weekly_sent_on") == week_key:
        return
    text = await asyncio.to_thread(PERF.weekly_report)
    await broadcast(context.bot, text)
    reports["weekly_sent_on"] = week_key
    save_reports(reports)


async def perf_update_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        changed = await asyncio.to_thread(PERF.update_open_outcomes)
        if changed and SUBSCRIBERS:
            lines = ["📋 تحديث نتائج إشارات:"]
            for r in changed[-5:]:
                lines.append(
                    f"• {r['symbol']}: {r.get('result')} | {r.get('pnl_pct'):+.2f}%"
                )
            await broadcast(context.bot, "\n".join(lines))
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
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("mywatch", cmd_mywatch))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("perf", cmd_perf))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("size", cmd_size))
    app.add_handler(CommandHandler("chart", cmd_chart))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("weekly", cmd_weekly))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    if app.job_queue:
        app.job_queue.run_repeating(live_scan_job, interval=LIVE_SCAN_SECONDS, first=25, name="live")
        app.job_queue.run_repeating(perf_update_job, interval=3600, first=120, name="perf")
        app.job_queue.run_repeating(daily_close_job, interval=300, first=40, name="daily-close")
        app.job_queue.run_repeating(weekly_report_job, interval=300, first=50, name="weekly")
    return app


def main() -> None:
    start_health_server()
    application = build_app()
    log.info(
        "جاهز | حد=%s | سقف=%s | كل %s د | إعلانات±%s",
        MIN_SCORE,
        DAILY_MAX,
        ALERT_EVERY_MINUTES,
        EARNINGS_DAYS,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
