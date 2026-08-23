#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بوت تقدير سكراب قطع السيارات — نسخة مطورة مجانية
Gemini + Telegram | مناسب لـ Render
"""

import os
import json
import logging
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_ID = 678180992

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id SERIAL PRIMARY KEY,
                    part_id INTEGER UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    weight_kg VARCHAR(20),
                    weight_avg DECIMAL(6,2),
                    type VARCHAR(50),
                    buy_min DECIMAL(8,2),
                    buy_max DECIMAL(8,2),
                    sell_min DECIMAL(8,2),
                    sell_max DECIMAL(8,2),
                    notes TEXT DEFAULT ''
                );
            """)
            cur.execute("SELECT COUNT(*) FROM prices;")
            count = cur.fetchone()[0]
            if count == 0:
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                for p in catalog:
                    cur.execute("""
                        INSERT INTO prices (part_id, name, weight_kg, weight_avg, type, 
                                           buy_min, buy_max, sell_min, sell_max, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (part_id) DO NOTHING;
                    """, (
                        p["id"], p["name"], p.get("weight_kg"), p.get("weight_avg"),
                        p.get("type"), p.get("buy_min"), p.get("buy_max"),
                        p.get("sell_min"), p.get("sell_max"), p.get("notes", "")
                    ))
                log.info("تم إدخال بيانات الكتالوج في قاعدة البيانات")

def get_catalog_from_db():
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM prices ORDER BY part_id;")
            return [dict(row) for row in cur.fetchall()]

def build_catalog_text(catalog):
    lines = []
    for p in catalog:
        line = (f"{p['part_id']}. {p['name']} | وزن {p['weight_kg']} كجم | نوع: {p['type']} | "
                f"بيع {p['sell_min']}-{p['sell_max']} ر.س | شراء {p['buy_min']}-{p['buy_max']} ر.س")
        if p.get("notes"):
            line += f" | ملاحظة: {p['notes']}"
        lines.append(line)
    return "\n".join(lines)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "catalog.json"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("scrap-bot")

# جلسات المستخدمين: تجميع صور متعددة
user_sessions = defaultdict(lambda: {"photos": [], "last_report": None, "ts": 0})

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

CATALOG_TEXT = "\n".join(
    f"{p['id']}. {p['name']} | وزن {p['weight_kg']} كجم | نوع: {p['type']} | "
    f"بيع {p['sell_sar']} ر.س | شراء {p['buy_sar']} ر.س"
    + (f" | ملاحظة: {p['notes']}" if p.get("notes") else "")
    for p in CATALOG
)

SYSTEM_PROMPT = f"""أنت خبير سكراب قطع سيارات في السعودية (سوق الرياض والمنطقة).
مهمتك: تحليل صور كومة سكراب وتحديد القطع الظاهرة من الجدول فقط، ثم تقدير أسعار الشراء والبيع.

جدول القطع (ريال سعودي):
{CATALOG_TEXT}

أسعار تقريبية إضافية للحديد المختلط غير المفرز (للكيلو):
- حديد عادي/زهر: شراء 0.4–0.8 | بيع 0.8–1.5
- ألمنيوم نظيف: شراء 3–6 | بيع 6–10
- نحاس: شراء 15–25 | بيع 25–35
- رصاص بطاريات: حسب البطارية في الجدول

قواعد صارمة:
1. لا تخترع قطعاً غير ظاهرة.
2. قدّر العدد والحالة (سليمة / مكسورة / صدئة / ناقصة).
3. إذا الصورة غير واضحة قل ذلك بصدق وخفّض مستوى الثقة.
4. الماكينة والقير: كن متحفظاً جداً واذكر أن السعر حسب الوزن الحقيقي.
5. البلاستيك والديكور قيمة منخفضة.
6. إذا أرسل المستخدم أكثر من صورة لنفس الكومة، اعتبرها زوايا مختلفة لنفس الكومة وادمج التقدير.
7. الأسعار تقريبية حسب الجدول المعطى فقط.

صيغة الرد الإلزامية (عربي واضح):

🔎 **القطع الظاهرة**
- اسم القطعة × العدد — الحالة — شراء: س–ص ر.س — بيع: ع–غ ر.س

📦 **غير واضح / مختلط**
- ...

💰 **إجمالي تقديري للكومة**
• شراء تقريبي: من X إلى Y ريال
• بيع سكراب تقريبي: من A إلى B ريال
• هامش تقريبي: ...

📊 **مستوى الثقة:** منخفض / متوسط / جيد
⚠️ **تنبيه:** تقدير بصري فقط بدون ميزان. السوق يتغير. لا تعتمد عليه كعرض ملزم.
📝 ملاحظات: ...
"""

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
else:
    model = None


def main_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📷 أضف صورة ثانية", callback_data="add_more"),
            InlineKeyboardButton("✅ انتهى التحليل", callback_data="finish"),
        ],
        [
            InlineKeyboardButton("📋 الجدول", callback_data="show_table"),
            InlineKeyboardButton("🔄 تحليل جديد", callback_data="new_analysis"),
        ],
        [
            InlineKeyboardButton("ℹ️ مساعدة", callback_data="help"),
        ],
    ])


def analyze_images(image_list: list, user_note: str = "") -> str:
    """image_list: list of (bytes, mime)"""
    if not model:
        raise RuntimeError("مفتاح GEMINI_API_KEY غير موجود")

    parts = []
    note = user_note.strip()
    if note:
        parts.append(f"ملاحظة المستخدم: {note}")
    if len(image_list) > 1:
        parts.append(f"يوجد {len(image_list)} صور لنفس الكومة من زوايا مختلفة. ادمجها في تقدير واحد.")
    else:
        parts.append("حلّل صورة كومة السكراب هذه.")

    parts.append("فرز القطع حسب الجدول وقدّر الشراء والبيع. كن صادقاً إذا غير واضح.")

    content = [ "\n".join(parts) ]
    for data, mime in image_list:
        content.append({"mime_type": mime, "data": data})

    response = model.generate_content(
        content,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2800,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("ما رجع رد من Gemini. جرب صورة أوضح أو انتظر (حدود مجانية).")
    return text.strip()


START_TEXT = """🔧 *بوت تقدير سكراب السيارات* (نسخة مطورة)

أرسل صورة كومة الحديد/القطع.

المميزات:
• فرز القطع حسب جدولك (40 قطعة)
• دعم *عدة صور* لنفس الكومة
• تقدير شراء وبيع بالريال
• أزرار سريعة بعد كل تحليل

📌 نصائح:
- صوّر من زاويتين أو ثلاث
- قرّب البطارية والرديتر والدينمو والجنوط والمحرك
- إضاءة كويسة

أوامر:
/start — البداية
/table — الجدول
/clear — مسح الصور المجمّعة
/help — مساعدة
"""

HELP_TEXT = """*طريقة الاستخدام*

1. أرسل صورة الكومة.
2. إذا تبي صورة ثانية/ثالثة لنفس الكومة أرسلها مباشرة أو اضغط «أضف صورة ثانية».
3. اضغط «✅ انتهى التحليل» عشان يدمج كل الصور ويعطيك تقرير واحد.
4. أو أرسل صورة واحدة بس وهو يحللها فوراً.

الأسعار من جدولك. التقدير بصري فقط.

إذا طلع خطأ حد مجاني (429) انتظر شوي أو لين بكرة.
"""


def format_table_chunks():
    lines = []
    for p in CATALOG:
        lines.append(
            f"{p['id']}. {p['name']}\n"
            f"   وزن {p['weight_kg']} | شراء {p['buy_sar']} | بيع {p['sell_sar']}"
        )
    full = "📋 *ملخص الجدول*\n\n" + "\n".join(lines)
    # split if needed
    if len(full) <= 4000:
        return [full]
    mid = len(CATALOG) // 2
    p1 = "📋 الجدول (1/2)\n\n" + "\n".join(
        f"{p['id']}. {p['name']} | شراء {p['buy_sar']} | بيع {p['sell_sar']}"
        for p in CATALOG[:mid]
    )
    p2 = "📋 الجدول (2/2)\n\n" + "\n".join(
        f"{p['id']}. {p['name']} | شراء {p['buy_sar']} | بيع {p['sell_sar']}"
        for p in CATALOG[mid:]
    )
    return [p1, p2]


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_sessions[uid] = {"photos": [], "last_report": None, "ts": time.time()}
    await update.message.reply_text(START_TEXT, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


async def cmd_table(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for chunk in format_table_chunks():
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_sessions[uid] = {"photos": [], "last_report": None, "ts": time.time()}
    await update.message.reply_text("تم مسح الصور المجمّعة. أرسل صورة جديدة.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return
    if not GEMINI_API_KEY:
        await update.message.reply_text("مفتاح GEMINI_API_KEY غير موجود في الإعدادات.")
        return

    uid = update.effective_user.id
    session = user_sessions[uid]
    # تنظيف جلسات قديمة (أكثر من ساعة)
    if time.time() - session.get("ts", 0) > 3600:
        session["photos"] = []
        session["last_report"] = None

    photo = update.message.photo[-1]
    caption = (update.message.caption or "").strip()
    status = await update.message.reply_text("⏳ جاري تحميل الصورة...")

    try:
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        tg_file = await photo.get_file()
        data = bytes(await tg_file.download_as_bytearray())
        session["photos"].append((data, "image/jpeg"))
        session["ts"] = time.time()
        if caption:
            session["note"] = caption

        count = len(session["photos"])

        if count == 1:
            # تحليل فوري للصورة الأولى
            await status.edit_text("⏳ جاري الفرز والتقدير...")
            report = analyze_images(session["photos"], session.get("note", ""))
            session["last_report"] = report
            if len(report) > 4000:
                await status.edit_text(report[:4000])
                await update.message.reply_text(report[4000:8000], reply_markup=main_keyboard())
            else:
                await status.edit_text(report, reply_markup=main_keyboard())
        else:
            await status.edit_text(
                f"✅ تم إضافة الصورة رقم {count}.\n"
                f"عندك الآن {count} صور.\n"
                "اضغط «✅ انتهى التحليل» عشان أدمجها كلها، أو أرسل صورة إضافية.",
                reply_markup=main_keyboard(),
            )
    except Exception as e:
        log.exception("photo")
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate" in err:
            msg = "⚠️ وصلت الحد المجاني لـ Gemini.\nانتظر ساعة أو لين بكرة وحاول مرة ثانية."
        else:
            msg = f"تعذر التحليل:\n{e}"
        await status.edit_text(msg)


async def handle_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not (doc.mime_type or "").startswith("image/"):
        return
    # treat as photo path by converting conceptually — reuse photo logic via download
    if not GEMINI_API_KEY:
        await update.message.reply_text("مفتاح GEMINI غير موجود.")
        return
    uid = update.effective_user.id
    session = user_sessions[uid]
    status = await update.message.reply_text("⏳ جاري تحميل الملف...")
    try:
        tg_file = await doc.get_file()
        data = bytes(await tg_file.download_as_bytearray())
        mime = doc.mime_type or "image/jpeg"
        session["photos"].append((data, mime))
        session["ts"] = time.time()
        count = len(session["photos"])
        if count == 1:
            await status.edit_text("⏳ جاري الفرز...")
            report = analyze_images(session["photos"])
            session["last_report"] = report
            await status.edit_text(report[:4000] if len(report) > 4000 else report, reply_markup=main_keyboard())
            if len(report) > 4000:
                await update.message.reply_text(report[4000:8000])
        else:
            await status.edit_text(
                f"✅ تمت إضافة الصورة ({count}). اضغط «انتهى التحليل» أو أرسل المزيد.",
                reply_markup=main_keyboard(),
            )
    except Exception as e:
        await status.edit_text(f"خطأ: {e}")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = update.effective_user.id
    session = user_sessions[uid]
    data = query.data

    if data == "show_table":
        for chunk in format_table_chunks():
            await query.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        return

    if data == "help":
        await query.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)
        return

    if data == "new_analysis":
        session["photos"] = []
        session["last_report"] = None
        session.pop("note", None)
        await query.message.reply_text("تم. أرسل صورة كومة جديدة.")
        return

    if data == "add_more":
        await query.message.reply_text("تمام، أرسل الصورة الإضافية الحين.")
        return

    if data == "finish":
        if not session["photos"]:
            await query.message.reply_text("ما عندك صور مجمّعة. أرسل صورة أولاً.")
            return
        status = await query.message.reply_text(
            f"⏳ جاري دمج {len(session['photos'])} صور وتحليل نهائي..."
        )
        try:
            await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
            report = analyze_images(session["photos"], session.get("note", ""))
            session["last_report"] = report
            # لا نمسح الصور تلقائياً عشان يقدر يعيد
            if len(report) > 4000:
                await status.edit_text(report[:4000])
                await query.message.reply_text(report[4000:8000], reply_markup=main_keyboard())
            else:
                await status.edit_text(report, reply_markup=main_keyboard())
        except Exception as e:
            log.exception("finish")
            err = str(e).lower()
            if "429" in err or "quota" in err:
                await status.edit_text("⚠️ الحد المجاني امتلأ. حاول لاحقاً.")
            else:
                await status.edit_text(f"فشل التحليل: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أرسل صورة الكومة.\n/help للمساعدة — /table للجدول — /clear للمسح",
        reply_markup=main_keyboard(),
    )


def main():
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())

    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN مطلوب")
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY فارغ")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("table", cmd_table))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Bot started | model=%s", GEMINI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
async def cmd_prices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("هذا الأمر للأدمن فقط.")
        return
    try:
        catalog = get_catalog_from_db()
        text = "📋 *الأسعار الحالية من قاعدة البيانات:*\n\n"
        for p in catalog:
            text += f"{p['part_id']}. {p['name']}\n   شراء: {p['buy_min']}-{p['buy_max']} | بيع: {p['sell_min']}-{p['sell_max']}\n"
            if len(text) > 3500:
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
                text = ""
        if text:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"خطأ: {e}")

async def cmd_update_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("هذا الأمر للأدمن فقط.")
        return
    
    # الاستخدام: /update_price 1 3 5 6 18
    # المعنى: رقم القطعة | شراء أدنى | شراء أعلى | بيع أدنى | بيع أعلى
    args = context.args
    if len(args) != 5:
        await update.message.reply_text(
            "الاستخدام:\n/update_price رقم_القطعة شراء_أدنى شراء_أعلى بيع_أدنى بيع_أعلى\n\n"
            "مثال:\n/update_price 1 3 5 6 18"
        )
        return
    
    try:
        part_id = int(args[0])
        buy_min = float(args[1])
        buy_max = float(args[2])
        sell_min = float(args[3])
        sell_max = float(args[4])
        
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE prices 
                    SET buy_min = %s, buy_max = %s, sell_min = %s, sell_max = %s
                    WHERE part_id = %s
                """, (buy_min, buy_max, sell_min, sell_max, part_id))
                
                if cur.rowcount == 0:
                    await update.message.reply_text("ما لقيت قطعة بهذا الرقم.")
                    return
        
        await update.message.reply_text(f"✅ تم تحديث القطعة رقم {part_id} بنجاح.")
    except Exception as e:
        await update.message.reply_text(f"خطأ: {e}")
def main():
    init_db()

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN غير موجود")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # الأوامر
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("table", cmd_table))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("prices", cmd_prices))
    app.add_handler(CommandHandler("update_price", cmd_update_price))

    # استقبال الصور
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_image))

    # الأزرار
    app.add_handler(CallbackQueryHandler(on_callback))

    log.info("البوت بدأ التشغيل...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
