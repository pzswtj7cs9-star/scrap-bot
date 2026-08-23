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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
ADMIN_ID = 678180992

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "catalog.json"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("scrap-bot")

user_sessions = defaultdict(lambda: {"photos": [], "last_report": None, "ts": 0})

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

def build_catalog_text():
    return "\n".join(
        f"{p['id']}. {p['name']} | وزن {p['weight_kg']} كجم | نوع: {p['type']} | "
        f"بيع {p['sell_sar']} ر.س | شراء {p['buy_sar']} ر.س"
        + (f" | ملاحظة: {p['notes']}" if p.get("notes") else "")
        for p in CATALOG
    )

CATALOG_TEXT = build_catalog_text()

SYSTEM_PROMPT = f"""أنت خبير سكراب قطع سيارات محترف في السعودية (سوق الرياض).
مهمتك: تحليل صور كومة السكراب بدقة عالية.

أولاً: حدد كل القطع الظاهرة من الجدول فقط، واحسب عدد كل نوع بدقة.
ثانياً: صنف القطع حسب نوع المعدن (حديد/زهر - ألمنيوم - نحاس - رصاص - مختلط - بلاستيك).
ثالثاً: قدر أسعار الشراء والبيع.

جدول القطع المعتمد (ريال سعودي):
{CATALOG_TEXT}

أسعار تقريبية إضافية للكيلو (إذا القطعة غير موجودة في الجدول):
- حديد عادي/زهر: شراء 0.4–0.8 | بيع 0.8–1.5
- ألمنيوم نظيف: شراء 3–6 | بيع 6–10
- نحاس: شراء 15–25 | بيع 25–35
- رصاص بطاريات: حسب البطارية

قواعد صارمة جداً:
1. لا تخترع قطعاً غير ظاهرة بوضوح.
2. احسب العدد بدقة (مثال: 3 هوبات، 2 رديتر، 1 دينمو...).
3. إذا القطعة مكسورة أو صدئة أو ناقصة اذكر ذلك.
4. إذا الصورة غير واضحة قل ذلك بصدق وخفّض الثقة.
5. الماكينة والقير: كن متحفظاً جداً واذكر أن السعر يعتمد على الوزن الحقيقي.
6. إذا أرسل أكثر من صورة، اعتبرها زوايا لنفس الكومة وادمجها.

صيغة الرد الإلزامية (التزم بها تماماً):

🔎 **القطع الظاهرة (مع العدد)**
- اسم القطعة × العدد — الحالة — شراء: س–ص ر.س — بيع: ع–غ ر.س

📦 **غير واضح / مختلط**
- ...

🔩 **فرز حسب نوع المعدن**
• حديد / زهر: ...
• ألمنيوم: ...
• نحاس: ...
• رصاص: ...
• مختلط / أخرى: ...

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

    content = ["\n".join(parts)]
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
• عدّ القطع بدقة + فرز حسب نوع المعدن
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
/prices — عرض الأسعار (أدمن)
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


async def cmd_prices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("هذا الأمر للأدمن فقط.")
        return
    text = "📋 *الأسعار الحالية:*\n\n"
    for p in CATALOG:
        text += f"{p['id']}. {p['name']}\n   شراء: {p['buy_sar']} | بيع: {p['sell_sar']}\n"
        if len(text) > 3500:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            text = ""
    if text:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_update_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("هذا الأمر للأدمن فقط.")
        return

    args = context.args
    if len(args) != 3:
        await update.message.reply_text(
            "الاستخدام:\n/update_price رقم_القطعة سعر_الشراء سعر_البيع\n\n"
            "مثال:\n/update_price 1 3-5 6-18"
        )
        return

    try:
        part_id = int(args[0])
        buy_sar = args[1]
        sell_sar = args[2]

        found = False
        for p in CATALOG:
            if p["id"] == part_id:
                p["buy_sar"] = buy_sar
                p["sell_sar"] = sell_sar
                found = True
                break

        if not found:
            await update.message.reply_text("ما لقيت قطعة بهذا الرقم.")
            return

        global CATALOG_TEXT, SYSTEM_PROMPT, model
        CATALOG_TEXT = build_catalog_text()

        # إعادة بناء الـ Prompt
        SYSTEM_PROMPT = f"""أنت خبير سكراب قطع سيارات محترف في السعودية (سوق الرياض).
مهمتك: تحليل صور كومة السكراب بدقة عالية.

أولاً: حدد كل القطع الظاهرة من الجدول فقط، واحسب عدد كل نوع بدقة.
ثانياً: صنف القطع حسب نوع المعدن (حديد/زهر - ألمنيوم - نحاس - رصاص - مختلط - بلاستيك).
ثالثاً: قدر أسعار الشراء والبيع.

جدول القطع المعتمد (ريال سعودي):
{CATALOG_TEXT}

أسعار تقريبية إضافية للكيلو (إذا القطعة غير موجودة في الجدول):
- حديد عادي/زهر: شراء 0.4–0.8 | بيع 0.8–1.5
- ألمنيوم نظيف: شراء 3–6 | بيع 6–10
- نحاس: شراء 15–25 | بيع 25–35
- رصاص بطاريات: حسب البطارية

قواعد صارمة جداً:
1. لا تخترع قطعاً غير ظاهرة بوضوح.
2. احسب العدد بدقة (مثال: 3 هوبات، 2 رديتر، 1 دينمو...).
3. إذا القطعة مكسورة أو صدئة أو ناقصة اذكر ذلك.
4. إذا الصورة غير واضحة قل ذلك بصدق وخفّض الثقة.
5. الماكينة والقير: كن متحفظاً جداً واذكر أن السعر يعتمد على الوزن الحقيقي.
6. إذا أرسل أكثر من صورة، اعتبرها زوايا لنفس الكومة وادمجها.

صيغة الرد الإلزامية (التزم بها تماماً):

🔎 **القطع الظاهرة (مع العدد)**
- اسم القطعة × العدد — الحالة — شراء: س–ص ر.س — بيع: ع–غ ر.س

📦 **غير واضح / مختلط**
- ...

🔩 **فرز حسب نوع المعدن**
• حديد / زهر: ...
• ألمنيوم: ...
• نحاس: ...
• رصاص: ...
• مختلط / أخرى: ...

💰 **إجمالي تقديري للكومة**
• شراء تقريبي: من X إلى Y ريال
• بيع سكراب تقريبي: من A إلى B ريال
• هامش تقريبي: ...

📊 **مستوى الثقة:** منخفض / متوسط / جيد
⚠️ **تنبيه:** تقدير بصري فقط بدون ميزان. السوق يتغير. لا تعتمد عليه كعرض ملزم.
📝 ملاحظات: ...
"""

        if GEMINI_API_KEY:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
            )

        await update.message.reply_text(
            f"✅ تم تحديث القطعة رقم {part_id}\nشراء: {buy_sar} | بيع: {sell_sar}\n\n"
            "ملاحظة: التغيير مؤقت لين يعمل البوت Restart."
        )
    except Exception as e:
        await update.message.reply_text(f"خطأ: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return
    if not GEMINI_API_KEY:
        await update.message.reply_text("مفتاح GEMINI_API_KEY غير موجود في الإعدادات.")
        return

    uid = update.effective_user.id
    session = user_sessions[uid]
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
    app.add_handler(CommandHandler("prices", cmd_prices))
    app.add_handler(CommandHandler("update_price", cmd_update_price))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Bot started | model=%s", GEMINI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
