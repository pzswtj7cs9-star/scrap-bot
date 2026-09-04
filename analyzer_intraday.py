"""
مسار المضاربة اللحظية (معزول عن السوينغ اليومي).
- اتجاه: ساعة
- تأكيد: 5 دقائق
- فلاتر جلسة: تجنب أول/آخر الجلسة، فوق الافتتاح أو VWAP، حجم نسبي
- سقف مستقل وتعلم مستقل عبر سجل منفصل من main
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from market import now_ny, REGULAR_OPEN, REGULAR_CLOSE
from stocks import MAX_AUTO_PRICE

# نافذة الإرسال اللحظي (بتوقيت نيويورك)
SKIP_OPEN_MIN = 20
SKIP_CLOSE_MIN = 20
INTRADAY_MIN_SCORE = 82


@dataclass
class IntradaySignal:
    symbol: str
    name: str
    price: float
    change_pct: float
    score: int
    grade: str
    buy_low: float
    buy_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_pct: float
    reward_r: float
    sl_method: str
    vwap_day_note: str
    above_open: bool
    vol_session_ok: bool
    reasons: list[str]
    warnings: list[str]
    mode: str = "intraday"
    entry_type: str = "دخول مبكر"
    entry_emoji: str = "🟢"
    structure_zone: str = "محايدة"
    quality_ok: bool = True
    live_ok: bool = True
    volume_ratio: float = 1.0
    regime: str = "neutral"
    factor_keys: list | None = None
    sma20: float = 0.0
    atr_pct: float = 0.0
    ext_sma20: float = 0.0


def session_window_ok(dt=None) -> tuple[bool, str]:
    """يمنع أول 20 دقيقة وآخر 20 دقيقة من الجلسة."""
    dt = dt or now_ny()
    t = dt.time()
    open_ok_after = time(9, 30 + SKIP_OPEN_MIN // 60, SKIP_OPEN_MIN % 60)
    # 9:50
    open_ok_after = time(9, 50)
    close_cut = time(15, 40)  # قبل الإغلاق بـ 20 دقيقة
    if t < open_ok_after:
        return False, "داخل أول 20 دقيقة — ضجيج افتتاح"
    if t > close_cut:
        return False, "آخر 20 دقيقة — تجنب التبييت"
    if t < REGULAR_OPEN or t > REGULAR_CLOSE:
        return False, "خارج الجلسة النظامية"
    return True, "نافذة لحظية مسموحة"


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def _grade(score: int) -> str:
    if score >= 95:
        return "A++"
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 75:
        return "B"
    return "C"


def analyze_intraday(symbol: str, name: str = "") -> Optional[IntradaySignal]:
    from market_data import fetch_intraday

    # ساعة للاتجاه
    h1 = fetch_intraday(symbol, interval="60m", period="10d")
    if h1 is None or len(h1) < 40:
        return None
    # 5د للتأكيد + VWAP اليوم
    m5 = fetch_intraday(symbol, interval="5m", period="5d")
    if m5 is None or len(m5) < 30:
        return None

    last_day = m5.index[-1].date()
    today_5 = m5[m5.index.date == last_day]
    if len(today_5) < 6:
        return None

    price = float(today_5["Close"].iloc[-1])
    if price <= 0 or price > float(MAX_AUTO_PRICE):
        return None

    day_open = float(today_5["Open"].iloc[0])
    prev_close = float(m5["Close"].iloc[-2]) if len(m5) > 1 else price
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0

    # VWAP اليوم
    vwap_s = _vwap(today_5)
    vwap_last = float(vwap_s.iloc[-1]) if pd.notna(vwap_s.iloc[-1]) else price
    above_vwap = price >= vwap_last
    vwap_note = "فوق VWAP اليوم" if above_vwap else "تحت VWAP اليوم"
    above_open = price >= day_open

    # حجم الجلسة النسبي: متوسط حجم شموع اليوم vs متوسط 5د السابق
    vol_today = float(today_5["Volume"].sum())
    bars = max(len(today_5), 1)
    avg_bar_today = vol_today / bars
    vol_hist = float(m5["Volume"].tail(60).mean() or 1)
    vol_session_ratio = avg_bar_today / vol_hist if vol_hist else 1.0
    vol_session_ok = vol_session_ratio >= 0.90

    # اتجاه الساعة
    hc = h1["Close"]
    ema20 = _ema(hc, 20)
    ema50 = _ema(hc, 50)
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    h_rsi = float(_rsi(hc, 14).iloc[-1])
    trend_up = price > e20 > e50 * 0.998 and h_rsi >= 45

    # تأكيد 5د
    c5 = today_5["Close"]
    e5 = float(_ema(c5, 20).iloc[-1])
    r5 = float(_rsi(c5, 14).iloc[-1])
    last_green = float(today_5["Close"].iloc[-1]) >= float(today_5["Open"].iloc[-1])
    mom = (price - float(c5.iloc[-6])) / float(c5.iloc[-6]) * 100 if len(c5) >= 6 else 0
    live_ok = price >= e5 * 0.998 and above_vwap and (last_green or mom > 0.05) and r5 < 78

    # سقوط حاد من قمة الجلسة
    session_high = float(today_5["High"].max())
    drop = (session_high - price) / session_high * 100 if session_high else 0
    dump = drop >= 2.5 and change_pct <= -1.2

    # —— تصنيف نوع الدخول اللحظي ——
    h_win = h1.tail(20)
    level_high = float(h_win["High"].iloc[:-1].max()) if len(h_win) > 3 else session_high
    was_below = float(h_win["Close"].iloc[-3]) < level_high * 0.998 if len(h_win) >= 3 else False
    breakout_now = price >= level_high * 1.001 and was_below
    prior_break = (
        float(h_win["High"].iloc[-8:-2].max()) >= level_high * 0.999 if len(h_win) >= 8 else False
    )
    near_level = abs(price - level_high) / max(price, 1e-9) * 100 <= 0.7
    e20_tmp = float(_ema(h1["Close"], 20).iloc[-1])
    ext_tmp = (price - e20_tmp) / e20_tmp * 100 if e20_tmp else 0.0
    failed = (
        float(today_5["High"].max()) >= level_high * 1.001
        and price < level_high * 0.997
        and not above_vwap
    ) or (dump and not above_vwap)
    retest = prior_break and near_level and price >= level_high * 0.997 and above_vwap and not failed
    early = above_vwap and above_open and not breakout_now and ext_tmp <= 2.2 and not failed

    if failed:
        entry_type, entry_emoji = "اختراق فاشل", "🔴"
    elif retest:
        entry_type, entry_emoji = "إعادة اختبار", "🟡"
    elif breakout_now and vol_session_ratio >= 1.0:
        entry_type, entry_emoji = "اختراق مؤكد", "🟢"
    elif early or (above_vwap and trend_up and ext_tmp <= 2.5):
        entry_type, entry_emoji = "دخول مبكر", "🟢"
    else:
        entry_type, entry_emoji = "دخول مبكر", "🟢"

    reasons = []
    warnings = []
    score = 50.0
    factors = []

    if trend_up:
        score += 18
        reasons.append("اتجاه الساعة صاعد (السعر وEMA20 فوق EMA50)")
        factors.append("h1_trend")
    else:
        warnings.append("اتجاه الساعة غير مؤكد")
        score -= 8

    if above_vwap:
        score += 10
        reasons.append("فوق VWAP اليوم")
        factors.append("vwap")
    else:
        warnings.append("تحت VWAP اليوم")
        score -= 12

    if above_open:
        score += 6
        reasons.append("فوق سعر افتتاح اليوم")
        factors.append("above_open")
    else:
        warnings.append("تحت افتتاح اليوم")
        score -= 4

    if live_ok:
        score += 12
        reasons.append("تأكيد 5 دقائق")
        factors.append("m5")
    else:
        warnings.append("لا تأكيد 5د كافٍ")
        score -= 10

    if vol_session_ok:
        score += 6
        reasons.append(f"حجم جلسة نسبي مقبول ({vol_session_ratio:.2f}x)")
        factors.append("vol_session")
    else:
        warnings.append("حجم الجلسة ضعيف نسبياً")
        score -= 6

    if 48 <= h_rsi <= 68:
        score += 5
        factors.append("rsi_h1")

    if dump:
        warnings.append("سقوط من قمة الجلسة")
        score -= 20

    if entry_type == "اختراق فاشل":
        warnings.append("اختراق فاشل — لا إرسال تلقائي")
        score -= 25
    elif entry_type == "اختراق مؤكد":
        score += 8
        reasons.append("اختراق مؤكد لمستوى الساعة")
        factors.append("breakout")
    elif entry_type == "إعادة اختبار":
        score += 5
        reasons.append("إعادة اختبار مستوى بعد اختراق")
        factors.append("retest")
    elif entry_type == "دخول مبكر":
        reasons.append("دخول مبكر فوق VWAP")
        factors.append("early")

    # امتداد عن ema20 الساعة
    ext = (price - e20) / e20 * 100 if e20 else 0
    if ext > 4.0:
        warnings.append("امتداد عن متوسط الساعة")
        score -= 8

    atr = float(_atr(h1, 14).iloc[-1] or price * 0.01)
    atr_pct = atr / price * 100
    if atr_pct > 6.0:
        warnings.append("تذبذب عالي")
        score -= 5

    score_i = int(max(0, min(100, round(score))))
    quality_ok = (
        (not dump)
        and (not failed)
        and entry_type != "اختراق فاشل"
        and ext <= 4.5
        and atr_pct <= 6.5
        and vol_session_ratio >= 0.85
    )

    # وقف أوسع شوي (1.5 ATR) + أهداف أقرب لنفس اليوم
    recent_low = float(today_5["Low"].tail(12).min())
    stop = min(price - 1.5 * atr, recent_low * 0.997)
    risk = price - stop
    if risk <= 0 or risk / price < 0.012:
        stop = price * (1 - 0.012)
        risk = price - stop
    if risk / price > 0.07:
        stop = price * 0.93
        risk = price - stop
    tp1 = price + risk * 1.2
    tp2 = price + risk * 1.8
    tp3 = price + risk * 2.6
    risk_pct = risk / price * 100
    reward_r = (tp2 - price) / risk if risk else 0

    buy_low = max(stop * 1.01, min(price * 0.995, e5))
    buy_high = price * 1.004

    return IntradaySignal(
        symbol=symbol,
        name=name or symbol,
        price=round(price, 4),
        change_pct=round(change_pct, 2),
        score=score_i,
        grade=_grade(score_i),
        buy_low=round(buy_low, 4),
        buy_high=round(buy_high, 4),
        stop_loss=round(stop, 4),
        tp1=round(tp1, 4),
        tp2=round(tp2, 4),
        tp3=round(tp3, 4),
        risk_pct=round(risk_pct, 2),
        reward_r=round(reward_r, 2),
        sl_method="ATR/قاع جلسة",
        vwap_day_note=vwap_note,
        above_open=above_open,
        vol_session_ok=vol_session_ok,
        reasons=reasons[:4],
        warnings=warnings[:3],
        quality_ok=quality_ok,
        live_ok=live_ok,
        volume_ratio=round(vol_session_ratio, 2),
        factor_keys=factors,
        sma20=round(e20, 4),
        atr_pct=round(atr_pct, 2),
        ext_sma20=round(ext, 2),
        entry_type=entry_type,
        entry_emoji=entry_emoji,
    )


def format_intraday_ar(sig: IntradaySignal, min_score: int = INTRADAY_MIN_SCORE) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    lines = [
        f"⚡ لحظي | {sig.symbol} | {sig.score}/100 | {sig.grade} | ساعة+5د",
        f"{sig.entry_emoji} نوع الدخول: {sig.entry_type}",
        f"{sig.name}",
        "—————————————",
        f"السعر: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
        f"شراء: {sig.buy_low:.2f} — {sig.buy_high:.2f}",
        f"وقف: {sig.stop_loss:.2f} ({sig.sl_method})  |  مخاطرة {sig.risk_pct:.2f}%",
        f"أهداف: {sig.tp1:.2f}  |  {sig.tp2:.2f}  |  {sig.tp3:.2f}",
        "—————————————",
        f"{sig.vwap_day_note} | افتتاح اليوم: {'فوق' if sig.above_open else 'تحت'} | حجم جلسة: {'نعم' if sig.vol_session_ok else 'لا'}",
    ]
    if sig.reasons:
        lines.append("لماذا: " + " | ".join(sig.reasons[:2]))
    if sig.warnings:
        lines.append("مخاطر: " + " | ".join(sig.warnings[:2]))
    if sig.score < min_score or not sig.live_ok or not sig.quality_ok:
        lines.append(f"تحت شرط الإرسال اللحظي ({min_score}+ / تأكيد / جودة)")
    lines.append("تحليل لحظي تعليمي — ليست توصية. يفضّل الخروج قبل الإغلاق.")
    return "\n".join(lines)


def scan_intraday(
    symbols: list[str],
    names: dict,
    min_score: int = INTRADAY_MIN_SCORE,
    limit: int = 8,
) -> list[IntradaySignal]:
    ok, _ = session_window_ok()
    if not ok:
        scan_intraday.last_window = _
        return []
    scan_intraday.last_window = "ok"

    # فلتر سوق خفيف عبر SPY الساعة
    try:
        spy = analyze_intraday("SPY", "SPY")
        if spy and spy.score < 55 and not spy.above_open:
            scan_intraday.last_window = "SPY لحظي ضعيف"
            # لا نوقف كلياً؛ نرفع الحد
            min_score = max(min_score, 88)
    except Exception:
        pass

    results: list[IntradaySignal] = []
    for sym in symbols:
        try:
            sig = analyze_intraday(sym, names.get(sym, sym))
            if not sig:
                continue
            if sig.score < min_score:
                continue
            if not sig.live_ok or not sig.quality_ok:
                continue
            # لا إرسال لاختراق فاشل
            if getattr(sig, "entry_type", "") == "اختراق فاشل":
                continue
            # شرط أقوى: فوق VWAP (مطلوب للإرسال)
            if "تحت" in sig.vwap_day_note:
                continue
            results.append(sig)
        except Exception:
            continue
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


scan_intraday.last_window = ""
