"""
محرك التحليل المحسّن:
- اتجاه يومي
- تأكيد إطار 15 دقيقة
- تأكيد لحظي 5 دقائق
- وقف خسارة محسّن (هيكل + ATR ديناميكي + قاع 15م)
- أوزان تكيفية من سجل الأداء
- فلتر نظام السوق (Market Regime)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from weights import WEIGHTS


@dataclass
class SignalResult:
    symbol: str
    name: str
    price: float
    change_pct: float
    score: int
    grade: str
    bias: str
    buy_low: float
    buy_high: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_pct: float
    reward_r: float
    rsi: float
    macd_hist: float
    sma20: float
    sma50: float
    sma200: float
    atr: float
    volume_ratio: float
    reasons: list[str]
    warnings: list[str]
    market_cap: Optional[str] = None
    live_price: Optional[float] = None
    live_rsi: Optional[float] = None
    live_vs_vwap: Optional[float] = None
    live_ok: bool = False
    m15_ok: bool = False
    rank_key: float = 0.0
    timeframe_note: str = "يومي"
    factor_keys: list[str] = field(default_factory=list)
    regime: str = "neutral"
    sl_method: str = ""
    quality_ok: bool = True
    atr_pct: float = 0.0
    ext_sma20: float = 0.0


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vp = (typical * df["Volume"]).cumsum()
    cum_v = df["Volume"].cumsum().replace(0, np.nan)
    return cum_vp / cum_v


def _fmt_cap(cap: Optional[float]) -> Optional[str]:
    if not cap or cap <= 0:
        return None
    if cap >= 1e12:
        return f"{cap / 1e12:.2f} تريليون $"
    if cap >= 1e9:
        return f"{cap / 1e9:.1f} مليار $"
    if cap >= 1e6:
        return f"{cap / 1e6:.0f} مليون $"
    return str(int(cap))


def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    from market_data import fetch_history as _fh

    df = _fh(symbol, period=period)
    if df is None or df.empty or len(df) < 60:
        raise ValueError(f"لا توجد بيانات كافية لـ {symbol}")
    return df.dropna(subset=["Close", "High", "Low", "Volume"])


def fetch_intraday(symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    from market_data import fetch_intraday as _fi

    df = _fi(symbol, period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"لا توجد بيانات لحظية لـ {symbol}")
    return df.dropna(subset=["Close", "High", "Low", "Volume"])


def fetch_info(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol).fast_info
        return {
            "last": getattr(info, "last_price", None),
            "market_cap": getattr(info, "market_cap", None),
            "currency": getattr(info, "currency", "USD"),
        }
    except Exception:
        return {}


def _grade(score: int) -> tuple[str, str]:
    if score >= 90:
        return "A++", "أقوى تأكيد صعود متاح اليوم"
    if score >= 85:
        return "A+", "تأكيد صعود عالي التوافق"
    if score >= 78:
        return "A", "إشارة جيدة لكنها تحت حد التنبيه"
    if score >= 68:
        return "B", "مراقبة / انتظار تأكيد"
    if score >= 55:
        return "C", "محايد"
    return "D", "لا توجد إشارة شراء"


def _compute_improved_stop(
    price: float,
    atr: float,
    swing_low: float,
    recent_low: float,
    m15_low: Optional[float],
    score: int,
) -> tuple[float, str]:
    """
    وقف خسارة محسّن:
    1) قاع هيكلي يومي
    2) ATR ديناميكي (أضيق مع الدرجات العالية)
    3) قاع إطار 15 دقيقة إن وُجد
    يختار الأكثر منطقية مع حدود دنيا/عليا.
    """
    # مضاعف ATR أقل عندما تكون الإشارة أقوى (ثقة أعلى → وقف أقرب)
    if score >= 90:
        atr_mult = 1.35
    elif score >= 85:
        atr_mult = 1.50
    else:
        atr_mult = 1.70

    structure_sl = swing_low - (0.15 * atr)
    atr_sl = price - (atr_mult * atr)
    recent_sl = recent_low - (0.1 * atr)

    candidates = [
        ("هيكل يومي", structure_sl),
        ("ATR", atr_sl),
        ("قاع حديث", recent_sl),
    ]
    if m15_low is not None and m15_low < price:
        candidates.append(("قاع 15د", m15_low - 0.05 * atr))

    # نفضّل الوقف الأعلى (أقرب للسعر) طالما أنه لا يقل عن حد أدنى
    min_dist = max(price * 0.010, atr * 0.65)   # ~1% أو 0.65 ATR
    max_dist = price * 0.085                     # لا نتجاوز ~8.5%

    valid = []
    for name, sl in candidates:
        dist = price - sl
        if min_dist <= dist <= max_dist:
            valid.append((name, sl, dist))

    if valid:
        # الأعلى سعراً = أقل مخاطرة مطلقة مع بقائه منطقياً
        valid.sort(key=lambda x: x[1], reverse=True)
        best_name, best_sl, _ = valid[0]
        return best_sl, best_name

    # fallback
    fallback = price - min(max_dist, max(min_dist, atr_mult * atr))
    return fallback, "افتراضي"


def analyze(symbol: str, name: str = "", with_live: bool = True) -> SignalResult:
    symbol = symbol.upper().strip()
    df = fetch_history(symbol)
    close = df["Close"]
    volume = df["Volume"]

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    rsi = _rsi(close, 14)
    atr = _atr(df, 14)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal_line = _ema(macd, 9)
    macd_hist = macd - signal_line
    vol_sma = volume.rolling(20).mean()

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = (price - prev) / prev * 100

    last_sma20 = float(sma20.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])
    last_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else last_sma50
    last_rsi = float(rsi.iloc[-1])
    last_atr = float(atr.iloc[-1])
    last_hist = float(macd_hist.iloc[-1])
    prev_hist = float(macd_hist.iloc[-2])
    vol_ratio = float(volume.iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] else 1.0

    swing_low = float(df["Low"].iloc[-12:].min())
    swing_high = float(df["High"].iloc[-20:].max())
    recent_low = float(df["Low"].iloc[-5:].min())

    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    factors: list[str] = []

    w = WEIGHTS.get  # اختصار

    # ---- الاتجاه اليومي ----
    uptrend_ok = False
    if price > last_sma200 and last_sma50 > last_sma200:
        score += w("uptrend_200")
        uptrend_ok = True
        reasons.append("الاتجاه اليومي صاعد (السعر ومتوسط 50 فوق 200)")
        factors.append("uptrend_200")
    elif price > last_sma200:
        score += w("price_above_200")
        reasons.append("السعر فوق متوسط 200 يوم")
        factors.append("price_above_200")
    else:
        warnings.append("السعر تحت متوسط 200 — الاتجاه طويل المدى ضعيف")

    stacked = False
    if last_sma20 > last_sma50 > last_sma200:
        score += w("stacked_ma")
        stacked = True
        reasons.append("ترتيب المتوسطات صاعد 20 > 50 > 200")
        factors.append("stacked_ma")
    elif last_sma20 > last_sma50:
        score += w("sma20_above_50")
        reasons.append("متوسط 20 فوق متوسط 50")
        factors.append("sma20_above_50")

    # ---- RSI ----
    if 50 <= last_rsi <= 64:
        score += w("rsi_healthy")
        reasons.append(f"RSI اليومي في زخم صحي ({last_rsi:.1f})")
        factors.append("rsi_healthy")
    elif 45 <= last_rsi < 50:
        score += w("rsi_near")
        reasons.append(f"RSI اليومي قريب من الانطلاق ({last_rsi:.1f})")
        factors.append("rsi_near")
    elif 64 < last_rsi <= 70:
        score += w("rsi_strong")
        reasons.append(f"RSI قوي ويقترب من الامتداد ({last_rsi:.1f})")
        warnings.append("لا تطارد القمة إذا امتد السعر أكثر")
        factors.append("rsi_strong")
    elif last_rsi > 74:
        warnings.append(f"RSI اليومي مرتفع جداً ({last_rsi:.1f})")
    else:
        warnings.append(f"RSI اليومي ضعيف ({last_rsi:.1f})")

    # ---- MACD ----
    if last_hist > 0 and last_hist > prev_hist:
        score += w("macd_expand")
        reasons.append("MACD اليومي صاعد ويتوسع")
        factors.append("macd_expand")
    elif last_hist > 0:
        score += w("macd_positive")
        reasons.append("MACD اليومي فوق الصفر")
        factors.append("macd_positive")
    else:
        warnings.append("MACD اليومي لا يؤكد الصعود بعد")

    # ---- الحجم ----
    if vol_ratio >= 1.35:
        score += w("vol_high")
        reasons.append(f"حجم يومي مرتفع ({vol_ratio:.2f}x)")
        factors.append("vol_high")
    elif vol_ratio >= 1.05:
        score += w("vol_ok")
        reasons.append("الحجم اليومي مقبول")
        factors.append("vol_ok")
    else:
        warnings.append("الحجم اليومي ضعيف")

    # ---- القرب من SMA20 ----
    ext_from_sma20 = (price - last_sma20) / last_sma20 * 100 if last_sma20 else 0
    if 0 <= ext_from_sma20 <= 4.0:
        score += w("near_sma20")
        reasons.append("السعر قريب من متوسط 20 (دخول أفضل من المطاردة)")
        factors.append("near_sma20")
    elif -2.2 <= ext_from_sma20 < 0:
        score += w("bounce_sma20")
        reasons.append("ارتكاز على متوسط 20")
        factors.append("bounce_sma20")
    elif ext_from_sma20 > 7:
        warnings.append("امتداد بعيد عن متوسط 20")
    else:
        score += 2

    if price > swing_low * 1.008:
        score += w("structure_hold")
        reasons.append("يحافظ على قاع هيكلي")
        factors.append("structure_hold")

    # ---- إطار 15 دقيقة ----
    m15_ok = False
    m15_low = None
    live_price = None
    live_rsi = None
    live_vs_vwap = None
    live_ok = False
    live_vol_ratio = 1.0
    live_momentum = 0.0

    if with_live:
        # --- 15m ---
        try:
            m15 = fetch_intraday(symbol, interval="15m", period="10d")
            last_day = m15.index[-1].date()
            today_m15 = m15[m15.index.date == last_day]
            if len(today_m15) >= 4:
                m15 = today_m15
            mclose = m15["Close"]
            m15_price = float(mclose.iloc[-1])
            m15_ema20 = float(_ema(mclose, 20).iloc[-1])
            m15_rsi = float(_rsi(mclose, 14).iloc[-1])
            m15_low = float(m15["Low"].iloc[-8:].min())
            m15_mom = (
                (m15_price - float(mclose.iloc[-4])) / float(mclose.iloc[-4]) * 100
                if len(mclose) >= 4
                else 0
            )
            above_m15_ema = m15_price >= m15_ema20 * 0.997
            rsi_m15_ok = 45 <= m15_rsi <= 72
            mom_ok = m15_mom > -0.15

            if above_m15_ema and rsi_m15_ok and mom_ok:
                score += w("m15_confirm")
                m15_ok = True
                reasons.append(f"تأكيد إطار 15د (فوق EMA20 + RSI {m15_rsi:.0f})")
                factors.append("m15_confirm")
            elif above_m15_ema or rsi_m15_ok:
                score += w("m15_partial")
                reasons.append("تأكيد جزئي على 15 دقيقة")
                factors.append("m15_partial")
            else:
                warnings.append("إطار 15د لم يؤكد الصعود بعد")
        except Exception:
            warnings.append("تعذر جلب إطار 15 دقيقة")

        # --- 5m ---
        try:
            intra = fetch_intraday(symbol, interval="5m", period="5d")
            last_day = intra.index[-1].date()
            today_bars = intra[intra.index.date == last_day]
            if len(today_bars) >= 8:
                intra = today_bars
            iclose = intra["Close"]
            live_price = float(iclose.iloc[-1])
            live_rsi_s = _rsi(iclose, 14)
            live_rsi = float(live_rsi_s.iloc[-1]) if pd.notna(live_rsi_s.iloc[-1]) else None
            ema20_i = float(_ema(iclose, 20).iloc[-1])
            vwap_s = _vwap(intra)
            vwap_last = float(vwap_s.iloc[-1]) if pd.notna(vwap_s.iloc[-1]) else ema20_i
            live_vs_vwap = (live_price - vwap_last) / vwap_last * 100 if vwap_last else 0
            vol_i = intra["Volume"]
            vol_avg = float(vol_i.rolling(12).mean().iloc[-1] or 1)
            live_vol_ratio = float(vol_i.iloc[-1] / vol_avg) if vol_avg else 1.0
            live_momentum = (
                (live_price - float(iclose.iloc[-6])) / float(iclose.iloc[-6]) * 100
                if len(iclose) >= 6
                else 0
            )
            last_green = float(intra["Close"].iloc[-1]) >= float(intra["Open"].iloc[-1])
            above_ema = live_price >= ema20_i * 0.998
            above_vwap = live_price >= vwap_last
            rsi_ok = live_rsi is not None and 48 <= live_rsi <= 72
            not_climax = live_rsi is None or live_rsi < 78

            live_points = 0.0
            if above_vwap and above_ema:
                live_points += w("m5_vwap_ema")
                reasons.append("السعر اللحظي فوق VWAP ومتوسط 20 لـ 5 دقائق")
                factors.append("m5_vwap_ema")
            elif above_vwap or above_ema:
                live_points += w("m5_partial")
                reasons.append("تأكيد جزئي على الإطار 5 دقائق")
                factors.append("m5_partial")
            else:
                warnings.append("الإطار اللحظي لم يؤكد بعد (تحت VWAP/متوسط 5د)")

            if last_green:
                live_points += w("m5_green")
                factors.append("m5_green")
            if rsi_ok:
                live_points += w("m5_rsi")
                reasons.append(f"RSI اللحظي متزن ({live_rsi:.1f})")
                factors.append("m5_rsi")
            elif live_rsi is not None and live_rsi > 78:
                warnings.append(f"تشبع لحظي ({live_rsi:.1f})")
            if live_vol_ratio >= 1.4:
                live_points += w("m5_vol")
                reasons.append(f"نشاط لحظي في الحجم ({live_vol_ratio:.2f}x)")
                factors.append("m5_vol")
            if live_momentum > 0.15:
                live_points += w("m5_momentum")
                factors.append("m5_momentum")

            live_ok = above_vwap and above_ema and not_climax and (last_green or live_momentum > 0)
            if live_ok:
                reasons.append("تأكيد لحظي للصعود على شموع 5 دقائق")
            score += live_points

            if live_price:
                change_pct = (live_price - prev) / prev * 100
                price = live_price
        except Exception:
            warnings.append("تعذر جلب التأكيد اللحظي الآن — تم الاعتماد على الإطار اليومي")

    # شرط الاتجاه اليومي للتنبيه العالي
    if not uptrend_ok and not stacked:
        score = min(score, 84)

    # ---- فلتر جودة الدخول ----
    # 1) امتداد عن متوسط 20  2) تذبذب ATR٪  3) حجم حقيقي
    atr_pct = (last_atr / price * 100) if price else 0.0
    quality_ok = True
    if ext_from_sma20 > 6.0:
        quality_ok = False
        warnings.append(f"ممتد بعيداً عن متوسط 20 ({ext_from_sma20:.1f}%) — تجنّب المطاردة")
        score = min(score, 84)
    if atr_pct > 5.5:
        quality_ok = False
        warnings.append(f"تذبذب عالي ATR={atr_pct:.1f}% — سهم هايج")
        score = min(score, 84)
    if vol_ratio < 0.95:
        quality_ok = False
        warnings.append(f"الحجم ضعيف جداً ({vol_ratio:.2f}x) — لا دعم شرائي كافٍ")
        score = min(score, 84)
    elif vol_ratio < 1.15 and quality_ok:
        # حجم مقبول بالكاد: لا يفشل الفلتر لكنه لا يُفضَّل للتنبيه التلقائي القوي
        warnings.append(f"الحجم دون المثالي ({vol_ratio:.2f}x)")

    score_i = int(max(0, min(100, round(score))))
    grade, bias = _grade(score_i)

    # ---- وقف الخسارة المحسّن ----
    stop_loss, sl_method = _compute_improved_stop(
        price=price,
        atr=last_atr,
        swing_low=swing_low,
        recent_low=recent_low,
        m15_low=m15_low,
        score=score_i,
    )

    ref = last_sma20 if last_sma20 else price
    buy_low = max(recent_low, ref * 0.987, stop_loss * 1.012)
    buy_high = min(price * 1.006, ref * 1.025)
    if buy_low >= buy_high:
        buy_low = price * 0.994
        buy_high = price * 1.004

    risk = price - stop_loss
    tp1 = price + risk * 1.6
    tp2 = price + risk * 2.6
    tp3 = min(
        price + risk * 4.0,
        swing_high * 1.03 if swing_high > price else price + risk * 4.0,
    )
    risk_pct = risk / price * 100 if price else 0
    reward_r = (tp2 - price) / risk if risk > 0 else 0

    info = fetch_info(symbol)
    cap = _fmt_cap(info.get("market_cap"))
    last_info = info.get("last")
    if last_info and not live_price:
        live_price = float(last_info)

    rank_key = (
        score_i * 10
        + (8 if live_ok else 0)
        + (5 if m15_ok else 0)
        + min(6.0, live_vol_ratio)
        + max(0.0, live_momentum)
    )

    tf_note = "يومي"
    if with_live:
        parts = ["يومي"]
        if m15_ok or True:
            parts.append("15د")
        parts.append("5د")
        tf_note = " + ".join(parts)

    return SignalResult(
        symbol=symbol,
        name=name or symbol,
        price=price,
        change_pct=change_pct,
        score=score_i,
        grade=grade,
        bias=bias,
        buy_low=buy_low,
        buy_high=buy_high,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        risk_pct=risk_pct,
        reward_r=reward_r,
        rsi=last_rsi,
        macd_hist=last_hist,
        sma20=last_sma20,
        sma50=last_sma50,
        sma200=last_sma200,
        atr=last_atr,
        volume_ratio=vol_ratio,
        reasons=reasons,
        warnings=warnings,
        market_cap=cap,
        live_price=live_price,
        live_rsi=live_rsi,
        live_vs_vwap=live_vs_vwap,
        live_ok=live_ok,
        m15_ok=m15_ok,
        rank_key=rank_key,
        timeframe_note=tf_note,
        factor_keys=factors,
        sl_method=sl_method,
        quality_ok=quality_ok,
        atr_pct=round(atr_pct, 2),
        ext_sma20=round(ext_from_sma20, 2),
    )


def format_signal_ar(sig: SignalResult, min_score: int = 85, rank: int | None = None) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    conf = []
    if sig.m15_ok:
        conf.append("15د")
    if sig.live_ok:
        conf.append("5د")
    live = "+".join(conf) if conf else "يومي"
    head = f"{sig.symbol} | {sig.score}/100 | {sig.grade} | {live}"
    if rank is not None:
        head = f"#{rank}  {head}"

    lines = [
        head,
        f"{sig.name}",
        f"السعر: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
        f"شراء: {sig.buy_low:.2f} — {sig.buy_high:.2f}",
        f"وقف: {sig.stop_loss:.2f} ({sig.sl_method})  |  مخاطرة {sig.risk_pct:.2f}%",
        f"أهداف: {sig.tp1:.2f}  |  {sig.tp2:.2f}  |  {sig.tp3:.2f}",
        f"عائد/مخاطرة ≈ 1:{sig.reward_r:.1f}",
    ]

    why = sig.reasons[:3]
    if why:
        lines.append("لماذا:")
        for r in why:
            lines.append(f"• {r}")

    risks = sig.warnings[:2]
    if sig.score < min_score or not sig.live_ok:
        risks.append(f"تحت شرط التنبيه ({min_score}+ وتأكيد لحظي)")
    if not getattr(sig, "quality_ok", True):
        risks.append("فشل فلتر جودة الدخول (امتداد/تذبذب/حجم)")
    if risks:
        lines.append("مخاطر:")
        for w in risks:
            lines.append(f"• {w}")

    lines.append("تحليل تعليمي — ليست توصية.")
    return "\n".join(lines)


def scan_symbols(
    symbols: list[str],
    names: dict,
    min_score: int = 85,
    require_live: bool = True,
    limit: int = 5,
    skip_earnings: bool = True,
    earnings_days: int = 2,
    require_m15: bool = False,
) -> list[SignalResult]:
    results: list[SignalResult] = []
    near_earn: list[str] = []
    try:
        from earnings import is_near_earnings
    except Exception:
        is_near_earnings = None

    # فلتر نظام السوق
    try:
        from market import get_market_regime

        regime = get_market_regime("SPY")
        effective_min = max(min_score, regime.get("min_score_adj", min_score))
    except Exception:
        effective_min = min_score
        regime = {"regime": "neutral", "allow_auto": True}

    for sym in symbols:
        try:
            if skip_earnings and is_near_earnings is not None:
                near, edt = is_near_earnings(sym, within_days=earnings_days)
                if near:
                    near_earn.append(f"{sym}({edt})")
                    continue
            sig = analyze(sym, names.get(sym, sym), with_live=True)
            sig.regime = regime.get("regime", "neutral")
            if sig.score < effective_min:
                continue
            if require_live and not sig.live_ok:
                continue
            if require_m15 and not sig.m15_ok:
                continue
            # فلتر جودة الدخول للتنبيه التلقائي
            if not getattr(sig, "quality_ok", True):
                continue
            if getattr(sig, "volume_ratio", 0) < 1.10:
                continue
            results.append(sig)
        except Exception:
            continue
    results.sort(key=lambda x: (x.score, x.rank_key), reverse=True)
    scan_symbols.last_skipped_earnings = near_earn
    scan_symbols.last_regime = regime
    return results[:limit]


def rank_all(symbols: list[str], names: dict) -> list[SignalResult]:
    results: list[SignalResult] = []
    for sym in symbols:
        try:
            results.append(analyze(sym, names.get(sym, sym), with_live=True))
        except Exception:
            continue
    results.sort(key=lambda x: (x.score, x.rank_key), reverse=True)
    return results
