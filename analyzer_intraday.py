"""
مسار المضاربة اللحظية (معزول عن السوينغ اليومي).
- اتجاه: ساعة
- تأكيد ناعم: 15 دقيقة
- دخول/زخم: 5 دقائق
- TP1: أقرب مقاومة/قمة سابقة مناسبة
- تعلم مستقل من نتائج اللحظي فقط
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from typing import Optional
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from market import now_ny, REGULAR_OPEN, REGULAR_CLOSE
from stocks import MAX_AUTO_PRICE

SKIP_OPEN_MIN = 20
SKIP_CLOSE_MIN = 20
INTRADAY_MIN_SCORE = 82

INTRADAY_LEARNING_FILE = Path("intraday_learning.jsonl")
LEARNING_MIN_SAMPLES = 20
LEARNING_LOOKBACK = 60
LEARNING_MAX_ADJUSTMENT = 4.0

# أخبار: اختياري عبر FINNHUB_API_KEY. إذا لم يوجد المفتاح لا يمنع التحليل.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
NEWS_LOOKBACK_HOURS = 6
NEWS_MOMENTUM_MIN_CHANGE = 4.0
NEWS_MOMENTUM_MIN_VOLUME = 1.5
NEWS_BLOCK_NEGATIVE = True
NEWS_HIGH_RISK_WORDS = {
    "bankruptcy", "chapter 11", "offering", "dilution", "investigation",
    "lawsuit", "fraud", "recall", "downgrade", "guidance cut",
    "layoff", "default", "restatement",
}
NEWS_POSITIVE_WORDS = {
    "acquisition", "acquire", "merger", "buyout", "takeover",
    "partnership", "contract", "approval", "award", "order",
    "investment", "funding", "strategic", "agreement",
}


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
    m15_state: str = "محايد"
    learning_adjustment: float = 0.0
    resistance_tp1: float = 0.0
    news_state: str = "neutral"
    news_title: str = ""
    news_source: str = ""
    breakout_quality: float = 0.0
    market_state: str = "السوق غير مؤكد"
    chop: bool = False


def _read_learning_records() -> list[dict]:
    if not INTRADAY_LEARNING_FILE.exists():
        return []
    rows: list[dict] = []
    try:
        with INTRADAY_LEARNING_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _append_learning_record(record: dict) -> None:
    INTRADAY_LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INTRADAY_LEARNING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _completed_learning(records: list[dict] | None = None) -> list[dict]:
    records = records if records is not None else _read_learning_records()
    completed = [
        r for r in records
        if r.get("record_type") == "outcome"
        and r.get("status") in {"tp1", "stop", "timeout"}
    ]
    return completed[-LEARNING_LOOKBACK:]


def _learning_adjustment(factors: list[str], entry_type: str) -> float:
    completed = _completed_learning()
    if len(completed) < LEARNING_MIN_SAMPLES:
        return 0.0

    overall_wins = sum(1 for r in completed if r.get("status") == "tp1")
    overall_rate = overall_wins / len(completed)

    relevant = [r for r in completed if r.get("factors")]
    factor_deltas: list[float] = []
    for factor in factors:
        subset = [r for r in relevant if factor in (r.get("factors") or [])]
        if len(subset) < 5:
            continue
        rate = sum(1 for r in subset if r.get("status") == "tp1") / len(subset)
        factor_deltas.append((rate - overall_rate) * 10.0)

    adjustment = float(np.mean(factor_deltas)) if factor_deltas else 0.0

    et_subset = [
        r for r in completed
        if str(r.get("entry_type") or "") == entry_type
    ]
    if len(et_subset) >= 5:
        et_rate = sum(1 for r in et_subset if r.get("status") == "tp1") / len(et_subset)
        adjustment += (et_rate - overall_rate) * 6.0

    return float(max(-LEARNING_MAX_ADJUSTMENT, min(LEARNING_MAX_ADJUSTMENT, adjustment)))


def register_intraday_signal(sig: IntradaySignal) -> str:
    """يحفظ لقطة الإشارة قبل إرسالها؛ لا يؤثر على سجل السوينغ."""
    signal_id = (
        f"{sig.symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )
    _append_learning_record(
        {
            "record_type": "signal",
            "signal_id": signal_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": sig.symbol,
            "entry": round(float(sig.price), 4),
            "stop_loss": round(float(sig.stop_loss), 4),
            "tp1": round(float(sig.tp1), 4),
            "score": int(sig.score),
            "grade": sig.grade,
            "entry_type": sig.entry_type,
            "factors": list(sig.factor_keys or []),
            "m15_state": sig.m15_state,
            "volume_ratio": float(sig.volume_ratio),
            "atr_pct": float(sig.atr_pct),
            "ext_sma20": float(sig.ext_sma20),
            "above_open": bool(sig.above_open),
            "vwap": sig.vwap_day_note,
            "news_state": sig.news_state,
            "breakout_quality": sig.breakout_quality,
            "market_state": sig.market_state,
            "chop": sig.chop,
            "status": "pending",
        }
    )
    return signal_id


def record_intraday_outcome(
    signal_id: str | None,
    symbol: str,
    status: str,
    exit_price: float | None = None,
    note: str = "",
) -> None:
    """يسجل نتيجة الإشارة مرة واحدة. TP1 هو نجاح لأن الخروج الكامل عند TP1."""
    if status not in {"tp1", "stop", "timeout"}:
        return

    records = _read_learning_records()
    target = None
    if signal_id:
        for r in reversed(records):
            if r.get("record_type") == "signal" and r.get("signal_id") == signal_id:
                target = r
                break
    if target is None:
        for r in reversed(records):
            if (
                r.get("record_type") == "signal"
                and r.get("symbol") == symbol
                and r.get("status") == "pending"
            ):
                target = r
                break
    if target is None:
        return

    sid = target.get("signal_id")
    if any(
        r.get("record_type") == "outcome"
        and r.get("signal_id") == sid
        for r in records
    ):
        return

    _append_learning_record(
        {
            "record_type": "outcome",
            "signal_id": sid,
            "created_at": target.get("created_at"),
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "entry": target.get("entry"),
            "stop_loss": target.get("stop_loss"),
            "tp1": target.get("tp1"),
            "score": target.get("score"),
            "grade": target.get("grade"),
            "entry_type": target.get("entry_type"),
            "factors": target.get("factors") or [],
            "m15_state": target.get("m15_state", "محايد"),
            "volume_ratio": target.get("volume_ratio", 0),
            "atr_pct": target.get("atr_pct", 0),
            "ext_sma20": target.get("ext_sma20", 0),
            "above_open": target.get("above_open", False),
            "vwap": target.get("vwap", ""),
            "news_state": target.get("news_state", "neutral"),
            "breakout_quality": target.get("breakout_quality", 0),
            "market_state": target.get("market_state", "السوق غير مؤكد"),
            "chop": target.get("chop", False),
            "status": status,
            "exit_price": exit_price,
            "note": note,
        }
    )
    target["status"] = "completed"


def _m15_confirmation(m15: pd.DataFrame, price: float) -> tuple[str, int]:
    """تأكيد ناعم: لا يمنع الإشارة وحده، لكنه يرفع/يخفض الثقة."""
    try:
        if m15 is None or len(m15) < 30:
            return "محايد", 0
        last_day = m15.index[-1].date()
        cur = m15[m15.index.date == last_day]
        if len(cur) < 6:
            cur = m15.tail(20)
        c = cur["Close"].astype(float)
        e20 = _ema(c, 20)
        e50 = _ema(c, 50)
        rsi = _rsi(c, 14)
        recent = c.tail(5)
        bullish = (
            price >= float(e20.iloc[-1]) * 0.998
            and float(e20.iloc[-1]) >= float(e50.iloc[-1]) * 0.997
            and float(rsi.iloc[-1]) >= 45
            and float(recent.iloc[-1]) >= float(recent.iloc[0])
        )
        bearish = (
            price < float(e20.iloc[-1]) * 0.997
            and float(e20.iloc[-1]) < float(e50.iloc[-1]) * 0.995
            and float(rsi.iloc[-1]) < 45
        )
        if bullish:
            return "داعم", 5
        if bearish:
            return "معاكس", -4
        return "محايد", 0
    except Exception:
        return "محايد", 0



def _fetch_recent_news(symbol: str) -> list[dict]:
    """مصدر أخبار اختياري. فشل المصدر لا يوقف البوت."""
    if not FINNHUB_API_KEY:
        return []
    try:
        end = datetime.now(timezone.utc).date()
        start = (datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)).date()
        params = urllib.parse.urlencode({
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": FINNHUB_API_KEY,
        })
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "IntradayScanner/1.0"})
        with urllib.request.urlopen(req, timeout=3.5) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _classify_news(symbol: str) -> tuple[str, str, str]:
    """
    يرجع: الحالة، العنوان، المصدر.
    positive_strong = خبر جوهري إيجابي مثل استحواذ.
    negative = خبر سلبي عالي المخاطر.
    neutral = لا خبر مؤثر/لا مصدر.
    """
    news = _fetch_recent_news(symbol)
    if not news:
        return "neutral", "", ""

    latest = news[0]
    title = str(latest.get("headline") or latest.get("title") or "").strip()
    source = str(latest.get("source") or "").strip()
    text = (title + " " + str(latest.get("summary") or "")).lower()

    if any(k in text for k in NEWS_HIGH_RISK_WORDS):
        return "negative", title, source
    if any(k in text for k in NEWS_POSITIVE_WORDS):
        # الاستحواذ/الاندماج/عرض الشراء: لا نمنعه، بل نطلب Momentum أقوى.
        if any(k in text for k in {"acquisition", "acquire", "merger", "buyout", "takeover"}):
            return "positive_strong", title, source
        return "positive", title, source
    return "neutral", title, source


def _chop_filter(today_5: pd.DataFrame, price: float, vwap: float) -> bool:
    """True = سوق متذبذب/Chop، فلا نطارد الإشارات."""
    try:
        c = today_5["Close"].astype(float)
        if len(c) < 12:
            return False
        e9 = _ema(c, 9)
        cross = ((c > e9).astype(int).diff().abs()).tail(12).sum()
        vwap_dist = abs(price - vwap) / price * 100 if price else 0
        ranges = (today_5["High"] - today_5["Low"]).astype(float)
        avg_range = ranges.tail(12).mean()
        if avg_range <= 0:
            return False
        tight = (ranges.tail(12).median() / avg_range) < 0.75
        return bool(cross >= 5 and vwap_dist < 0.7 and tight)
    except Exception:
        return False


def _breakout_quality(today_5: pd.DataFrame, level: float, price: float) -> tuple[bool, float]:
    """
    جودة آخر شمعة 5د فوق المقاومة + متابعة الشمعة السابقة.
    لا نعتبر لمس المستوى اختراقًا.
    """
    try:
        if level <= 0 or len(today_5) < 3:
            return False, 0.0
        b = today_5.iloc[-1]
        o, h, l, c = map(float, (b["Open"], b["High"], b["Low"], b["Close"]))
        rng = max(h - l, 1e-9)
        body = abs(c - o) / rng
        close_pos = (c - l) / rng
        upper_wick = (h - max(o, c)) / rng
        prior_close = float(today_5["Close"].iloc[-2])

        strong_close = c >= level * 1.001 and close_pos >= 0.70
        body_ok = body >= 0.45
        wick_ok = upper_wick <= 0.30
        follow = prior_close >= level * 0.997 or c >= prior_close * 1.002
        quality = (body * 0.4 + close_pos * 0.4 + (1 - min(upper_wick, 1)) * 0.2) * 100
        return bool(strong_close and body_ok and wick_ok and follow), float(quality)
    except Exception:
        return False, 0.0


def _market_alignment(fetch_intraday) -> tuple[bool, str]:
    """SPY + QQQ: اتجاه السوق العام يدعم/لا يدعم اللحظي."""
    states = []
    for sym in ("SPY", "QQQ"):
        try:
            d = fetch_intraday(sym, interval="5m", period="2d")
            if d is None or len(d) < 20:
                states.append(None)
                continue
            day = d.index[-1].date()
            cur = d[d.index.date == day]
            if len(cur) < 6:
                states.append(None)
                continue
            p = float(cur["Close"].iloc[-1])
            op = float(cur["Open"].iloc[0])
            vw = float(_vwap(cur).iloc[-1])
            states.append(p >= op and p >= vw)
        except Exception:
            states.append(None)
    known = [x for x in states if x is not None]
    if not known:
        return True, "السوق غير مؤكد"
    if all(known):
        return True, "SPY+QQQ داعمان"
    if not any(known):
        return False, "SPY+QQQ ضعيفان"
    return True, "SPY/QQQ مختلطان"


def session_window_ok(dt=None) -> tuple[bool, str]:
    dt = dt or now_ny()
    t = dt.time()
    open_ok_after = time(9, 50)
    close_cut = time(15, 40)
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


def _grade(score: int, strong: bool = False) -> str:
    if score >= 95 and strong:
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


def _find_prior_resistance(
    today_5: pd.DataFrame,
    h1: pd.DataFrame,
    price: float,
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []

    try:
        x = today_5.iloc[:-1].copy()
        if len(x) >= 8:
            highs = x["High"].astype(float)
            for i in range(2, len(highs) - 2):
                v = float(highs.iloc[i])
                if (
                    v >= float(highs.iloc[i - 1])
                    and v >= float(highs.iloc[i - 2])
                    and v >= float(highs.iloc[i + 1])
                    and v >= float(highs.iloc[i + 2])
                    and price * 1.008 <= v <= price * 1.07
                ):
                    candidates.append((v, "قمة 5د سابقة"))
    except Exception:
        pass

    try:
        x = h1.iloc[:-1].tail(30)
        if len(x) >= 5:
            highs = x["High"].astype(float)
            for i in range(2, len(highs) - 2):
                v = float(highs.iloc[i])
                if (
                    v >= float(highs.iloc[i - 1])
                    and v >= float(highs.iloc[i - 2])
                    and v >= float(highs.iloc[i + 1])
                    and v >= float(highs.iloc[i + 2])
                    and price * 1.008 <= v <= price * 1.07
                ):
                    candidates.append((v, "قمة ساعة سابقة"))
    except Exception:
        pass

    if not candidates:
        return 0.0, "هدف مخاطر احتياطي"
    return min(candidates, key=lambda x: x[0])


def analyze_intraday(symbol: str, name: str = "") -> Optional[IntradaySignal]:
    from market_data import fetch_intraday

    h1 = fetch_intraday(symbol, interval="60m", period="10d")
    if h1 is None or len(h1) < 40:
        return None

    m5 = fetch_intraday(symbol, interval="5m", period="5d")
    if m5 is None or len(m5) < 30:
        return None

    try:
        m15 = fetch_intraday(symbol, interval="15m", period="10d")
    except Exception:
        m15 = None

    last_day = m5.index[-1].date()
    today_5 = m5[m5.index.date == last_day]
    if len(today_5) < 6:
        return None

    price = float(today_5["Close"].iloc[-1])
    if price <= 0 or price > float(MAX_AUTO_PRICE):
        return None

    day_open = float(today_5["Open"].iloc[0])
    prev_days = m5[m5.index.date < last_day]
    prev_close = float(prev_days["Close"].iloc[-1]) if not prev_days.empty else price
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0

    vwap_s = _vwap(today_5)
    vwap_last = float(vwap_s.iloc[-1]) if pd.notna(vwap_s.iloc[-1]) else price
    above_vwap = price >= vwap_last
    vwap_note = "فوق VWAP اليوم" if above_vwap else "تحت VWAP اليوم"
    above_open = price >= day_open

    vol_today = float(today_5["Volume"].sum())
    bars = max(len(today_5), 1)
    avg_bar_today = vol_today / bars
    hist_5 = m5[m5.index.date < last_day].tail(120)
    vol_hist = (
        float(hist_5["Volume"].mean())
        if not hist_5.empty
        else float(m5["Volume"].tail(60).mean() or 1)
    )
    vol_session_ratio = avg_bar_today / vol_hist if vol_hist else 1.0
    vol_session_ok = vol_session_ratio >= 0.90

    hc = h1["Close"]
    ema20 = _ema(hc, 20)
    ema50 = _ema(hc, 50)
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    h_rsi = float(_rsi(hc, 14).iloc[-1])
    trend_up = price > e20 > e50 * 0.998 and h_rsi >= 45

    c5 = today_5["Close"]
    e5 = float(_ema(c5, 20).iloc[-1])
    r5 = float(_rsi(c5, 14).iloc[-1])
    last_green = float(today_5["Close"].iloc[-1]) >= float(today_5["Open"].iloc[-1])
    mom = (price - float(c5.iloc[-6])) / float(c5.iloc[-6]) * 100 if len(c5) >= 6 else 0.0
    live_ok = price >= e5 * 0.998 and above_vwap and (last_green or mom > 0.05) and r5 < 78

    m15_state, m15_points = _m15_confirmation(m15, price)
    news_state, news_title, news_source = _classify_news(symbol)
    market_ok, market_state = _market_alignment(fetch_intraday)
    chop = _chop_filter(today_5, price, vwap_last)

    session_high = float(today_5["High"].max())
    drop = (session_high - price) / session_high * 100 if session_high else 0
    dump = drop >= 2.5 and change_pct <= -1.2

    h_win = h1.tail(20)
    level_high = float(h_win["High"].iloc[:-1].max()) if len(h_win) > 3 else session_high
    was_below = float(h_win["Close"].iloc[-3]) < level_high * 0.998 if len(h_win) >= 3 else False
    breakout_now = price >= level_high * 1.001 and was_below
    prior_break = (
        float(h_win["High"].iloc[-8:-2].max()) >= level_high * 0.999
        if len(h_win) >= 8 else False
    )
    near_level = abs(price - level_high) / max(price, 1e-9) * 100 <= 0.7
    ext_tmp = (price - e20) / e20 * 100 if e20 else 0.0

    failed = (
        float(today_5["High"].max()) >= level_high * 1.001
        and price < level_high * 0.997
        and not above_vwap
    ) or (dump and not above_vwap)
    retest = prior_break and near_level and price >= level_high * 0.997 and above_vwap and not failed
    early = above_vwap and above_open and not breakout_now and ext_tmp <= 2.2 and not failed

    breakout_ok, breakout_quality = _breakout_quality(today_5, level_high, price)
    if failed:
        entry_type, entry_emoji = "اختراق فاشل", "🔴"
    elif retest:
        entry_type, entry_emoji = "إعادة اختبار", "🟡"
    elif breakout_now and breakout_ok and vol_session_ratio >= 1.0:
        entry_type, entry_emoji = "اختراق مؤكد", "🟢"
    elif early or (above_vwap and trend_up and ext_tmp <= 2.5):
        entry_type, entry_emoji = "دخول مبكر", "🟢"
    else:
        entry_type, entry_emoji = "دخول مبكر", "🟢"

    reasons: list[str] = []
    warnings: list[str] = []
    score = 42.0
    factors: list[str] = []

    if trend_up:
        score += 18
        reasons.append("اتجاه الساعة صاعد")
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
        reasons.append("فوق افتتاح اليوم")
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
        reasons.append(f"حجم جلسة {vol_session_ratio:.2f}x")
        factors.append("vol_session")
    else:
        warnings.append("حجم الجلسة ضعيف نسبياً")
        score -= 6

    if market_ok:
        score += 3
        factors.append("market")
        reasons.append(market_state)
    else:
        score -= 7
        warnings.append(market_state)

    if chop:
        score -= 12
        warnings.append("السوق متذبذب (Chop)")
        factors.append("chop")

    if breakout_now:
        if breakout_ok:
            score += 6
            factors.append("breakout_candle")
            reasons.append(f"قوة شمعة الاختراق {breakout_quality:.0f}/100")
        else:
            score -= 9
            warnings.append("اختراق بدون إغلاق/متابعة كافية")
            factors.append("weak_breakout")
    else:
        breakout_quality = 0.0

    if news_state == "negative":
        if NEWS_BLOCK_NEGATIVE:
            score -= 25
            warnings.append("خبر سلبي عالي المخاطر")
            factors.append("news_negative")
    elif news_state == "positive_strong":
        # لا نمنع الاستحواذ/الاندماج؛ نرفع المتطلبات بدل ذلك.
        score += 3
        factors.append("news_momentum")
        reasons.append("خبر إيجابي جوهري — وضع NEWS MOMENTUM")
    elif news_state == "positive":
        score += 1
        factors.append("news_positive")

    if m15_state == "داعم":
        score += m15_points
        reasons.append("15 دقيقة داعمة")
        factors.append("m15")
    elif m15_state == "معاكس":
        score += m15_points
        warnings.append("15 دقيقة معاكسة")
    else:
        reasons.append("15 دقيقة محايدة")
        factors.append("m15_neutral")

    if 48 <= h_rsi <= 68:
        score += 5
        factors.append("rsi_h1")

    if dump:
        warnings.append("سقوط من قمة الجلسة")
        score -= 20

    if entry_type == "اختراق فاشل":
        warnings.append("اختراق فاشل — لا إرسال")
        score -= 25
    elif entry_type == "اختراق مؤكد":
        score += 8
        reasons.append("اختراق مؤكد")
        factors.append("breakout")
    elif entry_type == "إعادة اختبار":
        score += 5
        reasons.append("إعادة اختبار مستوى")
        factors.append("retest")
    else:
        reasons.append("دخول مبكر فوق VWAP")
        factors.append("early")

    ext = (price - e20) / e20 * 100 if e20 else 0
    if ext > 4.0:
        warnings.append("امتداد عن متوسط الساعة")
        score -= 8
        factors.append("extended")

    atr = float(_atr(h1, 14).iloc[-1] or price * 0.01)
    atr_pct = atr / price * 100
    if atr_pct > 6.0:
        warnings.append("تذبذب عالي")
        score -= 5

    learning_adj = _learning_adjustment(factors, entry_type)
    if learning_adj:
        score += learning_adj
        reasons.append(f"تعلم سابق {learning_adj:+.1f}")

    strong_alignment = (
        trend_up
        and live_ok
        and above_vwap
        and m15_state != "معاكس"
        and vol_session_ratio >= 1.0
        and not dump
        and ext <= 4.0
    )

    if entry_type == "دخول مبكر":
        score = min(score, 94.0)
    elif entry_type == "إعادة اختبار":
        score = min(score, 97.0 if strong_alignment else 95.0)
    elif entry_type == "اختراق مؤكد":
        score = min(score, 100.0)
    else:
        score = min(score, 80.0)

    score_i = int(max(0, min(100, round(score))))
    strong_for_grade = (
        score_i >= 95
        and strong_alignment
        and entry_type in {"اختراق مؤكد", "إعادة اختبار"}
        and m15_state != "معاكس"
    )

    news_momentum_ok = True
    if news_state == "negative" and NEWS_BLOCK_NEGATIVE:
        news_momentum_ok = False
    if news_state == "positive_strong":
        news_momentum_ok = (
            change_pct >= NEWS_MOMENTUM_MIN_CHANGE
            and vol_session_ratio >= NEWS_MOMENTUM_MIN_VOLUME
            and above_vwap
            and breakout_ok
            and breakout_quality >= 60
            and m15_state != "معاكس"
            and market_ok
        )

    quality_ok = (
        (not dump)
        and (not failed)
        and entry_type != "اختراق فاشل"
        and ext <= 4.5
        and atr_pct <= 6.5
        and vol_session_ratio >= 0.85
        and not chop
        and news_momentum_ok
        and not (m15_state == "معاكس" and score_i < 92)
        and (market_ok or score_i >= 95)
    )

    recent_low = float(today_5["Low"].tail(12).min())
    stop = min(price - 1.5 * atr, recent_low * 0.997)
    risk = price - stop
    if risk <= 0 or risk / price < 0.012:
        stop = price * (1 - 0.012)
        risk = price - stop
    if risk / price > 0.07:
        stop = price * 0.93
        risk = price - stop

    resistance_tp1, resistance_source = _find_prior_resistance(today_5, h1, price)
    fallback_tp1 = price + risk * 1.2
    if resistance_tp1 > price and resistance_tp1 >= price * 1.008:
        tp1 = resistance_tp1
    else:
        tp1 = fallback_tp1
        resistance_source = "هدف مخاطر احتياطي"

    if tp1 > price * 1.07:
        tp1 = fallback_tp1
        resistance_source = "هدف مخاطر احتياطي"

    tp2 = max(tp1 + risk * 0.8, price + risk * 1.8)
    tp3 = max(tp2 + risk * 0.8, price + risk * 2.6)
    risk_pct = risk / price * 100
    reward_r = (tp1 - price) / risk if risk else 0.0
    tp1_distance_pct = (tp1 - price) / price * 100 if price else 0.0
    if tp1_distance_pct < 0.8:
        quality_ok = False
        warnings.append("TP1 قريب جدًا من الدخول")
    if reward_r < 1.0:
        quality_ok = False
        warnings.append("العائد إلى TP1 ضعيف")
    if news_state == "positive_strong" and news_momentum_ok:
        reasons.append("NEWS MOMENTUM مؤكد")
    buy_low = max(stop * 1.01, min(price * 0.995, e5))
    buy_high = price * 1.004

    if resistance_source != "هدف مخاطر احتياطي":
        reasons.append(f"TP1 مقاومة: {tp1:.2f}")
    else:
        warnings.append("لم توجد مقاومة قريبة مناسبة؛ TP1 احتياطي")

    return IntradaySignal(
        symbol=symbol,
        name=name or symbol,
        price=round(price, 4),
        change_pct=round(change_pct, 2),
        score=score_i,
        grade=_grade(score_i, strong=strong_for_grade),
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
        reasons=reasons[:5],
        warnings=warnings[:4],
        quality_ok=quality_ok,
        live_ok=live_ok,
        volume_ratio=round(vol_session_ratio, 2),
        factor_keys=factors,
        sma20=round(e20, 4),
        atr_pct=round(atr_pct, 2),
        ext_sma20=round(ext, 2),
        entry_type=entry_type,
        entry_emoji=entry_emoji,
        m15_state=m15_state,
        learning_adjustment=round(learning_adj, 2),
        resistance_tp1=round(tp1, 4),
        news_state=news_state,
        news_title=news_title,
        news_source=news_source,
        breakout_quality=round(breakout_quality, 1),
        market_state=market_state,
        chop=chop,
    )


def format_intraday_ar(sig: IntradaySignal, min_score: int = INTRADAY_MIN_SCORE) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    tp_source = "مقاومة/قمة سابقة" if sig.resistance_tp1 else "احتياطي"
    lines = [
        f"⚡ لحظي | {sig.symbol} | {sig.score}/100 | {sig.grade} | ساعة+15د+5د",
        f"{sig.entry_emoji} نوع الدخول: {sig.entry_type}",
        f"{sig.name}",
        "—————————————",
        f"السعر: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
        f"شراء: {sig.buy_low:.2f} — {sig.buy_high:.2f}",
        f"وقف: {sig.stop_loss:.2f} ({sig.sl_method}) | مخاطرة {sig.risk_pct:.2f}%",
        f"TP1: {sig.tp1:.2f} | TP2: {sig.tp2:.2f} | TP3: {sig.tp3:.2f}",
        f"مصدر TP1: {tp_source} | العائد إلى TP1: {sig.reward_r:.2f}R",
        "—————————————",
        f"{sig.vwap_day_note} | افتتاح: {'فوق' if sig.above_open else 'تحت'} | حجم: {sig.volume_ratio:.2f}x",
        f"15د: {sig.m15_state} | السوق: {sig.market_state} | تعلم: {sig.learning_adjustment:+.1f}",
        f"الأخبار: {sig.news_state} | جودة الاختراق: {sig.breakout_quality:.0f}/100",
    ]
    if sig.reasons:
        lines.append("لماذا: " + " | ".join(sig.reasons[:3]))
    if sig.warnings:
        lines.append("مخاطر: " + " | ".join(sig.warnings[:3]))
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

    try:
        spy = analyze_intraday("SPY", "SPY")
        if spy and spy.score < 55 and not spy.above_open:
            scan_intraday.last_window = "SPY لحظي ضعيف"
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
            if getattr(sig, "entry_type", "") == "اختراق فاشل":
                continue
            if "تحت" in sig.vwap_day_note:
                continue
            if sig.m15_state == "معاكس" and sig.score < 92:
                continue
            if sig.news_state == "negative":
                continue
            if sig.news_state == "positive_strong":
                if sig.entry_type not in {"اختراق مؤكد", "إعادة اختبار"}:
                    continue
                if sig.breakout_quality < 60:
                    continue
            results.append(sig)
        except Exception:
            continue

    rank = {"اختراق مؤكد": 0, "إعادة اختبار": 1, "دخول مبكر": 2}
    results.sort(key=lambda x: (rank.get(x.entry_type, 9), -x.score, -x.reward_r))
    return results[:limit]


scan_intraday.last_window = ""
