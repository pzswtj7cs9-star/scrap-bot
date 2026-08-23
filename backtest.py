"""باكتست مبسّط لاستراتيجية التأكيد (إطار يومي فقط — بدون 5 دقائق تاريخية كاملة)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from stocks import CORE_WATCHLIST


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
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class BTTrade:
    symbol: str
    entry_date: str
    entry: float
    stop: float
    tp1: float
    exit_date: str
    exit: float
    result: str
    pnl_pct: float


def _daily_score_row(df: pd.DataFrame, i: int) -> int:
    if i < 200:
        return 0
    close = df["Close"]
    price = float(close.iloc[i])
    sma20 = float(_sma(close, 20).iloc[i])
    sma50 = float(_sma(close, 50).iloc[i])
    sma200 = float(_sma(close, 200).iloc[i])
    rsi = float(_rsi(close).iloc[i])
    atr = float(_atr(df).iloc[i])
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    hist = macd - _ema(macd, 9)
    last_hist = float(hist.iloc[i])
    prev_hist = float(hist.iloc[i - 1])
    vol = df["Volume"]
    vol_ratio = float(vol.iloc[i] / vol.rolling(20).mean().iloc[i]) if vol.rolling(20).mean().iloc[i] else 1

    score = 0
    if price > sma200 and sma50 > sma200:
        score += 22
    elif price > sma200:
        score += 12
    if sma20 > sma50 > sma200:
        score += 14
    elif sma20 > sma50:
        score += 7
    if 48 <= rsi <= 65:
        score += 16
    elif 40 <= rsi < 48:
        score += 10
    elif 65 < rsi <= 72:
        score += 6
    if last_hist > 0 and last_hist > prev_hist:
        score += 16
    elif last_hist > 0:
        score += 9
    if vol_ratio >= 1.4:
        score += 12
    elif vol_ratio >= 1.05:
        score += 6
    ext = (price - sma20) / sma20 * 100 if sma20 else 0
    if 0 <= ext <= 4.5:
        score += 12
    elif -2.5 <= ext < 0:
        score += 10
    return int(max(0, min(100, score)))


def backtest_symbol(symbol: str, period: str = "1y", min_score: int = 85) -> list[BTTrade]:
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if df is None or len(df) < 220:
        return []
    trades: list[BTTrade] = []
    i = 200
    while i < len(df) - 5:
        score = _daily_score_row(df, i)
        if score < min_score:
            i += 1
            continue
        price = float(df["Close"].iloc[i])
        atr = float(_atr(df).iloc[i])
        swing_low = float(df["Low"].iloc[i - 12 : i + 1].min())
        stop = min(swing_low - 0.25 * atr, price - 1.8 * atr)
        stop = max(stop, price * 0.88)
        if price - stop < price * 0.012:
            stop = price - max(price * 0.012, atr * 0.8)
        risk = price - stop
        if risk <= 0:
            i += 1
            continue
        tp1 = price + risk * 1.6
        entry_date = str(df.index[i].date())
        result = "timeout"
        exit_price = float(df["Close"].iloc[min(i + 15, len(df) - 1)])
        exit_date = str(df.index[min(i + 15, len(df) - 1)].date())
        for j in range(i + 1, min(i + 16, len(df))):
            low = float(df["Low"].iloc[j])
            high = float(df["High"].iloc[j])
            if low <= stop:
                result = "stop"
                exit_price = stop
                exit_date = str(df.index[j].date())
                break
            if high >= tp1:
                result = "tp1"
                exit_price = tp1
                exit_date = str(df.index[j].date())
                break
        pnl = (exit_price - price) / price * 100
        trades.append(
            BTTrade(
                symbol=symbol,
                entry_date=entry_date,
                entry=price,
                stop=stop,
                tp1=tp1,
                exit_date=exit_date,
                exit=exit_price,
                result=result,
                pnl_pct=pnl,
            )
        )
        # تخطي فترة الصفقة لتقليل التداخل
        i += 8
    return trades


def run_backtest(symbols: list[str] | None = None, period: str = "1y", min_score: int = 85) -> str:
    symbols = symbols or CORE_WATCHLIST[:12]
    all_trades: list[BTTrade] = []
    for sym in symbols:
        try:
            all_trades.extend(backtest_symbol(sym, period=period, min_score=min_score))
        except Exception:
            continue
    if not all_trades:
        return "لم تُنتج إشارات كافية في الباكتست (ارفع الفترة أو خفّض الحد)."

    wins = [t for t in all_trades if t.pnl_pct > 0]
    stops = [t for t in all_trades if t.result == "stop"]
    tps = [t for t in all_trades if t.result == "tp1"]
    avg = sum(t.pnl_pct for t in all_trades) / len(all_trades)
    win_rate = len(wins) / len(all_trades) * 100

    lines = [
        "🧪 باكتست مبسّط (إطار يومي فقط)",
        f"الفترة: {period} | الحد: {min_score}+ | الأسهم: {len(symbols)}",
        f"عدد الصفقات: {len(all_trades)}",
        f"نسبة الصفقات الرابحة: {win_rate:.1f}%",
        f"متوسط العائد/صفقة: {avg:+.2f}%",
        f"وصول هدف1: {len(tps)} | ضرب الوقف: {len(stops)} | انتهاء المهلة: {len(all_trades)-len(tps)-len(stops)}",
        "",
        "⚠️ هذا تقريبي:",
        "• بدون تأكيد 5 دقائق التاريخي الكامل",
        "• بدون انزلاق سعري أو عمولة",
        "• لا يضمن النتائج المستقبلية",
        "",
        "عينة من آخر الصفقات:",
    ]
    for t in all_trades[-8:]:
        lines.append(
            f"  {t.symbol} {t.entry_date} → {t.result} {t.pnl_pct:+.1f}% (خروج {t.exit_date})"
        )
    return "\n".join(lines)
