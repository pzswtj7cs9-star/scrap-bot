"""طبقة بيانات موحّدة: Alpaca أولاً ثم Yahoo كاحتياطي."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger("halal-bot.data")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
# بيانات السوق دائماً من data.alpaca.markets حتى لو الحساب Paper
DATA_URL = os.getenv("APCA_DATA_URL", "https://data.alpaca.markets").rstrip("/")

_LAST_SOURCE = "none"
_LAST_ERROR = ""


def alpaca_configured() -> bool:
    return bool(APCA_KEY and APCA_SECRET)


def last_source() -> str:
    return _LAST_SOURCE


def last_error() -> str:
    return _LAST_ERROR


def _set_status(source: str, err: str = "") -> None:
    global _LAST_SOURCE, _LAST_ERROR
    _LAST_SOURCE = source
    _LAST_ERROR = err


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": APCA_KEY,
        "APCA-API-SECRET-KEY": APCA_SECRET,
    }


def _bars_to_df(bars: list) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    rows = []
    idx = []
    for b in bars:
        ts = b.get("t") or b.get("timestamp")
        if not ts:
            continue
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        dt = dt.tz_convert("America/New_York")
        idx.append(dt)
        rows.append(
            {
                "Open": float(b["o"]),
                "High": float(b["h"]),
                "Low": float(b["l"]),
                "Close": float(b["c"]),
                "Volume": float(b.get("v") or 0),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_alpaca_bars(symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None, limit: int = 10000) -> pd.DataFrame:
    if not alpaca_configured():
        raise RuntimeError("Alpaca keys missing")
    end = end or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    params = {
        "timeframe": timeframe,
        "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": min(limit, 10000),
        "adjustment": "split",
        "feed": "iex",  # مناسب للخطة المجانية
    }
    url = f"{DATA_URL}/v2/stocks/{symbol.upper()}/bars"
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:180]}")
    data = r.json() or {}
    bars = data.get("bars") or []
    # pagination بسيط
    next_token = data.get("next_page_token")
    while next_token and len(bars) < limit:
        params["page_token"] = next_token
        r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=20)
        if r.status_code >= 400:
            break
        data = r.json() or {}
        bars.extend(data.get("bars") or [])
        next_token = data.get("next_page_token")
    return _bars_to_df(bars)


def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """يومي: Alpaca ثم Yahoo."""
    days = {"6mo": 190, "1y": 400, "2y": 800, "5d": 10, "1mo": 40, "3mo": 100}.get(period, 400)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, "1Day", start)
            if df is not None and len(df) >= 30:
                _set_status("alpaca")
                return df
        except Exception as exc:
            log.warning("Alpaca daily %s: %s", symbol, exc)
            _set_status("yahoo", str(exc)[:120])

    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("Yahoo empty")
        _set_status("yahoo" if not alpaca_configured() else "yahoo-fallback")
        return df
    except Exception as exc:
        _set_status("none", str(exc)[:120])
        raise


def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """لحظي 5 دقائق: Alpaca ثم Yahoo."""
    days = 5 if period.endswith("d") else 5
    try:
        days = int(period.replace("d", ""))
    except Exception:
        days = 5
    start = datetime.now(timezone.utc) - timedelta(days=max(days, 3))

    tf_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
    alpaca_tf = tf_map.get(interval, "5Min")

    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, alpaca_tf, start)
            if df is not None and len(df) >= 10:
                _set_status("alpaca")
                return df
        except Exception as exc:
            log.warning("Alpaca intra %s: %s", symbol, exc)
            _set_status("yahoo", str(exc)[:120])

    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True, prepost=False)
        if df is None or df.empty:
            raise RuntimeError("Yahoo empty")
        _set_status("yahoo" if not alpaca_configured() else "yahoo-fallback")
        return df
    except Exception as exc:
        _set_status("none", str(exc)[:120])
        raise


def ping_sources() -> tuple[bool, str]:
    """فحص سريع لـ /health."""
    notes = []
    ok = False
    if alpaca_configured():
        try:
            df = fetch_alpaca_bars("SPY", "1Day", datetime.now(timezone.utc) - timedelta(days=10))
            if df is not None and not df.empty:
                last = float(df["Close"].iloc[-1])
                notes.append(f"Alpaca يعمل — SPY ≈ {last:.2f}")
                ok = True
                _set_status("alpaca")
            else:
                notes.append("Alpaca: فارغ")
        except Exception as exc:
            notes.append(f"Alpaca تعثر: {str(exc)[:80]}")
    else:
        notes.append("Alpaca: غير مُعد")

    try:
        info = yf.Ticker("SPY").fast_info
        last = getattr(info, "last_price", None)
        if last:
            notes.append(f"Yahoo يعمل — SPY ≈ {float(last):.2f}")
            ok = True
        else:
            # fallback history
            df = yf.Ticker("SPY").history(period="5d", interval="1d")
            if df is not None and not df.empty:
                notes.append(f"Yahoo يعمل — SPY ≈ {float(df['Close'].iloc[-1]):.2f}")
                ok = True
            else:
                notes.append("Yahoo: بدون سعر")
    except Exception as exc:
        notes.append(f"Yahoo تعثر: {str(exc)[:80]}")

    return ok, " | ".join(notes)
