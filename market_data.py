"""طبقة بيانات موحّدة: Alpaca أولاً ثم Yahoo كاحتياطي."""

from __future__ import annotations

import logging
import os
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger("halal-bot.data")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
DATA_URL = os.getenv("APCA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
# auto = جرّب SIP ثم IEX تلقائيًا. إذا لم يكن SIP متاحًا يعود إلى IEX المجاني.
APCA_FEED = os.getenv("APCA_FEED", "iex").strip().lower()
SIP_RETRY_COOLDOWN_MIN = 15
_SIP_DISABLED_UNTIL: datetime | None = None
_LAST_ALPACA_FEED = "iex"

_LAST_SOURCE = "none"
_LAST_ERROR = ""

# Central Alpaca request pacing shared by all threads/jobs in this process.
# The goal is to prevent bursty parallel scans from triggering HTTP 429.
APCA_MIN_REQUEST_INTERVAL = float(os.getenv("APCA_MIN_REQUEST_INTERVAL", "0.35"))
APCA_429_RETRIES = int(os.getenv("APCA_429_RETRIES", "3"))
_APCA_RATE_LOCK = threading.Lock()
_APCA_NEXT_REQUEST_AT = 0.0


def _alpaca_get(url: str, *, params: dict, timeout: float) -> requests.Response:
    """Rate-limited GET with bounded 429 backoff for every Alpaca call."""
    global _APCA_NEXT_REQUEST_AT
    last_exc = None
    for attempt in range(APCA_429_RETRIES + 1):
        with _APCA_RATE_LOCK:
            now = _time.monotonic()
            wait = _APCA_NEXT_REQUEST_AT - now
            if wait > 0:
                _time.sleep(wait)
            _APCA_NEXT_REQUEST_AT = _time.monotonic() + max(0.0, APCA_MIN_REQUEST_INTERVAL)
            try:
                r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=timeout)
            except Exception as exc:
                last_exc = exc
                continue

        if r.status_code != 429:
            return r

        if attempt >= APCA_429_RETRIES:
            return r

        retry_after = r.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(8.0, 1.0 * (2 ** attempt))
        except Exception:
            delay = min(8.0, 1.0 * (2 ** attempt))
        log.warning("Alpaca 429: انتظار %.1fs ثم إعادة المحاولة (%d/%d)", delay, attempt + 1, APCA_429_RETRIES)
        _time.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError("Alpaca request failed")


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


def _alpaca_request_bars(symbol: str, timeframe: str, start: datetime, end: datetime, limit: int, feed: str) -> pd.DataFrame:
    params = {
        "timeframe": timeframe,
        "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": min(limit, 10000),
        "adjustment": "split",
        "feed": feed,
    }
    url = f"{DATA_URL}/v2/stocks/{symbol.upper()}/bars"
    r = _alpaca_get(url, params=params, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:180]}")
    data = r.json() or {}
    bars = data.get("bars") or []
    next_token = data.get("next_page_token")
    while next_token and len(bars) < limit:
        params["page_token"] = next_token
        r = _alpaca_get(url, params=params, timeout=10)
        if r.status_code >= 400:
            break
        data = r.json() or {}
        bars.extend(data.get("bars") or [])
        next_token = data.get("next_page_token")
    return _bars_to_df(bars)


def fetch_alpaca_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: Optional[datetime] = None,
    limit: int = 10000,
) -> pd.DataFrame:
    global _SIP_DISABLED_UNTIL, _LAST_ALPACA_FEED
    if not alpaca_configured():
        raise RuntimeError("Alpaca keys missing")
    end = end or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    requested = APCA_FEED if APCA_FEED in {"iex", "sip", "delayed_sip"} else "auto"
    feeds = [requested] if requested != "auto" else []
    if requested == "auto":
        now = datetime.now(timezone.utc)
        if _SIP_DISABLED_UNTIL is None or now >= _SIP_DISABLED_UNTIL:
            feeds.append("sip")
        feeds.append("iex")

    last_exc = None
    for feed in feeds:
        try:
            df = _alpaca_request_bars(symbol, timeframe, start, end, limit, feed)
            if feed == "sip":
                _SIP_DISABLED_UNTIL = None
            _LAST_ALPACA_FEED = feed
            if df is not None:
                df.attrs["data_source"] = f"alpaca-{feed}"
                df.attrs["feed"] = feed
            _set_status(f"alpaca-{feed}")
            return df
        except Exception as exc:
            last_exc = exc
            # إذا لم تكن صلاحية SIP موجودة، لا نكرر الطلب في كل سهم/كل دقيقة.
            if feed == "sip" and APCA_FEED == "auto":
                _SIP_DISABLED_UNTIL = datetime.now(timezone.utc) + timedelta(minutes=SIP_RETRY_COOLDOWN_MIN)
                log.info("Alpaca SIP غير متاح حاليًا؛ الرجوع إلى IEX: %s", exc)
                continue
            if feed != feeds[-1]:
                continue
            raise
    raise last_exc or RuntimeError("Alpaca data unavailable")


def data_age_minutes(df: pd.DataFrame, now: Optional[datetime] = None) -> float:
    """عمر آخر شمعة بالدقائق، بناءً على timestamp الفعلي للفهرس."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return float("inf")
    ts = pd.Timestamp(df.index[-1])
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    now_ts = now_ts.tz_convert(ts.tz)
    age = (now_ts - ts).total_seconds() / 60.0
    # Timestamp مستقبلي بشكل غير منطقي = بيانات غير صالحة.
    if age < -2.0:
        return float("inf")
    return max(0.0, age)


def intraday_data_fresh(df: pd.DataFrame, interval: str, max_age_minutes: Optional[float] = None) -> tuple[bool, float]:
    defaults = {"1m": 4.0, "5m": 12.0, "15m": 25.0, "60m": 90.0, "1h": 90.0}
    age = data_age_minutes(df)
    limit = float(max_age_minutes if max_age_minutes is not None else defaults.get(interval, 12.0))
    return age <= limit, age


def fetch_latest_quote(symbol: str, feed: Optional[str] = None) -> dict:
    """أفضل Bid/Ask من Alpaca عند توفره، بدون اختراع قيم عند غياب الاقتباس."""
    if not alpaca_configured():
        return {}
    # في وضع auto نستخدم نفس الـfeed الذي نجح فعليًا مع آخر طلب شموع.
    # إذا لم ننجح بعد، البداية الآمنة هي IEX المجاني.
    use_feed = feed or (APCA_FEED if APCA_FEED in {"iex", "sip", "delayed_sip"} else _LAST_ALPACA_FEED)
    url = f"{DATA_URL}/v2/stocks/{symbol.upper()}/quotes/latest"
    params = {"feed": use_feed}
    r = _alpaca_get(url, params=params, timeout=5)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca quote {r.status_code}: {r.text[:160]}")
    data = r.json() or {}
    q = data.get("quote") or {}
    return {
        "bid": float(q.get("bp") or 0),
        "ask": float(q.get("ap") or 0),
        "timestamp": q.get("t"),
        "feed": use_feed,
    }


def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """يومي: Alpaca ثم Yahoo."""
    days = {"6mo": 190, "1y": 400, "2y": 800, "5d": 10, "1mo": 40, "3mo": 100}.get(period, 400)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, "1Day", start)
            if df is not None and len(df) >= 30:
                _set_status(df.attrs.get("data_source", "alpaca"))
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


def _period_days(period: str, default: int = 5) -> int:
    p = str(period or "").strip().lower()
    try:
        if p.endswith("d"):
            return max(1, int(p[:-1]))
        if p.endswith("mo"):
            return max(1, int(p[:-2]) * 31)
        if p.endswith("y"):
            return max(1, int(p[:-1]) * 365)
    except Exception:
        pass
    return default



def fetch_alpaca_bars_multi(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: Optional[datetime] = None,
    limit: int = 10000,
    chunk_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch many symbols in batches from Alpaca's multi-symbol bars endpoint.

    This is used by the scanners so 200 symbols do not become 400 individual
    HTTP requests.  The per-symbol fetch API remains unchanged for compatibility.
    """
    if not alpaca_configured():
        raise RuntimeError("Alpaca keys missing")
    end = end or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    clean = []
    seen = set()
    for sym in symbols or []:
        sym = str(sym).upper().strip()
        if sym and sym not in seen:
            clean.append(sym); seen.add(sym)

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(clean), max(1, int(chunk_size))):
        chunk = clean[i:i + max(1, int(chunk_size))]
        params = {
            "symbols": ",".join(chunk),
            "timeframe": timeframe,
            "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": min(limit, 10000),
            "adjustment": "split",
            "feed": APCA_FEED if APCA_FEED in {"iex", "sip", "delayed_sip"} else "iex",
        }
        url = f"{DATA_URL}/v2/stocks/bars"
        r = _alpaca_get(url, params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca multi {r.status_code}: {r.text[:180]}")
        data = r.json() or {}
        bars_by_symbol = data.get("bars") or {}
        next_token = data.get("next_page_token")
        while next_token:
            params["page_token"] = next_token
            r = _alpaca_get(url, params=params, timeout=15)
            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca multi page {r.status_code}: {r.text[:180]}")
            page = r.json() or {}
            for sym, bars in (page.get("bars") or {}).items():
                bars_by_symbol.setdefault(sym, []).extend(bars or [])
            next_token = page.get("next_page_token")

        for sym, bars in bars_by_symbol.items():
            df = _bars_to_df(bars or [])
            if df is not None and not df.empty:
                df.attrs["data_source"] = f"alpaca-{params['feed']}"
                df.attrs["feed"] = params["feed"]
                out[str(sym).upper()] = df

    _set_status(f"alpaca-{params['feed']}")
    return out

def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Unified timeframe loader. Keeps the old API but now correctly supports
    intraday + daily + weekly frames used by Daily V2."""
    interval = str(interval or "5m")
    period = str(period or "5d")

    tf_map = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "60m": "1Hour",
        "1h": "1Hour",
        "1Hour": "1Hour",
        "1d": "1Day",
        "1D": "1Day",
        "1wk": "1Week",
        "1w": "1Week",
        "1Week": "1Week",
    }
    alpaca_tf = tf_map.get(interval, "5Min")
    days = _period_days(period, 5)
    # Intraday needs a little calendar buffer for weekends/holidays;
    # daily/weekly use the requested lookback directly.
    if alpaca_tf in {"1Min", "5Min", "15Min", "1Hour"}:
        lookback_days = max(days + 3, 3)
    else:
        lookback_days = max(days + 5, 10)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, alpaca_tf, start)
            if df is not None and len(df) >= 10:
                _set_status("alpaca")
                return df
        except Exception as exc:
            log.warning("Alpaca intra %s %s: %s", symbol, interval, exc)
            _set_status("yahoo", str(exc)[:120])

    try:
        df = yf.Ticker(symbol).history(
            period=period, interval=interval, auto_adjust=True, prepost=False
        )
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
            df = yf.Ticker("SPY").history(period="5d", interval="1d")
            if df is not None and not df.empty:
                notes.append(f"Yahoo يعمل — SPY ≈ {float(df['Close'].iloc[-1]):.2f}")
                ok = True
            else:
                notes.append("Yahoo: بدون سعر")
    except Exception as exc:
        notes.append(f"Yahoo تعثر: {str(exc)[:80]}")

    return ok, " | ".join(notes)
