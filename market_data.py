"""طبقة بيانات موحّدة: Alpaca أولاً ثم Twelve Data كاحتياطي فقط."""

from __future__ import annotations

import logging
import json
import os
import threading
from collections import deque
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("halal-bot.data")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
DATA_URL = os.getenv("APCA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
# Alpaca execution/data feed is intentionally pinned to IEX for this deployment.
# Twelve Data is the only external fallback when Alpaca is unavailable or stale.
APCA_FEED = "iex"
_LAST_ALPACA_FEED = "iex"
TWELVE_API_KEY = (os.getenv("TWELVE_DATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY") or "").strip()
TWELVE_URL = os.getenv("TWELVE_DATA_URL", "https://api.twelvedata.com").rstrip("/")
TWELVE_TIMEOUT = float(os.getenv("TWELVE_DATA_TIMEOUT", "10"))
TWELVE_429_RETRIES = int(os.getenv("TWELVE_DATA_429_RETRIES", "2"))
_TWELVE_RATE_LOCK = threading.Lock()
_TWELVE_NEXT_REQUEST_AT = 0.0
# Once Twelve Data reports that the daily quota is exhausted, retrying every
# symbol is guaranteed to fail and only wastes time/credits. Keep a process-wide
# circuit breaker until the process restarts or an operator resets the quota.
_TWELVE_QUOTA_LOCK = threading.Lock()
_TWELVE_QUOTA_EXHAUSTED = False
_TWELVE_QUOTA_REASON = ""


_LAST_SOURCE = "none"
_LAST_ERROR = ""

# Central Alpaca request pacing shared by all threads/jobs in this process.
# The goal is to prevent bursty parallel scans from triggering HTTP 429.
APCA_MIN_REQUEST_INTERVAL = float(os.getenv("APCA_MIN_REQUEST_INTERVAL", "0.35"))
APCA_429_RETRIES = int(os.getenv("APCA_429_RETRIES", "3"))
_APCA_RATE_LOCK = threading.Lock()
_APCA_NEXT_REQUEST_AT = 0.0

# Rolling Alpaca IEX WebSocket cache for the dynamically selected intraday Top-30.
# This is a data-layer cache only; it never changes strategy/scoring logic.
_WS_LOCK = threading.RLock()
_WS_STOP = threading.Event()
_WS_THREAD = None
_WS_SYMBOLS: set[str] = set()
_WS_BARS: dict[str, deque] = {}
_WS_CONNECTED = False
_WS_LAST_ERROR = ""
_WS_MAX_BARS_PER_SYMBOL = 10
# The reader thread owns the live socket.  Desired symbols are updated by
# set_intraday_ws_symbols(); _ws_loop applies only the add/remove delta on the
# existing connection instead of closing and reconnecting on every Top-30 change.
_WS_SUBSCRIBED_SYMBOLS: set[str] = set()

def _ws_normalize_symbols(symbols) -> list[str]:
    out = []
    seen = set()
    for sym in symbols or []:
        s = str(sym).upper().strip()
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out

def _ws_store_bar(msg: dict) -> None:
    sym = str(msg.get("S") or msg.get("symbol") or "").upper().strip()
    ts = msg.get("t") or msg.get("timestamp")
    if not sym or not ts:
        return
    try:
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        dt = dt.tz_convert("America/New_York")
        row = {
            "Open": float(msg["o"]), "High": float(msg["h"]),
            "Low": float(msg["l"]), "Close": float(msg["c"]),
            "Volume": float(msg.get("v") or 0),
        }
    except Exception:
        return
    with _WS_LOCK:
        q = _WS_BARS.setdefault(sym, deque(maxlen=_WS_MAX_BARS_PER_SYMBOL))
        q.append((dt, row))

def _ws_loop() -> None:
    global _WS_CONNECTED, _WS_LAST_ERROR
    try:
        import websocket
    except Exception as exc:
        with _WS_LOCK:
            _WS_LAST_ERROR = f"websocket-client unavailable: {exc}"
        log.error("ALPACA IEX WS UNAVAILABLE | %s", _WS_LAST_ERROR)
        return

    url = "wss://stream.data.alpaca.markets/v2/iex"
    while not _WS_STOP.is_set():
        with _WS_LOCK:
            desired = set(_WS_SYMBOLS)
        if not desired or not alpaca_configured():
            _WS_STOP.wait(1.0)
            continue

        ws = None
        subscribed: set[str] = set()
        try:
            ws = websocket.create_connection(url, timeout=5, enable_multithread=True)
            ws.send(json.dumps({"action": "auth", "key": APCA_KEY, "secret": APCA_SECRET}))
            auth_raw = ws.recv()
            auth_payload = json.loads(auth_raw) if auth_raw else []
            if isinstance(auth_payload, dict):
                auth_payload = [auth_payload]
            if not any(isinstance(m, dict) and m.get("T") == "success" for m in (auth_payload or [])):
                raise RuntimeError(f"WebSocket auth failed: {auth_raw}")

            # Initial subscription for the current desired Top-30.
            ws.send(json.dumps({"action": "subscribe", "bars": sorted(desired)}))
            subscribed = set(desired)
            with _WS_LOCK:
                _WS_SUBSCRIBED_SYMBOLS.clear()
                _WS_SUBSCRIBED_SYMBOLS.update(subscribed)
                _WS_CONNECTED = True
                _WS_LAST_ERROR = ""
            log.info("ALPACA IEX WS CONNECTED | symbols=%d", len(subscribed))

            while not _WS_STOP.is_set():
                # Keep one persistent connection.  If Top-30 membership changes,
                # update only the delta on this same socket.
                with _WS_LOCK:
                    desired = set(_WS_SYMBOLS)

                to_remove = subscribed - desired
                to_add = desired - subscribed

                if to_remove:
                    ws.send(json.dumps({"action": "unsubscribe", "bars": sorted(to_remove)}))
                    subscribed -= to_remove

                if to_add:
                    ws.send(json.dumps({"action": "subscribe", "bars": sorted(to_add)}))
                    subscribed |= to_add

                if to_remove or to_add:
                    with _WS_LOCK:
                        _WS_SUBSCRIBED_SYMBOLS.clear()
                        _WS_SUBSCRIBED_SYMBOLS.update(subscribed)
                    log.info(
                        "ALPACA IEX WS SUBSCRIPTION UPDATED | symbols=%d | added=%d | removed=%d",
                        len(subscribed), len(to_add), len(to_remove),
                    )

                ws.settimeout(2.0)
                try:
                    raw = ws.recv()
                except Exception as exc:
                    if "timed out" in str(exc).lower():
                        continue
                    raise
                if not raw:
                    continue
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    payload = [payload]
                for msg in payload if isinstance(payload, list) else []:
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("T")
                    if msg_type == "b":
                        _ws_store_bar(msg)
                    elif msg_type == "subscription":
                        bars = msg.get("bars") or []
                        log.debug("ALPACA IEX WS SUBSCRIPTION ACK | bars=%d", len(bars))

        except Exception as exc:
            with _WS_LOCK:
                _WS_CONNECTED = False
                _WS_SUBSCRIBED_SYMBOLS.clear()
                _WS_LAST_ERROR = str(exc)[:240]
            log.warning("ALPACA IEX WS ERROR | %s", _WS_LAST_ERROR)
            _WS_STOP.wait(2.0)
        finally:
            with _WS_LOCK:
                _WS_CONNECTED = False
                _WS_SUBSCRIBED_SYMBOLS.clear()
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass

def set_intraday_ws_symbols(symbols: list[str]) -> None:
    """Update the rolling IEX WebSocket subscription for the current Top-30."""
    global _WS_THREAD
    clean = _ws_normalize_symbols(symbols)[:30]
    with _WS_LOCK:
        changed = set(clean) != _WS_SYMBOLS
        _WS_SYMBOLS.clear()
        _WS_SYMBOLS.update(clean)
        for sym in clean:
            _WS_BARS.setdefault(sym, deque(maxlen=_WS_MAX_BARS_PER_SYMBOL))
        if _WS_THREAD is None or not _WS_THREAD.is_alive():
            _WS_STOP.clear()
            _WS_THREAD = threading.Thread(target=_ws_loop, name="alpaca-iex-top30", daemon=True)
            _WS_THREAD.start()
    if changed:
        log.info("ALPACA IEX WS SUBSCRIPTION UPDATED | symbols=%d", len(clean))

def get_intraday_ws_bars(symbol: str) -> pd.DataFrame:
    """Return cached recent 1m IEX bars for one subscribed symbol."""
    sym = str(symbol).upper().strip()
    with _WS_LOCK:
        rows = list(_WS_BARS.get(sym, ()))
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame([row for _, row in rows], index=pd.DatetimeIndex([ts for ts, _ in rows]))
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data.attrs["data_source"] = "alpaca-iex-websocket"
    data.attrs["feed"] = "iex"
    return data

def websocket_5m_patch(symbol: str, base: pd.DataFrame) -> pd.DataFrame:
    """Patch only the latest completed 5m bucket from cached IEX 1m bars."""
    if base is None or base.empty:
        return base
    one = get_intraday_ws_bars(symbol)
    if one is None or one.empty:
        return base
    idx = pd.DatetimeIndex(one.index)
    bucket = idx.floor("5min")
    tmp = one.copy(); tmp["_bucket"] = bucket
    agg = tmp.groupby("_bucket", sort=True).agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum")
    )
    if agg.empty:
        return base
    latest = agg.index[-1]
    # Only replace a bucket when all five minute bars are present; otherwise
    # leave the historical REST candle untouched until the bucket completes.
    count = int((bucket == latest).sum())
    if count < 5:
        return base
    out = base.copy(); out.index = pd.DatetimeIndex(out.index)
    out = out[out.index.floor("5min") != latest]
    out = pd.concat([out, agg.loc[[latest]]])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs.update(base.attrs)
    out.attrs["data_source"] = "alpaca-iex+websocket"
    out.attrs["websocket_1m_count"] = count
    return out


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


def twelve_configured() -> bool:
    return bool(TWELVE_API_KEY)


def _twelve_get(endpoint: str, params: dict, timeout: float | None = None) -> requests.Response:
    """Twelve Data GET with pacing plus a quota-exhaustion circuit breaker."""
    global _TWELVE_NEXT_REQUEST_AT, _TWELVE_QUOTA_EXHAUSTED, _TWELVE_QUOTA_REASON
    with _TWELVE_QUOTA_LOCK:
        if _TWELVE_QUOTA_EXHAUSTED:
            raise RuntimeError(f"Twelve Data quota exhausted: {_TWELVE_QUOTA_REASON or 'daily API credits exhausted'}")
    url = f"{TWELVE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    params = dict(params or {})
    params.setdefault("apikey", TWELVE_API_KEY)
    last_exc = None
    for attempt in range(TWELVE_429_RETRIES + 1):
        with _TWELVE_RATE_LOCK:
            now = _time.monotonic()
            wait = _TWELVE_NEXT_REQUEST_AT - now
            if wait > 0:
                _time.sleep(wait)
            _TWELVE_NEXT_REQUEST_AT = _time.monotonic() + 0.25
            try:
                r = requests.get(url, params=params, timeout=timeout or TWELVE_TIMEOUT)
            except Exception as exc:
                last_exc = exc
                continue
        if r.status_code != 429:
            return r

        # A quota-exhausted 429 is not a transient rate-limit event. Do not
        # retry it and do not let every symbol repeat the same failed request.
        body = (r.text or "").lower()
        if "run out of api credits" in body or "api credits for the day" in body or "quota" in body:
            with _TWELVE_QUOTA_LOCK:
                _TWELVE_QUOTA_EXHAUSTED = True
                _TWELVE_QUOTA_REASON = (r.text or "daily API credits exhausted")[:180]
            log.error("Twelve Data quota exhausted; disabling further fallback requests for this process")
            raise RuntimeError("Twelve Data quota exhausted")

        if attempt >= TWELVE_429_RETRIES:
            return r
        retry_after = r.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(8.0, 1.0 * (2 ** attempt))
        except Exception:
            delay = min(8.0, 1.0 * (2 ** attempt))
        log.warning("Twelve Data 429: انتظار %.1fs ثم إعادة المحاولة (%d/%d)", delay, attempt + 1, TWELVE_429_RETRIES)
        _time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("Twelve Data request failed")


def _twelve_time_series(symbol: str, interval: str, start: datetime, end: datetime, outputsize: int = 5000) -> pd.DataFrame:
    """Fetch Twelve Data OHLCV and normalize it to the bot's DataFrame schema."""
    if not twelve_configured():
        raise RuntimeError("Twelve Data API key missing")
    td_interval = {
        "1m": "1min", "1Min": "1min", "5m": "5min", "5Min": "5min",
        "15m": "15min", "15Min": "15min", "60m": "1h", "1h": "1h", "1Hour": "1h",
        "1d": "1day", "1D": "1day", "1Day": "1day",
        "1wk": "1week", "1w": "1week", "1Week": "1week",
    }.get(interval, str(interval).lower())
    params = {
        "symbol": symbol.upper().strip(),
        "interval": td_interval,
        "start_date": start.astimezone(timezone.utc).isoformat(),
        "end_date": end.astimezone(timezone.utc).isoformat(),
        "outputsize": min(int(outputsize), 5000),
        "order": "asc",
        "timezone": "America/New_York",
    }
    r = _twelve_get("time_series", params)
    if r.status_code >= 400:
        raise RuntimeError(f"Twelve Data {r.status_code}: {r.text[:180]}")
    data = r.json() or {}
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data.get('message') or data.get('code') or 'no values'}")
    values = data.get("values") or []
    rows = []
    idx = []
    for v in values:
        try:
            ts = pd.Timestamp(v.get("datetime"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
            rows.append({
                "Open": float(v["open"]), "High": float(v["high"]),
                "Low": float(v["low"]), "Close": float(v["close"]),
                "Volume": float(v.get("volume") or 0),
            })
            idx.append(ts)
        except Exception:
            continue
    if not rows:
        raise RuntimeError("Twelve Data returned no valid bars")
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["data_source"] = "twelve-data"
    return df


def _twelve_fresh(df: pd.DataFrame, interval: str) -> tuple[bool, float]:
    """Hard freshness gate for fallback data; stale fallback is never returned."""
    if interval in {"1m", "5m", "15m", "60m", "1h", "1Hour", "1Min", "5Min", "15Min"}:
        return intraday_data_fresh(df, {"1Min":"1m","5Min":"5m","15Min":"15m","1Hour":"60m"}.get(interval, interval), None)
    age = data_age_minutes(df)
    limits = {"1d": 96 * 60, "1D": 96 * 60, "1Day": 96 * 60, "1wk": 10 * 24 * 60, "1w": 10 * 24 * 60, "1Week": 10 * 24 * 60}
    return age <= float(limits.get(interval, 96 * 60)), age


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
    global _LAST_ALPACA_FEED
    if not alpaca_configured():
        raise RuntimeError("Alpaca keys missing")
    end = end or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    feed = "iex"
    try:
        df = _alpaca_request_bars(symbol, timeframe, start, end, limit, feed)
        _LAST_ALPACA_FEED = feed
        if df is not None:
            df.attrs["data_source"] = "alpaca-iex"
            df.attrs["feed"] = "iex"
        _set_status("alpaca-iex")
        return df
    except Exception:
        raise


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
    """Latest Alpaca quote, pinned to IEX for deterministic execution data."""
    if not alpaca_configured():
        return {}
    use_feed = "iex"
    url = f"{DATA_URL}/v2/stocks/{symbol.upper()}/quotes/latest"
    r = _alpaca_get(url, params={"feed": use_feed}, timeout=5)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca quote {r.status_code}: {r.text[:160]}")
    data = r.json() or {}
    q = data.get("quote") or {}
    _set_status("alpaca-iex")
    return {
        "bid": float(q.get("bp") or 0),
        "ask": float(q.get("ap") or 0),
        "timestamp": q.get("t"),
        "feed": "iex",
    }

def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily history: Alpaca first, then Twelve Data only if Alpaca fails/stales."""
    days = {"6mo": 190, "1y": 400, "2y": 800, "5d": 10, "1mo": 40, "3mo": 100}.get(period, 400)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    last_errors = []

    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, "1Day", start)
            if df is not None and len(df) >= 30:
                fresh_ok, age = _twelve_fresh(df, "1Day")
                if fresh_ok:
                    _set_status(df.attrs.get("data_source", "alpaca-iex"))
                    return df
                last_errors.append(f"Alpaca stale age={age:.1f}m")
                log.warning("DATA STALE | %s | Alpaca daily age=%.1fm", symbol, float(age))
            else:
                last_errors.append("Alpaca daily incomplete")
        except Exception as exc:
            last_errors.append(f"Alpaca: {str(exc)[:100]}")
            log.warning("Alpaca daily %s: %s", symbol, exc)

    if twelve_configured():
        try:
            df = _twelve_time_series(symbol, "1Day", start, datetime.now(timezone.utc), outputsize=min(days + 10, 5000))
            if len(df) >= 30:
                fresh_ok, age = _twelve_fresh(df, "1Day")
                if fresh_ok:
                    _set_status("twelve-data")
                    log.info("DATA FALLBACK | %s | Twelve Data daily | age=%.1fm", symbol, float(age))
                    return df
                last_errors.append(f"Twelve Data stale age={age:.1f}m")
        except Exception as exc:
            last_errors.append(f"Twelve Data: {str(exc)[:100]}")
            log.warning("Twelve Data daily %s: %s", symbol, exc)

    _set_status("none", " | ".join(last_errors)[-240:])
    raise RuntimeError(f"No fresh daily data for {symbol}: {' | '.join(last_errors)}")



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
    global _LAST_ALPACA_FEED
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
    feed = "iex"
    for i in range(0, len(clean), max(1, int(chunk_size))):
        chunk = clean[i:i + max(1, int(chunk_size))]
        params = {
            "symbols": ",".join(chunk),
            "timeframe": timeframe,
            "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": min(limit, 10000),
            "adjustment": "split",
            "feed": feed,
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
                df.attrs["data_source"] = "alpaca-iex"
                df.attrs["feed"] = "iex"
                out[str(sym).upper()] = df
        _LAST_ALPACA_FEED = "iex"

    _set_status("alpaca-iex")
    return out


def _refresh_recent_5m_from_1m(symbol: str, base: pd.DataFrame, feed: Optional[str] = None) -> pd.DataFrame:
    """Refresh only completed 5m buckets from fresh Alpaca IEX 1m bars."""
    if base is None or base.empty or not alpaca_configured():
        return base
    try:
        now = datetime.now(timezone.utc)
        one_min = fetch_alpaca_bars(symbol, "1Min", now - timedelta(minutes=25), now, limit=200)
        if one_min is None or one_min.empty:
            return base
        ok, age = intraday_data_fresh(one_min, "1m", 4.0)
        if not ok:
            log.warning("DATA REFRESH 1m STALE | %s | age=%.1fm", symbol, float(age))
            return base
        idx = pd.DatetimeIndex(one_min.index)
        bucket = idx.floor("5min")
        tmp = one_min.copy(); tmp["_bucket"] = bucket
        counts = bucket.value_counts()
        completed = sorted([b for b, c in counts.items() if int(c) >= 5])
        if not completed:
            return base
        latest_bucket = completed[-1]
        agg = tmp.groupby("_bucket", sort=True).agg(
            Open=("Open", "first"), High=("High", "max"),
            Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum")
        )
        if latest_bucket not in agg.index:
            return base
        base2 = base.copy(); base2.index = pd.DatetimeIndex(base2.index)
        base2 = base2[base2.index.floor("5min") < latest_bucket]
        refreshed = pd.concat([base2, agg.loc[:latest_bucket]])
        refreshed = refreshed[~refreshed.index.duplicated(keep="last")].sort_index()
        refreshed.attrs.update(base.attrs)
        refreshed.attrs["data_source"] = str(base.attrs.get("data_source", "alpaca")) + "+1m-refresh"
        refreshed.attrs["recent_1m_age_min"] = float(age)
        log.info("DATA REFRESH 5m | %s | rebuilt_latest_completed_bucket=%s | 1m_age=%.1fm", symbol, str(latest_bucket), float(age))
        return refreshed
    except Exception as exc:
        log.warning("DATA REFRESH 5m FAILED | %s | %s", symbol, str(exc))
        return base

def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Unified timeframe loader with freshness-aware recent 5m repair."""
    interval = str(interval or "5m")
    period = str(period or "5d")

    tf_map = {
        "1m": "1Min", "5m": "5Min", "15m": "15Min",
        "60m": "1Hour", "1h": "1Hour", "1Hour": "1Hour",
        "1d": "1Day", "1D": "1Day", "1wk": "1Week",
        "1w": "1Week", "1Week": "1Week",
    }
    alpaca_tf = tf_map.get(interval, "5Min")
    days = _period_days(period, 5)
    if alpaca_tf in {"1Min", "5Min", "15Min", "1Hour"}:
        lookback_days = max(days + 3, 3)
    else:
        lookback_days = max(days + 5, 10)
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    last_errors = []
    if alpaca_configured():
        try:
            df = fetch_alpaca_bars(symbol, alpaca_tf, start)
            if df is not None and len(df) >= 10:
                fresh_ok, age = _twelve_fresh(df, interval)
                if alpaca_tf == "5Min" and not fresh_ok:
                    repaired = _refresh_recent_5m_from_1m(symbol, df)
                    if repaired is not None and not repaired.empty:
                        df = repaired
                        fresh_ok, age = _twelve_fresh(df, interval)
                log.info("DATA HEALTH | %s | %s | interval=%s | age=%.1fm | fresh=%s",
                          symbol, df.attrs.get("data_source", "alpaca-iex"), interval, float(age), bool(fresh_ok))
                if fresh_ok:
                    _set_status(df.attrs.get("data_source", "alpaca-iex"))
                    return df
                last_errors.append(f"Alpaca stale age={age:.1f}m")
        except Exception as exc:
            last_errors.append(f"Alpaca: {str(exc)[:100]}")
            log.warning("Alpaca intra %s %s: %s", symbol, interval, exc)

    if twelve_configured():
        try:
            # Keep fallback payloads tight. Twelve Data usage is quota-sensitive;
            # the analyzer only needs enough bars for the requested lookback,
            # not the 5000-bar maximum.
            if interval in {"5m", "5Min"}:
                td_outputsize = min(max(days * 78 + 30, 120), 800)
            elif interval in {"15m", "15Min"}:
                td_outputsize = min(max(days * 26 + 20, 80), 400)
            elif interval in {"60m", "1h", "1Hour"}:
                td_outputsize = min(max(days * 7 + 10, 60), 150)
            elif interval in {"1m", "1Min"}:
                td_outputsize = min(max(days * 390 + 30, 300), 2000)
            else:
                td_outputsize = min(max(days * 2 + 10, 30), 500)
            df = _twelve_time_series(symbol, interval, start, datetime.now(timezone.utc), outputsize=td_outputsize)
            if df is not None and len(df) >= 10:
                fresh_ok, age = _twelve_fresh(df, interval)
                log.info("DATA FALLBACK | %s | Twelve Data | interval=%s | age=%.1fm | fresh=%s",
                          symbol, interval, float(age), bool(fresh_ok))
                if fresh_ok:
                    _set_status("twelve-data")
                    return df
                last_errors.append(f"Twelve Data stale age={age:.1f}m")
        except Exception as exc:
            last_errors.append(f"Twelve Data: {str(exc)[:100]}")
            log.warning("Twelve Data intra %s %s: %s", symbol, interval, exc)

    _set_status("none", " | ".join(last_errors)[-240:])
    raise RuntimeError(f"No fresh {interval} data for {symbol}: {' | '.join(last_errors)}")


def ping_sources() -> tuple[bool, str]:
    """Health check for the live data chain: Alpaca -> Twelve Data."""
    notes = []
    ok = False
    try:
        df = fetch_intraday("SPY", interval="1d", period="5d")
        if df is not None and not df.empty:
            last = float(df["Close"].iloc[-1])
            source = str(df.attrs.get("data_source", last_source()))
            notes.append(f"مصدر البيانات يعمل ({source}) — SPY ≈ {last:.2f}")
            ok = True
    except Exception as exc:
        notes.append(f"مصدر البيانات تعثر: {str(exc)[:100]}")
    if not alpaca_configured():
        notes.append("Alpaca: غير مُعد")
    if not twelve_configured():
        notes.append("Twelve Data: API key غير مُعد")
    return ok, " | ".join(notes)
