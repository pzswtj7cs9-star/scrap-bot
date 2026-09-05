"""
???? ??????? ???????:
- ????? ????
- ????? ???? 15 ?????
- ????? ???? 5 ?????
- ??? ????? ????? (???? + ATR ???????? + ??? 15?)
- ????? ?????? ?? ??? ??????
- ???? ???? ????? (Market Regime)
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
   timeframe_note: str = "????"
   factor_keys: list[str] = field(default_factory=list)
   regime: str = "neutral"
   sl_method: str = ""
   quality_ok: bool = True
   atr_pct: float = 0.0
   ext_sma20: float = 0.0
   structure_zone: str = "??????"  # ????? / ????? / ?????? - ?????? ???
   pattern_note: str = ""
   pattern_target: float | None = None
   near_new_high: bool = False
   dist_from_high_pct: float = 0.0
   period_high: float = 0.0
   vwap_day_note: str = "-"
   vol_at_level: bool = False
   dump_from_peak: bool = False
   dump_note: str = ""
   dump_unrecovered: bool = False
   dumped_then_reclaimed: bool = False


def _detect_structure_zone(df: pd.DataFrame, price: float, vol_ratio: float) -> str:
   """????? ???? ????? ????? ?? ?????? - ?????? ??? ??? ???? ??? ???????."""
   try:
       window = df.tail(40)
       hi = float(window["High"].max())
       lo = float(window["Low"].min())
       if hi <= lo:
           return "??????"
       pos = (price - lo) / (hi - lo)
       # ?????: ????? ?????? + ??? ?? ????
       if pos <= 0.35 and vol_ratio >= 0.9:
           return "????? ?????"
       # ?????: ????? ??????
       if pos >= 0.75:
           return "?????/??? ??? ?????"
       return "??????"
   except Exception:
       return "??????"


def _detect_classical_patterns(df: pd.DataFrame, price: float) -> tuple[str, float | None]:
   """
   ????? ???????? ??????? (?????? ??? - ???? ????? ?????).
   ???? ???????: ???/??? ??????? ??? ??????? ????? ???? ???? ????/???? ????.
   """
   try:
       if len(df) < 35:
           return "", None
       tail = df.tail(40)
       highs = tail["High"].astype(float)
       lows = tail["Low"].astype(float)
       closes = tail["Close"].astype(float)
       hi = float(highs.max())
       lo = float(lows.min())
       rng = hi - lo
       if rng <= 0:
           return "", None

       last_hi = float(highs.iloc[-8:].max())
       last_lo = float(lows.iloc[-8:].min())
       mid_hi = float(highs.iloc[-20:-8].max())
       mid_lo = float(lows.iloc[-20:-8].min())
       old_hi = float(highs.iloc[-35:-20].max())
       old_lo = float(lows.iloc[-35:-20].min())

       # ??? ?????
       if abs(old_lo - last_lo) / price < 0.015 and last_hi > old_lo * 1.03 and price > (old_lo + last_hi) / 2:
           tgt = last_hi + abs(last_hi - old_lo)
           return "??? ????? ????? (??????)", round(tgt, 2)

       # ??? ??????
       if abs(old_hi - last_hi) / price < 0.015 and last_lo < old_hi * 0.97 and price < (old_hi + last_lo) / 2:
           tgt = last_lo - abs(old_hi - last_lo)
           return "??? ?????? ?????? (??????)", round(tgt, 2)

       # ??? ?????? ????: ???-???-???
       if mid_hi > old_hi * 1.02 and mid_hi > last_hi * 1.02 and abs(old_hi - last_hi) / price < 0.03:
           neck = min(mid_lo, last_lo)
           tgt = neck - (mid_hi - neck)
           return "??? ?????? ????? (??????)", round(tgt, 2)

       # ??? ?????? ?????
       if mid_lo < old_lo * 0.98 and mid_lo < last_lo * 0.98 and abs(old_lo - last_lo) / price < 0.03:
           neck = max(mid_hi, last_hi)
           tgt = neck + (neck - mid_lo)
           return "??? ?????? ????? ????? (??????)", round(tgt, 2)

       # ???? ????
       if (old_hi - last_hi) / rng > 0.15 and (last_lo - old_lo) / rng > 0.15:
           tgt = price + rng * 0.5
           return "???? ??????/???? (??????)", round(tgt, 2)

       # ??? ???? (?????? ???? ??? ?????)
       if last_hi < old_hi and last_lo < old_lo and (old_hi - last_hi) > (old_lo - last_lo) and price > last_hi:
           tgt = price + abs(old_hi - last_lo) * 0.6
           return "??? ???? / ??? ????? (??????)", round(tgt, 2)

       # ??? ???? (?????? ???? ??? ?????)
       if last_hi > old_hi and last_lo > old_lo and price < last_lo:
           tgt = price - abs(last_hi - old_lo) * 0.6
           return "??? ???? / ??? ????? (??????)", round(tgt, 2)

       # ??? ???? ??? ????
       pole = float(closes.iloc[-25]) 
       if price > pole * 1.08 and (last_hi - last_lo) / price < 0.04:
           tgt = price + abs(price - pole) * 0.7
           return "???/????? ??? ???? (??????)", round(tgt, 2)

       # ????/???? ???? ????
       h1, h2 = float(highs.iloc[-18]), float(highs.iloc[-6])
       l1, l2 = float(lows.iloc[-18]), float(lows.iloc[-6])
       if h2 < h1 and l2 < l1 and price > h2:
           tgt = price + abs(h1 - l1) * 0.8
           return "????/???? ???? ???? (??????)", round(tgt, 2)

       # ????/???? ???? ????
       if h2 > h1 and l2 > l1 and price < l2:
           tgt = price - abs(h2 - l2) * 0.8
           return "????/???? ???? ???? (??????)", round(tgt, 2)

       return "?? ???? ??? ??????? ????", None
   except Exception:
       return "", None


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
       return f"{cap / 1e12:.2f} ??????? $"
   if cap >= 1e9:
       return f"{cap / 1e9:.1f} ????? $"
   if cap >= 1e6:
       return f"{cap / 1e6:.0f} ????? $"
   return str(int(cap))


def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
   from market_data import fetch_history as _fh

   df = _fh(symbol, period=period)
   if df is None or df.empty or len(df) < 60:
       raise ValueError(f"?? ???? ?????? ????? ?? {symbol}")
   return df.dropna(subset=["Close", "High", "Low", "Volume"])


def _historic_high(symbol: str, fallback_df: pd.DataFrame, price: float) -> float:
   """???? ??? ?? ?? ????? ????? ?????? - ???? ?? ?????."""
   try:
       from market_data import fetch_history as _fh
       long_df = _fh(symbol, period="max")
       if long_df is not None and not long_df.empty and "High" in long_df.columns:
           return float(long_df["High"].max())
   except Exception:
       pass
   if fallback_df is not None and not fallback_df.empty:
       return float(fallback_df["High"].max())
   return price


def fetch_intraday(symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
   from market_data import fetch_intraday as _fi

   df = _fi(symbol, period=period, interval=interval)
   if df is None or df.empty:
       raise ValueError(f"?? ???? ?????? ????? ?? {symbol}")
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
       return "A++", "???? ????? ???? ???? ?????"
   if score >= 85:
       return "A+", "????? ???? ???? ???????"
   if score >= 78:
       return "A", "????? ???? ????? ??? ?? ???????"
   if score >= 68:
       return "B", "?????? / ?????? ?????"
   if score >= 55:
       return "C", "?????"
   return "D", "?? ???? ????? ????"


def _compute_improved_stop(
   price: float,
   atr: float,
   swing_low: float,
   recent_low: float,
   m15_low: Optional[float],
   score: int,
) -> tuple[float, str]:
   """
   ??? ????? ?????:
   1) ??? ????? ????
   2) ATR ???????? (???? ?? ??????? ???????)
   3) ??? ???? 15 ????? ?? ????
   ????? ?????? ?????? ?? ???? ????/????.
   """
   # ????? ATR ??? ????? ???? ??????? ???? (??? ???? ? ??? ????)
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
       ("???? ????", structure_sl),
       ("ATR", atr_sl),
       ("??? ????", recent_sl),
   ]
   if m15_low is not None and m15_low < price:
       candidates.append(("??? 15?", m15_low - 0.05 * atr))

   # ????? ????? ?????? (???? ?????) ????? ??? ?? ??? ?? ?? ????
   min_dist = max(price * 0.010, atr * 0.65)   # ~1% ?? 0.65 ATR
   max_dist = price * 0.085                     # ?? ?????? ~8.5%

   valid = []
   for name, sl in candidates:
       dist = price - sl
       if min_dist <= dist <= max_dist:
           valid.append((name, sl, dist))

   if valid:
       # ?????? ????? = ??? ?????? ????? ?? ????? ???????
       valid.sort(key=lambda x: x[1], reverse=True)
       best_name, best_sl, _ = valid[0]
       return best_sl, best_name

   # fallback
   fallback = price - min(max_dist, max(min_dist, atr_mult * atr))
   return fallback, "???????"


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
   # ??? ??????? ????? ?????: ???? ??? ??? ???? ??? ????? ????
   period_high = _historic_high(symbol, df, price)
   dist_from_high_pct = ((period_high - price) / period_high * 100) if period_high else 0.0
   near_new_high = dist_from_high_pct < 2.0
   recent_low = float(df["Low"].iloc[-5:].min())
   # ???? ??? ?? ??? ?????: ???? ???? ?????? ????? ??????
   dump_from_peak = False
   dump_note = ""
   try:
       win = df["High"].iloc[-15:]
       local_peak = float(win.max())
       peak_ago = int(len(win) - 1 - win.values.argmax())
       drop_peak = ((local_peak - price) / local_peak * 100) if local_peak else 0.0
       if peak_ago <= 3 and drop_peak >= 3.0 and change_pct <= -2.0:
           dump_from_peak = True
           dump_note = f"???? ?? ??? ????? {local_peak:.2f} ({drop_peak:.1f}% ???? {peak_ago} ???)"
   except Exception:
       pass

   score = 0.0
   reasons: list[str] = []
   warnings: list[str] = []
   factors: list[str] = []

   w = WEIGHTS.get  # ??????

   # ---- ??????? ?????? ----
   uptrend_ok = False
   if price > last_sma200 and last_sma50 > last_sma200:
       score += w("uptrend_200")
       uptrend_ok = True
       reasons.append("??????? ?????? ???? (????? ?????? 50 ??? 200)")
       factors.append("uptrend_200")
   elif price > last_sma200:
       score += w("price_above_200")
       reasons.append("????? ??? ????? 200 ???")
       factors.append("price_above_200")
   else:
       warnings.append("????? ??? ????? 200 - ??????? ???? ????? ????")

   stacked = False
   if last_sma20 > last_sma50 > last_sma200:
       score += w("stacked_ma")
       stacked = True
       reasons.append("????? ????????? ???? 20 > 50 > 200")
       factors.append("stacked_ma")
   elif last_sma20 > last_sma50:
       score += w("sma20_above_50")
       reasons.append("????? 20 ??? ????? 50")
       factors.append("sma20_above_50")

   # ---- RSI ----
   if 50 <= last_rsi <= 64:
       score += w("rsi_healthy")
       reasons.append(f"RSI ?????? ?? ??? ??? ({last_rsi:.1f})")
       factors.append("rsi_healthy")
   elif 45 <= last_rsi < 50:
       score += w("rsi_near")
       reasons.append(f"RSI ?????? ???? ?? ???????? ({last_rsi:.1f})")
       factors.append("rsi_near")
   elif 64 < last_rsi <= 70:
       score += w("rsi_strong")
       reasons.append(f"RSI ??? ?????? ?? ???????? ({last_rsi:.1f})")
       warnings.append("?? ????? ????? ??? ???? ????? ????")
       factors.append("rsi_strong")
   elif last_rsi > 74:
       warnings.append(f"RSI ?????? ????? ???? ({last_rsi:.1f})")
   else:
       warnings.append(f"RSI ?????? ???? ({last_rsi:.1f})")

   # ---- MACD ----
   if last_hist > 0 and last_hist > prev_hist:
       score += w("macd_expand")
       reasons.append("MACD ?????? ???? ??????")
       factors.append("macd_expand")
   elif last_hist > 0:
       score += w("macd_positive")
       reasons.append("MACD ?????? ??? ?????")
       factors.append("macd_positive")
   else:
       warnings.append("MACD ?????? ?? ???? ?????? ???")

   # ---- ????? ----
   if vol_ratio >= 1.35:
       score += w("vol_high")
       reasons.append(f"??? ???? ????? ({vol_ratio:.2f}x)")
       factors.append("vol_high")
   elif vol_ratio >= 1.05:
       score += w("vol_ok")
       reasons.append("????? ?????? ?????")
       factors.append("vol_ok")
   else:
       warnings.append("????? ?????? ????")

   # ---- ????? ?? SMA20 ----
   ext_from_sma20 = (price - last_sma20) / last_sma20 * 100 if last_sma20 else 0
   if 0 <= ext_from_sma20 <= 4.0:
       score += w("near_sma20")
       reasons.append("????? ???? ?? ????? 20 (???? ???? ?? ????????)")
       factors.append("near_sma20")
   elif -2.2 <= ext_from_sma20 < 0:
       score += w("bounce_sma20")
       reasons.append("?????? ??? ????? 20")
       factors.append("bounce_sma20")
   elif ext_from_sma20 > 7:
       warnings.append("?????? ???? ?? ????? 20")
   else:
       score += 2

   if price > swing_low * 1.008:
       score += w("structure_hold")
       reasons.append("????? ??? ??? ?????")
       factors.append("structure_hold")

   # ---- ???? 15 ????? ----
   m15_ok = False
   m15_low = None
   live_price = None
   live_rsi = None
   live_vs_vwap = None
   vwap_day_note = "-"
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
               reasons.append(f"????? ???? 15? (??? EMA20 + RSI {m15_rsi:.0f})")
               factors.append("m15_confirm")
           elif above_m15_ema or rsi_m15_ok:
               score += w("m15_partial")
               reasons.append("????? ???? ??? 15 ?????")
               factors.append("m15_partial")
           else:
               warnings.append("???? 15? ?? ???? ?????? ???")
       except Exception:
           warnings.append("???? ??? ???? 15 ?????")

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
           vwap_day_note = "??? VWAP ?????" if live_price >= vwap_last else "??? VWAP ?????"
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
               reasons.append("????? ?????? ??? VWAP ?????? 20 ?? 5 ?????")
               factors.append("m5_vwap_ema")
           elif above_vwap or above_ema:
               live_points += w("m5_partial")
               reasons.append("????? ???? ??? ?????? 5 ?????")
               factors.append("m5_partial")
           else:
               warnings.append("?????? ?????? ?? ???? ??? (??? VWAP/????? 5?)")

           if last_green:
               live_points += w("m5_green")
               factors.append("m5_green")
           if rsi_ok:
               live_points += w("m5_rsi")
               reasons.append(f"RSI ?????? ???? ({live_rsi:.1f})")
               factors.append("m5_rsi")
           elif live_rsi is not None and live_rsi > 78:
               warnings.append(f"???? ???? ({live_rsi:.1f})")
           if live_vol_ratio >= 1.4:
               live_points += w("m5_vol")
               reasons.append(f"???? ???? ?? ????? ({live_vol_ratio:.2f}x)")
               factors.append("m5_vol")
           if live_momentum > 0.15:
               live_points += w("m5_momentum")
               factors.append("m5_momentum")

           live_ok = above_vwap and above_ema and not_climax and (last_green or live_momentum > 0)
           if live_ok:
               reasons.append("????? ???? ?????? ??? ???? 5 ?????")
           score += live_points

           if live_price:
               change_pct = (live_price - prev) / prev * 100
               price = live_price
       except Exception:
           warnings.append("???? ??? ??????? ?????? ???? - ?? ???????? ??? ?????? ??????")

   # ??? ??????? ?????? ??????? ??????
   if not uptrend_ok and not stacked:
       score = min(score, 84)

   # ---- ???? ???? ?????? ----
   # 1) ?????? ?? ????? 20  2) ????? ATR?  3) ??? ?????
   atr_pct = (last_atr / price * 100) if price else 0.0
   quality_ok = True
   if ext_from_sma20 > 7.0:
       quality_ok = False
       warnings.append(f"???? ?????? ?? ????? 20 ({ext_from_sma20:.1f}%) - ????? ????????")
       score = min(score, 84)
   if atr_pct > 5.5:
       quality_ok = False
       warnings.append(f"????? ???? ATR={atr_pct:.1f}% - ??? ????")
       score = min(score, 84)
   if vol_ratio < 0.90:
       quality_ok = False
       warnings.append(f"????? ???? ???? ({vol_ratio:.2f}x) - ?? ??? ????? ????")
       score = min(score, 84)
   elif vol_ratio < 1.05 and quality_ok:
       warnings.append(f"????? ??? ??????? ({vol_ratio:.2f}x)")

   # ????? ?????? + ??? ?????? - ?????? ??? (?? ???? ??? ???????)
   structure_zone = _detect_structure_zone(df, price, vol_ratio)
   pattern_note, pattern_target = _detect_classical_patterns(df, price)

   score_i = int(max(0, min(100, round(score))))
   grade, bias = _grade(score_i)

   # ---- ??? ??????? ??????? ----
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
   tp3 = price + risk * 4.0
   if swing_high and swing_high > price:
       swing_tp = swing_high * 1.03
       if swing_tp > tp2:
           tp3 = min(tp3, swing_tp)
   if not (tp1 < tp2 < tp3):
       tp1, tp2, tp3 = price + risk * 1.6, price + risk * 2.6, price + risk * 4.0
   risk_pct = risk / price * 100 if price else 0
   reward_r = (tp2 - price) / risk if risk > 0 else 0

   def _near(level: float | None, pct: float = 1.2) -> bool:
       if not level:
           return False
       return abs(price - level) / price * 100 <= pct

   vol_at_level = bool(
       vol_ratio >= 1.00
       and (
           _near(last_sma20)
           or _near(swing_low)
           or _near(swing_high)
           or _near(recent_low)
       )
   )

   dump_unrecovered = False
   dumped_then_reclaimed = False
   try:
       win7 = df["High"].iloc[-7:]
       peak7 = float(win7.max())
       ago7 = int(len(win7) - 1 - win7.values.argmax())
       drop7 = ((peak7 - price) / peak7 * 100) if peak7 else 0.0
       dumped_week = ago7 <= 7 and drop7 >= 3.0
       reclaimed = (last_sma20 and price >= last_sma20 * 0.999) or (
           "???" in str(vwap_day_note or "")
       )
       dumped_then_reclaimed = bool(dumped_week and reclaimed)
       if dumped_week and not reclaimed:
           dump_unrecovered = True
           if not dump_note:
               dump_note = f"???? ?? ??? {peak7:.2f} ??? ?????? ??? VWAP/????? 20"
   except Exception:
       pass

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

   tf_note = "????"
   if with_live:
       parts = ["????"]
       if m15_ok or True:
           parts.append("15?")
       parts.append("5?")
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
       structure_zone=structure_zone,
       pattern_note=pattern_note,
       pattern_target=pattern_target,
       near_new_high=near_new_high,
       dist_from_high_pct=round(dist_from_high_pct, 2),
       period_high=round(period_high, 2),
       vwap_day_note=vwap_day_note,
       vol_at_level=vol_at_level,
       dump_from_peak=dump_from_peak,
       dump_note=dump_note,
       dump_unrecovered=dump_unrecovered,
       dumped_then_reclaimed=dumped_then_reclaimed,
   )


def format_signal_ar(sig: SignalResult, min_score: int = 84, rank: int | None = None) -> str:
   arrow = "?" if sig.change_pct >= 0 else "?"
   conf = []
   if sig.m15_ok:
       conf.append("15?")
   if sig.live_ok:
       conf.append("5?")
   live = "+".join(conf) if conf else "????"
   head = f"? ?????/???? | {sig.symbol} | {sig.score}/100 | {sig.grade} | {live}"
   if rank is not None:
       head = f"#{rank}  {head}"

   vol_lvl = "???" if getattr(sig, "vol_at_level", False) else "??"
   vwap_note = getattr(sig, "vwap_day_note", None) or "-"
   lines = [
       head,
       f"{sig.name}",
       "-------------",
       f"?????: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
       f"????: {sig.buy_low:.2f} - {sig.buy_high:.2f}",
       f"???: {sig.stop_loss:.2f} ({sig.sl_method})  |  ?????? {sig.risk_pct:.2f}%",
       f"?????: {sig.tp1:.2f}  |  {sig.tp2:.2f}  |  {sig.tp3:.2f}",
       "-------------",
       f"???????: {getattr(sig, 'structure_zone', '??????')} (?????? ???)",
   ]
   if getattr(sig, "pattern_note", ""):
       pt = getattr(sig, "pattern_target", None)
       if pt:
           lines.append(f"???: {sig.pattern_note} | ??? ?????? {pt:.2f}")
       else:
           lines.append(f"???: {sig.pattern_note}")
   if getattr(sig, "near_new_high", False):
       lines.append(
           f"??? ??? ??????? ?????: {getattr(sig, 'period_high', 0):.2f} | ??????? {getattr(sig, 'dist_from_high_pct', 0):.2f}%"
       )
   if getattr(sig, "dump_from_peak", False) or getattr(sig, "dump_unrecovered", False):
       lines.append(getattr(sig, "dump_note", "") or "???? ?? ??? ??? ?????? ???? - ????? ?????? ????????")
   lines.append("")
   lines.append(f"{vwap_note} | ????? ??? ???????: {vol_lvl}")
   lines.append("")
   lines.append("????? ?????? - ???? ?????.")
   return "\n".join(lines)


def scan_symbols(
   symbols: list[str],
   names: dict,
   min_score: int = 84,
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

   try:
       from stocks import MAX_AUTO_PRICE
   except Exception:
       MAX_AUTO_PRICE = 200.0

   # ???? ???? ?????
   try:
       from market import get_market_regime

       regime = get_market_regime("SPY")
       effective_min = max(min_score, regime.get("min_score_adj", min_score))
   except Exception:
       effective_min = min_score
       regime = {"regime": "neutral", "allow_auto": True}
   try:
       from auto_tune import AUTO
       effective_min = AUTO.effective_floor(int(effective_min))
       vol_gate = AUTO.volume_gate()
   except Exception:
       vol_gate = 1.00

   for sym in symbols:
       try:
           if skip_earnings and is_near_earnings is not None:
               near, edt = is_near_earnings(sym, within_days=earnings_days)
               if near:
                   near_earn.append(f"{sym}({edt})")
                   continue
           sig = analyze(sym, names.get(sym, sym), with_live=True)
           sig.regime = regime.get("regime", "neutral")
           # ?? ????? ??????? ????????
           if sig.price > float(MAX_AUTO_PRICE):
               continue
           if sig.score < effective_min:
               continue
           if require_live and not sig.live_ok:
               continue
           if require_m15 and not sig.m15_ok:
               continue
           # ???? ???? ?????? ??????? ????????
           if not getattr(sig, "quality_ok", True):
               continue
           if getattr(sig, "volume_ratio", 0) < vol_gate:
               continue
           if getattr(sig, "near_new_high", False):
               continue
           if getattr(sig, "dump_from_peak", False) or getattr(sig, "dump_unrecovered", False):
               continue
           try:
               from auto_tune import AUTO
               if getattr(AUTO, "require_strong_reclaim", False) and getattr(sig, "dumped_then_reclaimed", False):
                   above_vwap = "???" in str(getattr(sig, "vwap_day_note", "") or "")
                   above_ma = bool(sig.sma20 and sig.price >= sig.sma20 * 0.999)
                   if not (above_vwap and above_ma):
                       continue
           except Exception:
               pass
           try:
               from auto_tune import AUTO
               if AUTO.require_above_vwap and "???" in str(getattr(sig, "vwap_day_note", "") or ""):
                   continue
               if AUTO.require_vol_at_level and not getattr(sig, "vol_at_level", False):
                   continue
           except Exception:
               pass
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

