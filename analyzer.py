"""
محرك السوينغ/اليومي V2 — نفس تكتيك اليومي، مع أطر زمنية يومية/أسبوعية/4س.
- اتجاه: أسبوعي
- تأكيد ناعم: 4 ساعات
- دخول/زخم: يومي
- TP1: أقرب مقاومة/قمة يومية أو أسبوعية مناسبة
- تعلم مستقل من نتائج اليومي فقط
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import time as time_module
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from market import now_ny, REGULAR_OPEN, REGULAR_CLOSE, is_us_regular_session, session_label
from stocks import MAX_AUTO_PRICE

log = logging.getLogger("halal-bot.daily")

SKIP_OPEN_MIN = 0
SKIP_CLOSE_MIN = 0
DAILY_MIN_SCORE = 82

DAILY_LEARNING_FILE = Path("/var/data/daily_v2_learning.jsonl")
LEARNING_MIN_SAMPLES = 20
LEARNING_LOOKBACK = 60
LEARNING_MAX_ADJUSTMENT = 4.0
ADAPTIVE_POLICY_FILE = Path("/var/data/daily_v2_adaptive_policy.json")
ADAPTIVE_MIN_SAMPLES = 30
ADAPTIVE_CONFIRM_SAMPLES = 40
ADAPTIVE_MAX_CHANGE = 0.15
ADAPTIVE_BEST_FILE = Path("/var/data/daily_v2_adaptive_best.json")
ADAPTIVE_SHADOW_FILE = Path("/var/data/daily_v2_shadow_results.jsonl")
LEARNING_ALERT_FILE = Path("/var/data/daily_v2_learning_alert.json")

# Canonical list: the adaptive learner must track every real entry strategy.
ENTRY_TYPES = (
    "اختراق مؤكد", "إعادة اختبار", "دخول مبكر", "ارتداد VWAP", "ارتداد EMA20",
    "سحب سيولة", "اختراق نطاق الافتتاح", "استمرار الزخم", "ضغط ثم انفجار",
    "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة الفترة",
    "استعادة بعد فشل ORB", "استمرار ABC", "سحب سيولة مع Displacement",
)
PREFILTER_MAX_CANDIDATES = 50
ADAPTIVE_MIN_EDGE = 0.04
ADAPTIVE_MIN_COVERAGE = 0.45
ADAPTIVE_ROLLBACK_DROP = 0.06

# Self-adaptive market-regime layer: starts neutral, learns in shadow,
# and activates automatically only after an out-of-sample improvement.
REGIME_MIN_SAMPLES = 8
REGIME_WEIGHT_STEP = 0.03
REGIME_WEIGHT_MIN = 0.85
REGIME_WEIGHT_MAX = 1.15
REGIME_MIN_EDGE = 0.04
REGIME_MIN_COVERAGE = 0.45

# Adaptive Exit Engine: learns TP/SL behavior from MFE/MAE and time-to-result.
# It starts shadow-only and can activate automatically after OOS validation.
EXIT_MIN_SAMPLES = 12
EXIT_TP_STEP_R = 0.05
EXIT_SL_STEP = 0.03
EXIT_TP_MIN_R = 1.20
EXIT_TP_MAX_R = 1.80
EXIT_SL_MIN_MULT = 0.90
EXIT_SL_MAX_MULT = 1.10
EXIT_MIN_EDGE = 0.04
EXIT_MIN_COVERAGE = 0.45

# Trade Intelligence / Reason Engine.
# Shadow-only at first: classifies why a trade worked/failed and learns
# conditional patterns without changing the core trading rules.
REASON_MIN_SAMPLES = 12
REASON_ACTIVE_MIN_SAMPLES = 30
REASON_EDGE = 0.08
REASON_WEIGHT_STEP = 0.03
REASON_WEIGHT_MIN = 0.85
REASON_WEIGHT_MAX = 1.15

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
class DailySignal:
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
    vwap_note: str
    above_open: bool
    vol_ok: bool
    reasons: list[str]
    warnings: list[str]
    mode: str = "daily"
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
    h4_state: str = "محايد"
    learning_adjustment: float = 0.0
    resistance_tp1: float = 0.0
    news_state: str = "neutral"
    news_title: str = ""
    news_source: str = ""
    breakout_quality: float = 0.0
    market_state: str = "السوق غير مؤكد"
    chop: bool = False
    market_regime: str = "neutral"
    interaction_keys: list | None = None
    spread_pct: float = 0.0
    expected_slippage_pct: float = 0.0
    dollar_volume_3m: float = 0.0
    liquidity_ok: bool = True



# Backward-compatible name used by charting.py/performance.py.
SignalResult = DailySignal
def _read_learning_records() -> list[dict]:
    if not DAILY_LEARNING_FILE.exists():
        return []
    rows: list[dict] = []
    try:
        with DAILY_LEARNING_FILE.open("r", encoding="utf-8") as f:
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
    DAILY_LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DAILY_LEARNING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _completed_learning(records: list[dict] | None = None) -> list[dict]:
    records = records if records is not None else _read_learning_records()
    completed = [
        r for r in records
        if r.get("record_type") == "outcome"
        and r.get("status") in {"tp1", "stop", "timeout"}
    ]
    return completed[-LEARNING_LOOKBACK:]



def _default_adaptive_policy() -> dict:
    return {
        "version": 1,
        "generation": 0,
        "samples_at_update": 0,
        "approved": False,
        "weights": {
            "weekly_trend": 1.0, "h4": 1.0, "daily": 1.0, "vwap": 1.0,
            "vol_session": 1.0, "market": 1.0, "breakout": 1.0,
            "breakout_candle": 1.0, "retest": 1.0, "vwap_bounce": 1.0,
            "ema_pullback": 1.0, "liquidity_sweep": 1.0, "liquidity_displacement": 1.0, "orb": 1.0, "momentum_continuation": 1.0, "compression_expansion": 1.0, "bull_flag": 1.0, "resistance_reclaim": 1.0, "opening_drive_pullback": 1.0, "hod_reclaim": 1.0, "orb_failed_reclaim": 1.0, "abc_continuation": 1.0, "vwap_weekly_confluence": 1.0, "multi_level_confluence": 1.0, "early": 1.0,
            "news_momentum": 1.0,
        },
        "entry_limits": {"دخول مبكر": 94, "إعادة اختبار": 95, "ارتداد VWAP": 96, "ارتداد EMA20": 96, "سحب سيولة": 97, "ضغط ثم انفجار": 98, "استمرار الزخم": 97, "اختراق نطاق الافتتاح": 99, "اختراق مؤكد": 100, "علم صاعد": 98, "استعادة مستوى": 98, "دخول بعد Opening Drive": 98, "استعادة قمة الفترة": 98, "استعادة بعد فشل ORB": 99, "استمرار ABC": 98, "سحب سيولة مع Displacement": 99},
        "strategy_stats": {et: {"samples": 0, "wins": 0, "win_rate": 0.0} for et in ENTRY_TYPES},
        "min_volume_ratio": 0.85,
        "min_news_volume_ratio": 1.50,
        "min_news_change_pct": 4.0,
        "min_tp1_r": 1.20,
        "regime_weights": {
            regime: {et: 1.0 for et in ENTRY_TYPES}
            for regime in (
                "chop", "trend_clean", "trend_mixed", "market_weak",
                "news_momentum", "high_volatility", "neutral"
            )
        },
        "regime_active": False,
        "regime_activation_generation": None,
        "exit_policy": {
            regime: {
                et: {"tp1_r": 1.20, "sl_mult": 1.00}
                for et in ENTRY_TYPES
            }
            for regime in (
                "chop", "trend_clean", "trend_mixed", "market_weak",
                "news_momentum", "high_volatility", "neutral"
            )
        },
        "exit_active": False,
        "exit_activation_generation": None,
        "reason_policy": {
            "weights": {},
            "active": False,
            "generation": 0,
        },
        "history": [],
    }


def _load_adaptive_policy() -> dict:
    default = _default_adaptive_policy()
    try:
        if not ADAPTIVE_POLICY_FILE.exists():
            return default
        data = json.loads(ADAPTIVE_POLICY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default
        for k, v in default.items():
            data.setdefault(k, v)
        data.setdefault("weights", {})
        for k, v in default["weights"].items():
            data["weights"].setdefault(k, v)
        data.setdefault("entry_limits", {})
        for k, v in default["entry_limits"].items():
            data["entry_limits"].setdefault(k, v)
        data.setdefault("strategy_stats", {})
        for et, stat in default["strategy_stats"].items():
            data["strategy_stats"].setdefault(et, dict(stat))
        data.setdefault("regime_weights", {})
        for regime, et_map in default["regime_weights"].items():
            data["regime_weights"].setdefault(regime, {})
            for et, w in et_map.items():
                data["regime_weights"][regime].setdefault(et, w)
        data.setdefault("regime_active", False)
        data.setdefault("regime_activation_generation", None)
        data.setdefault("exit_policy", {})
        for regime, et_map in default["exit_policy"].items():
            data["exit_policy"].setdefault(regime, {})
            for et, vals in et_map.items():
                data["exit_policy"][regime].setdefault(et, dict(vals))
        data.setdefault("exit_active", False)
        data.setdefault("exit_activation_generation", None)
        data.setdefault("reason_policy", {"weights": {}, "active": False, "generation": 0})
        data["reason_policy"].setdefault("weights", {})
        data["reason_policy"].setdefault("active", False)
        data["reason_policy"].setdefault("generation", 0)
        return data
    except Exception:
        return default


def _save_adaptive_policy(policy: dict) -> None:
    ADAPTIVE_POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ADAPTIVE_POLICY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ADAPTIVE_POLICY_FILE)


def _adaptive_score_adjustment(
    factors: list[str],
    market_regime: str = "neutral",
    entry_type: str = "",
) -> float:
    """Adaptive adjustment plus self-learned regime/strategy multiplier."""
    try:
        p = _load_adaptive_policy()
        vals = [
            float(p.get("weights", {}).get(f, 1.0))
            for f in factors
            if f in p.get("weights", {})
        ]
        adjustment = (sum(vals) / len(vals) - 1.0) * 8.0 if vals else 0.0

        if p.get("regime_active") and entry_type:
            rw = (
                p.get("regime_weights", {})
                .get(market_regime, {})
                .get(entry_type, 1.0)
            )
            adjustment += (float(rw) - 1.0) * 8.0

        # Reason policy is intentionally conservative and does not invent a
        # new setup; it only nudges ranking after OOS approval.
        if p.get("reason_policy", {}).get("active") and entry_type:
            key = f"{market_regime}|{entry_type}"
            reason_map = p.get("reason_policy", {}).get("weights", {}).get(key, {})
            if reason_map:
                vals = [
                    float(v) for k, v in reason_map.items()
                    if k != "samples" and isinstance(v, (int, float))
                ]
                if vals:
                    adjustment += max(-1.0, min(1.0, (sum(vals) / len(vals) - 1.0) * 4.0))

        return max(-5.0, min(5.0, adjustment))
    except Exception:
        return 0.0


def _rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(r.get("status") == "tp1" for r in rows) / len(rows)


def _strategy_stats(rows: list[dict]) -> dict:
    """إحصاءات منفصلة لكل واحدة من استراتيجيات الدخول الـ16."""
    stats = {}
    for et in ENTRY_TYPES:
        subset = [r for r in rows if str(r.get("entry_type") or "") == et]
        wins = sum(1 for r in subset if r.get("status") == "tp1")
        stats[et] = {
            "samples": len(subset),
            "wins": wins,
            "win_rate": round(wins / len(subset), 4) if subset else 0.0,
        }
    return stats


def _build_candidate_policy(completed: list[dict], current: dict) -> dict | None:
    if len(completed) < ADAPTIVE_MIN_SAMPLES:
        return None
    recent = completed[-ADAPTIVE_CONFIRM_SAMPLES:]
    baseline = _rate(recent)
    candidate = json.loads(json.dumps(current))

    for factor in candidate["weights"]:
        subset = [r for r in recent if factor in (r.get("factors") or [])]
        if len(subset) < 6:
            continue
        r = _rate(subset)
        w = float(candidate["weights"].get(factor, 1.0))
        if r >= baseline + 0.10:
            candidate["weights"][factor] = min(1.0 + ADAPTIVE_MAX_CHANGE, w + 0.05)
        elif r <= baseline - 0.10:
            candidate["weights"][factor] = max(1.0 - ADAPTIVE_MAX_CHANGE, w - 0.05)

    for et in ENTRY_TYPES:
        subset = [r for r in recent if r.get("entry_type") == et]
        if len(subset) < 6:
            continue
        r = _rate(subset)
        lim = int(candidate["entry_limits"].get(et, 94))
        if r <= baseline - 0.12:
            candidate["entry_limits"][et] = max(88, lim - 1)
        elif r >= baseline + 0.12:
            candidate["entry_limits"][et] = min(100, lim + 1)

    low = [r for r in recent if float(r.get("volume_ratio", 9) or 9) < 1.2]
    if len(low) >= 8 and _rate(low) <= baseline - 0.10:
        candidate["min_volume_ratio"] = min(1.20, round(float(candidate.get("min_volume_ratio", .85)) + .05, 2))

    candidate["generation"] = int(current.get("generation", 0)) + 1
    candidate["samples_at_update"] = len(completed)
    return candidate


def _validate_candidate(candidate: dict, completed: list[dict]) -> tuple[bool, float, float, float]:
    recent = completed[-ADAPTIVE_CONFIRM_SAMPLES:]
    actual_rate = _rate(recent)
    selected = []
    for r in recent:
        factors = r.get("factors") or []
        pseudo_score = 50.0 + sum(2.0 * float(candidate.get("weights", {}).get(f, 1.0)) for f in factors)
        limit = float(candidate.get("entry_limits", {}).get(r.get("entry_type"), 94))
        if float(r.get("volume_ratio", 1.0) or 1.0) < float(candidate.get("min_volume_ratio", .85)):
            continue
        if r.get("news_state") == "negative":
            continue
        if pseudo_score >= min(82.0, limit):
            selected.append(r)
    if not selected:
        return False, actual_rate, 0.0, 0.0
    candidate_rate = _rate(selected)
    coverage = len(selected) / len(recent)
    approved = candidate_rate >= actual_rate + 0.04 and coverage >= 0.45
    return approved, actual_rate, candidate_rate, coverage



def _classify_regime(
    trend_up: bool,
    h4_state: str,
    chop: bool,
    market_ok: bool,
    atr_pct: float,
    news_state: str,
) -> str:
    if news_state == "positive_strong":
        return "news_momentum"
    if chop:
        return "chop"
    if not market_ok:
        return "market_weak"
    if trend_up and h4_state == "داعم" and atr_pct <= 4.5:
        return "trend_clean"
    if trend_up and h4_state != "معاكس":
        return "trend_mixed"
    if atr_pct > 8.0:
        return "high_volatility"
    return "neutral"


def _interaction_keys(
    entry_type: str,
    market_regime: str,
    h4_state: str,
    volume_ratio: float,
    breakout_quality: float,
) -> list[str]:
    keys = [f"type:{entry_type}", f"regime:{market_regime}", f"h4:{h4_state}"]
    if volume_ratio >= 1.5:
        keys.append("volume:strong")
    elif volume_ratio < 1.0:
        keys.append("volume:weak")
    if breakout_quality >= 70:
        keys.append("breakout:strong")
    elif 0 < breakout_quality < 55:
        keys.append("breakout:weak")
    if entry_type in ENTRY_TYPES and volume_ratio >= 1.5:
        keys.append("combo:breakout+volume")
    if h4_state == "داعم" and volume_ratio >= 1.2 and market_regime == "trend_clean":
        keys.append("combo:h4+volume+trend")
    if market_regime == "chop" and volume_ratio < 1.2:
        keys.append("combo:chop+weak_volume")
    return keys


def _recent_learning_rows() -> list[dict]:
    return [
        r for r in _read_learning_records()
        if r.get("record_type") == "outcome"
        and r.get("status") in {"tp1", "stop", "timeout"}
    ]


def _interaction_rate(rows: list[dict], key: str) -> tuple[float, int]:
    subset = [r for r in rows if key in (r.get("interactions") or [])]
    return _rate(subset), len(subset)


def _shadow_score_row(row: dict, policy: dict) -> bool:
    """قرار افتراضي محافظ للنظام الجديد، دون تغيير النتيجة التاريخية."""
    factors = row.get("factors") or []
    interactions = row.get("interactions") or []
    score = 50.0
    for f in factors:
        score += 2.0 * float(policy.get("weights", {}).get(f, 1.0))
    for k in interactions:
        score += 1.5 * float(policy.get("interaction_weights", {}).get(k, 1.0))
    et = row.get("entry_type")
    limit = float(policy.get("entry_limits", {}).get(et, 94))
    vol = float(row.get("volume_ratio", 1.0) or 1.0)
    if vol < float(policy.get("min_volume_ratio", .85)):
        return False
    if row.get("news_state") == "negative":
        return False

    if policy.get("regime_active") and et:
        regime = str(row.get("market_regime") or "neutral")
        rw = (
            policy.get("regime_weights", {})
            .get(regime, {})
            .get(et, 1.0)
        )
        score += (float(rw) - 1.0) * 8.0

    return score >= min(82.0, limit)


def _rollback_if_needed() -> dict:
    """إذا هبطت نتائج الجيل الحالي بقوة، يرجع تلقائيًا لأفضل جيل محفوظ."""
    p = _load_adaptive_policy()
    hist = p.get("history", [])
    if not hist:
        return {"status": "no_history"}
    current_rate = float(p.get("validation_new_rate", 0) or 0)
    best_path = ADAPTIVE_BEST_FILE
    if not best_path.exists():
        return {"status": "no_best"}
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_rate = float(best.get("validated_rate", 0) or 0)
        if best_rate > 0 and current_rate < best_rate - ADAPTIVE_ROLLBACK_DROP:
            _save_adaptive_policy(best["policy"])
            return {
                "status": "rollback",
                "from_generation": p.get("generation"),
                "to_generation": best["policy"].get("generation", 0),
            }
    except Exception:
        pass
    return {"status": "keep"}


def monthly_self_optimization() -> dict:
    """
    Monthly controlled optimization entry point.
    Reviews the full accumulated learning set once per month and returns a
    Telegram-friendly summary. It never changes core trading rules.
    """
    result = adaptive_retrain_if_ready(force_monthly=True)
    policy = _load_adaptive_policy()

    if result.get("status") in {"waiting", "waiting_oos", "unchanged"}:
        return {
            **result,
            "generation": int(policy.get("generation", 0)),
            "regime_active": bool(policy.get("regime_active", False)),
            "exit_active": bool(policy.get("exit_active", False)),
            "reason_active": bool(policy.get("reason_policy", {}).get("active", False)),
        }

    return {
        **result,
        "generation": int(policy.get("generation", 0)),
        "regime_active": bool(policy.get("regime_active", False)),
        "exit_active": bool(policy.get("exit_active", False)),
        "reason_active": bool(policy.get("reason_policy", {}).get("active", False)),
    }


def _build_regime_candidate(policy: dict, train: list[dict]) -> dict:
    """Learn strategy performance inside each market regime; shadow-only at first."""
    candidate = json.loads(json.dumps(policy))
    candidate.setdefault("regime_weights", {})
    regimes = (
        "chop", "trend_clean", "trend_mixed", "market_weak",
        "news_momentum", "high_volatility", "neutral"
    )

    for regime in regimes:
        candidate["regime_weights"].setdefault(regime, {})
        regime_rows = [
            r for r in train
            if str(r.get("market_regime") or "neutral") == regime
        ]
        regime_rate = _rate(regime_rows)

        for et in ENTRY_TYPES:
            subset = [
                r for r in regime_rows
                if str(r.get("entry_type") or "") == et
            ]
            if len(subset) < REGIME_MIN_SAMPLES:
                continue

            rate = _rate(subset)
            w = float(candidate["regime_weights"][regime].get(et, 1.0))

            if rate >= regime_rate + 0.10:
                w = min(REGIME_WEIGHT_MAX, w + REGIME_WEIGHT_STEP)
            elif rate <= regime_rate - 0.10:
                w = max(REGIME_WEIGHT_MIN, w - REGIME_WEIGHT_STEP)

            candidate["regime_weights"][regime][et] = round(w, 4)

    return candidate


def _classify_trade_reason(row: dict) -> str:
    """Explainable primary outcome reason; no effect on live trading."""
    status = str(row.get("status") or "").lower()
    mfe = float(row.get("mfe_pct") or 0.0)
    mae = abs(float(row.get("mae_pct") or 0.0))
    vr = float(row.get("volume_ratio") or 1.0)
    regime = str(row.get("market_regime") or "neutral")
    news = str(row.get("news_state") or "").lower()
    r = float(row.get("r_multiple") or 0.0)

    if "tp3" in status or "tp2" in status:
        return "strong_trend_followthrough"
    if "tp1" in status:
        if mfe > 2.0 * max(mae, 0.01):
            return "clean_momentum"
        return "tp1_reached_then_slowed"
    if "stop" in status:
        if mfe >= 0.8 and mae > max(mfe, 0.0) and vr < 1.0:
            return "weak_volume_stop"
        if regime == "chop":
            return "chop_stop"
        if news == "negative":
            return "negative_news_stop"
        if mfe >= 0.5:
            return "moved_then_reversed"
        return "immediate_setup_failure"
    if "timeout" in status:
        if mfe >= 1.0 and r <= 0.0:
            return "target_too_far_or_timing"
        if regime == "chop":
            return "chop_timeout"
        return "momentum_faded"
    return "other"


def _build_reason_candidate(policy: dict, train: list[dict]) -> dict:
    """Learn conditional outcome reasons without altering the core rules."""
    candidate = json.loads(json.dumps(policy))
    rp = candidate.setdefault("reason_policy", {"weights": {}, "active": False, "generation": 0})
    weights = rp.setdefault("weights", {})

    groups = {}
    for row in train:
        key = (
            str(row.get("market_regime") or "neutral"),
            str(row.get("entry_type") or ""),
        )
        groups.setdefault(key, []).append(row)

    for (regime, et), rows in groups.items():
        if not et or len(rows) < REASON_MIN_SAMPLES:
            continue

        wins = [r for r in rows if str(r.get("status") or "").lower() in ("tp1", "tp2", "tp3")]
        losses = [r for r in rows if str(r.get("status") or "").lower() in ("stop", "timeout")]
        if not wins and not losses:
            continue

        reason_counts = {}
        for r in rows:
            reason = _classify_trade_reason(r)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        key_name = f"{regime}|{et}"
        weights.setdefault(key_name, {})
        for reason, count in reason_counts.items():
            share = count / max(len(rows), 1)
            old_w = float(weights[key_name].get(reason, 1.0))
            if reason in ("clean_momentum", "strong_trend_followthrough") and share >= 0.25:
                old_w = min(REASON_WEIGHT_MAX, old_w + REASON_WEIGHT_STEP)
            elif reason in ("weak_volume_stop", "chop_stop", "negative_news_stop") and share >= 0.25:
                old_w = max(REASON_WEIGHT_MIN, old_w - REASON_WEIGHT_STEP)
            weights[key_name][reason] = round(old_w, 4)

        weights[key_name]["samples"] = len(rows)

    return candidate


def _build_exit_candidate(policy: dict, train: list[dict]) -> dict:
    """
    Learn TP1 R and structural-stop multiplier by regime + entry type.
    Uses completed trade path telemetry:
      MFE = max favorable excursion
      MAE = max adverse excursion
      time_to_result_min = time to closure/result
    This candidate is shadow-only until its OOS gate approves it.
    """
    candidate = json.loads(json.dumps(policy))
    candidate.setdefault("exit_policy", {})

    regimes = (
        "chop", "trend_clean", "trend_mixed", "market_weak",
        "news_momentum", "high_volatility", "neutral"
    )

    for regime in regimes:
        candidate["exit_policy"].setdefault(regime, {})
        regime_rows = [
            r for r in train
            if str(r.get("market_regime") or "neutral") == regime
        ]

        for et in ENTRY_TYPES:
            subset = [
                r for r in regime_rows
                if str(r.get("entry_type") or "") == et
            ]
            if len(subset) < EXIT_MIN_SAMPLES:
                continue

            # Convert excursion percentages into R using the original risk.
            mfe_r = []
            mae_r = []
            for r in subset:
                entry = float(r.get("entry") or 0)
                stop = float(r.get("stop_loss") or 0)
                if entry <= 0 or stop <= 0 or stop >= entry:
                    continue
                risk_pct = (entry - stop) / entry * 100.0
                if risk_pct <= 0:
                    continue
                mfe_r.append(float(r.get("mfe_pct") or 0.0) / risk_pct)
                mae_r.append(abs(float(r.get("mae_pct") or 0.0)) / risk_pct)

            if not mfe_r or not mae_r:
                continue

            current = candidate["exit_policy"][regime].get(
                et, {"tp1_r": 1.20, "sl_mult": 1.00}
            )
            tp1_r = float(current.get("tp1_r", 1.20))
            sl_mult = float(current.get("sl_mult", 1.00))

            # If the trade frequently reaches materially beyond TP1, allow a
            # slightly larger target. If it usually stalls before TP1, keep the
            # minimum 1.20R rather than shrinking below the strategy's core rule.
            median_mfe = float(np.median(mfe_r))
            hit_rate = sum(1 for x in mfe_r if x >= tp1_r) / len(mfe_r)
            if hit_rate >= 0.70 and median_mfe >= tp1_r + 0.20:
                tp1_r = min(EXIT_TP_MAX_R, tp1_r + EXIT_TP_STEP_R)
            elif hit_rate < 0.50 and median_mfe < tp1_r:
                tp1_r = max(EXIT_TP_MIN_R, tp1_r - EXIT_TP_STEP_R)

            # If MAE repeatedly approaches/exceeds the current structural risk
            # before the trade later succeeds, allow a small widening. If stops
            # are hit quickly with little favorable excursion, tighten instead.
            near_stop = sum(1 for x in mae_r if x >= 0.85) / len(mae_r)
            early_loss = sum(
                1 for r in subset
                if r.get("status") == "stop"
                and float(r.get("mfe_pct") or 0.0) <= 0.30
            ) / len(subset)

            if near_stop >= 0.60 and hit_rate >= 0.55:
                sl_mult = min(EXIT_SL_MAX_MULT, sl_mult + EXIT_SL_STEP)
            elif early_loss >= 0.55:
                sl_mult = max(EXIT_SL_MIN_MULT, sl_mult - EXIT_SL_STEP)

            candidate["exit_policy"][regime][et] = {
                "tp1_r": round(tp1_r, 3),
                "sl_mult": round(sl_mult, 3),
                "samples": len(subset),
                "median_mfe_r": round(median_mfe, 3),
                "near_stop_rate": round(near_stop, 3),
            }

    return candidate


def _apply_adaptive_exit(sig_price: float, structural_stop: float, regime: str,
                         entry_type: str, policy: dict, atr: float) -> tuple[float, float, float]:
    """
    Return (stop, tp1, tp1_r). Adaptive exit is inert until exit_active=True.
    Stop remains anchored to the structural stop and stays inside the global
    0.60%–4.50% risk band.
    """
    entry = float(sig_price)
    stop = float(structural_stop)
    risk = max(entry - stop, entry * 0.006)
    tp1_r = 1.20
    sl_mult = 1.00

    if policy.get("exit_active"):
        vals = (
            policy.get("exit_policy", {})
            .get(regime, {})
            .get(entry_type, {})
        )
        tp1_r = float(vals.get("tp1_r", 1.20))
        sl_mult = float(vals.get("sl_mult", 1.00))

    # Move the structural stop modestly around the original structural point.
    # Never cross entry and never exceed the global risk ceiling.
    base_gap = max(entry - stop, entry * 0.006)
    new_gap = base_gap * sl_mult
    new_gap = max(entry * 0.006, min(entry * 0.045, new_gap))
    new_stop = entry - new_gap

    tp1 = entry + new_gap * max(EXIT_TP_MIN_R, min(EXIT_TP_MAX_R, tp1_r))
    return new_stop, tp1, max(EXIT_TP_MIN_R, min(EXIT_TP_MAX_R, tp1_r))


def adaptive_retrain_if_ready(force_monthly: bool = False) -> dict:
    """يشغّل دورة التعلم الذاتي ويصدر نتيجة قابلة للإرسال إلى Telegram."""
    completed = _recent_learning_rows()
    policy = _load_adaptive_policy()

    if len(completed) < ADAPTIVE_MIN_SAMPLES:
        return {"status": "waiting", "samples": len(completed)}

    last_update = int(policy.get("samples_at_update", 0))
    if not force_monthly and len(completed) <= last_update:
        return {"status": "unchanged", "samples": len(completed)}

    # Normal learning stays bounded to the recent confirmation window.
    # Monthly optimization intentionally reviews the full accumulated dataset.
    recent = completed if force_monthly else completed[-ADAPTIVE_CONFIRM_SAMPLES:]
    # اختبار خارج العينة: الجزء الأحدث يبقى خارج التدريب حتى لا نعتمد على نفس البيانات.
    split = max(20, int(len(recent) * 0.70))
    train = recent[:split]
    test = recent[split:]
    if len(train) < 20 or len(test) < 10:
        return {"status": "waiting_oos", "samples": len(completed)}
    baseline_rate = _rate(train)
    candidate = json.loads(json.dumps(policy))
    candidate.setdefault("interaction_weights", {})
    # Always refresh per-strategy statistics so all 16 setups are observable.
    candidate["strategy_stats"] = _strategy_stats(completed)

    # Regime layer: learns separately for each market condition, but remains
    # shadow-only until its OOS gate proves that it improves the current policy.
    candidate = _build_regime_candidate(candidate, train)
    candidate = _build_exit_candidate(candidate, train)
    candidate = _build_reason_candidate(candidate, train)

    # العوامل
    for factor in candidate.get("weights", {}):
        subset = [r for r in train if factor in (r.get("factors") or [])]
        if len(subset) < 6:
            continue
        rate = _rate(subset)
        w = float(candidate["weights"].get(factor, 1.0))
        if rate >= baseline_rate + 0.10:
            candidate["weights"][factor] = min(1.0 + ADAPTIVE_MAX_CHANGE, w + 0.05)
        elif rate <= baseline_rate - 0.10:
            candidate["weights"][factor] = max(1.0 - ADAPTIVE_MAX_CHANGE, w - 0.05)

    # التركيبات
    keys = sorted({k for r in train for k in (r.get("interactions") or [])})
    for key in keys:
        rate, n = _interaction_rate(train, key)
        if n < 6:
            continue
        w = float(candidate["interaction_weights"].get(key, 1.0))
        if rate >= baseline_rate + 0.12:
            candidate["interaction_weights"][key] = min(1.0 + ADAPTIVE_MAX_CHANGE, w + 0.05)
        elif rate <= baseline_rate - 0.12:
            candidate["interaction_weights"][key] = max(1.0 - ADAPTIVE_MAX_CHANGE, w - 0.05)

    # نوع الدخول
    for et in ("دخول مبكر", "إعادة اختبار", "ارتداد VWAP", "ارتداد EMA20", "سحب سيولة", "سحب سيولة مع Displacement", "ضغط ثم انفجار", "استمرار الزخم", "اختراق نطاق الافتتاح", "اختراق مؤكد", "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة الفترة", "استعادة بعد فشل ORB", "استمرار ABC"):
        subset = [r for r in train if r.get("entry_type") == et]
        if len(subset) < 6:
            continue
        rate = _rate(subset)
        lim = int(candidate["entry_limits"].get(et, 94))
        if rate <= baseline_rate - 0.12:
            candidate["entry_limits"][et] = max(88, lim - 1)
        elif rate >= baseline_rate + 0.12:
            candidate["entry_limits"][et] = min(100, lim + 1)

    # نظام السوق والحجم
    for regime in ("chop", "trend_clean", "trend_mixed", "market_weak", "news_momentum"):
        subset = [r for r in train if r.get("market_regime") == regime]
        if len(subset) >= 6 and regime in {"chop", "market_weak"}:
            if _rate(subset) <= baseline_rate - 0.10:
                candidate["min_volume_ratio"] = min(
                    1.20, round(float(candidate.get("min_volume_ratio", .85)) + .05, 2)
                )

    low = [r for r in train if float(r.get("volume_ratio", 9) or 9) < 1.2]
    if len(low) >= 8 and _rate(low) <= baseline_rate - 0.10:
        candidate["min_volume_ratio"] = min(
            1.20, round(float(candidate.get("min_volume_ratio", .85)) + .05, 2)
        )

    candidate["generation"] = int(policy.get("generation", 0)) + 1
    candidate["samples_at_update"] = len(completed)

    current_selected = [r for r in test if _shadow_score_row(r, policy)]
    candidate_selected = [r for r in test if _shadow_score_row(r, candidate)]
    current_rate = _rate(current_selected)
    candidate_rate = _rate(candidate_selected)
    coverage = len(candidate_selected) / max(len(test), 1)

    regime_policy = json.loads(json.dumps(candidate))
    regime_policy["regime_active"] = True
    regime_selected = [r for r in test if _shadow_score_row(r, regime_policy)]
    regime_rate = _rate(regime_selected)
    regime_coverage = len(regime_selected) / max(len(test), 1)
    regime_improved = (
        regime_rate >= current_rate + REGIME_MIN_EDGE
        and regime_coverage >= REGIME_MIN_COVERAGE
        and len(regime_selected) >= 10
    )

    exit_policy = json.loads(json.dumps(candidate))
    exit_policy["exit_active"] = True
    exit_selected = [r for r in test if _shadow_score_row(r, exit_policy)]
    exit_rate = _rate(exit_selected)
    exit_coverage = len(exit_selected) / max(len(test), 1)
    exit_improved = (
        exit_rate >= current_rate + EXIT_MIN_EDGE
        and exit_coverage >= EXIT_MIN_COVERAGE
        and len(exit_selected) >= 10
    )

    # Reason Engine is evaluated separately. It never rewrites core rules;
    # it only becomes an optional ranking modifier after OOS proof.
    reason_policy = json.loads(json.dumps(candidate))
    reason_policy["reason_policy"]["active"] = True
    reason_selected = [r for r in test if _shadow_score_row(r, reason_policy)]
    reason_rate = _rate(reason_selected)
    reason_coverage = len(reason_selected) / max(len(test), 1)
    reason_improved = (
        reason_rate >= current_rate + REASON_EDGE
        and reason_coverage >= REGIME_MIN_COVERAGE
        and len(reason_selected) >= 10
    )

    approved = (
        candidate_rate >= current_rate + ADAPTIVE_MIN_EDGE
        and coverage >= ADAPTIVE_MIN_COVERAGE
        and len(candidate_selected) >= 10
    )

    result = {
        "generation": candidate["generation"],
        "samples": len(completed),
        "baseline_rate": round(baseline_rate, 4),
        "current_shadow_rate": round(current_rate, 4),
        "candidate_shadow_rate": round(candidate_rate, 4),
        "coverage": round(coverage, 4),
        "regime_shadow_rate": round(regime_rate, 4),
        "regime_coverage": round(regime_coverage, 4),
        "regime_improved": bool(regime_improved),
        "regime_active_before": bool(policy.get("regime_active", False)),
        "exit_shadow_rate": round(exit_rate, 4),
        "exit_coverage": round(exit_coverage, 4),
        "exit_improved": bool(exit_improved),
        "exit_active_before": bool(policy.get("exit_active", False)),
        "reason_shadow_rate": round(reason_rate, 4),
        "reason_coverage": round(reason_coverage, 4),
        "reason_improved": bool(reason_improved),
        "reason_active_before": bool(policy.get("reason_policy", {}).get("active", False)),
        "oos_samples": len(test),
        "train_samples": len(train),
        "approved": bool(approved),
        "at": datetime.now(timezone.utc).isoformat(),
    }

    # سجل Shadow دائم
    try:
        with ADAPTIVE_SHADOW_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception:
        pass

    if approved:
        candidate["approved"] = True
        candidate["validation_new_rate"] = candidate_rate
        candidate["validation_old_rate"] = current_rate

        # Automatic activation: no manual intervention.
        candidate["regime_active"] = bool(regime_improved or policy.get("regime_active", False))
        candidate["regime_activation_generation"] = (
            candidate["generation"] if regime_improved
            else policy.get("regime_activation_generation")
        )
        candidate["exit_active"] = bool(exit_improved or policy.get("exit_active", False))
        candidate["exit_activation_generation"] = (
            candidate["generation"] if exit_improved
            else policy.get("exit_activation_generation")
        )
        candidate["reason_policy"]["active"] = bool(
            reason_improved or policy.get("reason_policy", {}).get("active", False)
        )
        candidate["reason_policy"]["generation"] = (
            candidate["generation"] if reason_improved
            else policy.get("reason_policy", {}).get("generation", 0)
        )
        candidate["history"] = (policy.get("history", []) + [result])[-20:]
        _save_adaptive_policy(candidate)

        try:
            best = json.loads(ADAPTIVE_BEST_FILE.read_text(encoding="utf-8")) if ADAPTIVE_BEST_FILE.exists() else {}
            best_rate = float(best.get("validated_rate", 0) or 0)
            if candidate_rate > best_rate:
                ADAPTIVE_BEST_FILE.write_text(
                    json.dumps({
                        "validated_rate": candidate_rate,
                        "generation": candidate["generation"],
                        "policy": candidate,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass

        result["status"] = "approved"
        result["message"] = (
            f"🧠 تحديث التعلم الذاتي\n"
            f"تم تحليل {len(completed)} صفقة\n"
            f"الحالي: {current_rate*100:.1f}%\n"
            f"الجديد المتوقع: {candidate_rate*100:.1f}%\n"
            f"التغطية: {coverage*100:.1f}%\n"
            f"✅ تم اعتماد الجيل رقم {candidate['generation']}"
        )
    else:
        # The regime layer can graduate independently if it passes its own OOS gate.
        if regime_improved or exit_improved or reason_improved:
            policy["generation"] = int(policy.get("generation", 0)) + 1
            if regime_improved:
                policy["regime_weights"] = candidate.get("regime_weights", policy.get("regime_weights", {}))
                policy["regime_active"] = True
                policy["regime_activation_generation"] = policy["generation"]
            if exit_improved:
                policy["exit_policy"] = candidate.get("exit_policy", policy.get("exit_policy", {}))
                policy["exit_active"] = True
                policy["exit_activation_generation"] = policy["generation"]
            if reason_improved:
                policy["reason_policy"] = candidate.get(
                    "reason_policy", policy.get("reason_policy", {})
                )
                policy["reason_policy"]["active"] = True
                policy["reason_policy"]["generation"] = policy["generation"]
            policy["samples_at_update"] = len(completed)
            policy["history"] = (policy.get("history", []) + [result])[-20:]
            _save_adaptive_policy(policy)
            result["status"] = "regime_or_exit_approved"
            result["message"] = (
                f"🧠 Adaptive Learning\n"
                f"تم تحليل {len(completed)} صفقة\n"
                f"Regime: {'مفعل' if regime_improved else 'بدون تغيير'} | "
                f"Exit: {'مفعل' if exit_improved else 'بدون تغيير'}\n"
                f"الجيل {policy['generation']}"
            )
        else:
            policy["samples_at_update"] = len(completed)
            policy["history"] = (policy.get("history", []) + [result])[-20:]
            _save_adaptive_policy(policy)

            result["status"] = "rejected"
        result["message"] = (
            f"🧠 مراجعة التعلم الذاتي\n"
            f"تم تحليل {len(completed)} صفقة\n"
            f"الحالي: {current_rate*100:.1f}% | الجديد: {candidate_rate*100:.1f}%\n"
            f"❌ لم يتم اعتماد التعديل — السياسة الحالية بقيت كما هي"
        )

    # Rollback فعلي: لا يتم إلا إذا كان لدينا أفضل جيل وتراجع الأداء بقوة.
    rollback = _rollback_if_needed()
    result["rollback"] = rollback.get("status")
    if rollback.get("status") == "rollback":
        result["status"] = "rollback"
        result["message"] = (
            f"🔄 Rollback للتعلم الذاتي\n"
            f"الجيل {rollback.get('from_generation')} تراجع، فعاد البوت إلى "
            f"أفضل جيل محفوظ: {rollback.get('to_generation')}\n"
            f"🛡️ تم الإبقاء على أفضل سياسة"
        )

    try:
        LEARNING_ALERT_FILE.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return result


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


def register_daily_signal(sig: DailySignal) -> str:
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
            "h4_state": sig.h4_state,
            "volume_ratio": float(sig.volume_ratio),
            "atr_pct": float(sig.atr_pct),
            "ext_sma20": float(sig.ext_sma20),
            "above_open": bool(sig.above_open),
            "vwap": sig.vwap_note,
            "news_state": sig.news_state,
            "breakout_quality": sig.breakout_quality,
            "market_state": sig.market_state,
            "chop": sig.chop,
            "market_regime": sig.market_regime,
            "adaptive_generation": int(_load_adaptive_policy().get("generation", 0)),
            "exit_policy_active": bool(_load_adaptive_policy().get("exit_active", False)),
            "reason_engine_active": bool(
                _load_adaptive_policy().get("reason_policy", {}).get("active", False)
            ),
            "outcome_reason": _classify_trade_reason({**sig.__dict__, "status": ""}),
            "interactions": list(getattr(sig, "interaction_keys", []) or []),
            "status": "pending",
        }
    )
    return signal_id


def record_daily_outcome(
    signal_id: str | None,
    symbol: str,
    status: str,
    exit_price: float | None = None,
    note: str = "",
    mfe_pct: float | None = None,
    mae_pct: float | None = None,
    time_to_result_min: float | None = None,
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
            "h4_state": target.get("h4_state", "محايد"),
            "volume_ratio": target.get("volume_ratio", 0),
            "atr_pct": target.get("atr_pct", 0),
            "ext_sma20": target.get("ext_sma20", 0),
            "above_open": target.get("above_open", False),
            "vwap": target.get("vwap", ""),
            "news_state": target.get("news_state", "neutral"),
            "breakout_quality": target.get("breakout_quality", 0),
            "market_state": target.get("market_state", "السوق غير مؤكد"),
            "chop": target.get("chop", False),
            "market_regime": target.get("market_regime", "neutral"),
            "interactions": target.get("interactions", []) or [],
            "mfe_pct": round(float(mfe_pct if mfe_pct is not None else target.get("mfe_pct", 0.0) or 0.0), 4),
            "mae_pct": round(float(mae_pct if mae_pct is not None else target.get("mae_pct", 0.0) or 0.0), 4),
            "time_to_result_min": (
                round(float(time_to_result_min), 1)
                if time_to_result_min is not None
                else target.get("time_to_result_min")
            ),
            "status": status,
            "exit_price": exit_price,
            "note": note,
        }
    )
    target["status"] = "completed"


def _m15_confirmation(h4: pd.DataFrame, price: float) -> tuple[str, int]:
    """تأكيد ناعم 4 ساعات: نفس فكرة 15د في اللحظي، لكن على بنية 4س."""
    try:
        if h4 is None or len(h4) < 30:
            return "محايد", 0
        cur = h4.tail(30)
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
        req = urllib.request.Request(url, headers={"User-Agent": "DailyScanner/1.0"})
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


def _chop_filter(today_d: pd.DataFrame, price: float, vwap: float) -> bool:
    """True = سوق متذبذب/Chop، فلا نطارد الإشارات."""
    try:
        c = today_d["Close"].astype(float)
        if len(c) < 12:
            return False
        e9 = _ema(c, 9)
        cross = ((c > e9).astype(int).diff().abs()).tail(12).sum()
        vwap_dist = abs(price - vwap) / price * 100 if price else 0
        ranges = (today_d["High"] - today_d["Low"]).astype(float)
        avg_range = ranges.tail(12).mean()
        if avg_range <= 0:
            return False
        tight = (ranges.tail(12).median() / avg_range) < 0.75
        return bool(cross >= 5 and vwap_dist < 0.7 and tight)
    except Exception:
        return False


def _breakout_quality(today_d: pd.DataFrame, level: float, price: float) -> tuple[bool, float]:
    """
    جودة آخر شمعة يومي فوق المقاومة + متابعة الشمعة السابقة.
    لا نعتبر لمس المستوى اختراقًا.
    """
    try:
        if level <= 0 or len(today_d) < 3:
            return False, 0.0
        b = today_d.iloc[-1]
        o, h, l, c = map(float, (b["Open"], b["High"], b["Low"], b["Close"]))
        rng = max(h - l, 1e-9)
        body = abs(c - o) / rng
        close_pos = (c - l) / rng
        upper_wick = (h - max(o, c)) / rng
        prior_close = float(today_d["Close"].iloc[-2])

        strong_close = c >= level * 1.001 and close_pos >= 0.70
        body_ok = body >= 0.45
        wick_ok = upper_wick <= 0.30
        follow = prior_close >= level * 0.997 or c >= prior_close * 1.002
        quality = (body * 0.4 + close_pos * 0.4 + (1 - min(upper_wick, 1)) * 0.2) * 100
        return bool(strong_close and body_ok and wick_ok and follow), float(quality)
    except Exception:
        return False, 0.0


def _period_days(period: str, default: int = 5) -> int:
    """Convert common history periods (e.g. 5y, 6mo, 30d) to calendar days."""
    try:
        raw = str(period).strip().lower()
        if raw.endswith("y"):
            return max(1, int(float(raw[:-1]) * 365.25))
        if raw.endswith("mo"):
            return max(1, int(float(raw[:-2]) * 30.5))
        if raw.endswith("d"):
            return max(1, int(float(raw[:-1])))
        if raw.endswith("wk"):
            return max(1, int(float(raw[:-2]) * 7))
    except Exception:
        pass
    return int(default)


DAILY_MARKET_RETRY_ATTEMPTS = 4
DAILY_MARKET_RETRY_DELAYS = (0.0, 0.5, 1.0, 1.5)


def _market_alignment(fetch_intraday) -> tuple[bool, str]:
    """
    Daily SPY/QQQ market gate with explicit per-attempt diagnostics.

    The logging is diagnostic only; it does not change the daily trading rules:
    - incomplete/unavailable data is retried before being treated as unavailable;
    - mixed SPY/QQQ is allowed;
    - weak SPY+QQQ remains subject to the existing daily strong-stock override.
    """
    states = []

    for sym in ("SPY", "QQQ"):
        state = None
        for attempt in range(DAILY_MARKET_RETRY_ATTEMPTS):
            attempt_no = attempt + 1
            try:
                log.info(
                    "DAILY MARKET FETCH | %s | attempt %d/%d | interval=1d | period=6mo",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS,
                )
                d = fetch_intraday(sym, interval="1d", period="6mo")

                if d is None:
                    log.warning("DAILY MARKET FETCH | %s | attempt %d/%d | data=None", sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS)
                    raise ValueError("market data is None")

                if "Close" not in d.columns:
                    log.warning(
                        "DAILY MARKET FETCH | %s | attempt %d/%d | missing Close | columns=%s",
                        sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS, list(d.columns),
                    )
                    raise ValueError("Close column missing")

                c = pd.to_numeric(d["Close"], errors="coerce").dropna()
                log.info(
                    "DAILY MARKET FETCH | %s | attempt %d/%d | rows=%d | valid_close=%d",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS, len(d), len(c),
                )
                if len(c) < 50:
                    raise ValueError(f"insufficient daily closes: {len(c)} < 50")

                e20 = float(_ema(c, 20).iloc[-1])
                e50 = float(_ema(c, 50).iloc[-1])
                p = float(c.iloc[-1])
                if not (np.isfinite(e20) and np.isfinite(e50) and np.isfinite(p)):
                    raise ValueError("non-finite market values")

                state = bool(p >= e20 and e20 >= e50)
                log.info(
                    "DAILY MARKET RESULT | %s | attempt %d/%d | close=%.4f | ema20=%.4f | ema50=%.4f | state=%s",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS, p, e20, e50,
                    "داعم" if state else "ضعيف",
                )
                break

            except Exception as exc:
                log.warning(
                    "DAILY MARKET FETCH FAILED | %s | attempt %d/%d | %s",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS, exc,
                )
                if attempt < DAILY_MARKET_RETRY_ATTEMPTS - 1:
                    delay = DAILY_MARKET_RETRY_DELAYS[min(attempt + 1, len(DAILY_MARKET_RETRY_DELAYS) - 1)]
                    if delay > 0:
                        time_module.sleep(delay)

        states.append(state)
        log.info(
            "DAILY MARKET SYMBOL FINAL | %s | state=%s",
            sym, "داعم" if state is True else "ضعيف" if state is False else "غير متاح",
        )

    if any(x is None for x in states):
        log.warning(
            "DAILY MARKET FINAL | SPY=%s | QQQ=%s | ok=False | state=بيانات SPY/QQQ غير مكتملة بعد إعادة المحاولة",
            states[0], states[1],
        )
        return False, "بيانات SPY/QQQ غير مكتملة بعد إعادة المحاولة"

    if states[0] and states[1]:
        log.info("DAILY MARKET FINAL | SPY=داعم | QQQ=داعم | ok=True | state=SPY+QQQ داعمان يوميًا")
        return True, "SPY+QQQ داعمان يوميًا"
    if (not states[0]) and (not states[1]):
        log.info("DAILY MARKET FINAL | SPY=ضعيف | QQQ=ضعيف | ok=False | state=SPY+QQQ ضعيفان يوميًا")
        return False, "SPY+QQQ ضعيفان يوميًا"

    log.info("DAILY MARKET FINAL | SPY=%s | QQQ=%s | ok=True | state=SPY/QQQ مختلطان يوميًا", states[0], states[1])
    return True, "SPY/QQQ مختلطان يوميًا"


def session_window_ok(dt=None) -> tuple[bool, str]:
    """Daily engine: market must be a real US trading session; no intraday first/last-20 restriction."""
    dt = dt or now_ny()
    if not is_us_regular_session(dt):
        return False, session_label(dt)
    return True, "نافذة يومية مسموحة"


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
    """Daily anchored VWAP: rolling 20-bar volume-weighted typical price."""
    tp = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3
    vol = df["Volume"].astype(float).replace(0, np.nan)
    return (tp * vol).rolling(20, min_periods=5).sum() / vol.rolling(20, min_periods=5).sum()


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
    today_d: pd.DataFrame,
    weekly: pd.DataFrame,
    price: float,
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []

    try:
        x = today_d.iloc[:-1].copy()
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
                    candidates.append((v, "قمة يومي سابقة"))
    except Exception:
        pass

    try:
        x = weekly.iloc[:-1].tail(30)
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
                    candidates.append((v, "قمة أسبوعية سابقة"))
    except Exception:
        pass

    if not candidates:
        return 0.0, "هدف مخاطر احتياطي"
    return min(candidates, key=lambda x: x[0])


_QUOTE_CACHE: dict[str, tuple[datetime, dict]] = {}
QUOTE_CACHE_SECONDS = 30
MAX_SPREAD_PCT = 1.20
HARD_MAX_SPREAD_PCT = 2.00
MIN_DOLLAR_VOLUME_3M = 10_000_000.0

def _quote_liquidity(symbol: str, price: float) -> dict:
    """يومي: Bid/Ask من Alpaca فقط؛ لا نستخدم Yahoo كبديل للتنفيذ اليومي."""
    now = datetime.now(timezone.utc)
    cached = _QUOTE_CACHE.get(symbol)
    if cached and (now - cached[0]).total_seconds() < QUOTE_CACHE_SECONDS:
        return cached[1]
    result = {
        "ok": False,
        "spread_pct": 0.0,
        "slippage_pct": 0.0,
        "dollar_volume": 0.0,
        "quote_source": "none",
        "quote_age_min": float("inf"),
    }
    try:
        from market_data import fetch_latest_quote, data_age_minutes
        q = fetch_latest_quote(symbol)
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        px = float(price or 0)
        ts = q.get("timestamp")
        if ts:
            qdf = pd.DataFrame({"Close": [px]}, index=[pd.Timestamp(ts)])
            age = data_age_minutes(qdf)
            result["quote_age_min"] = age
        if bid > 0 and ask > bid and px > 0 and result["quote_age_min"] <= 2.0:
            spread = (ask - bid) / px * 100
            result["spread_pct"] = spread
            result["slippage_pct"] = spread / 2.0
            result["ok"] = spread <= HARD_MAX_SPREAD_PCT
            result["quote_source"] = "alpaca-" + str(q.get("feed") or "unknown")
    except Exception:
        pass
    _QUOTE_CACHE[symbol] = (now, result)
    return result


def analyze_daily(
    symbol: str,
    name: str = "",
    live: bool = True,
    market_context: tuple[bool, str] | None = None,
    preloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> Optional[DailySignal]:
    from market_data import fetch_intraday
    from market_data import intraday_data_fresh

    if preloaded is not None:
        weekly, daily = preloaded
    else:
        weekly = fetch_intraday(symbol, interval="1wk", period="5y")
        daily = fetch_intraday(symbol, interval="1d", period="2y")

    if weekly is None or len(weekly) < 60 or daily is None or len(daily) < 80:
        return None

    # Daily bars can be current during regular session. The engine is deliberately
    # tolerant of the last bar being today's partial candle.
    today_d = daily.tail(120).copy()
    last_day = daily.index[-1].date()
    try:
        h4 = fetch_intraday(symbol, interval="60m", period="60d")
        ok_h4, _ = intraday_data_fresh(h4, "60m", 240)
        if not ok_h4:
            h4 = None
    except Exception:
        h4 = None

    # Compatibility aliases keep the proven 16-setup intelligence readable.
    weekly = weekly.copy()
    daily = daily.copy()

    price = float(today_d["Close"].iloc[-1])
    if price <= 0 or price > float(MAX_AUTO_PRICE):
        return None

    day_open = float(today_d["Open"].iloc[-1])
    prev_days = daily[daily.index.date < last_day]
    prev_close = float(prev_days["Close"].iloc[-1]) if not prev_days.empty else price
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0

    vwap_s = _vwap(today_d)
    vwap_last = float(vwap_s.iloc[-1]) if pd.notna(vwap_s.iloc[-1]) else price
    above_vwap = price >= vwap_last
    vwap_note = "فوق VWAP 20 يوم" if above_vwap else "تحت VWAP 20 يوم"
    above_open = price >= day_open

    vol_today = float(today_d["Volume"].sum())
    bars = max(len(today_d), 1)
    avg_bar_today = vol_today / bars
    hist_5 = daily[daily.index.date < last_day].tail(120)
    vol_hist = (
        float(hist_5["Volume"].mean())
        if not hist_5.empty
        else float(daily["Volume"].tail(60).mean() or 1)
    )
    vol_ratio = avg_bar_today / vol_hist if vol_hist else 1.0
    vol_ok = vol_ratio >= 0.90

    hc = weekly["Close"]
    ema20 = _ema(hc, 20)
    ema50 = _ema(hc, 50)
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    h_rsi = float(_rsi(hc, 14).iloc[-1])
    trend_up = price > e20 > e50 * 0.998 and h_rsi >= 45

    c5 = today_d["Close"]
    e5 = float(_ema(c5, 20).iloc[-1])
    r5 = float(_rsi(c5, 14).iloc[-1])
    last_green = float(today_d["Close"].iloc[-1]) >= float(today_d["Open"].iloc[-1])
    mom = (price - float(c5.iloc[-6])) / float(c5.iloc[-6]) * 100 if len(c5) >= 6 else 0.0
    live_ok = price >= e5 * 0.998 and above_vwap and (last_green or mom > 0.05) and r5 < 78

    h4_state, h4_points = _m15_confirmation(h4, price)
    news_state, news_title, news_source = _classify_news(symbol)
    market_ok, market_state = market_context if market_context is not None else _market_alignment(fetch_intraday)
    chop = _chop_filter(today_d, price, vwap_last)

    session_high = float(today_d["High"].max())
    drop = (session_high - price) / session_high * 100 if session_high else 0
    dump = drop >= 2.5 and change_pct <= -1.2

    h_win = weekly.tail(20)
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
        float(today_d["High"].max()) >= level_high * 1.001
        and price < level_high * 0.997
        and not above_vwap
    ) or (dump and not above_vwap)
    retest = prior_break and near_level and price >= level_high * 0.997 and above_vwap and not failed

    # 1) VWAP Bounce/Reclaim: رجوع منظم إلى VWAP ثم استعادة المستوى.
    recent4 = today_d.tail(4)
    vwap_touch = False
    try:
        vwap_touch = bool((recent4["Low"].astype(float) <= vwap_last * 1.006).any())
    except Exception:
        vwap_touch = False
    vwap_bounce = (
        above_vwap and vwap_touch and not failed
        and last_green and (mom > 0.05)
        and vol_ratio >= 1.0
        and trend_up and h4_state != "معاكس"
    )

    # 2) EMA20 Pullback: ترند صاعد + تصحيح صحي إلى EMA20 + استعادة.
    ema_touch = False
    try:
        ema_touch = bool((recent4["Low"].astype(float) <= e5 * 1.006).any())
    except Exception:
        ema_touch = False
    ema_pullback = (
        trend_up and ema_touch and price >= e5 * 1.001
        and last_green and mom > 0.05
        and vol_ratio >= 1.0
        and h4_state != "معاكس" and not failed
    )

    # 3) Liquidity Sweep + Reclaim: كسر قاع قريب ثم استعادة المستوى بسرعة.
    support_level = 0.0
    liquidity_sweep = False
    try:
        support_window = today_d["Low"].astype(float).iloc[-12:-2]
        if len(support_window) >= 5:
            support_level = float(support_window.min())
            recent3 = today_d.tail(3)
            swept = (recent3["Low"].astype(float) < support_level * 0.998).any()
            reclaimed = price >= support_level * 1.002
            liquidity_sweep = bool(
                swept and reclaimed and last_green and mom > 0.05
                and vol_ratio >= 1.0 and h4_state != "معاكس"
                and not failed and above_vwap
            )
    except Exception:
        liquidity_sweep = False

    # 4) Liquidity Sweep + Displacement: سحب سيولة يتبعه اندفاع سعري واضح.
    liquidity_displacement = False
    try:
        if len(today_d) >= 8 and support_level > 0:
            cur = today_d.iloc[-1]
            prev3 = today_d.iloc[-4:-1]
            cur_o, cur_c = float(cur["Open"]), float(cur["Close"])
            cur_h, cur_l = float(cur["High"]), float(cur["Low"])
            cur_range = max(cur_h - cur_l, price * 0.0001)
            cur_body = abs(cur_c - cur_o)
            close_pos = (cur_c - cur_l) / cur_range
            prior_ranges = (prev3["High"].astype(float) - prev3["Low"].astype(float)).clip(lower=0)
            med_range = float(prior_ranges.median()) if len(prior_ranges) else 0.0
            swept = bool((today_d["Low"].astype(float).iloc[-5:-1] < support_level * 0.998).any())
            reclaimed = price >= support_level * 1.002
            displacement = bool(
                cur_c > cur_o and cur_body / cur_range >= 0.55 and close_pos >= 0.75
                and (med_range <= 0 or cur_range >= med_range * 1.35)
                and vol_ratio >= 1.25
            )
            liquidity_displacement = bool(
                swept and reclaimed and displacement and trend_up and above_vwap and above_open
                and h4_state != "معاكس" and market_ok and not failed
                and mom > 0.08 and ext_tmp <= 6.0
            )
    except Exception:
        liquidity_displacement = False

    # Daily equivalent of ORB: first 3 sessions of the current calendar month.
    # This preserves the tactic (range -> break -> follow-through) without pretending
    # that a 5-minute opening range exists on a daily chart.
    month_d = daily[daily.index.to_period("M") == daily.index[-1].to_period("M")] if isinstance(daily.index, pd.DatetimeIndex) else daily.tail(22)
    opening_range_high = float(month_d["High"].head(3).max()) if len(month_d) >= 3 else 0.0
    opening_range_low = float(month_d["Low"].head(3).min()) if len(month_d) >= 3 else 0.0

    # 5) Opening Range Breakout (ORB): اختراق نطاق بداية الشهر مع متابعة.
    orb_high = opening_range_high
    orb_breakout = False
    try:
        prior_orb = float(daily["Close"].iloc[-2]) < orb_high * 1.001 if orb_high > 0 and len(daily) >= 2 else False
        orb_breakout = bool(
            orb_high > 0 and price >= orb_high * 1.001 and prior_orb and last_green
            and vol_ratio >= 1.0 and above_vwap
            and h4_state != "معاكس" and not failed
        )
    except Exception:
        orb_breakout = False

    # 5) Momentum Continuation: استمرار دفعة صاعدة بدون مطاردة اختراق ضعيف.
    momentum_continuation = False
    try:
        tail5 = today_d.tail(4)
        closes = tail5["Close"].astype(float)
        opens = tail5["Open"].astype(float)
        highs = tail5["High"].astype(float)
        lows = tail5["Low"].astype(float)
        if len(tail5) >= 4:
            rising = bool(closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3])
            green_now = bool(closes.iloc[-1] >= opens.iloc[-1])
            body_now = abs(closes.iloc[-1] - opens.iloc[-1])
            range_now = max(highs.iloc[-1] - lows.iloc[-1], price * 0.0001)
            close_pos = (closes.iloc[-1] - lows.iloc[-1]) / range_now
            prior_move = (closes.iloc[-2] - closes.iloc[-4]) / max(closes.iloc[-4], 1e-9) * 100
            momentum_continuation = bool(
                trend_up and above_vwap and above_open and not failed and not breakout_now
                and h4_state != "معاكس" and market_ok
                and rising and green_now and prior_move >= 0.35
                and mom > 0.08 and vol_ratio >= 1.05
                and body_now / range_now >= 0.45 and close_pos >= 0.65
                and ext_tmp <= 6.0
            )
    except Exception:
        momentum_continuation = False

    # 6) Compression → Expansion: ضغط سعري ثم توسع مدعوم بالحجم.
    compression_expansion = False
    try:
        if len(today_d) >= 10:
            prev = today_d.iloc[-9:-1]
            cur = today_d.iloc[-1]
            prev_ranges = (prev["High"].astype(float) - prev["Low"].astype(float)).clip(lower=0)
            cur_range = max(float(cur["High"]) - float(cur["Low"]), price * 0.0001)
            med_range = float(prev_ranges.median()) if len(prev_ranges) else 0.0
            comp_range = float(prev["High"].max() - prev["Low"].min())
            comp_width_pct = comp_range / max(price, 1e-9) * 100
            cur_body = abs(float(cur["Close"]) - float(cur["Open"]))
            cur_pos = (float(cur["Close"]) - float(cur["Low"])) / cur_range
            expansion = cur_range >= max(med_range * 1.35, price * 0.003)
            compression = comp_width_pct <= 2.2 and med_range > 0
            compression_expansion = bool(
                compression and expansion and float(cur["Close"]) > float(cur["Open"])
                and cur_pos >= 0.70 and vol_ratio >= 1.20
                and above_vwap and above_open and trend_up
                and h4_state != "معاكس" and market_ok and not failed
                and cur_body / cur_range >= 0.45 and ext_tmp <= 7.0
            )
    except Exception:
        compression_expansion = False

    # مستويات اليوم السابق/الأسبوع السابق/بداية الشهر كعامل جودة.
    prev_day_high = 0.0
    prev_day_low = 0.0
    prev_close_level = 0.0
    try:
        if len(daily) >= 2:
            prev = daily.iloc[:-1].tail(20)
            prev_day_high = float(prev["High"].max())
            prev_day_low = float(prev["Low"].min())
            prev_close_level = float(daily["Close"].iloc[-2])
    except Exception:
        pass

    key_level_near = any(
        lvl > 0 and abs(price - lvl) / max(price, 1e-9) * 100 <= 0.60
        for lvl in (prev_day_high, prev_day_low, prev_close_level, orb_high, level_high)
    )



    # 5) Bull Flag: tight bullish consolidation after an impulsive move,
    # then a clean continuation trigger. Distinct from compression-expansion:
    # the prior leg must already be clearly bullish and the pullback must stay controlled.
    bull_flag = False
    try:
        if len(today_d) >= 12:
            impulse = today_d.iloc[-12:-6]
            flag = today_d.iloc[-6:-1]
            impulse_open = float(impulse["Open"].iloc[0])
            impulse_high = float(impulse["High"].max())
            impulse_gain = (impulse_high - impulse_open) / max(impulse_open, 1e-9) * 100
            flag_high = float(flag["High"].max())
            flag_low = float(flag["Low"].min())
            flag_range = (flag_high - flag_low) / max(flag_high, 1e-9) * 100
            breakout_flag = price >= flag_high * 1.001
            bull_flag = bool(
                impulse_gain >= 1.0
                and flag_range <= 2.0
                and breakout_flag
                and trend_up and above_vwap and above_open
                and h4_state != "معاكس" and market_ok and not failed
                and last_green and mom > 0.05
                and vol_ratio >= 1.05
                and ext_tmp <= 6.0
                and not orb_breakout
            )
    except Exception:
        bull_flag = False

    # 6) Resistance Reclaim: a previously established resistance is lost,
    # then reclaimed with confirmation. This is different from HOD reclaim:
    # the level can be an daily structural resistance, not necessarily today's high.
    resistance_reclaim = False
    reclaim_level = 0.0
    try:
        if len(today_d) >= 10:
            prior = today_d.iloc[-10:-2]
            reclaim_level = float(prior["High"].quantile(0.80))
            prev_close = float(today_d["Close"].iloc[-2])
            resistance_was_lost = prev_close < reclaim_level * 0.999
            reclaimed = price >= reclaim_level * 1.001
            touches = int((prior["High"] >= reclaim_level * 0.995).sum())
            resistance_reclaim = bool(
                touches >= 2
                and resistance_was_lost and reclaimed
                and trend_up and above_vwap and above_open
                and h4_state != "معاكس" and market_ok and not failed
                and last_green and mom > 0.05
                and vol_ratio >= 1.05
                and ext_tmp <= 6.0
            )
    except Exception:
        resistance_reclaim = False

    # 7) Opening Drive → Pullback: دفعة قوية في بداية الشهر ثم تصحيح منظم واستعادة.
    opening_drive_pullback = False
    drive_level = 0.0
    try:
        first6 = month_d.head(6)
        later = month_d.iloc[6:]
        if len(first6) >= 3:
            drive_open = float(first6["Open"].iloc[0])
            drive_high = float(first6["High"].max())
            drive_return = (drive_high - drive_open) / max(drive_open, 1e-9) * 100
            drive_level = drive_high
            if not later.empty and drive_return >= 2.0:
                recent = today_d.tail(3)
                recent_low = float(recent["Low"].min())
                pullback_from_high = (drive_high - recent_low) / max(drive_high, 1e-9) * 100
                reclaim_drive = price >= drive_high * 0.999
                controlled_pullback = 0.50 <= pullback_from_high <= 8.0
                not_chasing = ext_tmp <= 6.0
                opening_drive_pullback = bool(
                    trend_up and above_vwap and above_open
                    and h4_state != "معاكس" and market_ok and not failed
                    and drive_return >= 2.0 and controlled_pullback and reclaim_drive
                    and last_green and mom > 0.20 and vol_ratio >= 1.05
                    and not breakout_now and not orb_breakout and not_chasing
                )
    except Exception:
        opening_drive_pullback = False

    # 8) Period-High Reclaim: استعادة قمة الفترة (20 يومًا).
    # بعد تسجيل قمة يومية، يحصل تراجع تحت القمة ثم استعادة فعلية لها.
    # هذا ليس ORB: المستوى هنا هو HOD المتكوّن خلال الجلسة، وليس أول 15 دقيقة.
    hod_reclaim = False
    hod_level = 0.0
    try:
        if len(today_d) >= 8:
            prior = today_d.iloc[:-2]
            hod_level = float(prior["High"].max())
            if hod_level > 0:
                pullback_below_hod = float(today_d["Close"].iloc[-2]) < hod_level * 0.999
                reclaimed_hod = price >= hod_level * 1.001
                had_hod = float(prior["High"].max()) >= hod_level * 0.999
                hod_reclaim = bool(
                    had_hod and pullback_below_hod and reclaimed_hod
                    and above_vwap and above_open and trend_up
                    and h4_state != "معاكس" and market_ok and not failed
                    and last_green and mom > 0.05
                    and vol_ratio >= 1.05
                    and ext_tmp <= 6.0
                )
    except Exception:
        hod_reclaim = False

    # 9) ORB Failed Breakout -> Reclaim: مخصوص لفشل اختراق نطاق الافتتاح ثم استعادته.
    # مختلف عن سحب السيولة: المستوى هنا ORB High فقط، مع شرط اختراق سابق ثم فشل ثم reclaim.
    orb_failed_reclaim = False
    try:
        if len(today_d) >= 8 and orb_high > 0:
            post_orb = today_d.iloc[3:-1]
            broke = bool((post_orb["High"].astype(float) >= orb_high * 1.002).any())
            failure = bool((post_orb["Close"].astype(float) <= orb_high * 0.998).any())
            reclaim = price >= orb_high * 1.001
            orb_failed_reclaim = bool(
                broke and failure and reclaim
                and trend_up and above_vwap and above_open
                and h4_state != "معاكس" and market_ok and not failed
                and last_green and mom > 0.05
                and vol_ratio >= 1.05
                and ext_tmp <= 6.0
                and not orb_breakout
            )
    except Exception:
        orb_failed_reclaim = False

    # 10) ABC Pullback / 3-Wave Continuation: دفعة A، تصحيح B مضبوط، ثم C.
    # لا يكفي لمس EMA20؛ يجب أن تكون بنية A/B/C واضحة.
    abc_continuation = False
    try:
        if len(today_d) >= 12:
            a = today_d.iloc[-12:-8]
            b = today_d.iloc[-8:-4]
            c = today_d.iloc[-4:]
            a_open = float(a["Open"].iloc[0])
            a_high = float(a["High"].max())
            a_gain = (a_high - a_open) / max(a_open, 1e-9) * 100
            b_high = float(b["High"].max())
            b_low = float(b["Low"].min())
            b_retrace = (a_high - b_low) / max(a_high - a_open, price * 0.001) * 100
            c_high = float(c["High"].max())
            c_last_green = float(c["Close"].iloc[-1]) >= float(c["Open"].iloc[-1])
            c_break = c_high >= a_high * 0.999
            abc_continuation = bool(
                a_gain >= 0.70
                and 20.0 <= b_retrace <= 65.0
                and c_break and price >= a_high * 0.999
                and c_last_green and trend_up and above_vwap and above_open
                and h4_state != "معاكس" and market_ok and not failed
                and mom > 0.05 and vol_ratio >= 1.05
                and ext_tmp <= 6.0
            )
    except Exception:
        abc_continuation = False

    # طبقة Confluence: ليست نوع دخول جديداً، بل Bonus عند اجتماع VWAP + H1 + عدة مستويات.
    confluence_levels = []
    for lvl, label in ((vwap_last, "VWAP"), (e5, "EMA20"), (orb_high, "ORB"),
                       (level_high, "H1-Level"), (prev_close_level, "PrevClose")):
        try:
            if lvl > 0 and abs(price - float(lvl)) / max(price, 1e-9) * 100 <= 0.60:
                confluence_levels.append(label)
        except Exception:
            pass
    multi_level_confluence = len(set(confluence_levels)) >= 3
    vwap_weekly_confluence = bool(
        trend_up and above_vwap and e20 > e50
        and h4_state == "داعم" and not failed
    )

    early = above_vwap and above_open and not breakout_now and ext_tmp <= 2.2 and not failed

    breakout_ok, breakout_quality = _breakout_quality(today_d, level_high, price)
    orb_breakout_ok, orb_quality = _breakout_quality(today_d, orb_high, price) if orb_high > 0 else (False, 0.0)
    # Failed breakout is a rejection/filter condition, not an entry strategy.
    # If there is no separate recovery setup below, the candidate is discarded.
    if failed and not (retest or liquidity_displacement or liquidity_sweep or orb_failed_reclaim or abc_continuation or opening_drive_pullback or hod_reclaim or vwap_bounce or ema_pullback or orb_breakout or breakout_now or compression_expansion or momentum_continuation or bull_flag or resistance_reclaim):
        return None

    if retest:
        entry_type, entry_emoji = "إعادة اختبار", "🟡"
    elif orb_breakout and orb_breakout_ok:
        breakout_quality = max(breakout_quality, orb_quality)
        entry_type, entry_emoji = "اختراق نطاق الافتتاح", "🟢"
    elif breakout_now and breakout_ok and vol_ratio >= 1.0:
        entry_type, entry_emoji = "اختراق مؤكد", "🟢"
    elif liquidity_displacement:
        entry_type, entry_emoji = "سحب سيولة مع Displacement", "🟢"
    elif liquidity_sweep:
        entry_type, entry_emoji = "سحب سيولة", "🟢"
    elif compression_expansion:
        entry_type, entry_emoji = "ضغط ثم انفجار", "🟢"
    elif momentum_continuation:
        entry_type, entry_emoji = "استمرار الزخم", "🟢"
    elif bull_flag:
        entry_type, entry_emoji = "علم صاعد", "🟢"
    elif resistance_reclaim:
        entry_type, entry_emoji = "استعادة مستوى", "🟢"
    elif orb_failed_reclaim:
        entry_type, entry_emoji = "استعادة بعد فشل ORB", "🟢"
    elif abc_continuation:
        entry_type, entry_emoji = "استمرار ABC", "🟢"
    elif opening_drive_pullback:
        entry_type, entry_emoji = "دخول بعد Opening Drive", "🟢"
    elif hod_reclaim:
        entry_type, entry_emoji = "استعادة قمة الفترة", "🟢"
    elif vwap_bounce:
        entry_type, entry_emoji = "ارتداد VWAP", "🟢"
    elif ema_pullback:
        entry_type, entry_emoji = "ارتداد EMA20", "🟢"
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
        reasons.append("اتجاه الأسبوعي صاعد")
        factors.append("weekly_trend")
    else:
        warnings.append("اتجاه الأسبوعي غير مؤكد")
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
        reasons.append("تأكيد يومي")
        factors.append("daily")
    else:
        warnings.append("لا تأكيد يومي كافٍ")
        score -= 10

    if vol_ok:
        score += 6
        reasons.append(f"حجم جلسة {vol_ratio:.2f}x")
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
        warnings.append("السوق اليومي متذبذب (Chop)")
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

    if h4_state == "داعم":
        score += h4_points
        reasons.append("4 ساعات داعمة")
        factors.append("h4")
    elif h4_state == "معاكس":
        score += h4_points
        warnings.append("4 ساعات معاكسة")
    else:
        reasons.append("4 ساعات محايدة")
        factors.append("h4_neutral")

    if 48 <= h_rsi <= 68:
        score += 5
        factors.append("rsi_weekly")

    if dump:
        warnings.append("سقوط من قمة الفترة")
        score -= 20

    if entry_type == "اختراق مؤكد":
        score += 8
        reasons.append("اختراق مؤكد")
        factors.append("breakout")
    elif entry_type == "اختراق نطاق الافتتاح":
        score += 8
        reasons.append("اختراق نطاق الافتتاح ORB")
        factors.append("orb")
        factors.append("breakout")
    elif entry_type == "إعادة اختبار":
        score += 5
        reasons.append("إعادة اختبار مستوى")
        factors.append("retest")
    elif entry_type == "ارتداد VWAP":
        score += 6
        reasons.append("ارتداد واستعادة VWAP")
        factors.append("vwap_bounce")
    elif entry_type == "ارتداد EMA20":
        score += 6
        reasons.append("تصحيح صحي إلى EMA20")
        factors.append("ema_pullback")
    elif entry_type == "سحب سيولة مع Displacement":
        score += 10
        reasons.append("سحب سيولة ثم Displacement واستعادة قوية")
        factors.append("liquidity_sweep")
        factors.append("liquidity_displacement")
    elif entry_type == "سحب سيولة":
        score += 7
        reasons.append("سحب سيولة ثم استعادة المستوى")
        factors.append("liquidity_sweep")
    elif entry_type == "ضغط ثم انفجار":
        score += 8
        reasons.append("ضغط سعري ثم توسع بالحجم")
        factors.append("compression_expansion")
    elif entry_type == "استمرار الزخم":
        score += 7
        reasons.append("استمرار زخم بعد دفعة صاعدة")
        factors.append("momentum_continuation")
    elif entry_type == "علم صاعد":
        score += 9
        reasons.append("علم صاعد بعد دفعة قوية ثم استمرار")
        factors.append("bull_flag")
    elif entry_type == "استعادة مستوى":
        score += 9
        reasons.append("استعادة مقاومة بعد كسرها")
        factors.append("resistance_reclaim")
    elif entry_type == "استعادة بعد فشل ORB":
        score += 10
        reasons.append("فشل اختراق ORB ثم استعادة مؤكدة")
        factors.append("orb_failed_reclaim")
        factors.append("orb")
    elif entry_type == "استمرار ABC":
        score += 9
        reasons.append("بنية A/B/C: دفعة ثم تصحيح منظم ثم استمرار")
        factors.append("abc_continuation")
    elif entry_type == "دخول بعد Opening Drive":
        score += 9
        reasons.append("دفعة افتتاحية قوية ثم تراجع منظم واستعادة")
        factors.append("opening_drive_pullback")
    elif entry_type == "استعادة قمة الفترة":
        score += 9
        reasons.append("استعادة قمة الفترة بعد تراجع تحتها")
        factors.append("hod_reclaim")
    else:
        reasons.append("دخول مبكر فوق VWAP")
        factors.append("early")

    if key_level_near and "key_level" not in factors:
        factors.append("key_level")
        reasons.append("قرب مستوى سعري مهم")

    if vwap_weekly_confluence:
        score += 3
        factors.append("vwap_weekly_confluence")
        reasons.append("Confluence: VWAP + اتجاه الأسبوعي + 4س")
    if multi_level_confluence:
        score += min(5, 2 + len(set(confluence_levels)))
        factors.append("multi_level_confluence")
        reasons.append("تجمع مستويات: " + "/".join(confluence_levels[:4]))

    ext = (price - e20) / e20 * 100 if e20 else 0
    if ext > 4.0:
        warnings.append("امتداد عن متوسط الأسبوعي")
        score -= 8
        factors.append("extended")

    atr = float(_atr(weekly, 14).iloc[-1] or price * 0.01)
    atr_pct = atr / price * 100
    market_regime = _classify_regime(
        trend_up, h4_state, chop, market_ok, atr_pct, news_state
    )
    interaction_keys = _interaction_keys(
        entry_type, market_regime, h4_state, vol_ratio, 0.0
    )
    if atr_pct > 8.0:
        warnings.append("تذبذب عالي")
        score -= 5

    learning_adj = _learning_adjustment(factors, entry_type)
    adaptive_adj = _adaptive_score_adjustment(factors, market_regime, entry_type)
    total_learning_adj = learning_adj + adaptive_adj
    if total_learning_adj:
        score += total_learning_adj
        reasons.append(f"تعلم ذاتي {total_learning_adj:+.1f}")

    strong_alignment = (
        trend_up
        and live_ok
        and above_vwap
        and h4_state != "معاكس"
        and vol_ratio >= 1.0
        and not dump
        and ext <= 7.0
    )

    # Daily strong-stock override: weak market can be bypassed only when the
    # stock itself is exceptionally aligned. Mixed SPY/QQQ already passes the
    # normal market gate and therefore does not need an override. Missing market
    # data can never trigger this override.
    policy = _load_adaptive_policy()
    limits = policy.get("entry_limits", {})
    if entry_type == "دخول مبكر":
        score = min(score, float(limits.get("دخول مبكر", 94)))
    elif entry_type == "إعادة اختبار":
        score = min(score, float(limits.get("إعادة اختبار", 97.0 if strong_alignment else 95.0)))
    elif entry_type == "ارتداد VWAP":
        score = min(score, float(limits.get("ارتداد VWAP", 96.0)))
    elif entry_type == "ارتداد EMA20":
        score = min(score, float(limits.get("ارتداد EMA20", 96.0)))
    elif entry_type == "سحب سيولة مع Displacement":
        score = min(score, float(limits.get("سحب سيولة مع Displacement", 99.0)))
    elif entry_type == "سحب سيولة":
        score = min(score, float(limits.get("سحب سيولة", 97.0)))
    elif entry_type == "ضغط ثم انفجار":
        score = min(score, float(limits.get("ضغط ثم انفجار", 98.0)))
    elif entry_type == "استمرار الزخم":
        score = min(score, float(limits.get("استمرار الزخم", 97.0)))
    elif entry_type == "علم صاعد":
        score = min(score, float(limits.get("علم صاعد", 98.0)))
    elif entry_type == "استعادة مستوى":
        score = min(score, float(limits.get("استعادة مستوى", 98.0)))
    elif entry_type == "استعادة بعد فشل ORB":
        score = min(score, float(limits.get("استعادة بعد فشل ORB", 99.0)))
    elif entry_type == "استمرار ABC":
        score = min(score, float(limits.get("استمرار ABC", 98.0)))
    elif entry_type == "دخول بعد Opening Drive":
        score = min(score, float(limits.get("دخول بعد Opening Drive", 98.0)))
    elif entry_type == "استعادة قمة الفترة":
        score = min(score, float(limits.get("استعادة قمة الفترة", 98.0)))
    elif entry_type == "اختراق نطاق الافتتاح":
        score = min(score, float(limits.get("اختراق نطاق الافتتاح", 99.0)))
    elif entry_type == "اختراق مؤكد":
        score = min(score, float(limits.get("اختراق مؤكد", 100.0)))
    else:
        score = min(score, 80.0)

    score_i = int(max(0, min(100, round(score))))

    # Daily strong-stock override: weak market does not automatically block an exceptional stock.
    # The early-entry strategy is allowed if it independently reaches the strong daily threshold.
    # 93 is intentional: the daily "دخول مبكر" score cap is 94, so a 95/97 threshold
    # would make the override mathematically unreachable for that strategy.
    strong_stock_market_override = bool(
        (not market_ok)
        and market_state == "SPY+QQQ ضعيفان يوميًا"
        and score_i >= 93
        and strong_alignment
    )

    strong_for_grade = (
        score_i >= 95
        and strong_alignment
        and entry_type in {"اختراق مؤكد", "اختراق نطاق الافتتاح", "إعادة اختبار", "ارتداد VWAP", "ارتداد EMA20", "سحب سيولة", "ضغط ثم انفجار", "استمرار الزخم", "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة الفترة", "استعادة بعد فشل ORB", "استمرار ABC", "سحب سيولة مع Displacement"}
        and h4_state != "معاكس"
    )

    news_momentum_ok = True
    if news_state == "negative" and NEWS_BLOCK_NEGATIVE:
        news_momentum_ok = False
    if news_state == "positive_strong":
        news_momentum_ok = (
            change_pct >= float(policy.get("min_news_change_pct", NEWS_MOMENTUM_MIN_CHANGE))
            and vol_ratio >= float(policy.get("min_news_volume_ratio", NEWS_MOMENTUM_MIN_VOLUME))
            and above_vwap
            and (breakout_ok or entry_type != "دخول مبكر")
            and (breakout_quality >= 60 or entry_type != "دخول مبكر")
            and h4_state != "معاكس"
            and (market_ok or strong_stock_market_override)
        )

    quality_ok = (
        (not dump)
        and (not failed)
        and ext <= 8.0
        and atr_pct <= 8.0
        and vol_ratio >= float(policy.get("min_volume_ratio", 0.85))
        and not chop
        and news_momentum_ok
        and not (h4_state == "معاكس" and score_i < 92)
        and (market_ok or strong_stock_market_override)
    )

    recent_low = float(today_d["Low"].tail(12).min())

    # Structure-aware daily stop. The stop is placed behind the structure
    # that actually justifies the entry, then constrained to a practical
    # daily risk band of 0.60%–4.50%.
    stop_candidates = []
    if entry_type == "ارتداد VWAP":
        stop_candidates.append(vwap_last * 0.997)
    elif entry_type == "ارتداد EMA20":
        stop_candidates.append(e5 * 0.997)
    elif entry_type in {"سحب سيولة", "سحب سيولة مع Displacement"}:
        stop_candidates.append(support_level * 0.997 if support_level > 0 else recent_low * 0.997)
    elif entry_type in {"اختراق مؤكد", "اختراق نطاق الافتتاح"}:
        level = orb_high if entry_type == "اختراق نطاق الافتتاح" else level_high
        if level and level > 0:
            stop_candidates.append(level * 0.997)
    elif entry_type == "إعادة اختبار":
        level = level_high
        if level and level > 0:
            stop_candidates.append(level * 0.997)
    elif entry_type == "علم صاعد":
        stop_candidates.append(float(today_d["Low"].tail(5).min()) * 0.997)
    elif entry_type == "استعادة مستوى":
        stop_candidates.append(reclaim_level * 0.997 if reclaim_level > 0 else recent_low * 0.997)
    elif entry_type == "استعادة بعد فشل ORB":
        stop_candidates.append(orb_high * 0.997 if orb_high > 0 else recent_low * 0.997)
    elif entry_type == "استمرار ABC":
        stop_candidates.append(float(today_d["Low"].tail(4).min()) * 0.997)
    elif entry_type == "دخول بعد Opening Drive":
        stop_candidates.append(drive_level * 0.997 if drive_level > 0 else recent_low * 0.997)
    elif entry_type == "استعادة قمة الفترة":
        stop_candidates.append(hod_level * 0.997 if hod_level > 0 else recent_low * 0.997)
    elif entry_type in {"ضغط ثم انفجار", "استمرار الزخم"}:
        stop_candidates.append(float(today_d["Low"].tail(5).min()) * 0.997)

    # ATR remains the fallback/secondary safety reference.
    stop_candidates.append(price - 1.5 * atr)
    stop_candidates.append(recent_low * 0.997)

    # Choose the nearest valid structural stop below price.
    valid_stops = [x for x in stop_candidates if x > 0 and x < price]
    stop = max(valid_stops) if valid_stops else price * 0.985

    risk = price - stop
    min_risk = price * 0.015
    max_risk = price * 0.10
    if risk < min_risk:
        stop = price - min_risk
        risk = min_risk
    elif risk > max_risk:
        # Too-wide structures are rejected rather than hiding the risk.
        quality_ok = False
        warnings.append("وقف هيكلي واسع جدًا")

    resistance_tp1, resistance_source = _find_prior_resistance(today_d, weekly, price)

    # TP1 must be at least 1.20R. Prefer real resistance only when it clears
    # that threshold; otherwise use a risk-multiple fallback.
    fallback_tp1 = price + risk * 1.20
    if resistance_tp1 > price and resistance_tp1 >= fallback_tp1:
        tp1 = resistance_tp1
    else:
        tp1 = fallback_tp1
        resistance_source = "هدف مخاطر 1.20R"

    if tp1 <= price:
        quality_ok = False
        warnings.append("TP1 غير صالح")

    # Adaptive Exit Engine is inert until its own OOS validation activates it.
    policy_now = _load_adaptive_policy()
    adaptive_stop, adaptive_tp1, adaptive_tp1_r = _apply_adaptive_exit(
        price, stop, market_regime, entry_type, policy_now, atr
    )
    if policy_now.get("exit_active"):
        stop = adaptive_stop
        tp1 = adaptive_tp1
        risk = price - stop
        risk_pct_check = risk / price * 100 if price else 0.0
        if 1.50 <= risk_pct_check <= 10.00:
            warnings.append(f"Adaptive Exit: TP1={adaptive_tp1_r:.2f}R")
        else:
            # Safety: revert to the original structural stop if adaptive scaling
            # somehow leaves the allowed daily risk band.
            stop = max(stop_candidates)
            risk = price - stop
            tp1 = price + risk * 1.20

    tp2 = price + risk * 2.0
    tp3 = price + risk * 3.0
    risk_pct = risk / price * 100
    reward_r = (tp1 - price) / risk if risk else 0.0
    tp1_distance_pct = (tp1 - price) / price * 100 if price else 0.0
    if tp1_distance_pct < 0.8:
        quality_ok = False
        warnings.append("TP1 قريب جدًا من الدخول")
    if reward_r < float(policy.get("min_tp1_r", 1.2)):
        quality_ok = False
        warnings.append("العائد إلى TP1 ضعيف")
    if news_state == "positive_strong" and news_momentum_ok:
        reasons.append("NEWS MOMENTUM مؤكد")
    buy_low = max(stop * 1.01, min(price * 0.995, e5))
    buy_high = price * 1.004

    # فحص Spread/السيولة يُجرى في scan_daily للمرشحين فقط، حتى لا يبطئ تحليل كل الأسهم.
    liquidity = {"ok": True, "spread_pct": 0.0, "slippage_pct": 0.0, "dollar_volume": 0.0}
    liquidity_ok = True

    interaction_keys = _interaction_keys(
        entry_type, market_regime, h4_state, vol_ratio, breakout_quality
    )

    if resistance_source != "هدف مخاطر 1.20R":
        reasons.append(f"TP1 مقاومة: {tp1:.2f}")
    else:
        warnings.append("لم توجد مقاومة قريبة مناسبة؛ TP1 احتياطي")

    return DailySignal(
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
        sl_method="وقف هيكلي حسب نوع الدخول",
        vwap_note=vwap_note,
        above_open=above_open,
        vol_ok=vol_ok,
        reasons=reasons[:5],
        warnings=warnings[:4],
        quality_ok=quality_ok,
        live_ok=live_ok,
        volume_ratio=round(vol_ratio, 2),
        factor_keys=factors,
        sma20=round(e20, 4),
        atr_pct=round(atr_pct, 2),
        ext_sma20=round(ext, 2),
        entry_type=entry_type,
        entry_emoji=entry_emoji,
        h4_state=h4_state,
        learning_adjustment=round(total_learning_adj, 2),
        resistance_tp1=round(tp1, 4),
        news_state=news_state,
        news_title=news_title,
        news_source=news_source,
        breakout_quality=round(breakout_quality, 1),
        market_state=market_state,
        chop=chop,
        market_regime=market_regime,
        interaction_keys=interaction_keys,
        spread_pct=round(float(liquidity.get("spread_pct", 0) or 0), 3),
        expected_slippage_pct=round(float(liquidity.get("slippage_pct", 0) or 0), 3),
        dollar_volume_3m=round(float(liquidity.get("dollar_volume", 0) or 0), 0),
        liquidity_ok=liquidity_ok,
    )


def format_daily_ar(sig: DailySignal, min_score: int = DAILY_MIN_SCORE) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    tp_source = "مقاومة/قمة سابقة" if sig.resistance_tp1 else "احتياطي"
    lines = [
        f"⚡ يومي | {sig.symbol} | {sig.score}/100 | {sig.grade} | ساعة+4س+يومي",
        f"{sig.entry_emoji} نوع الدخول: {sig.entry_type}",
        f"{sig.name}",
        "—————————————",
        f"السعر: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
        f"شراء: {sig.buy_low:.2f} — {sig.buy_high:.2f}",
        f"وقف: {sig.stop_loss:.2f} ({sig.sl_method}) | مخاطرة {sig.risk_pct:.2f}%",
        f"TP1: {sig.tp1:.2f} | TP2: {sig.tp2:.2f} | TP3: {sig.tp3:.2f}",
        f"مصدر TP1: {tp_source} | العائد إلى TP1: {sig.reward_r:.2f}R",
        "—————————————",
        f"{sig.vwap_note} | افتتاح: {'فوق' if sig.above_open else 'تحت'} | حجم: {sig.volume_ratio:.2f}x",
        f"4س: {sig.h4_state} | السوق: {sig.market_state} | تعلم: {sig.learning_adjustment:+.1f}",
        f"الأخبار: {sig.news_state} | جودة الاختراق: {sig.breakout_quality:.0f}/100",
        f"Spread: {sig.spread_pct:.2f}% | انزلاق متوقع: {sig.expected_slippage_pct:.2f}% | السيولة: {'مناسبة' if sig.liquidity_ok else 'غير مناسبة'}",
    ]
    if sig.reasons:
        lines.append("لماذا: " + " | ".join(sig.reasons[:3]))
    if sig.warnings:
        lines.append("مخاطر: " + " | ".join(sig.warnings[:3]))
    if sig.score < min_score or not sig.live_ok or not sig.quality_ok:
        lines.append(f"تحت شرط الإرسال اليومي ({min_score}+ / تأكيد / جودة)")
    lines.append("تحليل يومي تعليمي — ليست توصية. يفضّل الخروج قبل الإغلاق.")
    return "\n".join(lines)



def get_learning_alert() -> dict | None:
    """يقرأ آخر تحديث تعلم ليتم إرساله من main.py إلى Telegram."""
    try:
        if not LEARNING_ALERT_FILE.exists():
            return None
        data = json.loads(LEARNING_ALERT_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("message") else None
    except Exception:
        return None



def _prefilter_daily(symbol: str) -> tuple[float, pd.DataFrame, pd.DataFrame] | None:
    """Stage 1: weekly + daily routing for the full universe."""
    try:
        from market_data import fetch_intraday, intraday_data_fresh
        weekly = fetch_intraday(symbol, interval="1wk", period="5y")
        daily = fetch_intraday(symbol, interval="1d", period="2y")
        if weekly is None or daily is None or len(weekly) < 60 or len(daily) < 80:
            return None
        price = float(daily["Close"].iloc[-1])
        if price <= 0 or price > float(MAX_AUTO_PRICE):
            return None
        wc = weekly["Close"].astype(float)
        dc = daily["Close"].astype(float)
        we20 = float(_ema(wc,20).iloc[-1]); we50 = float(_ema(wc,50).iloc[-1])
        de20 = float(_ema(dc,20).iloc[-1]); de50 = float(_ema(dc,50).iloc[-1])
        wrsi = float(_rsi(wc,14).iloc[-1]); drsi = float(_rsi(dc,14).iloc[-1])
        trend = price >= we20 * 0.99 and we20 >= we50 * 0.995 and drsi >= 42
        v = _vwap(daily.tail(60)); vw = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else price
        above_vwap = price >= vw * 0.995
        above_open = price >= float(daily["Open"].iloc[-1]) * 0.995
        hist = daily.iloc[:-1].tail(120)
        avg_cur = float(daily["Volume"].tail(5).mean())
        avg_hist = float(hist["Volume"].mean()) if not hist.empty else 1.0
        vol_ratio = avg_cur / avg_hist if avg_hist else 1.0
        mom = (price - float(dc.iloc[-6])) / max(float(dc.iloc[-6]),1e-9) * 100 if len(dc)>=6 else 0.0
        route = 0.0
        route += 3.0 if trend else 0.0
        route += 2.0 if above_vwap else 0.0
        route += 1.5 if above_open else 0.0
        route += min(2.5,max(0.0,mom))
        route += min(2.0,max(0.0,vol_ratio-0.75)*2.0)
        route += 1.0 if we20 > we50 else 0.0
        route -= 1.0 if wrsi >= 80 else 0.0
        if not trend and not above_vwap and mom <= 0:
            return None
        if vol_ratio < 0.55:
            return None
        return route, weekly, daily
    except Exception:
        return None



def _prefilter_daily_from_frames(symbol: str, weekly: pd.DataFrame, daily: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame] | None:
    try:
        if weekly is None or daily is None or len(weekly) < 60 or len(daily) < 80:
            return None
        price = float(daily["Close"].iloc[-1])
        if price <= 0 or price > float(MAX_AUTO_PRICE):
            return None
        wc = weekly["Close"].astype(float); dc = daily["Close"].astype(float)
        we20 = float(_ema(wc,20).iloc[-1]); we50 = float(_ema(wc,50).iloc[-1])
        wrsi = float(_rsi(wc,14).iloc[-1]); drsi = float(_rsi(dc,14).iloc[-1])
        trend = price >= we20 * 0.99 and we20 >= we50 * 0.995 and drsi >= 42
        v = _vwap(daily.tail(60)); vw = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else price
        above_vwap = price >= vw * 0.995
        above_open = price >= float(daily["Open"].iloc[-1]) * 0.995
        hist = daily.iloc[:-1].tail(120)
        avg_cur = float(daily["Volume"].tail(5).mean()); avg_hist = float(hist["Volume"].mean()) if not hist.empty else 1.0
        vol_ratio = avg_cur / avg_hist if avg_hist else 1.0
        mom = (price - float(dc.iloc[-6])) / max(float(dc.iloc[-6]),1e-9) * 100 if len(dc)>=6 else 0.0
        route = (3.0 if trend else 0.0) + (2.0 if above_vwap else 0.0) + (1.5 if above_open else 0.0)
        route += min(2.5,max(0.0,mom)) + min(2.0,max(0.0,vol_ratio-0.75)*2.0) + (1.0 if we20 > we50 else 0.0)
        route -= 1.0 if wrsi >= 80 else 0.0
        if not trend and not above_vwap and mom <= 0 or vol_ratio < 0.55:
            return None
        return route, weekly, daily
    except Exception:
        return None

def scan_daily(
    symbols: list[str],
    names: dict,
    min_score: int = DAILY_MIN_SCORE,
    limit: int = 8,
) -> list[DailySignal]:
    ok, reason = session_window_ok()
    if not ok:
        scan_daily.last_window = reason
        return []
    scan_daily.last_window = "ok"
    try:
        from market_data import fetch_intraday
        market_context = _market_alignment(fetch_intraday)
        if not market_context[0] and "ضعيفان" in market_context[1]:
            min_score = max(min_score, 88)
    except Exception as exc:
        log.warning("Daily market context unavailable after retries: %s", exc)
        market_context = (False, "بيانات SPY/QQQ غير متاحة")

    workers = min(8, max(2, len(symbols)))
    stage1=[]
    log.info("DAILY V2 SCAN: %d symbols loaded", len(symbols))
    try:
        from market_data import fetch_alpaca_bars_multi, alpaca_configured
        if alpaca_configured():
            days_w = _period_days("5y", 365)
            days_d = _period_days("2y", 365)
            now_utc = datetime.now(timezone.utc)
            weekly_map = fetch_alpaca_bars_multi(symbols, "1Week", now_utc - timedelta(days=days_w + 5), now_utc)
            daily_map = fetch_alpaca_bars_multi(symbols, "1Day", now_utc - timedelta(days=days_d + 5), now_utc)
            def _route_from_frames(sym):
                weekly = weekly_map.get(sym.upper()); daily = daily_map.get(sym.upper())
                if weekly is None or daily is None or len(weekly) < 60 or len(daily) < 80:
                    return None
                return _prefilter_daily_from_frames(sym, weekly, daily)
            for sym in symbols:
                item = _route_from_frames(sym)
                if item:
                    route, weekly, daily = item; stage1.append((route, sym, weekly, daily))
        else:
            raise RuntimeError("Alpaca not configured")
    except Exception as exc:
        log.warning("Daily batch scan unavailable; using per-symbol fallback: %s", exc)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_prefilter_daily,sym):sym for sym in symbols}
            for fut in as_completed(futures):
                sym=futures[fut]
                try: item=fut.result()
                except Exception: item=None
                if item:
                    route,weekly,daily=item; stage1.append((route,sym,weekly,daily))
    stage1.sort(key=lambda x:x[0], reverse=True)
    log.info("STAGE 1 DAILY: %d/%d passed", len(stage1), len(symbols))
    finalists=stage1[:max(PREFILTER_MAX_CANDIDATES, limit*5)]
    log.info("STAGE 2 DAILY: top %d", len(finalists))
    results=[]
    # Diagnostics only: these counters do NOT change any selection rule.
    # Each finalist is counted at the first rejection gate it fails so /scan
    # reports exactly where Stage 2 candidates disappear.
    stage2_rejects = {
        "exception": 0,
        "no_signal": 0,
        "score": 0,
        "live_ok": 0,
        "quality": 0,
        "negative_news": 0,
        "liquidity": 0,
        "stale_or_no_quote": 0,
    }
    def one(item):
        _,sym,weekly,daily=item
        try:
            return analyze_daily(sym,names.get(sym,sym),True,market_context,(weekly,daily))
        except Exception as exc:
            log.warning("DAILY STAGE 2 EXCEPTION | %s | %s", sym, str(exc))
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,x) for x in finalists]
        for fut in as_completed(futures):
            try:
                sig=fut.result()
            except Exception as exc:
                stage2_rejects["exception"] += 1
                log.warning("DAILY STAGE 2 FUTURE EXCEPTION | %s", str(exc))
                continue
            if not sig:
                stage2_rejects["no_signal"] += 1
                continue
            if sig.score < min_score:
                stage2_rejects["score"] += 1
                continue
            if not sig.live_ok:
                stage2_rejects["live_ok"] += 1
                continue
            if not sig.quality_ok:
                stage2_rejects["quality"] += 1
                continue
            if sig.news_state == "negative":
                stage2_rejects["negative_news"] += 1
                continue
            # Validate current bid/ask only for the Stage-2 finalists.
            # This keeps the full-universe scan fast while preventing Daily V2
            # from reporting a false "liquidity suitable" result.
            try:
                liq = _quote_liquidity(sig.symbol, sig.price)
                sig.spread_pct = round(float(liq.get("spread_pct", 0) or 0), 3)
                sig.expected_slippage_pct = round(float(liq.get("slippage_pct", 0) or 0), 3)
                sig.dollar_volume_3m = round(float(liq.get("dollar_volume", 0) or 0), 0)
                sig.liquidity_ok = bool(liq.get("ok", False))
                if not sig.liquidity_ok:
                    stage2_rejects["liquidity"] += 1
                    continue
                if liq.get("quote_source") == "none" or float(liq.get("quote_age_min", 999) or 999) > 2.0:
                    stage2_rejects["stale_or_no_quote"] += 1
                    continue
            except Exception as exc:
                stage2_rejects["liquidity"] += 1
                log.warning("DAILY LIQUIDITY CHECK FAILED | %s | %s", sig.symbol, str(exc))
                continue
            results.append(sig)

    log.info(
        "STAGE 2 DAILY RESULT: finalists=%d | qualified=%d | rejects=%s",
        len(finalists),
        len(results),
        stage2_rejects,
    )
    rank={et:i for i,et in enumerate(ENTRY_TYPES)}
    results.sort(key=lambda x:(
        -(float(x.score)+1.5*min(float(getattr(x,'reward_r',0) or 0),3.0)
          +2.0*("multi_level_confluence" in (getattr(x,'factor_keys',[]) or []))
          +1.5*("vwap_weekly_confluence" in (getattr(x,'factor_keys',[]) or []))
          -1.5*float(getattr(x,'spread_pct',0) or 0)
          -1.0*float(getattr(x,'expected_slippage_pct',0) or 0)
          -0.8*max(float(getattr(x,'ext_sma20',0) or 0)-3.0,0.0)),
        rank.get(x.entry_type,99),-float(x.score),-float(getattr(x,'reward_r',0) or 0)))
    return results[:limit]


scan_daily.last_window = ""


# Public compatibility API expected by the existing bot.
analyze = analyze_daily
format_signal_ar = format_daily_ar
scan_symbols = scan_daily
rank_all = lambda symbols, names: scan_daily(symbols, names, DAILY_MIN_SCORE, 5)

