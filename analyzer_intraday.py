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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from market import now_ny, REGULAR_OPEN, REGULAR_CLOSE
from stocks import MAX_AUTO_PRICE

log = logging.getLogger(__name__)

# Deployment marker: proves which analyzer_intraday build Render actually loaded.
INTRADAY_ANALYZER_VERSION = "REGIME_ADAPTIVE_PROFESSIONAL_MARKET_REGIME_V5"
log.info("INTRADAY ANALYZER VERSION | %s", INTRADAY_ANALYZER_VERSION)

SKIP_OPEN_MIN = 20
SKIP_CLOSE_MIN = 20
INTRADAY_MIN_SCORE = 82

INTRADAY_LEARNING_FILE = Path("/var/data/intraday_learning.jsonl")
LEARNING_MIN_SAMPLES = 20
LEARNING_LOOKBACK = 60
LEARNING_MAX_ADJUSTMENT = 4.0
ADAPTIVE_POLICY_FILE = Path("/var/data/intraday_adaptive_policy.json")
ADAPTIVE_MIN_SAMPLES = 30
ADAPTIVE_CONFIRM_SAMPLES = 40
ADAPTIVE_MAX_CHANGE = 0.15
STRATEGY_WEIGHT_MIN_SAMPLES = 20
STRATEGY_WEIGHT_STEP = 0.05
STRATEGY_WEIGHT_MIN_FACTOR = 0.50
STRATEGY_WEIGHT_MAX_FACTOR = 1.50
ADAPTIVE_BEST_FILE = Path("/var/data/intraday_adaptive_best.json")
ADAPTIVE_SHADOW_FILE = Path("/var/data/intraday_shadow_results.jsonl")
LEARNING_ALERT_FILE = Path("/var/data/intraday_learning_alert.json")

# Canonical list: the adaptive learner must track every real entry strategy.
ENTRY_TYPES = (
    "اختراق مؤكد", "إعادة اختبار", "دخول مبكر", "ارتداد VWAP", "ارتداد EMA20",
    "سحب سيولة", "اختراق نطاق الافتتاح", "استمرار الزخم", "ضغط ثم انفجار",
    "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة اليوم",
    "استعادة بعد فشل ORB", "استمرار ABC", "سحب سيولة مع Displacement",
)
PREFILTER_MAX_CANDIDATES = 50
# Stage-1 strategy-aware routing: reserve a small lane for every one of the 16 strategies.
PREFILTER_STRATEGY_TOP_K = 4
PREFILTER_STRATEGY_CAP = 100
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

# New dedicated market state: both SPY/QQQ are above session open but below VWAP.
# It is intentionally separate from mixed/weak and starts with a neutral learning weight.
POSITIVE_BELOW_VWAP_MAX_OPEN_GAP_PCT = 0.30
POSITIVE_BELOW_VWAP_MIN_SCORE = 85

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
# احتياطات التعلم الاحترافية
MONTHLY_MIN_SAMPLES = 100
STRATEGY_MIN_SAMPLES = 30
REGIME_STRATEGY_MIN_SAMPLES = 20
KILL_SWITCH_LOOKBACK = 30
KILL_SWITCH_MIN_DROP = 0.10
SIGNAL_DEDUP_MINUTES = 180

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
    entry_type: str = ""
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
    # سعر الدخول الفعلي وقت إرسال التنبيه؛ لا يغيّر سعر التحليل الأصلي
    alert_entry_price: float = 0.0
    m15_state: str = "محايد"
    learning_adjustment: float = 0.0
    resistance_tp1: float = 0.0
    news_state: str = "neutral"
    news_title: str = ""
    news_source: str = ""
    breakout_quality: float = 0.0
    market_state: str = "السوق غير مؤكد"
    market_condition: str = "غير مؤكد"
    market_relative_strength: float = 0.0
    market_avg_change: float = 0.0
    chop: bool = False
    market_regime: str = "neutral"
    interaction_keys: list | None = None
    spread_pct: float = 0.0
    expected_slippage_pct: float = 0.0
    dollar_volume_3m: float = 0.0
    liquidity_ok: bool = True
    diagnostic_reasons: list[str] | None = None
    # جميع الاستراتيجيات المطابقة فعليًا، وليس الاستراتيجية الأساسية فقط.
    matched_entry_types: list[str] | None = None
    # تقييم قوة كل استراتيجية مطابقة لاختيار الأقوى بدل أولوية الاسم فقط.
    strategy_scores: dict[str, float] | None = None
    strategy_component_scores: dict[str, dict[str, float]] | None = None


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
    # Adaptive learning uses the full accumulated history; the 100-new-trade
    # cycle gate controls when a new generation may be trained.
    return completed



def _default_adaptive_policy() -> dict:
    return {
        "version": 1,
        "generation": 0,
        "samples_at_update": 0,
        "approved": False,
        "weights": {
            "h1_trend": 1.0, "m15": 1.0, "m5": 1.0, "vwap": 1.0,
            "vol_session": 1.0, "market": 1.0, "breakout": 1.0,
            "breakout_candle": 1.0, "retest": 1.0, "vwap_bounce": 1.0,
            "ema_pullback": 1.0, "liquidity_sweep": 1.0, "liquidity_displacement": 1.0, "orb": 1.0, "momentum_continuation": 1.0, "compression_expansion": 1.0, "bull_flag": 1.0, "resistance_reclaim": 1.0, "opening_drive_pullback": 1.0, "hod_reclaim": 1.0, "orb_failed_reclaim": 1.0, "abc_continuation": 1.0, "vwap_h1_confluence": 1.0, "multi_level_confluence": 1.0, "early": 1.0,
            "news_momentum": 1.0,
        },
        "entry_limits": {"دخول مبكر": 94, "إعادة اختبار": 95, "ارتداد VWAP": 96, "ارتداد EMA20": 96, "سحب سيولة": 97, "ضغط ثم انفجار": 98, "استمرار الزخم": 97, "اختراق نطاق الافتتاح": 99, "اختراق مؤكد": 100, "علم صاعد": 98, "استعادة مستوى": 98, "دخول بعد Opening Drive": 98, "استعادة قمة اليوم": 98, "استعادة بعد فشل ORB": 99, "استمرار ABC": 98, "سحب سيولة مع Displacement": 99},
        "strategy_stats": {et: {"samples": 0, "wins": 0, "win_rate": 0.0} for et in ENTRY_TYPES},
        "strategy_weights": {},
        "strategy_weights_active": False,
        "strategy_weights_generation": 0,
        "min_volume_ratio": 0.85,
        "min_news_volume_ratio": 1.50,
        "min_news_change_pct": 4.0,
        "min_tp1_r": 1.20,
        "regime_weights": {
            regime: {et: 1.0 for et in ENTRY_TYPES}
            for regime in (
                "chop", "trend_clean", "trend_mixed", "market_weak",
                "market_positive_below_vwap",
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
                "market_positive_below_vwap",
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
        # Legacy learning knowledge is preserved inside the approved Adaptive
        # policy. It is no longer a second live-learning engine.
        "legacy_factor_bias": {},
        "history": [],
        "kill_switch": False,
        "kill_switch_reason": "",
        "kill_switch_at": None,
        "last_monthly_sample_count": 0,
        "last_monthly_period": "",
        "monthly_cycle_base_samples": 0,
        "monthly_cycle_total": 0,
        "monthly_cycle_initialized": False,
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
        # Older policy migration: start the first cumulative monthly cycle
        # from the existing Adaptive sample cursor, without deleting history.
        data.setdefault("monthly_cycle_base_samples", int(data.get("samples_at_update", 0) or 0))
        data.setdefault("monthly_cycle_total", 0)
        data.setdefault("monthly_cycle_initialized", False)
        data.setdefault("weights", {})
        for k, v in default["weights"].items():
            data["weights"].setdefault(k, v)
        data.setdefault("entry_limits", {})
        for k, v in default["entry_limits"].items():
            data["entry_limits"].setdefault(k, v)
        data.setdefault("strategy_stats", {})
        data.setdefault("strategy_weights", {})
        data.setdefault("strategy_weights_active", False)
        data.setdefault("strategy_weights_generation", 0)
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
        data.setdefault("legacy_factor_bias", {})
        data.setdefault("kill_switch", False)
        data.setdefault("kill_switch_reason", "")
        data.setdefault("kill_switch_at", None)
        data.setdefault("last_monthly_sample_count", 0)
        data.setdefault("last_monthly_period", "")
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
        if p.get("kill_switch"):
            return 0.0
        vals = []
        if p.get("approved"):
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

        # Preserve the useful semantics of the original _learning_adjustment:
        # factor edge vs overall baseline + primary-strategy edge. These values
        # are learned in shadow/OOS and only affect live scoring after approval.
        if p.get("approved"):
            fb = p.get("legacy_factor_bias", {}) or {}
            fvals = [float(fb[f]) for f in factors if f in fb]
            if fvals:
                adjustment += sum(fvals) / len(fvals)

        return max(-5.0, min(5.0, adjustment))
    except Exception:
        return 0.0


def _rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(r.get("status") == "tp1" for r in rows) / len(rows)


def _strategy_weight_defaults() -> dict:
    """Baseline weights for structural/Core components only.

    Confirmation components are scored separately in the fixed 30% Confirmation
    block, so they are intentionally not part of the adaptive Core weights.
    """
    return {
        "اختراق مؤكد": {"breakout_quality": 1.0},
        "اختراق نطاق الافتتاح": {"orb_quality": 1.0},
        "إعادة اختبار": {"prior_break": .30, "near_level": .40, "reclaim": .30},
        "ارتداد VWAP": {"touch": .50, "reclaim": .50},
        "ارتداد EMA20": {"touch": .50, "reclaim": .50},
        "سحب سيولة": {"sweep": 1.0},
        "سحب سيولة مع Displacement": {"sweep": .35, "displacement": .65},
        "ضغط ثم انفجار": {"match": 1.0},
        "استمرار الزخم": {"momentum": 1.0},
        "علم صاعد": {"impulse": .50, "flag": .50},
        "استعادة مستوى": {"match": .60, "reclaim": .40},
        "دخول بعد Opening Drive": {"drive": .60, "pullback": .40},
        "استعادة قمة اليوم": {"match": .60, "reclaim": .40},
        "استعادة بعد فشل ORB": {"failed_reclaim": .35, "orb_quality": .25, "reclaim": .40},
        "استمرار ABC": {"a": .30, "b": .30, "c_break": .40},
        "دخول مبكر": {"early_range": .40, "near_resistance": .30, "holding": .30},
    }


def _strategy_weights_for(policy: dict, name: str) -> dict:
    base = _strategy_weight_defaults().get(name, {})
    if not base:
        return {}
    raw = (policy.get("strategy_weights", {}) or {}).get(name, {})
    if not policy.get("strategy_weights_active") or not isinstance(raw, dict):
        return dict(base)
    vals = {k: float(raw.get(k, v)) for k, v in base.items()}
    total = sum(max(0.0, v) for v in vals.values())
    if total <= 0:
        return dict(base)
    return {k: max(0.0, v) / total for k, v in vals.items()}


def _weighted_strategy_score(components: dict[str, float], weights: dict[str, float]) -> float:
    if not components or not weights:
        return 0.0
    return sum(float(components.get(k, 0.0)) * float(w) for k, w in weights.items())


def _learn_strategy_weights(train: list[dict], current: dict) -> dict:
    """Learn component weights per strategy; only produces a candidate policy."""
    defaults = _strategy_weight_defaults()
    learned = json.loads(json.dumps(current.get("strategy_weights", {}) or {}))
    for et, base in defaults.items():
        rows = [r for r in train if str(r.get("entry_type") or "") == et and isinstance((r.get("strategy_component_scores") or {}).get(et), dict)]
        if len(rows) < STRATEGY_WEIGHT_MIN_SAMPLES:
            continue
        wins = [r for r in rows if r.get("status") == "tp1"]
        losses = [r for r in rows if r.get("status") in {"stop", "timeout"}]
        if len(wins) < 5 or len(losses) < 5:
            continue
        neww = {k: float((learned.get(et) or {}).get(k, v)) for k, v in base.items()}
        for k, bw in base.items():
            wm = sum(float((r.get("strategy_component_scores") or {}).get(et, {}).get(k, 0.0)) for r in wins) / len(wins)
            lm = sum(float((r.get("strategy_component_scores") or {}).get(et, {}).get(k, 0.0)) for r in losses) / len(losses)
            if wm - lm >= 8.0:
                neww[k] += STRATEGY_WEIGHT_STEP
            elif wm - lm <= -8.0:
                neww[k] -= STRATEGY_WEIGHT_STEP
            neww[k] = max(bw * STRATEGY_WEIGHT_MIN_FACTOR, min(bw * STRATEGY_WEIGHT_MAX_FACTOR, neww[k]))
        # Normalize while respecting broad safety bounds.
        total = sum(neww.values()) or 1.0
        neww = {k: v / total for k, v in neww.items()}
        learned[et] = neww
    return learned


def _strategy_weight_oos_quality(rows: list[dict], policy: dict) -> tuple[float, int]:
    scored = []
    for r in rows:
        et = str(r.get("entry_type") or "")
        comps = (r.get("strategy_component_scores") or {}).get(et)
        if not isinstance(comps, dict):
            continue
        score = _weighted_strategy_score(comps, _strategy_weights_for(policy, et))
        scored.append((score, r.get("status") == "tp1"))
    if len(scored) < 10:
        return 0.0, len(scored)
    wins = [s for s, w in scored if w]
    losses = [s for s, w in scored if not w]
    if not wins or not losses:
        return 0.0, len(scored)
    return (sum(wins) / len(wins)) - (sum(losses) / len(losses)), len(scored)


def _strategy_stats(rows: list[dict]) -> dict:
    """إحصاءات التعلم حسب الاستراتيجية الأساسية Primary فقط.

    ``matched_entry_types`` يبقى محفوظًا للبحث والتحليل متعدد الاستراتيجيات،
    لكن لا نكرر الصفقة نفسها داخل تعلم الأوزان؛ وإلا قد تتضخم عينة
    الاستراتيجية لمجرد أن الإشارة طابقت عدة setups متداخلة.
    """
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
        subset = [r for r in recent if str(r.get("entry_type") or "") == et]
        if len(subset) < STRATEGY_MIN_SAMPLES:
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
    m15_state: str,
    chop: bool,
    market_ok: bool,
    atr_pct: float,
    news_state: str,
    market_condition: str = "غير مؤكد",
) -> str:
    if news_state == "positive_strong":
        return "news_momentum"
    if chop:
        return "chop"
    if market_condition == "إيجابي_تحت_VWAP":
        return "market_positive_below_vwap"
    if not market_ok:
        return "market_weak"
    if trend_up and m15_state == "داعم" and atr_pct <= 4.5:
        return "trend_clean"
    if trend_up and m15_state != "معاكس":
        return "trend_mixed"
    if atr_pct > 6.0:
        return "high_volatility"
    return "neutral"


def _interaction_keys(
    entry_type: str,
    market_regime: str,
    m15_state: str,
    volume_ratio: float,
    breakout_quality: float,
) -> list[str]:
    keys = [f"type:{entry_type}", f"regime:{market_regime}", f"m15:{m15_state}"]
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
    if m15_state == "داعم" and volume_ratio >= 1.2 and market_regime == "trend_clean":
        keys.append("combo:m15+volume+trend")
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

    # Legacy learning semantics, consolidated into the single Adaptive policy.
    if policy.get("approved"):
        fb = policy.get("legacy_factor_bias", {}) or {}
        fvals = [float(fb[f]) for f in factors if f in fb]
        if fvals:
            score += sum(fvals) / len(fvals)

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
    Unified Adaptive optimization entry point.
    Reviews the full accumulated learning set once per 100 newly completed
    trades and returns a Telegram-friendly summary. It never changes core rules.
    """
    result = adaptive_retrain_if_ready(force_monthly=True)
    policy = _load_adaptive_policy()

    if result.get("status") in {"waiting", "waiting_oos", "waiting_cycle_samples", "unchanged"}:
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
            if len(subset) < REGIME_STRATEGY_MIN_SAMPLES:
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
            if len(subset) < REGIME_STRATEGY_MIN_SAMPLES:
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
    if policy.get("kill_switch"):
        return float(structural_stop), float(sig_price), 1.20
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


def _adaptive_kill_switch_state(completed: list[dict], policy: dict) -> tuple[bool, str]:
    """حماية تكيفية: تعطل طبقات التعلم فقط عند تدهور واضح، ولا توقف المحرك الأساسي."""
    if len(completed) < KILL_SWITCH_LOOKBACK:
        return False, ""
    recent = completed[-KILL_SWITCH_LOOKBACK:]
    recent_rate = _rate(recent)
    hist = completed[:-KILL_SWITCH_LOOKBACK]
    if len(hist) >= 30:
        baseline = _rate(hist[-60:])
    else:
        baseline = float(policy.get("validation_old_rate", recent_rate) or recent_rate)
    drop = baseline - recent_rate
    if drop >= KILL_SWITCH_MIN_DROP:
        return True, f"تراجع معدل النجاح {drop*100:.1f}% عن خط الأساس"
    return False, ""


def _monthly_completed_rows(completed: list[dict], period: str | None = None) -> list[dict]:
    """الصفقات المكتملة في شهر التقويم الحالي/المحدد فقط."""
    period = period or _monthly_period_key()
    rows = []
    for r in completed:
        stamp = str(r.get("closed_at") or r.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if dt.strftime("%Y-%m") == period:
                rows.append(r)
        except Exception:
            continue
    return rows


def _monthly_period_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _signal_duplicate_recent(sig, records: list[dict]) -> bool:
    """يمنع إعادة نفس الهيكل لنفس السهم خلال نافذة زمنية محددة."""
    now = datetime.now(timezone.utc)
    symbol = str(getattr(sig, "symbol", ""))
    entry_type = str(getattr(sig, "entry_type", ""))
    regime = str(getattr(sig, "market_regime", "neutral"))
    for r in reversed(records):
        if r.get("record_type") != "signal" or r.get("symbol") != symbol:
            continue
        if str(r.get("entry_type") or "") != entry_type or str(r.get("market_regime") or "neutral") != regime:
            continue
        if r.get("status") not in {"pending", "completed"}:
            continue
        try:
            created = datetime.fromisoformat(str(r.get("created_at")).replace("Z", "+00:00"))
            if (now - created).total_seconds() <= SIGNAL_DEDUP_MINUTES * 60:
                return True
        except Exception:
            continue
    return False


def adaptive_retrain_if_ready(force_monthly: bool = False) -> dict:
    """يشغّل دورة التعلم الذاتي ويصدر نتيجة قابلة للإرسال إلى Telegram."""
    completed = _recent_learning_rows()
    policy = _load_adaptive_policy()

    # Unified Adaptive cycle: evaluate once after every 100 newly completed
    # trades. Calendar months are NOT a learning boundary. The full accumulated
    # history remains available, so small-sample strategies carry forward and
    # can qualify in a later cycle without losing their observations.
    cycle_anchor = int(policy.get("adaptive_cycle_anchor_samples", policy.get("samples_at_update", 0)) or 0)
    if cycle_anchor > len(completed):
        cycle_anchor = len(completed)
    cycle_total = max(0, len(completed) - cycle_anchor)
    policy["adaptive_cycle_total"] = cycle_total
    if cycle_total < MONTHLY_MIN_SAMPLES:
        policy["last_monthly_sample_count"] = cycle_total
        _save_adaptive_policy(policy)
        return {"status": "waiting_cycle_samples", "samples": len(completed), "required": MONTHLY_MIN_SAMPLES, "cycle_total": cycle_total}

    kill, kill_reason = _adaptive_kill_switch_state(completed, policy)
    if kill:
        policy["kill_switch"] = True
        policy["kill_switch_reason"] = kill_reason
        policy["kill_switch_at"] = datetime.now(timezone.utc).isoformat()
        _save_adaptive_policy(policy)
        return {"status": "kill_switch", "samples": len(completed), "reason": kill_reason}
    elif policy.get("kill_switch"):
        policy["kill_switch"] = False
        policy["kill_switch_reason"] = ""
        _save_adaptive_policy(policy)

    if len(completed) < ADAPTIVE_MIN_SAMPLES:
        return {"status": "waiting", "samples": len(completed)}

    last_update = int(policy.get("samples_at_update", 0))
    if not force_monthly and len(completed) <= last_update:
        return {"status": "unchanged", "samples": len(completed)}

    # Every Adaptive generation uses the full accumulated dataset. This lets
    # strategies with small samples carry their observations into later cycles.
    recent = completed
    # اختبار خارج العينة: الجزء الأحدث يبقى خارج التدريب حتى لا نعتمد على نفس البيانات.
    split = max(20, int(len(recent) * 0.70))
    train = recent[:split]
    test = recent[split:]
    if len(train) < 20 or len(test) < 10:
        return {"status": "waiting_oos", "samples": len(completed)}
    baseline_rate = _rate(train)
    candidate = json.loads(json.dumps(policy))

    # Consolidate the original learning engine into Adaptive without losing its
    # information: same factor-vs-overall and primary-strategy-vs-overall idea,
    # but trained only on the in-sample partition and activated only after OOS.
    legacy_factor_bias = {}
    all_train_factors = sorted({f for r in train for f in (r.get("factors") or [])})
    for factor in all_train_factors:
        subset = [r for r in train if factor in (r.get("factors") or [])]
        if len(subset) >= 5:
            legacy_factor_bias[factor] = max(
                -LEARNING_MAX_ADJUSTMENT,
                min(LEARNING_MAX_ADJUSTMENT, (_rate(subset) - baseline_rate) * 10.0),
            )
    # Preserve the learned factor-vs-overall signal inside the single
    # Adaptive policy; it becomes live only if this candidate passes OOS.
    candidate["legacy_factor_bias"] = legacy_factor_bias
    candidate.setdefault("interaction_weights", {})
    # Always refresh per-strategy statistics so all 16 setups are observable.
    candidate["strategy_stats"] = _strategy_stats(completed)

    # Strategy-weight learning replaces the old live strategy bias. It is
    # optional and remains inactive until its own OOS test proves improvement.
    candidate["strategy_weights"] = _learn_strategy_weights(train, policy)

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
    for et in ("دخول مبكر", "إعادة اختبار", "ارتداد VWAP", "ارتداد EMA20", "سحب سيولة", "سحب سيولة مع Displacement", "ضغط ثم انفجار", "استمرار الزخم", "اختراق نطاق الافتتاح", "اختراق مؤكد", "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة اليوم", "استعادة بعد فشل ORB", "استمرار ABC"):
        subset = [r for r in train if str(r.get("entry_type") or "") == et]
        if len(subset) < STRATEGY_MIN_SAMPLES:
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

    current_sw_quality, sw_n = _strategy_weight_oos_quality(test, policy)
    candidate_sw_quality, _ = _strategy_weight_oos_quality(test, candidate)
    strategy_weights_improved = (
        sw_n >= 10 and candidate_sw_quality >= current_sw_quality + 2.0
    )
    candidate["strategy_weights_oos_quality"] = round(candidate_sw_quality, 4)
    candidate["strategy_weights_active"] = bool(strategy_weights_improved or policy.get("strategy_weights_active", False))
    candidate["strategy_weights_generation"] = (
        candidate["generation"] if strategy_weights_improved
        else int(policy.get("strategy_weights_generation", 0) or 0)
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
        "strategy_weights_oos_quality_before": round(current_sw_quality, 4),
        "strategy_weights_oos_quality_after": round(candidate_sw_quality, 4),
        "strategy_weights_improved": bool(strategy_weights_improved),
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
        candidate["last_monthly_sample_count"] = cycle_total
        candidate["adaptive_cycle_anchor_samples"] = len(completed)
        candidate["adaptive_cycle_total"] = 0
        candidate["monthly_cycle_total"] = 0
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
        # Strict all-or-nothing monthly approval: no adaptive sub-layer graduates
        # independently when the candidate as a whole fails its OOS gate.
        # Strategy-weight changes remain shadow-only unless the complete
        # candidate passes the existing global OOS gate.
        policy["samples_at_update"] = len(completed)
        # Consume this 100-trade review cycle even if the candidate is rejected;
        # the previous approved policy remains active until a later cycle wins.
        policy["adaptive_cycle_anchor_samples"] = len(completed)
        policy["adaptive_cycle_total"] = 0
        policy["last_monthly_sample_count"] = cycle_total
        policy["history"] = (policy.get("history", []) + [result])[-20:]
        _save_adaptive_policy(policy)
        result["status"] = "rejected"
        result["message"] = (
            f"🧠 مراجعة التعلم الذاتي\n"
            f"تم تحليل {len(completed)} صفقة\n"
            f"الحالي: {current_rate*100:.1f}% | الجديد: {candidate_rate*100:.1f}%\n"
            f"❌ لم يتم اعتماد أي تعديل — السياسة الحالية بقيت كما هي"
        )

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
            "entry": round(float(getattr(sig, "alert_entry_price", 0.0) or sig.price), 4),
            "analysis_price": round(float(sig.price), 4),
            "stop_loss": round(float(sig.stop_loss), 4),
            "tp1": round(float(sig.tp1), 4),
            "score": int(sig.score),
            "grade": sig.grade,
            "entry_type": sig.entry_type,
            "matched_entry_types": list(sig.matched_entry_types or []),
            "strategy_scores": dict(sig.strategy_scores or {}),
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
            "market_regime": sig.market_regime,
            "adaptive_generation": int(_load_adaptive_policy().get("generation", 0)),
            "exit_policy_active": bool(_load_adaptive_policy().get("exit_active", False)),
            "reason_engine_active": bool(
                _load_adaptive_policy().get("reason_policy", {}).get("active", False)
            ),
            "outcome_reason": _classify_trade_reason({**sig.__dict__, "status": ""}),
            "interactions": list(getattr(sig, "interaction_keys", []) or []),
            "decision_audit": {
                "matched_strategies": list(getattr(sig, "matched_entry_types", []) or []),
                "primary_strategy": str(sig.entry_type),
                "strategy_scores": dict(getattr(sig, "strategy_scores", {}) or {}),
            "strategy_component_scores": dict(getattr(sig, "strategy_component_scores", {}) or {}),
                "factors_positive": list(getattr(sig, "factor_keys", []) or []),
                "reasons": list(getattr(sig, "reasons", []) or []),
                "warnings": list(getattr(sig, "warnings", []) or []),
                "market_regime": str(getattr(sig, "market_regime", "neutral")),
                "score": int(getattr(sig, "score", 0) or 0),
                "entry": round(float(getattr(sig, "price", 0) or 0), 4),
                "stop": round(float(getattr(sig, "stop_loss", 0) or 0), 4),
                "tp1": round(float(getattr(sig, "tp1", 0) or 0), 4),
                "tp2": round(float(getattr(sig, "tp2", 0) or 0), 4),
                "tp3": round(float(getattr(sig, "tp3", 0) or 0), 4),
                "adaptive_generation": int(_load_adaptive_policy().get("generation", 0)),
            },
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
            "matched_entry_types": target.get("matched_entry_types") or [],
            "strategy_scores": target.get("strategy_scores") or {},
            "strategy_component_scores": target.get("strategy_component_scores") or {},
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


MARKET_RETRY_ATTEMPTS = 4
MARKET_RETRY_SLEEP_SECONDS = 0.25
MARKET_RELATIVE_CACHE_TTL_SECONDS = 300
_MARKET_RELATIVE_CACHE: tuple[float, float, float] | None = None


def _market_regime_from_state(market_state: str) -> str:
    state = str(market_state or "")
    if "داعمان" in state:
        return "قوي"
    if "إيجابيان تحت VWAP" in state:
        return "إيجابي_تحت_VWAP"
    if "ضعيفان" in state:
        return "ضعيف"
    if "مختلطان" in state:
        return "مختلط"
    return "غير مؤكد"


def get_intraday_market_context() -> tuple[bool, str, str]:
    """Return the exact Intraday SPY+QQQ market state used by the engine."""
    try:
        from market_data import fetch_intraday
        ok, state = _market_alignment(fetch_intraday)
        condition = _market_regime_from_state(state)
        return bool(ok), str(state), str(condition)
    except Exception as exc:
        log.warning("Intraday market context unavailable: %s", exc)
        return False, "بيانات SPY/QQQ غير متاحة", "غير مؤكد"


def _market_relative_returns(fetch_intraday) -> tuple[float, float, float]:
    import time
    global _MARKET_RELATIVE_CACHE
    now = time.time()
    if _MARKET_RELATIVE_CACHE and now - _MARKET_RELATIVE_CACHE[0] < MARKET_RELATIVE_CACHE_TTL_SECONDS:
        spy, qqq = _MARKET_RELATIVE_CACHE[1], _MARKET_RELATIVE_CACHE[2]
        return spy, qqq, (spy + qqq) / 2.0
    vals: dict[str, float] = {}
    for sym in ("SPY", "QQQ"):
        try:
            d = fetch_intraday(sym, interval="5m", period="3d")
            if d is None or len(d) < 6:
                vals[sym] = 0.0
                continue
            day = d.index[-1].date()
            cur = d[d.index.date == day]
            prev = d[d.index.date < day]
            if cur.empty or prev.empty:
                vals[sym] = 0.0
                continue
            prev_close = float(prev["Close"].iloc[-1])
            last = float(cur["Close"].iloc[-1])
            vals[sym] = (last - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        except Exception:
            vals[sym] = 0.0
    spy, qqq = float(vals.get("SPY", 0.0)), float(vals.get("QQQ", 0.0))
    _MARKET_RELATIVE_CACHE = (now, spy, qqq)
    return spy, qqq, (spy + qqq) / 2.0


def _market_alignment(fetch_intraday) -> tuple[bool, str]:
    """SPY + QQQ: جلب مستقل مع Retry وتسجيل واضح؛ الاختلاط لا يرفض السهم."""
    import time

    states = []
    for sym in ("SPY", "QQQ"):
        state = None
        for attempt in range(1, MARKET_RETRY_ATTEMPTS + 1):
            try:
                log.info("MARKET DATA | %s | attempt %d/%d", sym, attempt, MARKET_RETRY_ATTEMPTS)
                d = fetch_intraday(sym, interval="5m", period="2d")
                if d is None or len(d) < 20:
                    raise ValueError("market data unavailable/incomplete")
                day = d.index[-1].date()
                cur = d[d.index.date == day]
                if len(cur) < 6:
                    raise ValueError("market session data incomplete")
                p = float(cur["Close"].iloc[-1])
                op = float(cur["Open"].iloc[0])
                vw = float(_vwap(cur).iloc[-1])
                if p >= op and p >= vw:
                    state = "داعم"
                elif p >= op and p < vw and p >= op * (1.0 - POSITIVE_BELOW_VWAP_MAX_OPEN_GAP_PCT / 100.0):
                    state = "إيجابي_تحت_VWAP"
                else:
                    state = "ضعيف"
                log.info(
                    "MARKET DATA | %s | success | bars=%d | close=%.4f | open=%.4f | vwap=%.4f | state=%s",
                    sym, len(cur), p, op, vw, state,
                )
                break
            except Exception as exc:
                log.warning(
                    "MARKET DATA | %s | failed attempt %d/%d | %s",
                    sym, attempt, MARKET_RETRY_ATTEMPTS, str(exc),
                )
                if attempt < MARKET_RETRY_ATTEMPTS:
                    time.sleep(MARKET_RETRY_SLEEP_SECONDS)
        states.append(state)

    if states == ["داعم", "داعم"]:
        result = (True, "SPY+QQQ داعمان")
    elif states == ["إيجابي_تحت_VWAP", "إيجابي_تحت_VWAP"]:
        result = (True, "SPY+QQQ إيجابيان تحت VWAP")
    elif states == ["ضعيف", "ضعيف"]:
        result = (False, "SPY+QQQ ضعيفان")
    elif all(x is None for x in states):
        result = (False, "السوق غير مؤكد")
    elif any(x is None for x in states):
        result = (False, "بيانات السوق غير مكتملة")
    else:
        # أي مزيج بين داعم/إيجابي تحت VWAP/ضعيف يبقى سوقًا مختلطًا.
        result = (True, "SPY/QQQ مختلطان")

    log.info(
        "MARKET RESULT | SPY=%s | QQQ=%s | ok=%s | state=%s",
        states[0] if states[0] is not None else "غير متوفر",
        states[1] if states[1] is not None else "غير متوفر",
        result[0], result[1],
    )
    return result


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


_QUOTE_CACHE: dict[str, tuple[datetime, dict]] = {}
QUOTE_CACHE_SECONDS = 30
MAX_SPREAD_PCT = 0.80
HARD_MAX_SPREAD_PCT = 1.20
MIN_DOLLAR_VOLUME_3M = 10_000_000.0

def _quote_liquidity(symbol: str, price: float) -> dict:
    """لحظي: Bid/Ask من Alpaca فقط؛ لا نستخدم Yahoo كبديل للتنفيذ اللحظي."""
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


def get_live_entry_price(symbol: str) -> float:
    """يأخذ سعرًا لحظيًا واحدًا وقت الإرسال من Alpaca (متوسط Bid/Ask).
    مستقل عن سعر التحليل ولا يعيد حساب الوقف أو الأهداف.
    """
    try:
        from market_data import fetch_latest_quote, data_age_minutes
        q = fetch_latest_quote(symbol)
        bid = float(q.get("bid") or 0)
        ask = float(q.get("ask") or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0
        ts = q.get("timestamp")
        if ts:
            mid = (bid + ask) / 2.0
            qdf = pd.DataFrame({"Close": [mid]}, index=[pd.Timestamp(ts)])
            if data_age_minutes(qdf) > 2.0:
                return 0.0
        return round((bid + ask) / 2.0, 4)
    except Exception:
        return 0.0


def analyze_intraday(
    symbol: str,
    name: str = "",
    market_context: tuple[bool, str] | None = None,
    preloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> Optional[IntradaySignal]:
    from market_data import fetch_intraday
    from market_data import intraday_data_fresh

    if preloaded is not None:
        h1, m5 = preloaded
    else:
        h1 = fetch_intraday(symbol, interval="60m", period="10d")
        m5 = fetch_intraday(symbol, interval="5m", period="5d")

    ok_h1, _ = intraday_data_fresh(h1, "60m", 90)
    if h1 is None or len(h1) < 40 or not ok_h1:
        return None

    ok_m5, _ = intraday_data_fresh(m5, "5m", 12)
    if m5 is None or len(m5) < 30 or not ok_m5:
        return None

    try:
        m15 = fetch_intraday(symbol, interval="15m", period="10d")
        ok_m15, _ = intraday_data_fresh(m15, "15m", 25)
        if not ok_m15:
            m15 = None
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
    market_ok, market_state = market_context if market_context is not None else _market_alignment(fetch_intraday)
    market_condition = _market_regime_from_state(market_state)

    chop = _chop_filter(today_5, price, vwap_last)

    session_high = float(today_5["High"].max())
    drop = (session_high - price) / session_high * 100 if session_high else 0
    dump = drop >= 2.5 and change_pct <= -1.2

    h_win = h1.tail(20)
    level_high = float(h_win["High"].iloc[:-1].max()) if len(h_win) > 3 else session_high
    was_below = float(h_win["Close"].iloc[-3]) < level_high * 0.998 if len(h_win) >= 3 else False
    breakout_now = price >= level_high * 1.001 and was_below
    # A real retest requires a prior close above the broken resistance,
    # followed by a return toward that same level. A mere historical touch is
    # not enough to label the setup as Retest.
    prior_break = (
        bool((h_win["Close"].astype(float).iloc[-8:-2] >= level_high * 1.001).any())
        if len(h_win) >= 8 else False
    )
    near_level = abs(price - level_high) / max(price, 1e-9) * 100 <= 0.7
    ext_tmp = (price - e20) / e20 * 100 if e20 else 0.0

    market_rel_spy = market_rel_qqq = market_rel_avg = 0.0
    stock_relative_strength = 0.0
    relative_strength_ok = False
    if market_condition == "ضعيف":
        market_rel_spy, market_rel_qqq, market_rel_avg = _market_relative_returns(fetch_intraday)
        stock_relative_strength = change_pct - market_rel_avg
        relative_strength_ok = bool(
            change_pct >= 0.75 and stock_relative_strength >= 1.25
        )

    # نظام السوق هو الذي يحدد بوابة الدخول: قوي/مختلط/ضعيف.
    mixed_market_ok = bool(
        market_condition == "مختلط"
        and trend_up and live_ok and above_vwap and above_open
        and m15_state == "داعم" and vol_session_ratio >= 1.0
        and not dump and not chop and ext_tmp <= 4.0
    )

    # بوابة اكتشاف الإشارات: تسمح ببناء/تقييم setup في الأنظمة الثلاثة.
    # لا تعني القبول النهائي؛ السوق المختلط/الضعيف سيُحسم لاحقاً بعد اكتمال
    # الدرجة وشروط الجودة والاستثناء الخاص بالسهم القوي.
    setup_market_permission = market_condition in {"قوي", "مختلط", "إيجابي_تحت_VWAP", "ضعيف"}

    # بوابة السوق التمهيدية للـScore: السوق القوي مسموح مباشرة، والمختلط فقط
    # إذا اجتاز شروطه التمهيدية. السوق الضعيف لا يأخذ مكافأة السوق قبل حسم
    # استثناء السهم القوي بعد اكتمال الدرجة.
    positive_below_vwap_ok = bool(
        market_condition == "إيجابي_تحت_VWAP"
        and trend_up
        and live_ok
        and above_vwap
        and above_open
        and m15_state != "معاكس"
        and vol_session_ratio >= 0.90
        and not dump
        and not chop
        and ext_tmp <= 4.5
    )

    market_permission = bool(
        market_condition == "قوي"
        or mixed_market_ok
        or positive_below_vwap_ok
    )

    failed = (
        float(today_5["High"].max()) >= level_high * 1.001
        and price < level_high * 0.997
        and not above_vwap
    ) or (dump and not above_vwap)

    # Strategy Core: structural setup only. Generic confirmations are evaluated
    # separately below and contribute to Strategy Score instead of preventing a match.
    retest = prior_break and near_level and price >= level_high * 0.997

    recent4 = today_5.tail(4)
    vwap_touch = False
    try:
        vwap_touch = bool((recent4["Low"].astype(float) <= vwap_last * 1.006).any())
    except Exception:
        vwap_touch = False
    vwap_bounce = vwap_touch

    ema_touch = False
    try:
        ema_touch = bool((recent4["Low"].astype(float) <= e5 * 1.006).any())
    except Exception:
        ema_touch = False
    ema_pullback = ema_touch and price >= e5 * 1.001

    support_level = 0.0
    liquidity_sweep = False
    try:
        support_window = today_5["Low"].astype(float).iloc[-12:-2]
        if len(support_window) >= 5:
            support_level = float(support_window.min())
            recent3 = today_5.tail(3)
            swept = (recent3["Low"].astype(float) < support_level * 0.998).any()
            reclaimed = price >= support_level * 1.002
            liquidity_sweep = bool(swept and reclaimed)
    except Exception:
        liquidity_sweep = False

    liquidity_displacement = False
    try:
        if len(today_5) >= 8 and support_level > 0:
            cur = today_5.iloc[-1]
            prev3 = today_5.iloc[-4:-1]
            cur_o, cur_c = float(cur["Open"]), float(cur["Close"])
            cur_h, cur_l = float(cur["High"]), float(cur["Low"])
            cur_range = max(cur_h - cur_l, price * 0.0001)
            cur_body = abs(cur_c - cur_o)
            close_pos = (cur_c - cur_l) / cur_range
            prior_ranges = (prev3["High"].astype(float) - prev3["Low"].astype(float)).clip(lower=0)
            med_range = float(prior_ranges.median()) if len(prior_ranges) else 0.0
            swept = bool((today_5["Low"].astype(float).iloc[-5:-1] < support_level * 0.998).any())
            reclaimed = price >= support_level * 1.002
            displacement = bool(
                cur_c > cur_o and cur_body / cur_range >= 0.55 and close_pos >= 0.75
                and (med_range <= 0 or cur_range >= med_range * 1.35)
            )
            liquidity_displacement = bool(swept and reclaimed and displacement)
    except Exception:
        liquidity_displacement = False

    orb_high = 0.0
    orb_breakout = False
    prior_orb = False
    try:
        if len(today_5) >= 6:
            orb = today_5.iloc[:3]
            orb_high = float(orb["High"].max())
            prior_orb = float(today_5["Close"].iloc[-2]) < orb_high * 1.001
            orb_breakout = bool(price >= orb_high * 1.001 and prior_orb)
    except Exception:
        orb_breakout = False

    momentum_continuation = False
    try:
        tail5 = today_5.tail(4)
        closes = tail5["Close"].astype(float)
        opens = tail5["Open"].astype(float)
        highs = tail5["High"].astype(float)
        lows = tail5["Low"].astype(float)
        if len(tail5) >= 4:
            rising = bool(closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3])
            green_now = bool(closes.iloc[-1] >= opens.iloc[-1])
            body_now = abs(closes.iloc[-1] - opens.iloc[-1])
            range_now = max(highs.iloc[-1] - lows.iloc[-1], price * 0.0001)
            momentum_close_pos = (closes.iloc[-1] - lows.iloc[-1]) / range_now
            momentum_body_ok = body_now / range_now >= 0.45
            momentum_close_ok = momentum_close_pos >= 0.65
            prior_move = (closes.iloc[-2] - closes.iloc[-4]) / max(closes.iloc[-4], 1e-9) * 100
            momentum_continuation = bool(rising and prior_move >= 0.35 and mom > 0.08)
    except Exception:
        momentum_continuation = False

    compression_expansion = False
    try:
        if len(today_5) >= 10:
            prev = today_5.iloc[-9:-1]
            cur = today_5.iloc[-1]
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
                compression and expansion
            )
    except Exception:
        compression_expansion = False

    prev_day_high = 0.0
    prev_day_low = 0.0
    prev_close_level = 0.0
    try:
        if len(h1) >= 30:
            prior = h1.iloc[:-20]
            if len(prior) >= 5:
                prev_day_high = float(prior["High"].max())
                prev_day_low = float(prior["Low"].min())
                prev_close_level = float(prior["Close"].iloc[-1])
    except Exception:
        pass

    key_level_near = any(
        lvl > 0 and abs(price - lvl) / max(price, 1e-9) * 100 <= 0.45
        for lvl in (prev_day_high, prev_day_low, prev_close_level, orb_high, level_high)
    )

    bull_flag = False
    try:
        if len(today_5) >= 12:
            impulse = today_5.iloc[-12:-6]
            flag = today_5.iloc[-6:-1]
            impulse_open = float(impulse["Open"].iloc[0])
            impulse_high = float(impulse["High"].max())
            impulse_gain = (impulse_high - impulse_open) / max(impulse_open, 1e-9) * 100
            flag_high = float(flag["High"].max())
            flag_low = float(flag["Low"].min())
            flag_range = (flag_high - flag_low) / max(flag_high, 1e-9) * 100
            breakout_flag = price >= flag_high * 1.001
            bull_flag = bool(impulse_gain >= 1.0 and flag_range <= 2.0 and breakout_flag)
    except Exception:
        bull_flag = False

    resistance_reclaim = False
    reclaim_level = 0.0
    try:
        if len(today_5) >= 10:
            prior = today_5.iloc[-10:-2]
            reclaim_level = float(prior["High"].quantile(0.80))
            prev_close = float(today_5["Close"].iloc[-2])
            resistance_was_lost = prev_close < reclaim_level * 0.999
            reclaimed = price >= reclaim_level * 1.001
            touches = int((prior["High"] >= reclaim_level * 0.995).sum())
            resistance_reclaim = bool(touches >= 2 and resistance_was_lost and reclaimed)
    except Exception:
        resistance_reclaim = False

    opening_drive_pullback = False
    drive_level = 0.0
    try:
        if len(today_5) >= 8:
            first6 = today_5.iloc[:6]
            later = today_5.iloc[6:]
            drive_open = float(first6["Open"].iloc[0])
            drive_high = float(first6["High"].max())
            drive_return = (drive_high - drive_open) / max(drive_open, 1e-9) * 100
            drive_level = drive_high
            if not later.empty and drive_return >= 1.0:
                recent = today_5.tail(3)
                recent_low = float(recent["Low"].min())
                pullback_from_high = (drive_high - recent_low) / max(drive_high, 1e-9) * 100
                reclaim_drive = price >= drive_high * 0.999
                controlled_pullback = 0.25 <= pullback_from_high <= 2.5
                opening_drive_pullback = bool(drive_return >= 1.0 and controlled_pullback and reclaim_drive)
    except Exception:
        opening_drive_pullback = False

    hod_reclaim = False
    hod_level = 0.0
    try:
        if len(today_5) >= 8:
            prior = today_5.iloc[:-2]
            hod_level = float(prior["High"].max())
            if hod_level > 0:
                pullback_below_hod = float(today_5["Close"].iloc[-2]) < hod_level * 0.999
                reclaimed_hod = price >= hod_level * 1.001
                had_hod = float(prior["High"].max()) >= hod_level * 0.999
                hod_reclaim = bool(had_hod and pullback_below_hod and reclaimed_hod)
    except Exception:
        hod_reclaim = False

    orb_failed_reclaim = False
    try:
        if len(today_5) >= 8 and orb_high > 0:
            post_orb = today_5.iloc[3:-1]
            broke = bool((post_orb["High"].astype(float) >= orb_high * 1.002).any())
            failure = bool((post_orb["Close"].astype(float) <= orb_high * 0.998).any())
            reclaim = price >= orb_high * 1.001
            orb_failed_reclaim = bool(broke and failure and reclaim)
    except Exception:
        orb_failed_reclaim = False

    abc_continuation = False
    try:
        if len(today_5) >= 12:
            a = today_5.iloc[-12:-8]
            b = today_5.iloc[-8:-4]
            c = today_5.iloc[-4:]
            a_open = float(a["Open"].iloc[0])
            a_high = float(a["High"].max())
            a_gain = (a_high - a_open) / max(a_open, 1e-9) * 100
            b_low = float(b["Low"].min())
            b_retrace = (a_high - b_low) / max(a_high - a_open, price * 0.001) * 100
            c_high = float(c["High"].max())
            c_last_green = float(c["Close"].iloc[-1]) >= float(c["Open"].iloc[-1])
            c_break = c_high >= a_high * 0.999
            abc_continuation = bool(a_gain >= 0.70 and 20.0 <= b_retrace <= 65.0
                                    and c_break and price >= a_high * 0.999)
    except Exception:
        abc_continuation = False

    # Early Entry core: pre-breakout compression/holding near meaningful resistance.
    early = False
    try:
        recent3 = today_5.tail(3)
        early_range = (float(recent3["High"].max()) - float(recent3["Low"].min())) / max(price, 1e-9) * 100
        early_near_resistance = level_high > 0 and abs(price - level_high) / max(price, 1e-9) * 100 <= 1.5
        early_holding = float(recent3["Close"].iloc[-1]) >= float(recent3["Close"].iloc[0])
        early = bool(not breakout_now and early_near_resistance and early_range <= 1.5 and early_holding)
    except Exception:
        early = False

    breakout_ok, breakout_quality = _breakout_quality(today_5, level_high, price)
    orb_breakout_ok, orb_quality = _breakout_quality(today_5, orb_high, price) if orb_high > 0 else (False, 0.0)

    # Strategy-specific confirmations are defined once inside _strategy_strength.
    # They affect the 30% Confirmation block and never block Core matching.

    # Preserve the original interaction rule for Early Entry: it is only eligible
    # when no other structural setup is already present.
    if early:
        early = not any((retest, orb_breakout, breakout_now, liquidity_displacement,
                         liquidity_sweep, compression_expansion, momentum_continuation,
                         bull_flag, resistance_reclaim, orb_failed_reclaim,
                         abc_continuation, opening_drive_pullback, hod_reclaim,
                         vwap_bounce, ema_pullback))

    # Capture the complete strategy context only after strategy confirmations exist.
    strategy_ctx = locals().copy()

    matched_entry_types: list[str] = []
    if retest:
        matched_entry_types.append("إعادة اختبار")
    if orb_breakout:
        matched_entry_types.append("اختراق نطاق الافتتاح")
        breakout_quality = max(breakout_quality, orb_quality)
    if breakout_now:
        matched_entry_types.append("اختراق مؤكد")
    if liquidity_displacement:
        matched_entry_types.append("سحب سيولة مع Displacement")
    if liquidity_sweep:
        matched_entry_types.append("سحب سيولة")
    if compression_expansion:
        matched_entry_types.append("ضغط ثم انفجار")
    if momentum_continuation:
        matched_entry_types.append("استمرار الزخم")
    if bull_flag:
        matched_entry_types.append("علم صاعد")
    if resistance_reclaim:
        matched_entry_types.append("استعادة مستوى")
    if orb_failed_reclaim:
        matched_entry_types.append("استعادة بعد فشل ORB")
    if abc_continuation:
        matched_entry_types.append("استمرار ABC")
    if opening_drive_pullback:
        matched_entry_types.append("دخول بعد Opening Drive")
    if hod_reclaim:
        matched_entry_types.append("استعادة قمة اليوم")
    if vwap_bounce:
        matched_entry_types.append("ارتداد VWAP")
    if ema_pullback:
        matched_entry_types.append("ارتداد EMA20")
    if early:
        matched_entry_types.append("دخول مبكر")

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
    vwap_h1_confluence = bool(
        trend_up and above_vwap and e20 > e50
        and m15_state == "داعم" and not failed
    )

    # Early Entry Core was already defined above; do not redefine it here with
    # confirmation/safety conditions, otherwise those conditions would become
    # hidden Core gates again.

    # قوة الاستراتيجية: كل تطابق يحصل على تقييم مستقل من جودة setup الحالية.
    # هذا التقييم لا يستبدل Score النهائي؛ وظيفته اختيار أقوى استراتيجية
    # عندما تتطابق عدة استراتيجيات على السهم نفسه.
    strategy_scores: dict[str, float] = {}
    policy_for_strategy = _load_adaptive_policy()
    strategy_stats = policy_for_strategy.get("strategy_stats", {})

    def _competition_fail_reasons(name: str) -> list[str]:
        """Diagnostic-only blockers for the Competition Audit.

        Reads only values already calculated by the analyzer. It never changes
        matching, scoring, ranking, alerts, exits, filters, or learning.
        """
        c = strategy_ctx
        out: list[str] = []

        def v(key, default=None):
            return c.get(key, default)

        def add(label: str, ok: bool):
            if not bool(ok):
                out.append(label)

        def common(*, mom_min=None, vol_min=None, ext_max=None,
                   trend=True, vwap=True, opening=False, green=True,
                   market=True, state=True, no_failed=True):
            if trend:
                add("الاتجاه ليس صاعدًا", v("trend_up", False))
            if vwap:
                add("السعر ليس فوق VWAP", v("above_vwap", False))
            if opening:
                add("السعر ليس فوق الافتتاح", v("above_open", False))
            if green:
                add("آخر شمعة ليست خضراء", v("last_green", False))
            if market:
                add("صلاحية السوق/الإعداد غير متحققة", v("setup_market_permission", v("market_ok", False)))
            if state:
                add("حالة الإطار معاكسة", str(v("m15_state", v("h4_state", "محايد"))) != "معاكس")
            if no_failed:
                add("يوجد failed/rejection", not v("failed", False))
            if mom_min is not None:
                try: add(f"الزخم <= {mom_min:.2f}", float(v("mom", 0.0) or 0.0) > mom_min)
                except Exception: add(f"الزخم <= {mom_min:.2f}", False)
            if vol_min is not None:
                try: add(f"الحجم أقل من {vol_min:.2f}x", float(v("vol_session_ratio", v("vol_ratio", 0.0)) or 0.0) >= vol_min)
                except Exception: add(f"الحجم أقل من {vol_min:.2f}x", False)
            if ext_max is not None:
                try: add(f"الامتداد أكبر من {ext_max:.1f}%", float(v("ext_tmp", 999.0) or 999.0) <= ext_max)
                except Exception: add(f"الامتداد أكبر من {ext_max:.1f}%", False)

        # Exact/near-exact gates for the simpler setups.
        if name == "إعادة اختبار":
            add("لا يوجد اختراق سابق للمستوى", v("prior_break", False))
            add("السعر ليس قريبًا من المستوى", v("near_level", False))
            try:
                p, lvl = float(v("price", 0.0) or 0.0), float(v("level_high", 0.0) or 0.0)
                add("لم تتم استعادة 99.7% من المستوى", lvl > 0 and p >= lvl * 0.997)
            except Exception: add("تعذر فحص استعادة المستوى", False)
            common(vwap=True, opening=False, green=False, market=False, state=False, no_failed=True)

        elif name == "اختراق نطاق الافتتاح":
            try:
                p, lvl = float(v("price", 0.0) or 0.0), float(v("orb_high", 0.0) or 0.0)
                add("ORB غير صالح", lvl > 0)
                add("لم يحدث اختراق ORB >= 0.1%", lvl > 0 and p >= lvl * 1.001)
            except Exception: add("تعذر فحص ORB", False)
            add("الإغلاق السابق ليس تحت ORB", v("prior_orb", False))
            common(mom_min=None, vol_min=1.0, vwap=True, opening=False, green=True, market=False, state=True, no_failed=True)

        elif name == "اختراق مؤكد":
            add("لا يوجد breakout_now", v("breakout_now", False))
            add("جودة الاختراق غير كافية", v("breakout_ok", False))
            try: add("الحجم أقل من 1.00x", float(v("vol_session_ratio", v("vol_ratio", 0.0)) or 0.0) >= 1.0)
            except Exception: add("تعذر فحص الحجم", False)

        elif name == "ارتداد VWAP":
            add("السعر ليس فوق VWAP", v("above_vwap", False))
            add("لم يحدث لمس VWAP", v("vwap_touch", False))
            add("آخر شمعة ليست خضراء", v("last_green", False))
            common(mom_min=0.05, vol_min=1.0, trend=True, vwap=False, opening=False, green=False, market=False, state=True, no_failed=True)

        elif name == "ارتداد EMA20":
            add("الاتجاه ليس صاعدًا", v("trend_up", False))
            add("لم يحدث لمس EMA20", v("ema_touch", False))
            try:
                p, ema = float(v("price", 0.0) or 0.0), float(v("e5", v("e20", 0.0)) or 0.0)
                add("لم تتم استعادة EMA20", ema > 0 and p >= ema * 1.001)
            except Exception: add("تعذر فحص استعادة EMA20", False)
            common(mom_min=0.05, vol_min=1.0, trend=False, vwap=False, opening=False, green=True, market=False, state=True, no_failed=True)

        elif name == "دخول مبكر":
            add("الاتجاه ليس صاعدًا", v("trend_up", False))
            add("السعر ليس فوق VWAP", v("above_vwap", False))
            add("السعر ليس فوق الافتتاح", v("above_open", False))
            add("يوجد breakout_now", not v("breakout_now", False))
            add("ليس قريبًا من المقاومة", v("early_near_resistance", False))
            limit = 1.5 if "m15_state" in c else 3.0
            try: add(f"نطاق الدخول المبكر أكبر من {limit:.1f}%", float(v("early_range", 999.0) or 999.0) <= limit)
            except Exception: add("تعذر فحص early_range", False)
            add("Holding غير إيجابي", v("early_holding", False))
            common(mom_min=0.03, vol_min=0.95, trend=False, vwap=False, opening=False, green=True, market=True, state=True, no_failed=True, ext_max=2.2)
            others = {
                "retest":"إعادة الاختبار", "orb_breakout":"ORB", "breakout_now":"الاختراق المؤكد",
                "liquidity_displacement":"سحب السيولة + Displacement", "liquidity_sweep":"سحب السيولة",
                "compression_expansion":"ضغط ثم انفجار", "momentum_continuation":"استمرار الزخم",
                "bull_flag":"العلم الصاعد", "resistance_reclaim":"استعادة مستوى",
                "orb_failed_reclaim":"استعادة بعد فشل ORB", "abc_continuation":"ABC",
                "opening_drive_pullback":"Opening Drive", "hod_reclaim":"استعادة القمة",
                "vwap_bounce":"ارتداد VWAP", "ema_pullback":"ارتداد EMA20"
            }
            for key, label in others.items():
                add(f"يوجد trigger لـ {label}", not v(key, False))

        else:
            # Composite setups: report the structural trigger plus the common
            # confirmation gates. This is deliberately conservative: we do not
            # invent a sub-blocker when the original analyzer did not expose it.
            trigger = {
                "سحب سيولة":"liquidity_sweep",
                "سحب سيولة مع Displacement":"liquidity_displacement",
                "ضغط ثم انفجار":"compression_expansion",
                "استمرار الزخم":"momentum_continuation",
                "علم صاعد":"bull_flag",
                "استعادة مستوى":"resistance_reclaim",
                "دخول بعد Opening Drive":"opening_drive_pullback",
                "استعادة قمة اليوم":"hod_reclaim",
                                "استعادة بعد فشل ORB":"orb_failed_reclaim",
                "استمرار ABC":"abc_continuation",
            }.get(name)
            if trigger:
                add("البنية/الـtrigger الرئيسي غير مكتمل", v(trigger, False))

            if name == "سحب سيولة":
                common(mom_min=0.05, vol_min=1.0, opening=False, ext_max=None)
            elif name == "سحب سيولة مع Displacement":
                common(mom_min=0.08, vol_min=1.25, opening=True, ext_max=6.0 if "h4_state" in c else 3.5)
            elif name == "ضغط ثم انفجار":
                common(mom_min=0.05, vol_min=1.20, opening=True, ext_max=4.0 if "m15_state" in c else 6.0)
            elif name == "استمرار الزخم":
                common(mom_min=0.08, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
                add("يوجد breakout_now", not v("breakout_now", False))
            elif name == "علم صاعد":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
                add("يوجد ORB breakout", not v("orb_breakout", False))
            elif name == "استعادة مستوى":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
            elif name == "دخول بعد Opening Drive":
                common(mom_min=0.05 if "m15_state" in c else 0.20, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
                add("يوجد breakout_now", not v("breakout_now", False))
                add("يوجد ORB breakout", not v("orb_breakout", False))
            elif name in {"استعادة قمة اليوم"}:
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
            elif name == "استعادة بعد فشل ORB":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
                add("يوجد ORB breakout حالي", not v("orb_breakout", False))
            elif name == "استمرار ABC":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)

        if not out:
            out.append("لم يكتمل trigger الاستراتيجية رغم عدم توفر blocker أدق")
        # Keep the audit readable; the caller already limits the displayed list.
        return out

    # Storage is initialized before the diagnostic-only zero-match audit so
    # _strategy_strength() can safely run for all 16 strategies.
    strategy_component_scores: dict[str, dict[str, float]] = {}

    def _strategy_strength(name: str) -> float:
        """Independent 100-point setup-quality score; stock quality is scored separately."""
        def clip(x, lo=0.0, hi=100.0):
            return max(lo, min(hi, float(x)))

        c = strategy_ctx
        price_v = float(c.get("price", 0.0) or 0.0)
        vr = float(c.get("vol_session_ratio", c.get("vol_ratio", 1.0)) or 1.0)
        green = bool(c.get("last_green", False))
        mom = float(c.get("mom", 0.0) or 0.0)
        mstate = str(c.get("m15_state", c.get("h4_state", "محايد")))

        if name == "اختراق مؤكد":
            bq = clip(c.get("breakout_quality", 0.0))
            vol = clip((vr - 0.75) / 0.75 * 100)
            q = 0.55*bq + 0.25*vol + 0.20*(100 if green else 0)
        elif name == "اختراق نطاق الافتتاح":
            oq = clip(c.get("orb_quality", 0.0))
            vol = clip((vr - 0.75) / 0.75 * 100)
            q = 0.55*oq + 0.25*vol + 0.20*(100 if green else 0)
        elif name == "إعادة اختبار":
            prior = 100 if c.get("prior_break", False) else 0
            level = float(c.get("level_high", 0.0) or 0.0)
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 0.7
            near = clip((0.7-dist)/0.7*100)
            reclaim = clip((price_v/max(level,1e-9)-0.997)/0.004*100) if level > 0 else 0
            q = 0.25*prior + 0.35*near + 0.25*reclaim + 0.15*clip((vr-0.8)/0.6*100)
        elif name == "ارتداد VWAP":
            touch = 100 if c.get("vwap_touch", False) else 0
            vwap = float(c.get("vwap_last", 0.0) or 0.0)
            reclaim = clip((price_v/max(vwap,1e-9)-0.998)/0.004*100) if vwap > 0 else 0
            q = 0.30*touch + 0.30*reclaim + 0.20*(100 if green else 0) + 0.10*clip((mom-0.05)/0.20*100) + 0.10*clip((vr-0.9)/0.6*100)
        elif name == "ارتداد EMA20":
            touch = 100 if c.get("ema_touch", False) else 0
            ema = float(c.get("e20", 0.0) or 0.0)
            reclaim = clip((price_v/max(ema,1e-9)-1.001)/0.004*100) if ema > 0 else 0
            q = 0.30*touch + 0.30*reclaim + 0.20*(100 if green else 0) + 0.10*(100 if mstate == "داعم" else 0) + 0.10*clip((vr-0.9)/0.6*100)
        elif name == "سحب سيولة":
            sweep = 100 if c.get("liquidity_sweep", False) else 0
            q = 0.35*sweep + 0.25*clip((vr-0.9)/0.7*100) + 0.20*(100 if green else 0) + 0.20*clip((mom-0.05)/0.25*100)
        elif name == "سحب سيولة مع Displacement":
            sweep = 100 if c.get("liquidity_sweep", False) else 0
            disp = 100 if c.get("liquidity_displacement", False) else 0
            q = 0.20*sweep + 0.40*disp + 0.20*clip((vr-1.0)/0.75*100) + 0.10*clip((mom-0.08)/0.25*100) + 0.10*(100 if green else 0)
        elif name == "ضغط ثم انفجار":
            match = 100 if c.get("compression_expansion", False) else 0
            q = 0.40*match + 0.25*clip((vr-1.2)/0.8*100) + 0.20*(100 if green else 0) + 0.15*clip((mom-0.05)/0.25*100)
        elif name == "استمرار الزخم":
            q = 0.45*clip((mom-0.08)/0.50*100) + 0.30*clip((vr-1.0)/0.75*100) + 0.15*(100 if green else 0) + 0.10*(100 if mstate == "داعم" else 0)
        elif name == "علم صاعد":
            impulse = clip((float(c.get("impulse_gain",0.0) or 0.0)-1.0)/2.0*100)
            flag = clip((2.0-float(c.get("flag_range",2.0) or 2.0))/1.5*100)
            q = 0.30*impulse + 0.30*flag + 0.20*(100 if green else 0) + 0.20*clip((vr-0.9)/0.7*100)
        elif name == "استعادة مستوى":
            match = 100 if c.get("resistance_reclaim", False) else 0
            level = float(c.get("reclaim_level", 0.0) or 0.0)
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 1.0
            reclaim = clip((0.6-dist)/0.6*100)
            q = 0.35*match + 0.25*reclaim + 0.20*(100 if green else 0) + 0.20*clip((vr-0.9)/0.7*100)
        elif name == "دخول بعد Opening Drive":
            drive = clip((float(c.get("drive_return",0.0) or 0.0)-1.0)/2.0*100)
            pb = float(c.get("pullback_from_high", 99.0) or 99.0)
            pull = clip((2.5-abs(pb-1.0))/1.5*100)
            q = 0.35*drive + 0.25*pull + 0.20*(100 if green else 0) + 0.20*clip((vr-0.9)/0.7*100)
        elif name in {"استعادة قمة اليوم"}:
            match = 100 if c.get("hod_reclaim", False) else 0
            level = float(c.get("hod_level", 0.0) or 0.0)
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 1.0
            reclaim = clip((0.6-dist)/0.6*100)
            q = 0.35*match + 0.25*reclaim + 0.20*(100 if green else 0) + 0.20*clip((vr-0.9)/0.7*100)
        elif name == "استعادة بعد فشل ORB":
            failed = 100 if c.get("orb_failed_reclaim", False) else 0
            # Failure and reclaim are structural components; do not reward merely having orb_high.
            orbq = clip(float(c.get("orb_quality", 0.0) or 0.0))
            orb_high = float(c.get("orb_high", 0.0) or 0.0)
            reclaim = clip((price_v/max(orb_high,1e-9)-1.001)/0.004*100) if orb_high > 0 else 0
            q = 0.25*failed + 0.25*orbq + 0.30*reclaim + 0.10*clip((mom-0.05)/0.25*100) + 0.10*clip((vr-0.9)/0.7*100)
        elif name == "استمرار ABC":
            a = clip((float(c.get("a_gain",0.0) or 0.0)-0.70)/1.5*100)
            b = clip(100-abs(float(c.get("b_retrace",42.5) or 42.5)-42.5)/22.5*100)
            cb = 100 if c.get("c_break", False) else 0
            q = 0.25*a + 0.25*b + 0.30*cb + 0.10*clip((mom-0.05)/0.25*100) + 0.10*clip((vr-0.9)/0.7*100)
        else:  # دخول مبكر
            er = float(c.get("early_range",3.0) or 3.0)
            early_range = clip((3.0-er)/2.0*100)
            near = 100 if c.get("early_near_resistance", False) else 0
            holding = 100 if c.get("early_holding", False) else 0
            q = 0.30*early_range + 0.25*near + 0.20*holding + 0.15*clip((mom-0.05)/0.25*100) + 0.10*(100 if green else 0)

        # Raw 0-100 component values are kept separate from the weighted score.
        if name == "اختراق مؤكد":
            components = {"breakout_quality": bq, "volume": vol, "candle": 100 if green else 0}
        elif name == "اختراق نطاق الافتتاح":
            components = {"orb_quality": oq, "volume": vol, "candle": 100 if green else 0}
        elif name == "إعادة اختبار":
            components = {"prior_break": prior, "near_level": near, "reclaim": reclaim, "volume": clip((vr-0.8)/0.6*100)}
        elif name == "ارتداد VWAP":
            components = {"touch": touch, "reclaim": reclaim, "candle": 100 if green else 0, "momentum": clip((mom-0.05)/0.20*100), "volume": clip((vr-0.9)/0.6*100)}
        elif name == "ارتداد EMA20":
            components = {"touch": touch, "reclaim": reclaim, "candle": 100 if green else 0, "higher_tf": 100 if mstate == "داعم" else 0, "volume": clip((vr-0.9)/0.6*100)}
        elif name == "سحب سيولة":
            components = {"sweep": sweep, "volume": clip((vr-0.9)/0.7*100), "candle": 100 if green else 0, "momentum": clip((mom-0.05)/0.25*100)}
        elif name == "سحب سيولة مع Displacement":
            components = {"sweep": sweep, "displacement": disp, "volume": clip((vr-1.0)/0.75*100), "momentum": clip((mom-0.08)/0.25*100), "candle": 100 if green else 0}
        elif name == "ضغط ثم انفجار":
            components = {"match": match, "volume": clip((vr-1.2)/0.8*100), "candle": 100 if green else 0, "momentum": clip((mom-0.05)/0.25*100)}
        elif name == "استمرار الزخم":
            components = {"momentum": clip((mom-0.08)/0.50*100), "volume": clip((vr-1.0)/0.75*100), "candle": 100 if green else 0, "higher_tf": 100 if mstate == "داعم" else 0}
        elif name == "علم صاعد":
            components = {"impulse": impulse, "flag": flag, "candle": 100 if green else 0, "volume": clip((vr-0.9)/0.7*100)}
        elif name == "استعادة مستوى":
            components = {"match": match, "reclaim": reclaim, "candle": 100 if green else 0, "volume": clip((vr-0.9)/0.7*100)}
        elif name == "دخول بعد Opening Drive":
            components = {"drive": drive, "pullback": pull, "candle": 100 if green else 0, "volume": clip((vr-0.9)/0.7*100)}
        elif name in {"استعادة قمة اليوم"}:
            components = {"match": match, "reclaim": reclaim, "candle": 100 if green else 0, "volume": clip((vr-0.9)/0.7*100)}
        elif name == "استعادة بعد فشل ORB":
            components = {"failed_reclaim": failed, "orb_quality": orbq, "reclaim": reclaim, "momentum": clip((mom-0.05)/0.25*100), "volume": clip((vr-0.9)/0.7*100)}
        elif name == "استمرار ABC":
            components = {"a": a, "b": b, "c_break": cb, "momentum": clip((mom-0.05)/0.25*100), "volume": clip((vr-0.9)/0.7*100)}
        else:
            components = {"early_range": early_range, "near_resistance": near, "holding": holding, "momentum": clip((mom-0.05)/0.25*100), "candle": 100 if green else 0}

        # Core score uses only structural components. Generic confirmations such as
        # volume/candle/momentum/higher-TF are deliberately excluded here so they are
        # counted once, in the 30-point Confirmation block below.
        core_keys = {
            "اختراق مؤكد": {"breakout_quality"},
            "اختراق نطاق الافتتاح": {"orb_quality"},
            "إعادة اختبار": {"prior_break", "near_level", "reclaim"},
            "ارتداد VWAP": {"touch", "reclaim"},
            "ارتداد EMA20": {"touch", "reclaim"},
            "سحب سيولة": {"sweep"},
            "سحب سيولة مع Displacement": {"sweep", "displacement"},
            "ضغط ثم انفجار": {"match"},
            "استمرار الزخم": {"momentum"},
            "علم صاعد": {"impulse", "flag"},
            "استعادة مستوى": {"match", "reclaim"},
            "دخول بعد Opening Drive": {"drive", "pullback"},
            "استعادة قمة اليوم": {"match", "reclaim"},
                        "استعادة بعد فشل ORB": {"failed_reclaim", "orb_quality", "reclaim"},
            "استمرار ABC": {"a", "b", "c_break"},
            "دخول مبكر": {"early_range", "near_resistance", "holding"},
        }.get(name, set())
        all_weights = _strategy_weights_for(policy_for_strategy, name)
        core_weights = {k: w for k, w in all_weights.items() if k in core_keys}
        total_core = sum(core_weights.values()) or 1.0
        core_weights = {k: w / total_core for k, w in core_weights.items()}
        base_q = _weighted_strategy_score(components, core_weights)

        # Confirmation is a separate, explicit 30-point block. Each condition is
        # weighted by its role for that strategy; weights sum to 100% per strategy.
        confirmation_weights = {
            "اختراق مؤكد": {"volume": .60, "candle": .40},
            "اختراق نطاق الافتتاح": {"above_vwap": .30, "m15": .20, "volume": .30, "candle": .20},
            "إعادة اختبار": {"above_vwap": .45, "volume": .25, "candle": .15, "m15": .15},
            "ارتداد VWAP": {"trend": .25, "m15": .20, "candle": .15, "momentum": .20, "volume": .20},
            "ارتداد EMA20": {"m15": .20, "candle": .15, "volume": .20, "momentum": .20, "trend": .25},
            "سحب سيولة": {"m15": .20, "above_vwap": .20, "candle": .15, "momentum": .20, "volume": .25},
            "سحب سيولة مع Displacement": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "momentum": .15, "volume": .15, "candle": .10},
            "ضغط ثم انفجار": {"above_vwap": .15, "above_open": .10, "trend": .15, "m15": .10, "market": .10, "volume": .20, "candle": .10, "momentum": .10},
            "استمرار الزخم": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .10, "no_breakout": .15},
            "علم صاعد": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .10, "no_orb": .15},
            "استعادة مستوى": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .15},
            "دخول بعد Opening Drive": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .10, "no_breakout": .075, "no_orb": .075},
            "استعادة قمة اليوم": {"above_vwap": .15, "above_open": .10, "trend": .15, "m15": .10, "market": .10, "volume": .15, "candle": .15},
            "استعادة بعد فشل ORB": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .10, "no_orb": .15},
            "استمرار ABC": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .10, "momentum": .15},
            "دخول مبكر": {"trend": .15, "above_vwap": .15, "above_open": .10, "m15": .10, "market": .10, "volume": .15, "candle": .15},
        }.get(name, {})
        confirmation_values = {
            "trend": 100.0 if bool(c.get("trend_up", False)) else 0.0,
            "above_vwap": 100.0 if bool(c.get("above_vwap", False)) else 0.0,
            "above_open": 100.0 if bool(c.get("above_open", False)) else 0.0,
            "m15": 100.0 if mstate != "معاكس" else 0.0,
            "market": 100.0 if bool(c.get("setup_market_permission", False)) else 0.0,
            "volume": clip((vr - 0.90) / 0.60 * 100),
            "momentum": clip((mom - 0.05) / 0.25 * 100),
            "candle": 100.0 if green else 0.0,
            "no_breakout": 100.0 if not bool(c.get("breakout_now", False)) else 0.0,
            "no_orb": 100.0 if not bool(c.get("orb_breakout", False)) else 0.0,
        }
        confirmation_score = sum(confirmation_values[k] * w for k, w in confirmation_weights.items())
        q = 0.70 * base_q + 0.30 * confirmation_score
        strategy_component_scores[name] = dict(components)
        strategy_component_scores[name]["confirmation_score"] = confirmation_score

        # Strategy performance statistics are used by the Adaptive learner;
        # they no longer add a separate live +/-3 bias to the Strategy Score.
        return round(max(0.0, min(100.0, q)), 2)

    if not matched_entry_types:
        # DIAGNOSTIC ONLY: when all 16 canonical strategies fail, emit the same
        # full competition detail before returning. This does not change the
        # actual no-match behavior: the analyzer still returns None.
        _zero_scores: dict[str, float] = {}
        _zero_details: list[str] = []
        for _et in ENTRY_TYPES:
            try:
                _zero_scores[_et] = float(_strategy_strength(_et))
            except Exception:
                _zero_scores[_et] = 0.0
        _zero_ranked = sorted(
            ENTRY_TYPES,
            key=lambda _et: (_zero_scores.get(_et, 0.0), -ENTRY_TYPES.index(_et)),
            reverse=True,
        )
        for _rank, _et in enumerate(_zero_ranked, 1):
            _why = ";".join(_competition_fail_reasons(_et)[:4])
            _zero_details.append(
                f"{_rank}. {_et}:FAIL(score={_zero_scores.get(_et, 0.0):.1f}; blockers={_why})"
            )
        log.info(
            "INTRADAY STRATEGY COMPETITION AUDIT V2 | %s | primary=NONE | matched=0/16 | "
            "ranking=DIAGNOSTIC_ONLY",
            symbol,
        )
        for _detail in _zero_details:
            log.info(
                "INTRADAY STRATEGY COMPETITION DETAIL | %s | %s",
                symbol,
                _detail,
            )
        # DIAGNOSTIC ONLY: summarize the most frequent blockers across all 16
        # failed strategies. This does not change the no-match decision.
        from collections import Counter as _Counter
        _blocker_counts = _Counter()
        for _et in ENTRY_TYPES:
            try:
                _blocker_counts.update(_competition_fail_reasons(_et))
            except Exception:
                pass
        _top_blockers = ";".join(
            f"{_reason}={_count}" for _reason, _count in _blocker_counts.most_common(6)
        ) or "unavailable"
        log.info(
            "INTRADAY NO_SIGNAL | %s | reason=no_strategy_match | top_blockers=%s",
            symbol,
            _top_blockers,
        )
        # Preserve the original trading behavior exactly.
        return None

    def _strategy_identity(name: str) -> float:
        """Structural identity score, separate from generic market quality.

        Common filters (trend/VWAP/open/volume/momentum/extension) are deliberately
        not counted here. This score measures only the defining setup structure and
        is used as a deterministic tie-break for overlapping matches.
        """
        identity = {
            "اختراق مؤكد": (100.0 if breakout_ok else 0.0) + min(20.0, float(breakout_quality) * 0.20),
            "اختراق نطاق الافتتاح": (100.0 if orb_breakout_ok else 0.0) + min(20.0, float(orb_quality) * 0.20),
            "إعادة اختبار": 55.0 + (25.0 if prior_break else 0.0) + (20.0 if near_level else 0.0),
            "ارتداد VWAP": 100.0 if vwap_touch else 0.0,
            "ارتداد EMA20": 100.0 if ema_touch else 0.0,
            "سحب سيولة مع Displacement": 100.0 if liquidity_displacement else 0.0,
            "سحب سيولة": 100.0 if liquidity_sweep else 0.0,
            "ضغط ثم انفجار": 100.0 if compression_expansion else 0.0,
            "استمرار الزخم": 100.0 if momentum_continuation else 0.0,
            "علم صاعد": 100.0 if bull_flag else 0.0,
            "استعادة مستوى": 100.0 if resistance_reclaim else 0.0,
            "استعادة بعد فشل ORB": 100.0 if orb_failed_reclaim else 0.0,
            "استمرار ABC": 100.0 if abc_continuation else 0.0,
            "دخول بعد Opening Drive": 100.0 if opening_drive_pullback else 0.0,
            "استعادة قمة اليوم": 100.0 if hod_reclaim else 0.0,
            "دخول مبكر": 45.0,
        }
        return round(max(0.0, min(120.0, identity.get(name, 0.0))), 2)

    strategy_identity_scores = {et: _strategy_identity(et) for et in matched_entry_types}
    strategy_scores = {et: _strategy_strength(et) for et in matched_entry_types}
    entry_order = {et: i for i, et in enumerate(ENTRY_TYPES)}
    # عند تداخل الاستراتيجيات، لا نحذف أي match. إذا تعادلت القوة، نفضّل
    # الاستراتيجية ذات الهوية البنيوية الأوضح بدل أن يحسم ترتيب ENTRY_TYPES
    # الاختيار بشكل اعتباطي. هذا لا يغيّر التعلم أو عدد matches.
    strategy_specificity = {
        "سحب سيولة مع Displacement": 16,
        "استعادة بعد فشل ORB": 15,
        "دخول بعد Opening Drive": 14,
        "استمرار ABC": 13,
        "علم صاعد": 12,
        "ضغط ثم انفجار": 11,
        "اختراق نطاق الافتتاح": 10,
        "اختراق مؤكد": 9,
        "إعادة اختبار": 8,
        "سحب سيولة": 7,
        "استعادة قمة اليوم": 6,
        "استعادة مستوى": 5,
        "ارتداد VWAP": 4,
        "ارتداد EMA20": 3,
        "استمرار الزخم": 2,
        "دخول مبكر": 1,
    }
    entry_type = max(
        matched_entry_types,
        key=lambda et: (
            strategy_scores.get(et, 0.0),
            strategy_identity_scores.get(et, 0.0),
            strategy_specificity.get(et, 0),
            -entry_order.get(et, 999),
        ),
    )

    # DIAGNOSTIC ONLY: print the already-computed strategy decision immediately
    # after primary selection. This does not alter matching, scoring, ranking,
    # selection, or any trading rule.
    _audit_scores = dict(strategy_scores or {})
    _audit_scores_text = ", ".join(
        f"{k}={float(v):.1f}"
        for k, v in sorted(
            _audit_scores.items(),
            key=lambda kv: (-float(kv[1]), str(kv[0]))
        )
    )
    _audit_tiebreak_text = ", ".join(
        f"{k}:score={float(strategy_scores.get(k, 0.0)):.1f}"
        f"/identity={float(strategy_identity_scores.get(k, 0.0)):.1f}"
        f"/specificity={int(strategy_specificity.get(k, 0))}"
        f"/order={int(entry_order.get(k, 999))}"
        for k in sorted(
            matched_entry_types,
            key=lambda et: (
                -float(strategy_scores.get(et, 0.0)),
                -float(strategy_identity_scores.get(et, 0.0)),
                -int(strategy_specificity.get(et, 0)),
                int(entry_order.get(et, 999)),
            )
        )
    )

    # COMPETITION AUDIT V2 — DIAGNOSTIC ONLY.
    # This section evaluates ALL 16 strategies for observability and produces a
    # partial competition score even when a strategy did not fully match.
    # It MUST NOT change matched_entry_types, strategy_scores, entry_type,
    # alerts, exits, adaptive learning, market filters, or any trading rule.
    _competition_scores_all: dict[str, float] = {}
    _competition_details: list[str] = []
    for _et in ENTRY_TYPES:
        try:
            # Reuse the score already calculated by the real selection path
            # for matched strategies. Only unmatched strategies need a
            # diagnostic-only strength calculation here. This avoids doing
            # duplicate scoring work and does not change the trading decision.
            if _et in strategy_scores:
                _competition_scores_all[_et] = float(strategy_scores[_et])
            else:
                _competition_scores_all[_et] = float(_strategy_strength(_et))
        except Exception:
            _competition_scores_all[_et] = 0.0

    _competition_ranked = sorted(
        ENTRY_TYPES,
        key=lambda _et: (
            _competition_scores_all.get(_et, 0.0),
            strategy_identity_scores.get(_et, 0.0) if _et in matched_entry_types else 0.0,
            -entry_order.get(_et, 999),
        ),
        reverse=True,
    )

    for _rank, _et in enumerate(_competition_ranked, 1):
        _is_match = _et in matched_entry_types
        _score = _competition_scores_all.get(_et, 0.0)
        if _is_match:
            _competition_details.append(
                f"{_rank}. {_et}:MATCH(score={_score:.1f})"
            )
        else:
            _why = ";".join(_competition_fail_reasons(_et)[:4])
            _competition_details.append(
                f"{_rank}. {_et}:FAIL(score={_score:.1f}; blockers={_why})"
            )

    log.info(
        "INTRADAY STRATEGY COMPETITION AUDIT V2 | %s | primary=%s | matched=%s/16 | "
        "ranking=DIAGNOSTIC_ONLY",
        symbol,
        entry_type,
        len(matched_entry_types),
    )
    for _detail in _competition_details:
        log.info(
            "INTRADAY STRATEGY COMPETITION DETAIL | %s | %s",
            symbol,
            _detail,
        )

    log.info(
        "INTRADAY STRATEGY AUDIT | %s | primary=%s | matched=%s | scores=%s | "
        "tiebreak=%s",
        symbol,
        entry_type,
        list(matched_entry_types or []),
        _audit_scores_text or "none",
        _audit_tiebreak_text or "none",
    )

    entry_emoji = "🟡" if entry_type == "إعادة اختبار" else "🟢"

    reasons: list[str] = []
    warnings: list[str] = []
    # Live score is the selected Strategy Score (70% Core + 30% Confirmation).
    # Generic context below is diagnostic/adaptive metadata only; it must not
    # add another score layer and double-count Confirmation.
    score = float(strategy_scores.get(entry_type, 0.0))
    factors: list[str] = []

    if trend_up:
        reasons.append("اتجاه الساعة صاعد")
        factors.append("h1_trend")
    else:
        warnings.append("اتجاه الساعة غير مؤكد")

    if above_vwap:
        reasons.append("فوق VWAP اليوم")
        factors.append("vwap")
    else:
        warnings.append("تحت VWAP اليوم")

    if above_open:
        reasons.append("فوق افتتاح اليوم")
        factors.append("above_open")
    else:
        warnings.append("تحت افتتاح اليوم")

    if live_ok:
        reasons.append("تأكيد 5 دقائق")
        factors.append("m5")
    else:
        warnings.append("لا تأكيد 5د كافٍ")

    if vol_session_ok:
        reasons.append(f"حجم جلسة {vol_session_ratio:.2f}x")
        factors.append("vol_session")
    else:
        warnings.append("حجم الجلسة ضعيف نسبياً")

    if market_permission:
        if market_ok:
            factors.append("market")
            reasons.append(market_state)
        else:
            factors.append("market_override")
            reasons.append("السهم أقوى من السوق رغم ضعف/اختلاط SPY+QQQ")
    else:
        warnings.append(market_state)

    if chop:
        warnings.append("السوق متذبذب (Chop)")
        factors.append("chop")

    if breakout_now:
        if breakout_ok:
            factors.append("breakout_candle")
            reasons.append(f"قوة شمعة الاختراق {breakout_quality:.0f}/100")
        else:
            warnings.append("اختراق بدون إغلاق/متابعة كافية")
            factors.append("weak_breakout")
    else:
        breakout_quality = 0.0

    if news_state == "negative":
        if NEWS_BLOCK_NEGATIVE:
            warnings.append("خبر سلبي عالي المخاطر")
            factors.append("news_negative")
    elif news_state == "positive_strong":
        factors.append("news_momentum")
        reasons.append("خبر إيجابي جوهري — وضع NEWS MOMENTUM")
    elif news_state == "positive":
        factors.append("news_positive")

    if m15_state == "داعم":
        reasons.append("15 دقيقة داعمة")
        factors.append("m15")
    elif m15_state == "معاكس":
        warnings.append("15 دقيقة معاكسة")
    else:
        reasons.append("15 دقيقة محايدة")
        factors.append("m15_neutral")

    if 48 <= h_rsi <= 68:
        factors.append("rsi_h1")

    if dump:
        warnings.append("سقوط من قمة الجلسة")

    if entry_type == "اختراق مؤكد":
        reasons.append("اختراق مؤكد")
        factors.append("breakout")
    elif entry_type == "اختراق نطاق الافتتاح":
        reasons.append("اختراق نطاق الافتتاح ORB")
        factors.append("orb")
        factors.append("breakout")
    elif entry_type == "إعادة اختبار":
        reasons.append("إعادة اختبار مستوى")
        factors.append("retest")
    elif entry_type == "ارتداد VWAP":
        reasons.append("ارتداد واستعادة VWAP")
        factors.append("vwap_bounce")
    elif entry_type == "ارتداد EMA20":
        reasons.append("تصحيح صحي إلى EMA20")
        factors.append("ema_pullback")
    elif entry_type == "سحب سيولة مع Displacement":
        reasons.append("سحب سيولة ثم Displacement واستعادة قوية")
        factors.append("liquidity_sweep")
        factors.append("liquidity_displacement")
    elif entry_type == "سحب سيولة":
        reasons.append("سحب سيولة ثم استعادة المستوى")
        factors.append("liquidity_sweep")
    elif entry_type == "ضغط ثم انفجار":
        reasons.append("ضغط سعري ثم توسع بالحجم")
        factors.append("compression_expansion")
    elif entry_type == "استمرار الزخم":
        reasons.append("استمرار زخم بعد دفعة صاعدة")
        factors.append("momentum_continuation")
    elif entry_type == "علم صاعد":
        reasons.append("علم صاعد بعد دفعة قوية ثم استمرار")
        factors.append("bull_flag")
    elif entry_type == "استعادة مستوى":
        reasons.append("استعادة مقاومة بعد كسرها")
        factors.append("resistance_reclaim")
    elif entry_type == "استعادة بعد فشل ORB":
        reasons.append("فشل اختراق ORB ثم استعادة مؤكدة")
        factors.append("orb_failed_reclaim")
        factors.append("orb")
    elif entry_type == "استمرار ABC":
        reasons.append("بنية A/B/C: دفعة ثم تصحيح منظم ثم استمرار")
        factors.append("abc_continuation")
    elif entry_type == "دخول بعد Opening Drive":
        reasons.append("دفعة افتتاحية قوية ثم تراجع منظم واستعادة")
        factors.append("opening_drive_pullback")
    elif entry_type == "استعادة قمة اليوم":
        reasons.append("استعادة قمة اليوم بعد تراجع تحتها")
        factors.append("hod_reclaim")
    else:
        reasons.append("دخول مبكر فوق VWAP")
        factors.append("early")

    if key_level_near and "key_level" not in factors:
        factors.append("key_level")
        reasons.append("قرب مستوى سعري مهم")

    if vwap_h1_confluence:
        factors.append("vwap_h1_confluence")
        reasons.append("Confluence: VWAP + اتجاه الساعة + 15د")
    if multi_level_confluence:
        factors.append("multi_level_confluence")
        reasons.append("تجمع مستويات: " + "/".join(confluence_levels[:4]))

    ext = (price - e20) / e20 * 100 if e20 else 0
    if ext > 4.0:
        warnings.append("امتداد عن متوسط الساعة")
        factors.append("extended")

    atr = float(_atr(h1, 14).iloc[-1] or price * 0.01)
    atr_pct = atr / price * 100
    market_regime = _classify_regime(
        trend_up, m15_state, chop, market_ok, atr_pct, news_state, market_condition
    )
    interaction_keys = _interaction_keys(
        entry_type, market_regime, m15_state, vol_session_ratio, 0.0
    )
    if atr_pct > 6.0:
        warnings.append("تذبذب عالي")

    # Legacy learning remains available for historical research only.
    # The new Adaptive layer is the sole learning modifier for live scoring.
    learning_adj = 0.0
    adaptive_adj = _adaptive_score_adjustment(factors, market_regime, entry_type)
    total_learning_adj = learning_adj + adaptive_adj
    if total_learning_adj:
        score += total_learning_adj
        reasons.append(f"تعلم ذاتي {total_learning_adj:+.1f}")

    strong_alignment = (
        trend_up
        and live_ok
        and above_vwap
        and m15_state != "معاكس"
        and vol_session_ratio >= 1.0
        and not dump
        and ext <= 4.0
    )

    policy = _load_adaptive_policy()
    limits = policy.get("entry_limits", {})
    # نحفظ الدرجة قبل سقف نوع الاستراتيجية لاستخدامها في استثناء السوق.
    # سقف الاستراتيجية يبقى كما هو للـScore المعروض والترتيب.
    override_score = float(score)
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
    elif entry_type == "استعادة قمة اليوم":
        score = min(score, float(limits.get("استعادة قمة اليوم", 98.0)))
    elif entry_type == "اختراق نطاق الافتتاح":
        score = min(score, float(limits.get("اختراق نطاق الافتتاح", 99.0)))
    elif entry_type == "اختراق مؤكد":
        score = min(score, float(limits.get("اختراق مؤكد", 100.0)))
    else:
        score = min(score, 80.0)

    score_i = int(max(0, min(100, round(score))))

    # بعد اكتمال الدرجة نطبق شروط نظام السوق الفعلي.
    if market_condition == "ضعيف":
        strong_stock_market_override = bool(
            relative_strength_ok
            and strong_alignment
            and vol_session_ratio >= 1.25
            and not chop
            and ext_tmp <= 3.5
            and override_score >= 92.0
            and entry_type != "دخول مبكر"
        )
    else:
        strong_stock_market_override = False

    if market_condition == "مختلط":
        mixed_market_ok = bool(
            score_i >= 85
            and trend_up and live_ok and above_vwap and above_open
            and m15_state == "داعم" and vol_session_ratio >= 1.0
            and not dump and not chop and ext_tmp <= 4.0
        )
    else:
        mixed_market_ok = False

    if market_condition == "إيجابي_تحت_VWAP":
        positive_below_vwap_ok = bool(
            score_i >= POSITIVE_BELOW_VWAP_MIN_SCORE
            and trend_up and live_ok and above_vwap and above_open
            and m15_state != "معاكس"
            and vol_session_ratio >= 0.90
            and not dump and not chop and ext_tmp <= 4.5
        )
    else:
        positive_below_vwap_ok = False

    market_permission = bool(
        market_condition == "قوي"
        or mixed_market_ok
        or positive_below_vwap_ok
        or strong_stock_market_override
    )
    strong_for_grade = (
        score_i >= 95
        and strong_alignment
        and entry_type in {"اختراق مؤكد", "اختراق نطاق الافتتاح", "إعادة اختبار", "ارتداد VWAP", "ارتداد EMA20", "سحب سيولة", "ضغط ثم انفجار", "استمرار الزخم", "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة اليوم", "استعادة بعد فشل ORB", "استمرار ABC", "سحب سيولة مع Displacement"}
        and m15_state != "معاكس"
    )

    news_momentum_ok = True
    if news_state == "negative" and NEWS_BLOCK_NEGATIVE:
        news_momentum_ok = False
    if news_state == "positive_strong":
        news_momentum_ok = (
            change_pct >= float(policy.get("min_news_change_pct", NEWS_MOMENTUM_MIN_CHANGE))
            and vol_session_ratio >= float(policy.get("min_news_volume_ratio", NEWS_MOMENTUM_MIN_VOLUME))
            and above_vwap
            and (breakout_ok or entry_type != "دخول مبكر")
            and (breakout_quality >= 60 or entry_type != "دخول مبكر")
            and m15_state != "معاكس"
            and market_permission
        )

    quality_ok = (
        (not dump)
        and (not failed)
        and ext <= 4.5
        and atr_pct <= 6.5
        and vol_session_ratio >= float(policy.get("min_volume_ratio", 0.85))
        and not chop
        and news_momentum_ok
        and not (m15_state == "معاكس" and score_i < 92)
        and market_permission
    )

    recent_low = float(today_5["Low"].tail(12).min())

    # Structure-aware intraday stop. The stop is placed behind the structure
    # that actually justifies the entry, then constrained to a practical
    # intraday risk band of 0.60%–4.50%.
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
        stop_candidates.append(float(today_5["Low"].tail(5).min()) * 0.997)
    elif entry_type == "استعادة مستوى":
        stop_candidates.append(reclaim_level * 0.997 if reclaim_level > 0 else recent_low * 0.997)
    elif entry_type == "استعادة بعد فشل ORB":
        stop_candidates.append(orb_high * 0.997 if orb_high > 0 else recent_low * 0.997)
    elif entry_type == "استمرار ABC":
        stop_candidates.append(float(today_5["Low"].tail(4).min()) * 0.997)
    elif entry_type == "دخول بعد Opening Drive":
        stop_candidates.append(drive_level * 0.997 if drive_level > 0 else recent_low * 0.997)
    elif entry_type == "استعادة قمة اليوم":
        stop_candidates.append(hod_level * 0.997 if hod_level > 0 else recent_low * 0.997)
    elif entry_type in {"ضغط ثم انفجار", "استمرار الزخم"}:
        stop_candidates.append(float(today_5["Low"].tail(5).min()) * 0.997)

    # ATR remains the fallback/secondary safety reference.
    stop_candidates.append(price - 1.5 * atr)
    stop_candidates.append(recent_low * 0.997)

    # Choose the nearest valid structural stop below price.
    valid_stops = [x for x in stop_candidates if x > 0 and x < price]
    stop = max(valid_stops) if valid_stops else price * 0.985

    risk = price - stop
    min_risk = price * 0.006
    max_risk = price * 0.045
    if risk < min_risk:
        stop = price - min_risk
        risk = min_risk
    elif risk > max_risk:
        # Too-wide structures are rejected rather than hiding the risk.
        quality_ok = False
        warnings.append("وقف هيكلي واسع جدًا")

    resistance_tp1, resistance_source = _find_prior_resistance(today_5, h1, price)

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
        if 0.60 <= risk_pct_check <= 4.50:
            warnings.append(f"Adaptive Exit: TP1={adaptive_tp1_r:.2f}R")
        else:
            # Safety: revert to the original structural stop if adaptive scaling
            # somehow leaves the allowed intraday risk band.
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
    if reward_r + 1e-9 < float(policy.get("min_tp1_r", 1.2)):
        quality_ok = False
        warnings.append("العائد إلى TP1 ضعيف")
    if news_state == "positive_strong" and news_momentum_ok:
        reasons.append("NEWS MOMENTUM مؤكد")
    buy_low = max(stop * 1.01, min(price * 0.995, e5))
    buy_high = price * 1.004

    # فحص Spread/السيولة يُجرى في scan_intraday للمرشحين فقط، حتى لا يبطئ تحليل كل الأسهم.
    liquidity = {"ok": True, "spread_pct": 0.0, "slippage_pct": 0.0, "dollar_volume": 0.0}
    liquidity_ok = True

    interaction_keys = _interaction_keys(
        entry_type, market_regime, m15_state, vol_session_ratio, breakout_quality
    )

    if resistance_source != "هدف مخاطر 1.20R":
        reasons.append(f"TP1 مقاومة: {tp1:.2f}")
    else:
        warnings.append("لم توجد مقاومة قريبة مناسبة؛ TP1 احتياطي")

    # تشخيص فقط — لا يغيّر أي شرط تداول. يوضح بالضبط لماذا لم يتجاوز
    # المرشح Stage 2، مع فصل أسباب الجودة/السوق/الدرجة/التأكيد.
    diagnostic_reasons: list[str] = []
    if score_i < INTRADAY_MIN_SCORE:
        diagnostic_reasons.append(f"score<{INTRADAY_MIN_SCORE}")
    if not live_ok:
        diagnostic_reasons.append("live_ok=False")
    if not market_permission:
        diagnostic_reasons.append("market_block")
    if market_condition == "مختلط" and not mixed_market_ok:
        diagnostic_reasons.append("mixed_conditions")
    if market_condition == "إيجابي_تحت_VWAP" and not positive_below_vwap_ok:
        diagnostic_reasons.append("positive_below_vwap_conditions")
    if market_condition == "ضعيف":
        if not relative_strength_ok:
            diagnostic_reasons.append("weak_relative_strength")
        if override_score < 92.0:
            diagnostic_reasons.append("weak_score<92")
        if entry_type == "دخول مبكر":
            diagnostic_reasons.append("weak_early_entry")
    if "تحت" in vwap_note:
        diagnostic_reasons.append("below_vwap")
    if m15_state == "معاكس" and score_i < 92:
        diagnostic_reasons.append("m15_contrary")
    if news_state == "negative":
        diagnostic_reasons.append("negative_news")
    if dump:
        diagnostic_reasons.append("dump")
    if failed:
        diagnostic_reasons.append("failed_breakout")
    if ext > 4.5:
        diagnostic_reasons.append("extension>4.5%")
    if atr_pct > 6.5:
        diagnostic_reasons.append("atr>6.5%")
    if vol_session_ratio < float(policy.get("min_volume_ratio", 0.85)):
        diagnostic_reasons.append("volume<policy")
    if chop:
        diagnostic_reasons.append("chop")
    if tp1_distance_pct < 0.8:
        diagnostic_reasons.append("tp1_distance<0.8%")
    if reward_r + 1e-9 < float(policy.get("min_tp1_r", 1.2)):
        diagnostic_reasons.append("tp1_r<1.20")
    if risk > price * 0.045:
        diagnostic_reasons.append("wide_stop>4.5%")

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
        sl_method="وقف هيكلي حسب نوع الدخول",
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
        learning_adjustment=round(total_learning_adj, 2),
        resistance_tp1=round(tp1, 4),
        news_state=news_state,
        news_title=news_title,
        news_source=news_source,
        breakout_quality=round(breakout_quality, 1),
        market_state=market_state,
        market_condition=market_condition,
        market_relative_strength=round(stock_relative_strength, 2),
        market_avg_change=round(market_rel_avg, 2),
        chop=chop,
        market_regime=market_regime,
        interaction_keys=interaction_keys,
        spread_pct=round(float(liquidity.get("spread_pct", 0) or 0), 3),
        expected_slippage_pct=round(float(liquidity.get("slippage_pct", 0) or 0), 3),
        dollar_volume_3m=round(float(liquidity.get("dollar_volume", 0) or 0), 0),
        liquidity_ok=liquidity_ok,
        diagnostic_reasons=diagnostic_reasons,
        matched_entry_types=matched_entry_types,
        strategy_scores=strategy_scores,
        strategy_component_scores=strategy_component_scores,
    )



def _intraday_diagnostic_metrics(sig: IntradaySignal, min_score: int = INTRADAY_MIN_SCORE) -> str:
    """Numeric diagnostic only; never changes trading decisions."""
    try:
        policy = _load_adaptive_policy()
    except Exception:
        policy = {}
    min_vol = float(policy.get("min_volume_ratio", 0.85) or 0.85)
    min_r = float(policy.get("min_tp1_r", 1.20) or 1.20)
    price = float(getattr(sig, "price", 0.0) or 0.0)
    tp1 = float(getattr(sig, "tp1", 0.0) or 0.0)
    stop = float(getattr(sig, "stop_loss", 0.0) or 0.0)
    tp1_dist = ((tp1 - price) / price * 100.0) if price > 0 else 0.0
    stop_risk = ((price - stop) / price * 100.0) if price > 0 and stop > 0 else 0.0
    return (
        f"regime={getattr(sig, 'market_condition', 'غير مؤكد')} "
        f"| RS={float(getattr(sig, 'market_relative_strength', 0.0) or 0.0):+.2f}% "
        f"| MKT={float(getattr(sig, 'market_avg_change', 0.0) or 0.0):+.2f}% "
        f"| score={float(sig.score):.0f}/{min_score} "
        f"| vol={float(getattr(sig, 'volume_ratio', 0.0) or 0.0):.2f}x/{min_vol:.2f}x "
        f"| ext={float(getattr(sig, 'ext_sma20', 0.0) or 0.0):.2f}%/4.50% "
        f"| ATR={float(getattr(sig, 'atr_pct', 0.0) or 0.0):.2f}%/6.50% "
        f"| TP1dist={tp1_dist:.2f}%/0.80% "
        f"| TP1R={float(getattr(sig, 'reward_r', 0.0) or 0.0):.2f}R/{min_r:.2f}R "
        f"| risk={stop_risk:.2f}%/4.50%"
    )


def format_intraday_ar(sig: IntradaySignal, min_score: int = INTRADAY_MIN_SCORE) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    market_map = {
        "قوي": "🟢 قوي",
        "إيجابي_تحت_VWAP": "🟠 إيجابي",
        "مختلط": "🟡 مختلط",
        "ضعيف": "🔴 ضعيف",
        "غير مؤكد": "⚪️ غير مؤكد",
    }
    market_condition = str(getattr(sig, "market_condition", "") or "")
    market_label = market_map.get(market_condition)
    if market_label is None:
        state = str(getattr(sig, "market_state", "") or "")
        if "داعمان" in state:
            market_label = "🟢 قوي"
        elif "إيجابيان" in state:
            market_label = "🟠 إيجابي"
        elif "مختلطان" in state:
            market_label = "🟡 مختلط"
        elif "ضعيفان" in state:
            market_label = "🔴 ضعيف"
        else:
            market_label = "⚪️ غير مؤكد"

    lines = [
        f"⚡ لحظي | {sig.symbol} | {sig.score}/100 | {sig.grade} | ساعة+15د+5د",
        f"{market_label} | نظام السوق",
        f"{sig.entry_emoji} الدخول: {sig.entry_type}",
        f"{sig.name}",
        "—————————————",
        f"السعر: {(getattr(sig, 'alert_entry_price', 0.0) or sig.price):.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
        f"شراء: {sig.buy_low:.2f} — {sig.buy_high:.2f}",
        f"وقف: {sig.stop_loss:.2f} | مخاطرة: {sig.risk_pct:.2f}%",
        f"TP1: {sig.tp1:.2f} | {sig.reward_r:.2f}R",
        f"TP2: {sig.tp2:.2f}",
        f"TP3: {sig.tp3:.2f}",
    ]
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



def _prefilter_intraday(
    symbol: str,
    preloaded: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame] | None:
    """Stage 1: cheap H1+5m routing with a lane for each of the 16 strategies.

    Stage 1 never declares a trade strategy. It only estimates which strategies
    deserve Stage-2 inspection using data already available here. The exact
    strategy gates remain in analyze_intraday(), where 15m/full setup data exists.
    """
    try:
        from market_data import fetch_intraday, intraday_data_fresh
        if preloaded is not None:
            h1, m5 = preloaded
        else:
            h1 = fetch_intraday(symbol, interval="60m", period="10d")
            m5 = fetch_intraday(symbol, interval="5m", period="5d")
        ok_h1, _ = intraday_data_fresh(h1, "60m", 90)
        ok_m5, _ = intraday_data_fresh(m5, "5m", 12)
        if h1 is None or m5 is None or len(h1) < 40 or len(m5) < 30 or not ok_h1 or not ok_m5:
            return None

        last_day = m5.index[-1].date()
        today = m5[m5.index.date == last_day]
        if len(today) < 6:
            return None
        price = float(today["Close"].iloc[-1])
        if price <= 0 or price > float(MAX_AUTO_PRICE):
            return None

        hc = h1["Close"]
        e20 = float(_ema(hc, 20).iloc[-1])
        e50 = float(_ema(hc, 50).iloc[-1])
        h_rsi = float(_rsi(hc, 14).iloc[-1])
        trend = price > e20 > e50 * 0.998 and h_rsi >= 45

        vwap_s = _vwap(today)
        vwap = float(vwap_s.iloc[-1]) if pd.notna(vwap_s.iloc[-1]) else price
        vwap_dist = (price - vwap) / max(vwap, 1e-9) * 100
        above_vwap = price >= vwap * 0.996
        above_open = price >= float(today["Open"].iloc[0]) * 0.997
        last_green = float(today["Close"].iloc[-1]) >= float(today["Open"].iloc[-1])
        c5 = today["Close"]
        mom = (price - float(c5.iloc[-6])) / max(float(c5.iloc[-6]), 1e-9) * 100
        r5 = float(_rsi(c5, 14).iloc[-1])

        hist = m5[m5.index.date < last_day].tail(120)
        avg_today = float(today["Volume"].mean())
        avg_hist = float(hist["Volume"].mean()) if not hist.empty else float(m5["Volume"].tail(60).mean() or 1)
        vol_ratio = avg_today / avg_hist if avg_hist else 1.0

        # Cheap structural proxies. These are ROUTING signals only; Stage 2 is authoritative.
        prev_high = float(hist["High"].max()) if not hist.empty else price
        prev_low = float(hist["Low"].min()) if not hist.empty else price
        day_high = float(today["High"].max())
        day_low = float(today["Low"].min())
        recent = today.tail(min(12, len(today)))
        recent_high = float(recent["High"].max())
        recent_low = float(recent["Low"].min())
        recent_range_pct = (recent_high - recent_low) / max(price, 1e-9) * 100
        prior_range = today.iloc[:-6] if len(today) > 12 else today.iloc[:-3]
        prior_high = float(prior_range["High"].max()) if not prior_range.empty else day_high
        prior_low = float(prior_range["Low"].min()) if not prior_range.empty else day_low
        first_or = today.head(min(3, len(today)))
        orb_high = float(first_or["High"].max()) if not first_or.empty else price
        orb_low = float(first_or["Low"].min()) if not first_or.empty else price
        last_low = float(today["Low"].iloc[-1])
        last_high = float(today["High"].iloc[-1])
        last_open = float(today["Open"].iloc[-1])
        last_close = float(today["Close"].iloc[-1])
        last_range_pct = abs(last_high - last_low) / max(last_close, 1e-9) * 100
        near_vwap = abs(price - vwap) / max(vwap, 1e-9) * 100 <= 0.8
        near_ema20 = abs(price - e20) / max(e20, 1e-9) * 100 <= 1.0
        near_prev_high = abs(price - prev_high) / max(price, 1e-9) * 100 <= 1.2
        near_orb = abs(price - orb_high) / max(price, 1e-9) * 100 <= 1.2
        strong_volume = vol_ratio >= 1.15
        strong_momentum = mom >= 0.35
        pullback = last_close >= last_open and (last_low <= price * 0.997 or near_vwap or near_ema20)
        displacement = strong_volume and last_range_pct >= 0.8 and last_close > last_open
        sweep_low = last_low < min(prior_low, orb_low) * 1.001 and last_close > last_open
        compression = recent_range_pct <= 2.5 and (strong_momentum or strong_volume)
        prior_push = (price - float(today["Close"].iloc[max(0, len(today)-18)])) / max(price, 1e-9) * 100 if len(today) >= 6 else 0.0
        bull_flag_proxy = prior_push >= 1.0 and recent_range_pct <= 2.5 and above_vwap
        opening_drive = len(today) >= 8 and float(first_or["Close"].iloc[-1]) > float(first_or["Open"].iloc[0]) and pullback
        hod_reclaim = price >= recent_high * 0.998 and strong_volume
        resistance_reclaim = price >= prev_high * 0.998 or near_prev_high
        abc_proxy = trend and above_vwap and pullback and strong_momentum

        # Every strategy gets a routing score. No score here can create a signal.
        route_by_strategy = {
            "اختراق مؤكد": (8 if price >= prev_high * 0.998 else 0) + (3 if strong_volume else 0) + (2 if last_green else 0),
            "إعادة اختبار": (5 if near_prev_high else 0) + (3 if pullback else 0) + (2 if above_vwap else 0),
            "دخول مبكر": (4 if above_vwap else 0) + (3 if above_open else 0) + (2 if trend else 0) + (1 if not strong_momentum else 0),
            "ارتداد VWAP": (6 if near_vwap else 0) + (3 if last_green else 0) + (2 if trend else 0),
            "ارتداد EMA20": (6 if near_ema20 else 0) + (3 if trend else 0) + (2 if last_green else 0),
            "سحب سيولة": (7 if sweep_low else 0) + (3 if above_vwap else 0) + (2 if strong_volume else 0),
            "اختراق نطاق الافتتاح": (7 if price >= orb_high * 0.998 else 0) + (3 if strong_volume else 0) + (2 if above_vwap else 0),
            "استمرار الزخم": (5 if strong_momentum else 0) + (3 if trend else 0) + (2 if strong_volume else 0),
            "ضغط ثم انفجار": (6 if compression else 0) + (3 if strong_volume else 0) + (2 if last_green else 0),
            "علم صاعد": (6 if bull_flag_proxy else 0) + (3 if trend else 0) + (2 if above_vwap else 0),
            "استعادة مستوى": (6 if resistance_reclaim else 0) + (3 if above_vwap else 0) + (2 if last_green else 0),
            "دخول بعد Opening Drive": (6 if opening_drive else 0) + (3 if trend else 0) + (2 if pullback else 0),
            "استعادة قمة اليوم": (6 if hod_reclaim else 0) + (3 if strong_volume else 0) + (2 if trend else 0),
            "استعادة بعد فشل ORB": (6 if near_orb else 0) + (3 if pullback else 0) + (2 if above_vwap else 0),
            "استمرار ABC": (6 if abc_proxy else 0) + (3 if trend else 0) + (2 if strong_volume else 0),
            "سحب سيولة مع Displacement": (7 if sweep_low and displacement else 0) + (3 if trend else 0) + (2 if above_vwap else 0),
        }

        # General route score remains useful for overall ranking.
        route_score = max(route_by_strategy.values()) + (2.0 if trend else 0.0)
        route_score += 1.5 if above_vwap else 0.0
        route_score += 1.0 if above_open else 0.0
        route_score += min(2.0, max(0.0, mom))
        route_score += min(2.0, max(0.0, vol_ratio - 0.75) * 2.0)
        route_score -= max(0.0, vwap_dist - 3.0) * 0.5
        route_score -= 1.0 if r5 >= 82 else 0.0

        # Only hard-fail unusable data/liquidity. Do NOT discard a valid setup
        # merely because it is not a generic trend/VWAP/momentum candidate.
        if vol_ratio < 0.65:
            return None
        return route_score, route_by_strategy, h1, m5
    except Exception:
        return None


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
        # IMPORTANT: scan_intraday has its own scope; fetch_intraday must be
        # imported here before calling _market_alignment. Without this import
        # the old code raised NameError, silently fell back to "السوق غير مؤكد",
        # and every Stage-2 signal failed market_permission.
        from market_data import fetch_intraday
        market_context = _market_alignment(fetch_intraday)
        market_regime = _market_regime_from_state(market_context[1])
        if market_regime == "ضعيف":
            scan_intraday.last_window = "SPY+QQQ لحظيًا ضعيفان"
            min_score = max(min_score, 88)
        elif market_regime == "إيجابي_تحت_VWAP":
            scan_intraday.last_window = "SPY+QQQ إيجابيان تحت VWAP"
            min_score = max(min_score, POSITIVE_BELOW_VWAP_MIN_SCORE)
        elif market_regime == "مختلط":
            min_score = max(min_score, 85)
        log.info(
            "INTRADAY MARKET REGIME | regime=%s | state=%s | min_score=%d",
            market_regime, market_context[1], min_score,
        )
    except Exception as exc:
        log.warning("INTRADAY MARKET CONTEXT FAILED | %s", str(exc))
        market_context = (False, "السوق غير مؤكد")

    workers = min(8, max(2, len(symbols)))
    stage1: list[tuple[float, dict[str, float], str, pd.DataFrame, pd.DataFrame]] = []

    log.info("INTRADAY SCAN: %d symbols loaded", len(symbols))

    # Stage 1 bulk load: two multi-symbol requests (H1 + 5m) instead of
    # hundreds of per-symbol requests. If bulk loading fails, fall back to
    # the existing per-symbol path, which is protected by market_data rate limiting.
    bulk_h1: dict[str, pd.DataFrame] = {}
    bulk_m5: dict[str, pd.DataFrame] = {}
    try:
        from market_data import fetch_alpaca_bars_multi, alpaca_configured
        from datetime import datetime, timedelta, timezone
        if alpaca_configured():
            end = datetime.now(timezone.utc)
            bulk_h1 = fetch_alpaca_bars_multi(
                symbols, "1Hour", end - timedelta(days=13), end=end, chunk_size=50
            )
            bulk_m5 = fetch_alpaca_bars_multi(
                symbols, "5Min", end - timedelta(days=8), end=end, chunk_size=50
            )
            log.info(
                "STAGE 1 BULK: H1=%d symbols, 5m=%d symbols",
                len(bulk_h1), len(bulk_m5),
            )
    except Exception as exc:
        log.warning("STAGE 1 bulk load failed; fallback to per-symbol: %s", exc)
        bulk_h1, bulk_m5 = {}, {}

    # Stage 1 — H1 + 5m only for the full universe.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _prefilter_intraday,
                sym,
                (bulk_h1.get(sym), bulk_m5.get(sym))
                if sym in bulk_h1 and sym in bulk_m5 else None,
            ): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                item = fut.result()
            except Exception:
                item = None
            if item:
                route_score, route_by_strategy, h1, m5 = item
                stage1.append((route_score, route_by_strategy, sym, h1, m5))

    stage1.sort(key=lambda x: x[0], reverse=True)
    log.info(
        "STAGE 1: %d/%d completed; %d candidates passed prefilter",
        len(stage1), len(symbols), len(stage1),
    )
    # Strategy-aware routing: keep the strongest overall names AND reserve a
    # small candidate lane for every strategy. This prevents a generic Stage-1
    # score from starving a valid setup type before analyze_intraday() sees it.
    base_n = max(PREFILTER_MAX_CANDIDATES, limit * 5)
    selected: dict[str, tuple] = {sym: item for item in stage1[:base_n] for sym in [item[2]]}
    for et in ENTRY_TYPES:
        ranked_for_strategy = sorted(
            stage1, key=lambda item: float(item[1].get(et, 0.0)), reverse=True
        )
        for item in ranked_for_strategy[:PREFILTER_STRATEGY_TOP_K]:
            selected[item[2]] = item
    finalists = sorted(
        selected.values(),
        key=lambda item: (float(item[0]), max(item[1].values()) if item[1] else 0.0),
        reverse=True,
    )[:max(base_n, min(PREFILTER_STRATEGY_CAP, base_n + len(ENTRY_TYPES) * PREFILTER_STRATEGY_TOP_K))]
    log.info(
        "STAGE 2: top %d candidates selected; strategy-aware routing reserved lanes for %d strategies",
        len(finalists), len(ENTRY_TYPES),
    )

    # Stage 2 — only finalists receive 15m + full setup/confluence/news analysis.
    results: list[IntradaySignal] = []
    rejection_counts = {
        "no_signal": 0,
        "score": 0,
        "live_ok": 0,
        "quality": 0,
        "below_vwap": 0,
        "m15_contrary": 0,
        "negative_news": 0,
        "liquidity": 0,
    }
    rejection_samples: list[str] = []
    def _one_stage2(item):
        _, _, sym, h1, m5 = item
        try:
            return analyze_intraday(
                sym,
                names.get(sym, sym),
                market_context=market_context,
                preloaded=(h1, m5),
            )
        except Exception as exc:
            log.warning("INTRADAY STAGE 2 EXCEPTION | %s | %s", sym, str(exc))
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_stage2, item) for item in finalists]
        for fut in as_completed(futures):
            sig = fut.result()
            if not sig:
                rejection_counts["no_signal"] += 1
                continue
            # تشخيص أول سبب فعلي للرفض، مع الاحتفاظ بأسباب الإشارة كلها داخل
            # diagnostic_reasons حتى نعرف هل المشكلة درجة أم جودة أم سوق...
            reasons = list(getattr(sig, "diagnostic_reasons", []) or [])

            if sig.score < min_score:
                rejection_counts["score"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: score={sig.score}<{min_score} reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if not sig.live_ok:
                rejection_counts["live_ok"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: live_ok=False reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if not sig.quality_ok:
                rejection_counts["quality"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: quality=False reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if "تحت" in sig.vwap_day_note:
                rejection_counts["below_vwap"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: below_vwap reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if sig.m15_state == "معاكس" and sig.score < 92:
                rejection_counts["m15_contrary"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: m15_contrary score={sig.score} reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if sig.news_state == "negative":
                rejection_counts["negative_news"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: negative_news reasons={reasons} | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            liq = _quote_liquidity(sig.symbol, sig.price)
            sig.spread_pct = round(float(liq.get("spread_pct", 0) or 0), 3)
            sig.expected_slippage_pct = round(float(liq.get("slippage_pct", 0) or 0), 3)
            sig.dollar_volume_3m = round(float(liq.get("dollar_volume", 0) or 0), 0)
            sig.liquidity_ok = bool(liq.get("ok", True))
            if sig.spread_pct > MAX_SPREAD_PCT:
                sig.warnings.append(f"Spread مرتفع {sig.spread_pct:.2f}%")
            if not sig.liquidity_ok:
                rejection_counts["liquidity"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: liquidity=False spread={sig.spread_pct:.3f}% slippage={sig.expected_slippage_pct:.3f}% | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            if liq.get("quote_source") == "none" or float(liq.get("quote_age_min", 999) or 999) > 2.0:
                rejection_counts["liquidity"] += 1
                rejection_samples.append(
                    f"{sig.symbol}: stale_or_no_quote source={liq.get('quote_source')} age={float(liq.get('quote_age_min', 999) or 999):.2f}m | {_intraday_diagnostic_metrics(sig, min_score)}"
                )
                continue
            results.append(sig)
            log.info(
                "INTRADAY QUALIFIED METRICS | %s | %s",
                sig.symbol,
                _intraday_diagnostic_metrics(sig, min_score),
            )

    log.info(
        "STAGE 2: %d deep candidates completed; %d qualified signals | rejects=%s",
        len(finalists), len(results), rejection_counts,
    )
    if rejection_samples:
        for sample in rejection_samples[:20]:
            log.info("INTRADAY REJECT DETAIL | %s", sample)

    records_for_dedup = _read_learning_records()
    before_dedup = len(results)
    results = [sig for sig in results if not _signal_duplicate_recent(sig, records_for_dedup)]
    log.info("SIGNAL DEDUP | removed=%d | remaining=%d", before_dedup - len(results), len(results))
    rank = {et: i for i, et in enumerate(ENTRY_TYPES)}
    results.sort(key=lambda x: (
        -(float(x.score) + 1.5 * min(float(getattr(x, "reward_r", 0) or 0), 3.0)
                    - 1.5 * float(getattr(x, "spread_pct", 0) or 0)
          - 1.0 * float(getattr(x, "expected_slippage_pct", 0) or 0)
          - 0.8 * max(float(getattr(x, "ext_sma20", 0) or 0) - 2.0, 0.0)),
        rank.get(x.entry_type, 99), -float(x.score), -float(getattr(x, "reward_r", 0) or 0)
    ))
    return results[:limit]


scan_intraday.last_window = ""
