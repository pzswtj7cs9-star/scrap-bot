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
from threading import Lock

import numpy as np
import pandas as pd

from market import now_ny, REGULAR_OPEN, REGULAR_CLOSE, is_us_regular_session, session_label
from stocks import MAX_AUTO_PRICE

log = logging.getLogger("halal-bot.daily")

DAILY_ANALYZER_VERSION = "20260921-214500-FINAL-AUDIT-CUMULATIVE-LIVEGATE"
log.info("DAILY ANALYZER VERSION | %s", DAILY_ANALYZER_VERSION)

SKIP_OPEN_MIN = 0
SKIP_CLOSE_MIN = 0
DAILY_MIN_SCORE = 82

DAILY_AUDIT_FILE = Path("/var/data/daily_audit_counters.json")
_DAILY_AUDIT_LOCK = Lock()
# Runtime-only bridge: analyze_daily() records the exact zero-match blockers so
# scan_daily() can persist them in the cumulative audit. Diagnostic only.
_DAILY_NO_SIGNAL_AUDIT: dict[str, dict[str, int]] = {}
_DAILY_NO_SIGNAL_LOCK = Lock()
_DAILY_DATA_AUDIT: dict[str, dict[str, int]] = {}
_DAILY_DATA_AUDIT_LOCK = Lock()

def _daily_data_audit_record(symbol: str, *reasons: str) -> None:
    if not symbol:
        return
    with _DAILY_DATA_AUDIT_LOCK:
        dst = _DAILY_DATA_AUDIT.setdefault(str(symbol).upper(), {})
        for reason in reasons:
            if reason:
                _daily_bump(dst, str(reason))

def _daily_data_audit_pop(symbol: str) -> dict[str, int]:
    with _DAILY_DATA_AUDIT_LOCK:
        return dict(_DAILY_DATA_AUDIT.pop(str(symbol).upper(), {}) or {})


def _new_daily_audit() -> dict:
    return {
        "schema_version": 1,
        "updated_at": None,
        "scans": 0,
        "stage1": {"passed": 0, "rejected": 0, "reasons": {}},
        "stage2": {
            "deep_candidates": 0, "qualified": 0,
            "gates": {},
            "reason_counts_by_gate": {},
            "diagnostic_reason_counts_by_gate": {},
        },
        "data": {
            "stage1": {},
            "stage2": {},
            "timeframes": {},
        },
        "strategy": {"matched": {}, "failed": {}, "blockers": {}},
    }

def _load_daily_audit() -> dict:
    base = _new_daily_audit()
    try:
        if not DAILY_AUDIT_FILE.exists():
            return base
        raw = json.loads(DAILY_AUDIT_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return base
        for key, value in raw.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    except Exception as exc:
        log.warning("DAILY AUDIT LOAD FAILED | %s", str(exc))
        return base

def _save_daily_audit(data: dict) -> None:
    try:
        DAILY_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = DAILY_AUDIT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(DAILY_AUDIT_FILE)
    except Exception as exc:
        log.warning("DAILY AUDIT SAVE FAILED | %s", str(exc))

def _daily_bump(counter: dict[str, int], reason: str, amount: int = 1) -> None:
    key = str(reason or "unknown")
    counter[key] = int(counter.get(key, 0) or 0) + int(amount or 0)

def _commit_daily_audit(*, stage1_counts=None, stage1_passed=0, stage1_rejected=0,
                        stage2_deep=0, stage2_qualified=0, stage2_gate_counts=None,
                        stage2_gate_reasons=None, stage2_gate_diag=None,
                        data_stage1=None, data_stage2=None,
                        strategy_matched=None, strategy_failed=None, strategy_blockers=None) -> dict:
    with _DAILY_AUDIT_LOCK:
        data = _load_daily_audit()
        data["scans"] = int(data.get("scans", 0) or 0) + 1
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        st1 = data.setdefault("stage1", {"passed": 0, "rejected": 0, "reasons": {}})
        st1["passed"] = int(st1.get("passed", 0) or 0) + int(stage1_passed or 0)
        st1["rejected"] = int(st1.get("rejected", 0) or 0) + int(stage1_rejected or 0)
        for reason, count in (stage1_counts or {}).items():
            _daily_bump(st1.setdefault("reasons", {}), reason, count)
        st2 = data.setdefault("stage2", {})
        st2["deep_candidates"] = int(st2.get("deep_candidates", 0) or 0) + int(stage2_deep or 0)
        st2["qualified"] = int(st2.get("qualified", 0) or 0) + int(stage2_qualified or 0)
        for gate, count in (stage2_gate_counts or {}).items():
            _daily_bump(st2.setdefault("gates", {}), gate, count)
        for gate, reasons in (stage2_gate_reasons or {}).items():
            dst = st2.setdefault("reason_counts_by_gate", {}).setdefault(gate, {})
            for reason, count in reasons.items():
                _daily_bump(dst, reason, count)
        for gate, reasons in (stage2_gate_diag or {}).items():
            dst = st2.setdefault("diagnostic_reason_counts_by_gate", {}).setdefault(gate, {})
            for reason, count in reasons.items():
                _daily_bump(dst, reason, count)
        data_audit = data.setdefault("data", {"stage1": {}, "stage2": {}, "timeframes": {}})
        for reason, count in (data_stage1 or {}).items():
            _daily_bump(data_audit.setdefault("stage1", {}), reason, count)
        for reason, count in (data_stage2 or {}).items():
            _daily_bump(data_audit.setdefault("stage2", {}), reason, count)
            if ":" in str(reason):
                tf, subreason = str(reason).split(":", 1)
                _daily_bump(data_audit.setdefault("timeframes", {}).setdefault(tf, {}), subreason, count)
        strat = data.setdefault("strategy", {})
        for key, count in (strategy_matched or {}).items():
            _daily_bump(strat.setdefault("matched", {}), key, count)
        for key, count in (strategy_failed or {}).items():
            _daily_bump(strat.setdefault("failed", {}), key, count)
        for strategy, reasons in (strategy_blockers or {}).items():
            dst = strat.setdefault("blockers", {}).setdefault(strategy, {})
            for reason, count in reasons.items():
                _daily_bump(dst, reason, count)
        _save_daily_audit(data)
        return data


# Central execution / market-regime configuration. Keep global safety thresholds
# here so changing one policy value cannot leave a stale duplicate elsewhere.
QUOTE_MAX_AGE_MIN = 2.0
MAX_SPREAD_PCT = 0.80          # warning threshold
HARD_MAX_SPREAD_PCT = 1.20     # hard reject threshold
DAILY_MARKET_SCORE_BY_STATE = {
    "قوي": 82,
    "إيجابي_تحت_VWAP": 85,
    "مختلط": 86,
    "ضعيف": 92,
    "غير مؤكد": 92,
}

DAILY_LEARNING_FILE = Path("/var/data/daily_v2_learning.jsonl")
LEARNING_MIN_SAMPLES = 20
LEARNING_LOOKBACK = 60
LEARNING_MAX_ADJUSTMENT = 4.0
ADAPTIVE_POLICY_FILE = Path("/var/data/daily_v2_adaptive_policy.json")
ADAPTIVE_MIN_SAMPLES = 30
ADAPTIVE_CONFIRM_SAMPLES = 40
ADAPTIVE_MAX_CHANGE = 0.15
STRATEGY_WEIGHT_MIN_SAMPLES = 20
STRATEGY_WEIGHT_STEP = 0.05
STRATEGY_WEIGHT_MIN_FACTOR = 0.50
STRATEGY_WEIGHT_MAX_FACTOR = 1.50
ADAPTIVE_BEST_FILE = Path("/var/data/daily_v2_adaptive_best.json")
ADAPTIVE_SHADOW_FILE = Path("/var/data/daily_v2_shadow_results.jsonl")
LEARNING_ALERT_FILE = Path("/var/data/daily_v2_learning_alert.json")

# Canonical list: the adaptive learner must track every real entry strategy.
# Stage-1 routing limits. The protected-lane cap is derived from these values
# and the canonical strategy list so it cannot drift if the strategy count changes.
PREFILTER_MAX_CANDIDATES = 50
PREFILTER_STRATEGY_TOP_K = 4
ENTRY_TYPES = (
    "اختراق مؤكد", "إعادة اختبار", "دخول مبكر", "ارتداد VWAP", "ارتداد EMA20",
    "سحب سيولة", "اختراق نطاق الافتتاح", "استمرار الزخم", "ضغط ثم انفجار",
    "علم صاعد", "استعادة مستوى", "دخول بعد Opening Drive", "استعادة قمة اليوم",
    "استعادة بعد فشل ORB", "استمرار ABC", "سحب سيولة مع Displacement",
    "استمرار/استعادة الفجوة", "استعادة بعد فشل كسر دعم", "ارتداد بعد تفوق نسبي",
)
PREFILTER_STRATEGY_CAP = PREFILTER_MAX_CANDIDATES + (len(ENTRY_TYPES) * PREFILTER_STRATEGY_TOP_K)
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

# Final daily risk safety band: keep aligned with the documented 0.60%–4.50% policy.
DAILY_MIN_RISK_PCT = 0.60
DAILY_MAX_RISK_PCT = 4.50

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

# Setup freshness is expressed in completed daily bars.
STRATEGY_FRESHNESS_BARS = {
    "اختراق مؤكد": 3,
    "إعادة اختبار": 5,
    "دخول مبكر": 3,
    "ارتداد VWAP": 4,
    "ارتداد EMA20": 4,
    "سحب سيولة": 3,
    "اختراق نطاق الافتتاح": 3,
    "استمرار الزخم": 3,
    "ضغط ثم انفجار": 8,
    "علم صاعد": 6,
    "استعادة مستوى": 4,
    "دخول بعد Opening Drive": 6,
    "استعادة قمة اليوم": 4,
    "استعادة بعد فشل ORB": 4,
    "استمرار ABC": 4,
    "سحب سيولة مع Displacement": 3,
    "استمرار/استعادة الفجوة": 3,
    "استعادة بعد فشل كسر دعم": 5,
    "ارتداد بعد تفوق نسبي": 4,
}

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
    entry_type: str = ""
    entry_emoji: str = "🟢"
    matched_entry_types: list[str] | None = None
    strategy_scores: dict[str, float] | None = None
    strategy_component_scores: dict[str, dict[str, float]] | None = None
    structure_zone: str = "محايدة"
    quality_ok: bool = True
    live_ok: bool = True
    # Diagnostic-only: exact failed sub-conditions of the live gate.
    live_gate_reasons: list[str] | None = None
    volume_ratio: float = 1.0
    regime: str = "neutral"
    factor_keys: list | None = None
    sma20: float = 0.0
    atr_pct: float = 0.0
    ext_sma20: float = 0.0
    # سعر التنفيذ النهائي وقت قبول التنبيه؛ لا يغيّر سعر التحليل أو Stop/TP.
    alert_entry_price: float = 0.0
    h4_state: str = "محايد"
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
    liquidity_ok: bool = True
    quality_reasons: list[str] | None = None

    # Raw score used for Global eligibility; strategy entry_limits never block alerts.
    raw_score: float = 0.0



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
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    # Learning records are the source of truth for outcomes/cycle counting.
    # Flush + fsync so a process crash cannot silently leave the last outcome
    # only in the OS buffer. This does not change learning logic.
    with DAILY_LEARNING_FILE.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


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
            "weekly_trend": 1.0, "h4": 1.0, "daily": 1.0, "vwap": 1.0,
            "vol_session": 1.0, "market": 1.0, "breakout": 1.0,
            "breakout_candle": 1.0, "retest": 1.0, "vwap_bounce": 1.0,
            "ema_pullback": 1.0, "liquidity_sweep": 1.0, "liquidity_displacement": 1.0, "orb": 1.0, "momentum_continuation": 1.0, "compression_expansion": 1.0, "bull_flag": 1.0, "resistance_reclaim": 1.0, "opening_drive_pullback": 1.0, "hod_reclaim": 1.0, "orb_failed_reclaim": 1.0, "abc_continuation": 1.0, "vwap_weekly_confluence": 1.0, "multi_level_confluence": 1.0, "early": 1.0,
            "news_momentum": 1.0,
        },
        "entry_limits": {"دخول مبكر": 94, "إعادة اختبار": 95, "ارتداد VWAP": 96, "ارتداد EMA20": 96, "سحب سيولة": 97, "ضغط ثم انفجار": 98, "استمرار الزخم": 97, "اختراق نطاق الافتتاح": 99, "اختراق مؤكد": 100, "علم صاعد": 98, "استعادة مستوى": 98, "دخول بعد Opening Drive": 98, "استعادة قمة اليوم": 98, "استعادة بعد فشل ORB": 99, "استمرار ABC": 98, "سحب سيولة مع Displacement": 99, "استمرار/استعادة الفجوة": 98, "استعادة بعد فشل كسر دعم": 98, "ارتداد بعد تفوق نسبي": 97},
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

    Confirmation is separate and contributes only through the fixed 30% block.
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
        "استمرار/استعادة الفجوة": {"gap_quality": .35, "gap_hold": .25, "gap_trigger": .40},
        "استعادة بعد فشل كسر دعم": {"support_quality": .25, "breakdown_quality": .35, "breakdown_reclaim": .40},
        "ارتداد بعد تفوق نسبي": {"rs_strength": .30, "rs_pullback": .25, "rs_higher_low": .20, "rs_trigger": .25},
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
    """إحصاءات منفصلة لكل واحدة من استراتيجيات الدخول المعتمدة."""
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
    approved = candidate_rate >= actual_rate + ADAPTIVE_MIN_EDGE and coverage >= ADAPTIVE_MIN_COVERAGE
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
            # Roll back adaptive policy parameters only. Preserve the current
            # learning-cycle state so an older policy cannot rewind the 100-trade
            # cycle anchor and accidentally trigger a premature retrain.
            best_policy = json.loads(json.dumps(best["policy"]))
            for _cycle_key in (
                "adaptive_cycle_anchor_samples",
                "adaptive_cycle_total",
                "last_monthly_sample_count",
            ):
                if _cycle_key in p:
                    best_policy[_cycle_key] = p[_cycle_key]
            _save_adaptive_policy(best_policy)
            return {
                "status": "rollback",
                "from_generation": p.get("generation"),
                "to_generation": best["policy"].get("generation", 0),
            }
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)
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
    risk = max(entry - stop, entry * (DAILY_MIN_RISK_PCT / 100.0))
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
    base_gap = max(entry - stop, entry * (DAILY_MIN_RISK_PCT / 100.0))
    new_gap = base_gap * sl_mult
    new_gap = max(entry * (DAILY_MIN_RISK_PCT / 100.0), min(entry * (DAILY_MAX_RISK_PCT / 100.0), new_gap))
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


# Setup lifecycle is diagnostic metadata only. It documents the exact
# trigger/consumption/reset contract without changing strategy matching.
SETUP_LIFECYCLE_RULES = {
    "اختراق مؤكد": ("إغلاق مؤكد فوق مستوى الاختراق", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا عند تكوّن مستوى/اختراق جديد"),
    "إعادة اختبار": ("اختراق سابق ثم عودة للمستوى واستعادة", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد اختراق جديد للمستوى"),
    "دخول مبكر": ("ضغط قبل الاختراق قرب المقاومة", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد انتهاء البنية وتكوّن ضغط جديد"),
    "ارتداد VWAP": ("لمس VWAP ثم استعادة", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Setup VWAP جديد"),
    "ارتداد EMA20": ("Pullback إلى EMA20 ثم استعادة", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Pullback جديد"),
    "سحب سيولة": ("Sweep ثم Reclaim لمستوى السيولة", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Sweep جديد"),
    "اختراق نطاق الافتتاح": ("إغلاق مؤكد فوق ORB", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا مع ORB/جلسة جديدة"),
    "استمرار الزخم": ("Impulse ثم Pause ثم Resume", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد دورة Momentum جديدة"),
    "ضغط ثم انفجار": ("Compression ثم Expansion ثم Breakout", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Compression جديدة"),
    "علم صاعد": ("Impulse ثم Flag ثم Breakout", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Flag جديدة"),
    "استعادة مستوى": ("فقد مستوى ثم Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد فقد/استعادة جديدة"),
    "دخول بعد Opening Drive": ("Opening Drive ثم Pullback ثم Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Drive جديد"),
    "استعادة قمة اليوم": ("فقد HOD ثم Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد HOD جديد وفقده ثم Reclaim"),
    "استعادة بعد فشل ORB": ("ORB Break ثم Failure ثم Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Failure جديد"),
    "استمرار ABC": ("A ثم B ثم C Break", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد موجة ABC جديدة"),
    "سحب سيولة مع Displacement": ("Sweep ثم Reclaim ثم Displacement", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد Sweep/Displacement جديد"),
    "استمرار/استعادة الفجوة": ("Gap >=2% ثم Hold/Failure ثم Continuation أو Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا مع فجوة جلسة جديدة"),
    "استعادة بعد فشل كسر دعم": ("Support متعدد اللمس ثم Close دون الدعم ثم Reclaim", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد تكوّن دعم جديد وفشل جديد"),
    "ارتداد بعد تفوق نسبي": ("تفوق مستمر مقابل SPY وQQQ ثم Pullback مضبوط ثم Trigger", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا بعد موجة تفوق جديدة"),
}

def _setup_lifecycle(sig) -> dict:
    et = str(getattr(sig, "entry_type", "") or "")
    trigger, consumed, reset = SETUP_LIFECYCLE_RULES.get(
        et, ("Trigger الاستراتيجية", "يُستهلك عند تسجيل الإشارة", "يتاح مجددًا عند تكوّن Setup جديد")
    )
    return {"trigger": trigger, "consumed": consumed, "reset": reset, "freshness_bars": int(STRATEGY_FRESHNESS_BARS.get(et, 3))}


def _setup_fingerprint(sig) -> str:
    """Stable setup identity for lifecycle/reset tracking; diagnostic metadata only."""
    et = str(getattr(sig, "entry_type", "") or "")
    symbol = str(getattr(sig, "symbol", "") or "")
    regime = str(getattr(sig, "market_regime", "neutral") or "neutral")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stop = float(getattr(sig, "stop_loss", 0.0) or 0.0)
    return f"{symbol}|{et}|{regime}|{day}|anchor:{stop:.4f}"


def _signal_duplicate_recent(sig, records: list[dict]) -> bool:
    """يمنع إعادة نفس الهيكل لنفس السهم خلال نافذة زمنية محددة."""
    now = datetime.now(timezone.utc)
    symbol = str(getattr(sig, "symbol", ""))
    entry_type = str(getattr(sig, "entry_type", ""))
    regime = str(getattr(sig, "market_regime", "neutral"))
    current_fp = _setup_fingerprint(sig)
    for r in reversed(records):
        if r.get("record_type") != "signal" or r.get("symbol") != symbol:
            continue
        if str(r.get("entry_type") or "") != entry_type or str(r.get("market_regime") or "neutral") != regime:
            continue
        if r.get("status") not in {"pending", "completed"}:
            continue
        stored_fp = str(r.get("setup_fingerprint") or "")
        if stored_fp and stored_fp == current_fp:
            return True
        # Backward compatibility for historical records without fingerprinting.
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
    # Always refresh per-strategy statistics so all canonical setups are observable.
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
    for et in ENTRY_TYPES:
        subset = [
            r for r in train
            if str(r.get("entry_type") or "") == et
        ]
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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)

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
        except Exception as exc:
            log.debug("DAILY non-critical nested fallback exception: %s", exc)

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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)

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


def _score_ledger_for_signal(sig: DailySignal) -> dict:
    """Diagnostic-only final score ledger; never changes trading decisions."""
    primary = str(getattr(sig, "entry_type", "") or "")
    rows = dict(getattr(sig, "strategy_component_scores", {}) or {})
    row = dict(rows.get(primary, {}) or {})
    base = float(row.get("final_strategy_score", getattr(sig, "score", 0)) or 0.0)
    adaptive = float(getattr(sig, "learning_adjustment", 0.0) or 0.0)
    pre_cap = base + adaptive
    final_score = float(getattr(sig, "score", 0) or 0.0)
    return {
        "primary_strategy": primary,
        "core_score": float(row.get("core_score", 0.0) or 0.0),
        "core_contribution_70pct": float(row.get("core_contribution_70pct", 0.0) or 0.0),
        "confirmation_score": float(row.get("confirmation_score", 0.0) or 0.0),
        "confirmation_contribution_30pct": float(row.get("confirmation_contribution_30pct", 0.0) or 0.0),
        "base_strategy_score": base,
        "adaptive_adjustment": adaptive,
        "score_before_strategy_cap": pre_cap,
        "final_score": final_score,
        "strategy_cap_applied": bool(final_score + 1e-9 < pre_cap),
        "score_conservation_check": round(float(row.get("core_contribution_70pct", 0.0) or 0.0) + float(row.get("confirmation_contribution_30pct", 0.0) or 0.0) - base, 8),
        "safety_gates": {
            "quality_ok": bool(getattr(sig, "quality_ok", False)),
            "live_ok": bool(getattr(sig, "live_ok", False)),
            "liquidity_ok": bool(getattr(sig, "liquidity_ok", False)),
            "spread_hard_reject_pass": float(getattr(sig, "spread_pct", 0.0) or 0.0) <= 1.20,
        },
    }


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
            "execution_entry": round(float(getattr(sig, "alert_entry_price", 0.0) or sig.price), 4),
            "stop_loss": round(float(sig.stop_loss), 4),
            "tp1": round(float(sig.tp1), 4),
            "score": int(sig.score),
            "grade": sig.grade,
            "entry_type": sig.entry_type,
            "setup_fingerprint": _setup_fingerprint(sig),
            "matched_entry_types": list(getattr(sig, "matched_entry_types", []) or []),
            "strategy_scores": dict(getattr(sig, "strategy_scores", {}) or {}),
            "strategy_component_scores": dict(getattr(sig, "strategy_component_scores", {}) or {}),
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
            "decision_audit": {
                "matched_strategies": list(getattr(sig, "matched_entry_types", []) or []),
                "primary_strategy": str(sig.entry_type),
                "strategy_competition": {
                    "matched_count": len(list(getattr(sig, "matched_entry_types", []) or [])),
                    "matched_strategies": list(getattr(sig, "matched_entry_types", []) or []),
                    "non_primary_strategies": [
                        str(x) for x in (getattr(sig, "matched_entry_types", []) or [])
                        if str(x) != str(sig.entry_type)
                    ],
                    "primary_selection": "highest_strategy_score_then_identity_then_specificity_then_entry_order",
                },
                "strategy_scores": dict(getattr(sig, "strategy_scores", {}) or {}),
                "score_ledger": {
                    str(k): {
                        "core_score": float((v or {}).get("core_score", 0.0) or 0.0),
                        "core_contribution_70pct": float((v or {}).get("core_contribution_70pct", 0.0) or 0.0),
                        "confirmation_score": float((v or {}).get("confirmation_score", 0.0) or 0.0),
                        "confirmation_contribution_30pct": float((v or {}).get("confirmation_contribution_30pct", 0.0) or 0.0),
                        "final_strategy_score": float((v or {}).get("final_strategy_score", 0.0) or 0.0),
                    } for k, v in (getattr(sig, "strategy_component_scores", {}) or {}).items()
                },
                "final_score_ledger": _score_ledger_for_signal(sig),
                "setup_fingerprint": _setup_fingerprint(sig),
                "setup_lifecycle": _setup_lifecycle(sig),
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
    # Outcome must be linked to the exact signal.
    # Never fall back to the latest pending signal for the same symbol,
    # because multiple pending signals can exist for one symbol.
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
            "matched_entry_types": target.get("matched_entry_types") or [target.get("entry_type")],
            "strategy_scores": target.get("strategy_scores") or {},
            "strategy_component_scores": target.get("strategy_component_scores") or {},
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



def _build_4h_from_60m(df: pd.DataFrame) -> pd.DataFrame | None:
    """حوّل شموع 60 دقيقة إلى شموع 4 ساعات فعلية من جلسة التداول النظامية.
    يتم تكوين كل شمعة 4س من أربع شموع 60د متتالية في نفس يوم التداول،
    مع تجاهل المجموعة الجزئية الأخيرة إذا لم تكتمل 4 شموع.
    """
    try:
        if df is None or df.empty:
            return None

        x = df.copy()
        x = x.sort_index()

        # توحيد أسماء الأعمدة إذا كانت MultiIndex.
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]

        needed = ["Open", "High", "Low", "Close"]
        if any(c not in x.columns for c in needed):
            return None

        # نحتاج فقط لشموع الجلسة النظامية؛ نستبعد أي pre/post-market
        # إن كانت موجودة في المصدر.
        if getattr(x.index, "tz", None) is not None:
            local_idx = x.index.tz_convert("America/New_York")
        else:
            local_idx = x.index

        session_mask = (
            (local_idx.time >= pd.Timestamp("09:30").time()) &
            (local_idx.time < pd.Timestamp("16:00").time())
        )
        x = x.loc[session_mask].copy()
        if x.empty:
            return None

        if getattr(x.index, "tz", None) is not None:
            session_day = x.index.tz_convert("America/New_York").date
        else:
            session_day = x.index.date

        # أربع شموع 60د متتالية = شمعة 4 ساعات.
        x["_session_day"] = session_day
        x["_bar_no"] = x.groupby("_session_day").cumcount()
        x["_group"] = x["_bar_no"] // 4

        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
        if "Volume" in x.columns:
            agg["Volume"] = "sum"

        g = x.groupby(["_session_day", "_group"], sort=True)
        counts = g["Close"].count()
        valid_groups = counts[counts >= 4].index

        y = g.agg(agg).loc[valid_groups]
        if y.empty:
            return None

        # استخدم نهاية آخر ساعة داخل كل مجموعة كزمن الشمعة 4س.
        end_times = x.groupby(["_session_day", "_group"], sort=True).apply(
            lambda z: z.index[-1]
        ).loc[valid_groups]
        y.index = pd.DatetimeIndex(end_times.values)

        y = y.sort_index()
        y = y[~y.index.duplicated(keep="last")]
        return y
    except Exception:
        return None


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
        bar_idx = -2 if is_us_regular_session(now_ny()) and len(today_d) >= 2 else -1
        b = today_d.iloc[bar_idx]
        o, h, l, c = map(float, (b["Open"], b["High"], b["Low"], b["Close"]))
        rng = max(h - l, 1e-9)
        body = abs(c - o) / rng
        close_pos = (c - l) / rng
        upper_wick = (h - max(o, c)) / rng
        prior_close = float(today_d["Close"].iloc[bar_idx - 1])

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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)
    return int(default)


DAILY_MARKET_RETRY_ATTEMPTS = 4
DAILY_MARKET_RETRY_DELAYS = (0.0, 0.5, 1.0, 1.5)
DAILY_POSITIVE_EMA50_BUFFER_PCT = 0.50


def _market_alignment(fetch_intraday) -> tuple[bool, str]:
    """
    Daily SPY/QQQ market regime with an explicit positive-below-VWAP state:
      قوي               = كلاهما فوق EMA20 و EMA20 >= EMA50
      إيجابي_تحت_VWAP  = كلاهما إيجابيان سعريًا لكن كلاهما تحت VWAP اليومي
      إيجابي            = كلاهما إيجابيان سعريًا، لكن ليسا كلاهما تحت VWAP
      مختلط             = أحدهما قوي/إيجابي والآخر ضعيف، أو حالات إيجابية غير متطابقة
      ضعيف              = كلاهما ضعيف
      غير مؤكد          = بيانات غير مكتملة
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
                    raise ValueError("market data is None")
                if not {"Close", "High", "Low", "Volume"}.issubset(d.columns):
                    raise ValueError("required OHLCV columns missing")

                c = pd.to_numeric(d["Close"], errors="coerce")
                h = pd.to_numeric(d["High"], errors="coerce")
                l = pd.to_numeric(d["Low"], errors="coerce")
                vol = pd.to_numeric(d["Volume"], errors="coerce")
                valid = pd.concat([c, h, l, vol], axis=1).dropna()
                if len(valid) < 50:
                    raise ValueError(f"insufficient daily OHLCV rows: {len(valid)} < 50")

                c = valid["Close"]
                e20 = float(_ema(c, 20).iloc[-1])
                e50 = float(_ema(c, 50).iloc[-1])
                p = float(c.iloc[-1])
                vwap_series = _vwap(valid)
                vwap = float(vwap_series.iloc[-1])

                if not all(np.isfinite(x) for x in (e20, e50, p, vwap)) or vwap <= 0:
                    raise ValueError("non-finite market values or VWAP")

                strong = bool(p >= e20 and e20 >= e50)
                positive = bool(
                    not strong
                    and p >= e50 * (1.0 - DAILY_POSITIVE_EMA50_BUFFER_PCT / 100.0)
                )

                if strong:
                    state = "داعم"
                elif positive and p < vwap:
                    state = "إيجابي_تحت_VWAP"
                elif positive:
                    state = "إيجابي"
                else:
                    state = "ضعيف"

                log.info(
                    "DAILY MARKET RESULT | %s | attempt %d/%d | close=%.4f | ema20=%.4f | ema50=%.4f | vwap20=%.4f | state=%s",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS,
                    p, e20, e50, vwap, state,
                )
                break

            except Exception as exc:
                log.warning(
                    "DAILY MARKET FETCH FAILED | %s | attempt %d/%d | %s",
                    sym, attempt_no, DAILY_MARKET_RETRY_ATTEMPTS, exc,
                )
                if attempt < DAILY_MARKET_RETRY_ATTEMPTS - 1:
                    delay = DAILY_MARKET_RETRY_DELAYS[
                        min(attempt + 1, len(DAILY_MARKET_RETRY_DELAYS) - 1)
                    ]
                    if delay > 0:
                        time_module.sleep(delay)

        states.append(state)
        log.info(
            "DAILY MARKET SYMBOL FINAL | %s | state=%s",
            sym, state or "غير متاح",
        )

    if any(x is None for x in states):
        return False, "بيانات SPY/QQQ غير مكتملة بعد إعادة المحاولة"

    if states[0] == "داعم" and states[1] == "داعم":
        log.info("DAILY MARKET FINAL | SPY=داعم | QQQ=داعم | ok=True | state=SPY+QQQ داعمان يوميًا")
        return True, "SPY+QQQ داعمان يوميًا"

    if states[0] == "ضعيف" and states[1] == "ضعيف":
        log.info("DAILY MARKET FINAL | SPY=ضعيف | QQQ=ضعيف | ok=False | state=SPY+QQQ ضعيفان يوميًا")
        return False, "SPY+QQQ ضعيفان يوميًا"

    if states[0] == "إيجابي_تحت_VWAP" and states[1] == "إيجابي_تحت_VWAP":
        log.info("DAILY MARKET FINAL | SPY/QQQ إيجابيان وتحت VWAP | ok=True")
        return True, "SPY+QQQ إيجابيان وتحت VWAP يوميًا"

    # Any other combination, including one positive benchmark above VWAP, is
    # deliberately treated as mixed. This keeps the daily policy to the five
    # explicit regimes: strong / positive-below-VWAP / mixed / weak / unknown.
    log.info(
        "DAILY MARKET FINAL | SPY=%s | QQQ=%s | ok=True | state=SPY/QQQ مختلطان يوميًا",
        states[0], states[1],
    )
    return True, "SPY/QQQ مختلطان يوميًا"

def get_daily_market_context() -> tuple[bool, str, str]:
    """Return the exact Daily market state used by the Daily engine.

    This is a read-only public wrapper for Main so display/gating does not
    use the separate generic market.py regime.
    Returns: (market_ok, market_state, market_condition).
    """
    try:
        from market_data import fetch_intraday
        ok, state = _market_alignment(fetch_intraday)
        condition = _daily_market_condition(state)
        return bool(ok), str(state), str(condition)
    except Exception as exc:
        log.warning("Daily market context unavailable: %s", exc)
        return False, "بيانات SPY/QQQ غير متاحة", "غير مؤكد"


def _daily_market_condition(market_state: str) -> str:
    """Map the benchmark state to the daily policy regime.

    This is deliberately separate from the internal adaptive ``market_regime``
    used for learning so the execution gate always reflects the actual
    SPY/QQQ condition detected before the scan.
    """
    state = str(market_state or "")
    if "داعمان" in state:
        return "قوي"
    if "إيجابيان وتحت VWAP" in state:
        return "إيجابي_تحت_VWAP"
    if "إيجابيان" in state:
        return "إيجابي"
    if "ضعيفان" in state:
        return "ضعيف"
    if "مختلطان" in state:
        return "مختلط"
    return "غير مؤكد"


def session_window_ok(dt=None) -> tuple[bool, str]:
    """Daily engine: market must be a real US trading session; no intraday first/last-20 restriction."""
    dt = dt or now_ny()
    if not is_us_regular_session(dt):
        return False, session_label(dt)
    return True, "نافذة يومية مسموحة"


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI with SMA seed and explicit 100/50 edge handling."""
    c = close.astype(float)
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    out = pd.Series(np.nan, index=c.index, dtype=float)
    if len(c) <= n:
        return out
    avg_up = up.iloc[1:n + 1].mean()
    avg_dn = dn.iloc[1:n + 1].mean()

    def _value(gain, loss):
        if loss == 0 and gain > 0:
            return 100.0
        if loss == 0 and gain == 0:
            return 50.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    out.iloc[n] = _value(avg_up, avg_dn)
    alpha = 1.0 / n
    for i in range(n + 1, len(c)):
        avg_up = ((n - 1) * avg_up + float(up.iloc[i])) / n
        avg_dn = ((n - 1) * avg_dn + float(dn.iloc[i])) / n
        out.iloc[i] = _value(avg_up, avg_dn)
    return out


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR(14): seed with SMA(TR, n), then Wilder RMA."""
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out = pd.Series(np.nan, index=tr.index, dtype=float)
    if len(tr) < n:
        return out
    first = tr.iloc[:n].mean()
    out.iloc[n - 1] = first
    alpha = 1.0 / n
    for i in range(n, len(tr)):
        out.iloc[i] = out.iloc[i - 1] + alpha * (tr.iloc[i] - out.iloc[i - 1])
    return out


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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)

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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)

    if not candidates:
        return 0.0, "هدف مخاطر احتياطي"
    return min(candidates, key=lambda x: x[0])


_QUOTE_CACHE: dict[str, tuple[datetime, dict]] = {}
QUOTE_CACHE_SECONDS = 30
def _quote_liquidity(symbol: str, price: float) -> dict:
    """يومي: Bid/Ask من Alpaca فقط؛ لا نستخدم Yahoo كبديل للتنفيذ اليومي."""
    now = datetime.now(timezone.utc)
    cached = _QUOTE_CACHE.get(symbol)
    if cached:
        cache_age = (now - cached[0]).total_seconds()
        cached_result = cached[1]
        cached_quote_age = float(cached_result.get("quote_age_min", float("inf")) or float("inf"))
        # Cache duration is only an optimization; the hard quote-freshness rule
        # remains <=2 minutes on every cache hit.
        effective_quote_age = cached_quote_age + (cache_age / 60.0)
        if cache_age < QUOTE_CACHE_SECONDS and effective_quote_age <= QUOTE_MAX_AGE_MIN:
            cached_result = dict(cached_result)
            cached_result["quote_age_min"] = effective_quote_age
            return cached_result
    result = {
        "ok": False,
        "spread_pct": 0.0,
        "slippage_pct": 0.0,
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
        if bid > 0 and ask > bid and px > 0 and result["quote_age_min"] <= QUOTE_MAX_AGE_MIN:
            mid = (bid + ask) / 2.0
            spread = (ask - bid) / mid * 100.0
            result["spread_pct"] = spread
            result["slippage_pct"] = (ask - mid) / mid * 100.0
            result["ok"] = spread <= HARD_MAX_SPREAD_PCT
            result["quote_source"] = "alpaca-" + str(q.get("feed") or "unknown")
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)
    _QUOTE_CACHE[symbol] = (now, result)
    return result


def _final_execution_snapshot(symbol: str, reference_price: float) -> dict:
    """لقطة تنفيذ نهائية مستقلة عن التحليل. لا تعيد حساب Stop/TP."""
    result = {
        "ok": False,
        "entry_price": 0.0,
        "spread_pct": float("inf"),
        "quote_age_min": float("inf"),
        "quote_source": "none",
    }
    try:
        from market_data import fetch_latest_quote, data_age_minutes
        q = fetch_latest_quote(symbol)
        bid = float(q.get("bid") or 0)
        ask = float(q.get("ask") or 0)
        if bid <= 0 or ask <= bid or float(reference_price or 0) <= 0:
            return result
        mid = (bid + ask) / 2.0
        ts = q.get("timestamp")
        if ts:
            qdf = pd.DataFrame({"Close": [mid]}, index=[pd.Timestamp(ts)])
            result["quote_age_min"] = float(data_age_minutes(qdf))
        if result["quote_age_min"] > QUOTE_MAX_AGE_MIN:
            return result
        spread = (ask - bid) / mid * 100.0
        result.update({
            "spread_pct": spread,
            "entry_price": round(mid, 4),
            "quote_source": "alpaca-" + str(q.get("feed") or "unknown"),
            "ok": spread <= HARD_MAX_SPREAD_PCT,
        })
    except Exception as exc:
        log.debug("DAILY FINAL EXECUTION SNAPSHOT FAILED | %s | %s", symbol, str(exc))
    return result


_DAILY_MARKET_RET_CACHE = {"ts": 0.0, "spy": None, "qqq": None}

def _daily_market_relative_returns(fetch_intraday) -> tuple[float | None, float | None]:
    """Return current daily % change for SPY/QQQ with a short cache.

    Used only by the daily weak-market exception. It does not alter the normal
    market gate; it provides the benchmark returns needed to measure whether a
    stock is materially outperforming a weak market.
    """
    now_ts = time_module.time()
    cached = _DAILY_MARKET_RET_CACHE
    if (now_ts - float(cached.get("ts", 0.0))) < 300.0 and cached.get("spy") is not None and cached.get("qqq") is not None:
        return float(cached["spy"]), float(cached["qqq"])

    vals = {}
    for sym in ("SPY", "QQQ"):
        try:
            d = fetch_intraday(sym, interval="1d", period="10d")
            c = pd.to_numeric(d["Close"], errors="coerce").dropna() if d is not None and "Close" in d.columns else pd.Series(dtype=float)
            if len(c) >= 2 and float(c.iloc[-2]) > 0:
                vals[sym] = (float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2]) * 100.0
            else:
                vals[sym] = None
        except Exception as exc:
            log.warning("DAILY RELATIVE MARKET FETCH FAILED | %s | %s", sym, str(exc))
            vals[sym] = None

    if vals.get("SPY") is not None and vals.get("QQQ") is not None:
        cached.update({"ts": now_ts, "spy": vals["SPY"], "qqq": vals["QQQ"]})
    return vals.get("SPY"), vals.get("QQQ")


def _daily_volume_ratio_time_of_day(symbol: str, fetch_intraday, now_ny) -> float:
    """Daily relative volume through the same session time.

    Compares today's cumulative 5-minute volume from the regular-session open
    through the current time with the average cumulative volume through that
    same time across prior sessions. Falls back to 1.0 when intraday history
    is unavailable so missing data cannot manufacture a weak/strong reading.
    """
    try:
        bars = fetch_intraday(symbol, interval="5m", period="30d")
        if bars is None or bars.empty or "Volume" not in bars.columns:
            return 1.0
        df = bars.copy().sort_index()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_ny = idx.tz_convert("America/New_York")
        df.index = idx_ny
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)

        now = pd.Timestamp(now_ny)
        if now.tzinfo is None:
            now = now.tz_localize("America/New_York")
        else:
            now = now.tz_convert("America/New_York")
        session_date = now.date()
        session_open = now.normalize() + pd.Timedelta(hours=9, minutes=30)
        if now < session_open:
            return 1.0

        # Use only regular-session bars through the current clock time.
        work = df[(df.index.time >= pd.Timestamp("09:30").time()) &
                  (df.index.time <= pd.Timestamp("16:00").time()) &
                  (df.index <= now)]
        if work.empty:
            return 1.0

        current = work[work.index.date == session_date]["Volume"].sum()
        if current <= 0:
            return 1.0

        cutoff = now.time()
        prior = []
        for day, group in work.groupby(work.index.date):
            if day >= session_date:
                continue
            g = group[group.index.time <= cutoff]
            if len(g) == 0:
                continue
            # Require a meaningful sample and use cumulative volume to this
            # exact time, not the full-day total.
            prior.append(float(g["Volume"].sum()))

        if not prior:
            return 1.0
        baseline = float(pd.Series(prior[-20:]).mean())
        return float(current / baseline) if baseline > 0 else 1.0
    except Exception as exc:
        log.debug("DAILY time-of-day volume ratio fallback | %s | %s", symbol, exc)
        return 1.0


def _aligned_relative_strength_metrics(
    stock_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    lookback: int = 5,
    persistence_bars: int = 5,
) -> tuple[float, float, float, float, bool]:
    """Return aligned stock-vs-SPY/QQQ performance using completed bars only.

    Returns (vs_spy_pct, vs_qqq_pct, persistence_pct, stock_return_pct, valid).
    Alignment is timestamp-based; no positional mixing of unrelated bars.
    """
    try:
        def _prep(df):
            if df is None or len(df) == 0 or "Close" not in df.columns:
                return None
            x = df[["Close"]].copy()
            idx = pd.to_datetime(x.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            else:
                idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
            x.index = idx
            x = x[~x.index.duplicated(keep="last")].sort_index()
            x["Close"] = pd.to_numeric(x["Close"], errors="coerce")
            return x.dropna(subset=["Close"])

        st = _prep(stock_df); sp = _prep(spy_df); qq = _prep(qqq_df)
        if st is None or sp is None or qq is None:
            return 0.0, 0.0, 0.0, 0.0, False
        x = st.rename(columns={"Close":"stock"}).join(
            sp.rename(columns={"Close":"spy"}), how="inner"
        ).join(qq.rename(columns={"Close":"qqq"}), how="inner").dropna()
        if len(x) < max(lookback + 1, persistence_bars + 1):
            return 0.0, 0.0, 0.0, 0.0, False
        x = x.iloc[:-1] if len(x) > 1 else x
        if len(x) < lookback + 1:
            return 0.0, 0.0, 0.0, 0.0, False
        sret = (float(x["stock"].iloc[-1]) / float(x["stock"].iloc[-1-lookback]) - 1.0) * 100.0
        pret = (float(x["spy"].iloc[-1]) / float(x["spy"].iloc[-1-lookback]) - 1.0) * 100.0
        qret = (float(x["qqq"].iloc[-1]) / float(x["qqq"].iloc[-1-lookback]) - 1.0) * 100.0
        edge = x[["stock","spy","qqq"]].pct_change().dropna() * 100.0
        edge = edge.assign(rs=edge["stock"] - (edge["spy"] + edge["qqq"]) / 2.0)
        edge = edge.tail(persistence_bars)
        persistence = float((edge["rs"] > 0.0).mean() * 100.0) if len(edge) >= 2 else 0.0
        return sret - pret, sret - qret, persistence, sret, True
    except Exception as exc:
        log.debug("Relative-strength alignment fallback: %s", exc)
        return 0.0, 0.0, 0.0, 0.0, False


def _detect_professional_new_setups(
    df: pd.DataFrame,
    closed_idx: int,
    prev_close: float,
    session_open: float,
    atr: float,
    setup_profile: str = "intraday",
) -> dict:
    """Detect the three additional long setups from completed candles only.

    Daily uses completed 60m session structure with shorter intraday-session windows;
    Intraday uses completed 5m structure. Both remain completed-bar only.
    """
    out = {
        "gap_setup": False, "gap_mode": "", "gap_pct": 0.0, "gap_mid": 0.0,
        "gap_pullback_high": 0.0,
        "failed_breakdown_reclaim": False, "failed_breakdown_support": 0.0,
        "failed_breakdown_low": 0.0, "failed_breakdown_bars": 0,
        "failed_breakdown_depth_atr": 0.0, "failed_breakdown_touches": 0,
        "rs_pullback": False, "rs_higher_low": False, "rs_reference_gain": 0.0,
        "rs_pullback_pct": 0.0, "rs_pullback_high": 0.0,
    }
    try:
        _profile = str(setup_profile or "intraday").lower()
        _min_bars = 5 if _profile == "daily" else 6
        if df is None:
            return out
        _closed_pos = len(df) + int(closed_idx) if int(closed_idx) < 0 else int(closed_idx)
        if _closed_pos < 0:
            return out
        closed = df.iloc[:_closed_pos + 1].copy().dropna(subset=["Open","High","Low","Close"])
        if len(closed) < _min_bars:
            return out
        if len(closed) < _min_bars or prev_close <= 0 or session_open <= 0:
            return out
        if atr <= 0:
            return out
        last = closed.iloc[-1]
        gap_pct = (session_open - prev_close) / prev_close * 100.0
        out["gap_pct"] = gap_pct
        # 17 — bullish Gap Continuation / Gap Reclaim.
        if gap_pct >= 2.0:
            gap_abs = session_open - prev_close
            gap_mid = prev_close + gap_abs * 0.50
            out["gap_mid"] = gap_mid
            opening_phase = closed.iloc[:min(3, len(closed))]
            initial_hold = float(opening_phase["Close"].min()) >= gap_mid
            open_hold = bool((closed["Close"].astype(float) >= session_open * 0.997).all())
            pre = closed.iloc[:-1]
            pullback_high = float(pre["High"].iloc[1:].max()) if len(pre) >= 3 else 0.0
            pullback_low = float(pre["Low"].iloc[1:].min()) if len(pre) >= 3 else session_open
            had_pullback = bool(pullback_high > 0 and pullback_low < pullback_high * 0.997 and pullback_low >= session_open * 0.997)
            continuation = bool(initial_hold and open_hold and had_pullback and float(last["Close"]) >= pullback_high * 1.001)
            lost_upper = bool(len(closed) >= 4 and (closed["Close"].astype(float).iloc[2:-1] <= gap_mid).any())
            no_invalidation = bool(float(closed["Close"].min()) >= prev_close * 0.998)
            reclaim = bool(lost_upper and no_invalidation and float(last["Close"]) >= session_open * 1.001)
            if continuation:
                out.update(gap_setup=True, gap_mode="continuation", gap_pullback_high=pullback_high)
            elif reclaim:
                out.update(gap_setup=True, gap_mode="reclaim", gap_pullback_high=float(closed["High"].iloc[2:-1].max()))
        # 18 — actual close below a repeatedly tested support, then reclaim.
        # Support is formed BEFORE the breakdown window; the breakdown bars
        # themselves must not be allowed to redefine the support level.
        # Use a compact, session-feasible structure: 3 bars form support,
        # followed by up to 3 bars for the breakdown/reclaim sequence.
        # This prevents the Daily 60m version from requiring more bars than
        # a regular US session can provide.
        base = closed.iloc[-6:-3]
        lows = base["Low"].astype(float)
        support = 0.0; touches_best = 0
        # Choose a pre-formed support with the greatest number of touches.
        # This avoids letting the breakdown candles redefine the level.
        for level in sorted({round(float(x), 6) for x in lows.tolist() if float(x) > 0}, reverse=True):
            touches = int(((lows >= level * 0.996) & (lows <= level * 1.004)).sum())
            if touches >= 2 and touches > touches_best:
                support, touches_best = level, touches
        if support > 0 and touches_best >= 2 and len(closed) >= 6:
            bw = closed.iloc[-3:]
            pre_reclaim = bw.iloc[:-1]
            bi = next((i for i in range(len(pre_reclaim)) if float(pre_reclaim["Close"].iloc[i]) <= support * 0.998), None)
            current_reclaims = bool(float(last["Close"]) >= support * 1.001)
            if bi is not None and current_reclaims:
                after = bw.iloc[bi:]
                failure_low = float(after["Low"].min())
                depth_atr = max(0.0, (support - failure_low) / max(atr, 1e-9))
                bars = len(after)
                ok = bool(1 <= bars <= 5 and depth_atr <= 2.0)
                out.update(failed_breakdown_reclaim=ok, failed_breakdown_support=support,
                           failed_breakdown_low=failure_low, failed_breakdown_bars=bars,
                           failed_breakdown_depth_atr=depth_atr, failed_breakdown_touches=touches_best)
        # 19 — price-action portion of Relative Strength Pullback.
        # Daily: 3-bar impulse + 1-bar pullback + current trigger.
        # Intraday: 4-bar impulse + 1-bar pullback + current trigger.
        _rs_impulse = 3 if _profile == "daily" else 4
        _rs_pullback = 1
        _rs_total = _rs_impulse + _rs_pullback + 1
        prior_move = closed.iloc[-_rs_total:-(1 + _rs_pullback)]
        pullback = closed.iloc[-(1 + _rs_pullback):-1]
        if len(prior_move) >= _rs_impulse and len(pullback) >= _rs_pullback:
            ref_open = float(prior_move["Open"].iloc[0]); ref_high = float(prior_move["High"].max())
            ref_gain = (ref_high - ref_open) / max(ref_open, 1e-9) * 100.0
            ref_range = max(ref_high - float(prior_move["Low"].min()), 1e-9)
            pb_low = float(pullback["Low"].min()); pb_high = float(pullback["High"].max())
            pb_pct = (ref_high - pb_low) / ref_range * 100.0
            held_hl = pb_low > float(prior_move["Low"].min()) * 0.998
            trigger = float(last["Close"]) >= pb_high * 1.001
            out.update(rs_reference_gain=ref_gain, rs_pullback_pct=pb_pct, rs_pullback_high=pb_high,
                       rs_higher_low=held_hl,
                       rs_pullback=bool(ref_gain >= 1.0 and 0.0 < pb_pct <= 50.0 and held_hl and trigger))
    except Exception as exc:
        log.warning("DAILY professional setup detection failed: %s", exc)
        return out
    return out


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
    # During the regular session the latest daily bar is still forming.
    # Structural breakout/continuation triggers use the latest CLOSED candle.
    closed_idx = -2 if is_us_regular_session(now_ny()) and len(today_d) >= 2 else -1
    closed_prev_idx = closed_idx - 1 if abs(closed_idx) <= len(today_d) - 1 else -2
    try:
        h4_60m = fetch_intraday(symbol, interval="60m", period="60d")
        ok_h4, h4_age_min = intraday_data_fresh(h4_60m, "60m", 240)
        if h4_60m is None:
            _daily_data_audit_record(symbol, "60m:h4_missing")
        elif len(h4_60m) < 30:
            _daily_data_audit_record(symbol, "60m:h4_bars<30")
        if not ok_h4:
            _daily_data_audit_record(symbol, f"60m:h4_stale(age={h4_age_min:.1f}m>240m)")
            h4 = None
        else:
            # مهم: المصدر يعطينا 60د؛ نحوله فعليًا إلى 4س قبل
            # تمريره إلى _m15_confirmation، حتى لا تُحسب مؤشرات
            # "4H" على شموع 60د.
            h4 = _build_4h_from_60m(h4_60m)
            if h4 is None or len(h4) < 30:
                _daily_data_audit_record(symbol, "4h:h4_bars<30_after_resample")
                h4 = None
    except Exception as exc:
        _daily_data_audit_record(symbol, "60m:h4_fetch_or_resample_exception")
        h4 = None
        log.warning("DAILY 4H DATA FAILED | %s | %s", symbol, str(exc))

    # Compatibility aliases keep the proven setup intelligence readable.
    weekly = weekly.copy()
    daily = daily.copy()
    for _tf_name, _df in (("weekly", weekly), ("daily", daily)):
        try:
            _ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in _df.columns]
            if _ohlc_cols and _df[_ohlc_cols].tail(3).isna().any().any():
                _daily_data_audit_record(symbol, f"{_tf_name}:nan_ohlc_observed")
        except Exception:
            pass

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

    # Time-of-day relative volume: compare cumulative volume through the
    # current session time with the same point in prior sessions.
    vol_ratio = _daily_volume_ratio_time_of_day(symbol, fetch_intraday, now_ny)
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
    _closed_d = today_d.iloc[:closed_idx + 1].copy()
    _closed_c5 = _closed_d["Close"].astype(float)
    e5_closed = float(_ema(_closed_c5, 20).iloc[-1]) if len(_closed_c5) else e5
    vwap_closed_s = _vwap(_closed_d)
    vwap_last_closed = float(vwap_closed_s.iloc[-1]) if len(vwap_closed_s) and pd.notna(vwap_closed_s.iloc[-1]) else vwap_last
    last_green = float(today_d["Close"].iloc[closed_idx]) >= float(today_d["Open"].iloc[closed_idx])
    mom = (price - float(c5.iloc[-6])) / float(c5.iloc[-6]) * 100 if len(c5) >= 6 else 0.0
    live_ok = price >= e5 * 0.998 and above_vwap and (last_green or mom > 0.05) and r5 < 78
    # AUDIT ONLY: decompose the exact live gate without changing its logic.
    live_gate_reasons: list[str] = []
    if price < e5 * 0.998:
        live_gate_reasons.append("below_ema20_gate")
    if not above_vwap:
        live_gate_reasons.append("below_vwap_gate")
    if (not last_green) and mom <= 0.05:
        live_gate_reasons.append("last_candle_not_green")
        live_gate_reasons.append("momentum<=0.05")
    if r5 >= 78:
        live_gate_reasons.append("rsi5>=78")

    h4_state, h4_points = _m15_confirmation(h4, price)
    news_state, news_title, news_source = _classify_news(symbol)
    market_ok, market_state = market_context if market_context is not None else _market_alignment(fetch_intraday)
    market_condition = _daily_market_condition(market_state)
    log.info(
        "DAILY MARKET REGIME | condition=%s | state=%s | market_ok=%s",
        market_condition, market_state, market_ok,
    )

    spy_daily_pct = qqq_daily_pct = None
    if market_condition == "ضعيف":
        spy_daily_pct, qqq_daily_pct = _daily_market_relative_returns(fetch_intraday)
    market_avg_pct = ((spy_daily_pct + qqq_daily_pct) / 2.0) if spy_daily_pct is not None and qqq_daily_pct is not None else None
    relative_strength_pct = (change_pct - market_avg_pct) if market_avg_pct is not None else None

    # Weak market: allow only stocks showing genuine relative strength.
    relative_strength_ok = bool(
        market_condition == "ضعيف"
        and change_pct >= 0.75
        and relative_strength_pct is not None
        and relative_strength_pct >= 1.25
    )
    chop = _chop_filter(today_d, price, vwap_last)

    session_high = float(today_d["High"].max())
    drop = (session_high - price) / session_high * 100 if session_high else 0
    dump = drop >= 2.5 and change_pct <= -1.2

    h_win = weekly.tail(20)
    level_high = float(h_win["High"].iloc[:-1].max()) if len(h_win) > 3 else session_high
    closed_breakout_close = float(today_d["Close"].iloc[closed_idx])
    was_below = float(h_win["Close"].iloc[-3]) < level_high * 0.998 if len(h_win) >= 3 else False
    breakout_now = closed_breakout_close >= level_high * 1.001 and was_below
    # A real retest requires a prior close above the broken resistance,
    # followed by a return toward that same level. A mere historical touch is
    # not enough to label the setup as Retest.
    break_window = today_d.iloc[max(0, closed_idx - 5):closed_idx]
    prior_break = bool(
        len(break_window) > 0
        and (break_window["Close"].astype(float) >= level_high * 1.001).any()
    )
    retest_window = today_d.iloc[max(0, closed_idx - 3):closed_idx] if "closed_idx" in locals() else today_d.tail(4)
    retest_touch = bool(
        len(retest_window) > 0
        and (retest_window["Low"].astype(float) <= level_high * 1.007).any()
        and (retest_window["High"].astype(float) >= level_high * 0.993).any()
    )
    near_level = retest_touch
    ext_tmp = (price - e20) / e20 * 100 if e20 else 0.0

    failed = (
        float(today_d["High"].max()) >= level_high * 1.001
        and price < level_high * 0.997
        and not above_vwap
    ) or (dump and not above_vwap)
    # Latest completed candle close; define before any strategy uses it.
    closed_close = float(today_d["Close"].iloc[closed_idx])

    retest = prior_break and near_level and closed_close >= level_high * 0.997

    # 1) VWAP Bounce/Reclaim: رجوع منظم إلى VWAP ثم استعادة المستوى.
    recent4 = today_d.iloc[max(0, closed_idx - 4):closed_idx]
    vwap_touch = False
    try:
        vwap_touch = bool((recent4["Low"].astype(float) <= vwap_last * 1.006).any())
    except Exception:
        vwap_touch = False
    vwap_bounce = bool(vwap_touch and closed_close >= vwap_last_closed * 1.001)

    # 2) EMA20 Pullback: ترند صاعد + تصحيح صحي إلى EMA20 + استعادة.
    ema_touch = False
    try:
        ema_touch = bool((recent4["Low"].astype(float) <= e5 * 1.006).any())
    except Exception:
        ema_touch = False
    ema_pullback = bool(ema_touch and closed_close >= e5_closed * 1.001)

    # 3) Liquidity Sweep + Reclaim: كسر قاع قريب ثم استعادة المستوى بسرعة.
    support_level = 0.0
    liquidity_sweep = False
    try:
        support_window = today_d["Low"].astype(float).iloc[-10:-2]
        if len(support_window) >= 5:
            support_level = float(support_window.min())
            recent3 = today_d.iloc[max(0, closed_idx - 3):closed_idx]
            swept = len(recent3) > 0 and (recent3["Low"].astype(float) < support_level * 0.998).any()
            reclaimed = closed_close >= support_level * 1.002
            liquidity_sweep = bool(swept and reclaimed)
    except Exception:
        liquidity_sweep = False

    # 4) Liquidity Sweep + Displacement: سحب سيولة يتبعه اندفاع سعري واضح.
    liquidity_displacement = False
    try:
        if len(today_d) >= 8 and support_level > 0:
            cur = today_d.iloc[closed_idx]
            cur_pos = len(today_d) + closed_idx
            prev3 = today_d.iloc[max(0, cur_pos - 3):cur_pos]
            cur_o, cur_c = float(cur["Open"]), float(cur["Close"])
            cur_h, cur_l = float(cur["High"]), float(cur["Low"])
            cur_range = max(cur_h - cur_l, price * 0.0001)
            cur_body = abs(cur_c - cur_o)
            close_pos = (cur_c - cur_l) / cur_range
            prior_ranges = (prev3["High"].astype(float) - prev3["Low"].astype(float)).clip(lower=0)
            med_range = float(prior_ranges.median()) if len(prior_ranges) else 0.0
            cur_pos = len(today_d) + closed_idx
            sweep_start = max(0, cur_pos - 3)
            swept = bool((today_d["Low"].astype(float).iloc[sweep_start:cur_pos] < support_level * 0.998).any())
            reclaimed = cur_c >= support_level * 1.002
            # Core displacement is price structure only. Volume is a
            # confirmation factor and is scored separately in the 30% block.
            displacement = bool(
                cur_c > cur_o and cur_body / cur_range >= 0.55 and close_pos >= 0.75
                and (med_range <= 0 or cur_range >= med_range * 1.35)
            )
            liquidity_displacement = bool(swept and reclaimed and displacement)
    except Exception:
        liquidity_displacement = False

    # Daily timeframe adaptation of ORB: first 3 completed sessions of the current
    # month form the opening range. The current session is excluded from the range.
    month_d = daily[daily.index.to_period("M") == daily.index[-1].to_period("M")] if isinstance(daily.index, pd.DatetimeIndex) else daily.tail(22)
    completed_month = month_d.iloc[:-1] if len(month_d) >= 2 else month_d.iloc[0:0]
    opening_range_high = float(completed_month["High"].head(3).max()) if len(completed_month) >= 3 else 0.0
    opening_range_low = float(completed_month["Low"].head(3).min()) if len(completed_month) >= 3 else 0.0

    # 5) Opening Range Breakout (ORB): monthly opening-range breakout on daily timeframe.
    orb_high = opening_range_high
    orb_breakout = False
    try:
        closed_close = float(today_d["Close"].iloc[closed_idx])
        prior_close = float(today_d["Close"].iloc[closed_prev_idx])
        prior_orb = bool(orb_high > 0 and prior_close < orb_high * 1.001)
        orb_breakout = bool(orb_high > 0 and closed_close >= orb_high * 1.001 and prior_orb)
    except Exception:
        orb_breakout = False

    # 5) Momentum Continuation: استمرار دفعة صاعدة بدون مطاردة اختراق ضعيف.
    momentum_continuation = False
    try:
        cur_pos = len(today_d) + closed_idx
        if cur_pos >= 5:
            # True continuation = impulse -> controlled pause/pullback -> resume.
            impulse = today_d.iloc[cur_pos - 5:cur_pos - 3]
            pause = today_d.iloc[cur_pos - 3:cur_pos - 1]
            resume = today_d.iloc[cur_pos - 1:cur_pos].iloc[0]

            impulse_open = float(impulse["Open"].iloc[0])
            impulse_close = float(impulse["Close"].iloc[-1])
            impulse_high = float(impulse["High"].max())
            impulse_low = float(impulse["Low"].min())
            impulse_gain = (impulse_close - impulse_open) / max(impulse_open, 1e-9) * 100
            impulse_range_pct = (impulse_high - impulse_low) / max(price, 1e-9) * 100

            pause_high = float(pause["High"].max())
            pause_low = float(pause["Low"].min())
            pause_close = float(pause["Close"].iloc[-1])
            pause_range_pct = (pause_high - pause_low) / max(price, 1e-9) * 100
            pause_hold = pause_low >= impulse_close * 0.995

            resume_open = float(resume["Open"])
            resume_close = float(resume["Close"])
            resume_green = resume_close >= resume_open
            resume_above_pause = resume_close >= pause_close * 1.001
            momentum_continuation = bool(
                impulse_gain >= 0.35
                and impulse_range_pct > 0
                and pause_range_pct <= max(1.50, impulse_range_pct * 0.90)
                and pause_hold
                and resume_green
                and resume_above_pause
                and mom > 0.08
            )
    except Exception:
        momentum_continuation = False

    # 6) Compression → Expansion: ضغط سعري ثم توسع مدعوم بالحجم.
    compression_expansion = False
    try:
        if len(today_d) >= 10:
            cur_pos = len(today_d) + closed_idx
            prev = today_d.iloc[max(0, cur_pos - 8):cur_pos]
            cur = today_d.iloc[closed_idx]
            prev_ranges = (prev["High"].astype(float) - prev["Low"].astype(float)).clip(lower=0)
            cur_range = max(float(cur["High"]) - float(cur["Low"]), price * 0.0001)
            med_range = float(prev_ranges.median()) if len(prev_ranges) else 0.0
            comp_range = float(prev["High"].max() - prev["Low"].min())
            comp_width_pct = comp_range / max(price, 1e-9) * 100
            cur_body = abs(float(cur["Close"]) - float(cur["Open"]))
            cur_pos = (float(cur["Close"]) - float(cur["Low"])) / cur_range
            expansion = cur_range >= max(med_range * 1.35, price * 0.003)
            compression = comp_width_pct <= 2.2 and med_range > 0
            comp_high = float(prev["High"].max()) if len(prev) else 0.0
            breakout_from_compression = bool(
                comp_high > 0 and float(cur["Close"]) >= comp_high * 1.001
            )
            # Core = compression -> expansion -> breakout.
            # Volume/VWAP/open/trend/HTF/market/extension remain confirmations
            # or final safety gates and must not prevent Core matching.
            compression_expansion = bool(
                compression and expansion and breakout_from_compression
                and float(cur["Close"]) > float(cur["Open"])
                and cur_pos >= 0.70
                and cur_body / cur_range >= 0.45
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
    except Exception as exc:
        log.debug("DAILY non-critical fallback exception: %s", exc)

    key_level_near = any(
        lvl > 0 and abs(price - lvl) / max(price, 1e-9) * 100 <= 0.60
        for lvl in (prev_day_high, prev_day_low, prev_close_level, orb_high, level_high)
    )



    # 5) Bull Flag: tight bullish consolidation after an impulsive move,
    # then a clean continuation trigger. Distinct from compression-expansion:
    # the prior leg must already be clearly bullish and the pullback must stay controlled.
    bull_flag = False
    try:
        if len(today_d) >= 8:
            impulse = today_d.iloc[-8:-4]
            flag = today_d.iloc[-4:-1]
            impulse_open = float(impulse["Open"].iloc[0])
            impulse_high = float(impulse["High"].max())
            impulse_gain = (impulse_high - impulse_open) / max(impulse_open, 1e-9) * 100
            flag_high = float(flag["High"].max())
            flag_low = float(flag["Low"].min())
            impulse_range = max(impulse_high - float(impulse["Low"].min()), price * 0.001)
            flag_retrace = (impulse_high - flag_low) / impulse_range * 100
            flag_range = (flag_high - flag_low) / max(flag_high, 1e-9) * 100
            flag_tight = flag_range <= 2.0 and flag_retrace <= 50.0
            breakout_flag = closed_close >= flag_high * 1.001
            bull_flag = bool(impulse_gain >= 1.0 and flag_tight and breakout_flag)
    except Exception:
        bull_flag = False

    # 6) Resistance Reclaim: a previously established resistance is lost,
    # then reclaimed with confirmation. This is different from HOD reclaim:
    # the level can be an daily structural resistance, not necessarily today's high.
    resistance_reclaim = False
    reclaim_level = 0.0
    try:
        if len(today_d) >= 8:
            prior = today_d.iloc[-8:-2]
            reclaim_level = float(prior["High"].quantile(0.80))
            cur_pos = len(today_d) + closed_idx if closed_idx < 0 else closed_idx
            prev_pos = cur_pos - 1
            prev_close = float(today_d["Close"].iloc[prev_pos]) if prev_pos >= 0 else 0.0
            resistance_was_lost = prev_pos >= 0 and prev_close < reclaim_level * 0.999
            reclaimed = closed_close >= reclaim_level * 1.001
            touches = int((prior["High"] >= reclaim_level * 0.995).sum())
            resistance_reclaim = bool(touches >= 2 and resistance_was_lost and reclaimed)
    except Exception:
        resistance_reclaim = False

    # 7) Opening Drive → Pullback: use the first completed 60m bars of the
    # current regular session as the opening drive, then require a controlled
    # pullback and a CLOSED 60m reclaim. Daily remains the primary engine;
    # 60m data is used only to define this intraday structure precisely.
    opening_drive_pullback = False
    drive_level = 0.0
    try:
        if h4_60m is not None and len(h4_60m) >= 4:
            idx60 = pd.DatetimeIndex(h4_60m.index)
            idx60_ny = idx60.tz_convert("America/New_York") if idx60.tz is not None else idx60.tz_localize("America/New_York")
            session_mask = (
                (idx60_ny.date == now_ny().date())
                & (idx60_ny.time >= REGULAR_OPEN)
                & (idx60_ny.time <= REGULAR_CLOSE)
            )
            session60 = h4_60m.loc[session_mask].copy()
            session_closed60 = session60.iloc[:-1] if is_us_regular_session(now_ny()) and len(session60) >= 2 else session60
            if len(session_closed60) >= 3:
                drive_bars = session_closed60.iloc[:2]
                later_bars = session_closed60.iloc[2:]
                drive_open = float(drive_bars["Open"].iloc[0])
                drive_high = float(drive_bars["High"].max())
                drive_return = (drive_high - drive_open) / max(drive_open, 1e-9) * 100
                drive_level = drive_high
                latest60 = session_closed60.iloc[-1]
                prior60 = session_closed60.iloc[:-1]
                recent_low = float(prior60.tail(3)["Low"].min()) if len(prior60) else float(latest60["Low"])
                pullback_from_high = (drive_high - recent_low) / max(drive_high, 1e-9) * 100
                reclaim_drive = float(latest60["Close"]) >= drive_high * 0.999
                controlled_pullback = 0.50 <= pullback_from_high <= 8.0
                not_chasing = ext_tmp <= 6.0
                opening_drive_pullback = bool(
                    len(later_bars) > 0 and drive_return >= 2.0
                    and controlled_pullback and reclaim_drive and not_chasing
                )
    except Exception:
        opening_drive_pullback = False

    # 8) Period-High Reclaim: استعادة قمة اليوم.
    # HOD is derived from the current regular-session 60m structure, not from
    # an arbitrary six-day Daily window. The reclaim is evaluated on a CLOSED 60m bar.
    hod_reclaim = False
    hod_level = 0.0
    try:
        if h4_60m is not None and len(h4_60m) >= 4:
            idx60 = pd.DatetimeIndex(h4_60m.index)
            idx60_ny = idx60.tz_convert("America/New_York") if idx60.tz is not None else idx60.tz_localize("America/New_York")
            session_mask = (
                (idx60_ny.date == now_ny().date())
                & (idx60_ny.time >= REGULAR_OPEN)
                & (idx60_ny.time <= REGULAR_CLOSE)
            )
            session60 = h4_60m.loc[session_mask].copy()
            session_closed60 = session60.iloc[:-1] if is_us_regular_session(now_ny()) and len(session60) >= 2 else session60
            if len(session_closed60) >= 3:
                prior60 = session_closed60.iloc[:-1]
                latest60 = session_closed60.iloc[-1]
                hod_level = float(prior60["High"].max()) if len(prior60) else 0.0
                recent_pullback = prior60.tail(3)
                pullback_below_hod = bool(
                    hod_level > 0 and len(recent_pullback) > 0
                    and (recent_pullback["Close"].astype(float) < hod_level * 0.999).any()
                )
                reclaimed_hod = bool(hod_level > 0 and float(latest60["Close"]) >= hod_level * 1.001)
                hod_reclaim = bool(pullback_below_hod and reclaimed_hod)
    except Exception:
        hod_reclaim = False

    # 9) ORB Failed Breakout -> Reclaim: مخصوص لفشل اختراق نطاق الافتتاح ثم استعادته.
    # مختلف عن سحب السيولة: المستوى هنا ORB High فقط، مع شرط اختراق سابق ثم فشل ثم reclaim.
    orb_failed_reclaim = False
    try:
        if orb_high > 0:
            _month_completed = completed_month.copy()
            _post_orb = _month_completed.iloc[3:] if len(_month_completed) >= 4 else _month_completed.iloc[0:0]
            post_orb = _post_orb
            broke = bool(len(post_orb) and (post_orb["High"].astype(float) >= orb_high * 1.002).any())
            failure = bool(len(post_orb) and (post_orb["Close"].astype(float) <= orb_high * 0.998).any())
            reclaim = closed_close >= orb_high * 1.001
            orb_failed_reclaim = bool(broke and failure and reclaim)
    except Exception:
        orb_failed_reclaim = False

    # 10) ABC Pullback / 3-Wave Continuation: دفعة A، تصحيح B مضبوط، ثم C.
    # لا يكفي لمس EMA20؛ يجب أن تكون بنية A/B/C واضحة.
    abc_continuation = False
    try:
        if len(today_d) >= 9:
            closed_today_d = today_d.iloc[:closed_idx + 1]
            a = closed_today_d.iloc[-9:-6]
            b = closed_today_d.iloc[-6:-3]
            c = closed_today_d.iloc[-3:]
            a_open = float(a["Open"].iloc[0])
            a_high = float(a["High"].max())
            a_gain = (a_high - a_open) / max(a_open, 1e-9) * 100
            b_high = float(b["High"].max())
            b_low = float(b["Low"].min())
            b_retrace = (a_high - b_low) / max(a_high - a_open, price * 0.001) * 100
            c_high = float(c["High"].max())
            c_last_green = float(c["Close"].iloc[-1]) >= float(c["Open"].iloc[-1])
            c_close = float(c["Close"].iloc[-1])
            c_break = c_close >= a_high * 1.001
            abc_continuation = bool(
                a_gain >= 0.70
                and 20.0 <= b_retrace <= 65.0
                and c_break and price >= a_high * 0.999
                and c_last_green
            )
    except Exception:
        abc_continuation = False

    # 17/18/19 — additional professional setup cores use completed 60m session structure.
    new_setup_df = None
    try:
        if h4_60m is not None and len(h4_60m) >= 12:
            _idx60 = pd.DatetimeIndex(h4_60m.index)
            _idx60_ny = _idx60.tz_convert("America/New_York") if _idx60.tz is not None else _idx60.tz_localize("America/New_York")
            _mask60 = ((_idx60_ny.date == now_ny().date()) & (_idx60_ny.time >= REGULAR_OPEN) & (_idx60_ny.time <= REGULAR_CLOSE))
            _session60 = h4_60m.loc[_mask60].copy()
            new_setup_df = _session60.iloc[:-1] if is_us_regular_session(now_ny()) and len(_session60) >= 2 else _session60
    except Exception as exc:
        log.debug("DAILY new-strategy session fallback: %s", exc)
    if new_setup_df is None or len(new_setup_df) < 5:
        new_setup_df = today_d.iloc[0:0].copy()
    # New strategies use their actual 60m structure, so their ATR must also be
    # 60m-based (not Daily ATR). Read the last completed 60m bar from the full
    # history so ATR(14) is already seeded when the session setup appears.
    _daily_setup_atr = 0.0
    try:
        _atr60 = _atr(h4_60m, 14) if h4_60m is not None else None
        if _atr60 is not None and new_setup_df is not None and len(new_setup_df):
            _last60_idx = new_setup_df.index[-1]
            if _last60_idx in _atr60.index and pd.notna(_atr60.loc[_last60_idx]):
                _daily_setup_atr = float(_atr60.loc[_last60_idx])
    except Exception:
        _daily_setup_atr = 0.0
    new_setups = _detect_professional_new_setups(new_setup_df, len(new_setup_df)-1, prev_close, day_open, _daily_setup_atr, setup_profile="daily")
    gap_setup = bool(new_setups["gap_setup"]); gap_mode = str(new_setups["gap_mode"] or ""); gap_pct = float(new_setups["gap_pct"] or 0.0); gap_mid = float(new_setups["gap_mid"] or 0.0); gap_pullback_high = float(new_setups["gap_pullback_high"] or 0.0)
    failed_breakdown_reclaim = bool(new_setups["failed_breakdown_reclaim"]); failed_breakdown_support = float(new_setups["failed_breakdown_support"] or 0.0); failed_breakdown_low = float(new_setups["failed_breakdown_low"] or 0.0); failed_breakdown_bars = int(new_setups["failed_breakdown_bars"] or 0); failed_breakdown_depth_atr = float(new_setups["failed_breakdown_depth_atr"] or 0.0); failed_breakdown_touches = int(new_setups["failed_breakdown_touches"] or 0)
    rs_pullback = bool(new_setups["rs_pullback"]); rs_higher_low = bool(new_setups["rs_higher_low"]); rs_reference_gain = float(new_setups["rs_reference_gain"] or 0.0); rs_pullback_pct = float(new_setups["rs_pullback_pct"] or 0.0); rs_pullback_high = float(new_setups["rs_pullback_high"] or 0.0)
    rs_strategy_ok = False; rs_vs_spy = rs_vs_qqq = 0.0; rs_persistence = 0.0
    if rs_pullback:
        try:
            spy60 = fetch_intraday("SPY", interval="60m", period="10d")
            qqq60 = fetch_intraday("QQQ", interval="60m", period="10d")
            rs_vs_spy, rs_vs_qqq, rs_persistence, _rs_stock_return, _rs_valid = _aligned_relative_strength_metrics(
                new_setup_df, spy60, qqq60, lookback=3, persistence_bars=3
            )
            rs_strategy_ok = bool(_rs_valid and rs_vs_spy >= 1.0 and rs_vs_qqq >= 1.0 and rs_persistence >= 60.0)
        except Exception as exc:
            log.debug("DAILY RS strategy benchmark fallback: %s", exc)
    rs_pullback=bool(rs_pullback and rs_strategy_ok)

    # طبقة Confluence: ليست نوع دخول جديداً، بل Bonus عند اجتماع VWAP + H1 + عدة مستويات.
    confluence_levels = []
    for lvl, label in ((vwap_last, "VWAP"), (e5, "EMA20"), (orb_high, "ORB"),
                       (level_high, "H1-Level"), (prev_close_level, "PrevClose")):
        try:
            if lvl > 0 and abs(price - float(lvl)) / max(price, 1e-9) * 100 <= 0.60:
                confluence_levels.append(label)
        except Exception as exc:
            log.debug("DAILY non-critical nested fallback exception: %s", exc)
    multi_level_confluence = len(set(confluence_levels)) >= 3
    vwap_weekly_confluence = bool(
        trend_up and above_vwap and e20 > e50
        and h4_state == "داعم" and not failed
    )

    # Early Entry is itself a structural setup, not a score fallback:
    # pre-breakout compression/holding under a meaningful resistance, with
    # improving price action and no already-confirmed strategy trigger.
    early = False
    try:
        recent3 = today_d.iloc[:closed_idx + 1].tail(3)
        early_range = (float(recent3["High"].max()) - float(recent3["Low"].min())) / max(price, 1e-9) * 100
        early_near_resistance = level_high > 0 and abs(price - level_high) / max(price, 1e-9) * 100 <= 1.5
        early_holding = float(recent3["Close"].iloc[-1]) >= float(recent3["Close"].iloc[0])
        early = bool(
            not breakout_now
            and early_near_resistance and early_range <= 3.0 and early_holding
            and not (retest or orb_breakout or breakout_now or liquidity_displacement
                     or liquidity_sweep or compression_expansion or momentum_continuation
                     or bull_flag or resistance_reclaim or orb_failed_reclaim
                     or abc_continuation or opening_drive_pullback or hod_reclaim
                     or vwap_bounce or ema_pullback or gap_setup or failed_breakdown_reclaim
                     or rs_pullback)
        )
    except Exception:
        early = False

    breakout_ok, breakout_quality = _breakout_quality(today_d, level_high, price)
    orb_breakout_ok, orb_quality = _breakout_quality(today_d, orb_high, price) if orb_high > 0 else (False, 0.0)
    # Failed breakout is a rejection/filter condition, not an entry strategy.
    # If there is no separate recovery setup below, the candidate is discarded.
    if failed and not (retest or liquidity_displacement or liquidity_sweep or orb_failed_reclaim or abc_continuation or opening_drive_pullback or hod_reclaim or vwap_bounce or ema_pullback or orb_breakout or breakout_now or compression_expansion or momentum_continuation or bull_flag or resistance_reclaim or gap_setup or failed_breakdown_reclaim or rs_pullback):
        return None

    # Multi-label strategy detection: every strategy that genuinely matches is
    # recorded. One primary strategy is still selected for the alert/exit rules,
    # using the existing precedence so overlapping setups remain deterministic.
    matched_entry_types = []
    if retest:
        matched_entry_types.append("إعادة اختبار")
    if orb_breakout:
        matched_entry_types.append("اختراق نطاق الافتتاح")
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
    if gap_setup:
        matched_entry_types.append("استمرار/استعادة الفجوة")
    if failed_breakdown_reclaim:
        matched_entry_types.append("استعادة بعد فشل كسر دعم")
    if rs_pullback:
        matched_entry_types.append("ارتداد بعد تفوق نسبي")
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

    # Score each matched strategy independently, like the intraday engine.
    strategy_scores: dict[str, float] = {}
    try:
        policy_for_strategy = _load_adaptive_policy()
        strategy_stats = policy_for_strategy.get("strategy_stats", {})
    except Exception:
        strategy_stats = {}

    strategy_ctx = locals().copy()


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
                # add() records a blocker when its second argument is False.
                # Therefore the argument must represent the GOOD state here.
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
                # Audit must mirror the Retest Core trigger: latest completed candle close.
                p, lvl = float(v("closed_close", 0.0) or 0.0), float(v("level_high", 0.0) or 0.0)
                add("لم تتم استعادة 99.7% من المستوى", lvl > 0 and p >= lvl * 0.997)
            except Exception: add("تعذر فحص استعادة المستوى", False)
            common(vwap=True, opening=False, green=False, market=False, state=False, no_failed=True)

        elif name == "اختراق نطاق الافتتاح":
            try:
                # Audit mirrors the ORB Core trigger: latest completed candle close.
                p, lvl = float(v("closed_close", 0.0) or 0.0), float(v("orb_high", 0.0) or 0.0)
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
                # Audit mirrors the EMA20 Core trigger: latest completed candle close.
                p, ema = float(v("closed_close", 0.0) or 0.0), float(v("e5", v("e20", 0.0)) or 0.0)
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
                "vwap_bounce":"ارتداد VWAP", "ema_pullback":"ارتداد EMA20",
                "gap_setup":"استمرار/استعادة الفجوة", "failed_breakdown_reclaim":"استعادة بعد فشل كسر دعم",
                "rs_pullback":"ارتداد بعد تفوق نسبي"
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
                "استمرار/استعادة الفجوة":"gap_setup",
                "استعادة بعد فشل كسر دعم":"failed_breakdown_reclaim",
                "ارتداد بعد تفوق نسبي":"rs_pullback",
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
            elif name in {"استعادة قمة اليوم", "استعادة قمة اليوم"}:
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
            elif name == "استعادة بعد فشل ORB":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
                add("يوجد ORB breakout حالي", not v("orb_breakout", False))
            elif name == "استمرار ABC":
                common(mom_min=0.05, vol_min=1.05, opening=True, ext_max=3.5 if "m15_state" in c else 6.0)
            elif name == "استمرار/استعادة الفجوة":
                add("الفجوة أقل من 2.00%", float(v("gap_pct",0.0) or 0.0) >= 2.0)
                add("لم يكتمل Hold/Failure للفجوة", bool(v("gap_setup",False)))
                common(mom_min=0.03, vol_min=0.90, opening=False, ext_max=6.0 if "m15_state" not in c else 4.5)
            elif name == "استعادة بعد فشل كسر دعم":
                add("الدعم لا يملك لمسَين على الأقل", int(v("failed_breakdown_touches",0) or 0) >= 2)
                add("نافذة الفشل تجاوزت 5 شموع", 1 <= int(v("failed_breakdown_bars",0) or 0) <= 5)
                add("الهبوط تجاوز 2 ATR", float(v("failed_breakdown_depth_atr",99.0) or 99.0) <= 2.0)
                add("لم تتم استعادة الدعم", bool(v("failed_breakdown_reclaim",False)))
                common(mom_min=0.03, vol_min=0.90, opening=False, ext_max=6.0 if "m15_state" not in c else 4.5)
            elif name == "ارتداد بعد تفوق نسبي":
                add("التفوق مقابل SPY أقل من 1.00%", float(v("rs_vs_spy",0.0) or 0.0) >= 1.0)
                add("التفوق مقابل QQQ أقل من 1.00%", float(v("rs_vs_qqq",0.0) or 0.0) >= 1.0)
                add("استمرارية التفوق أقل من 60%", float(v("rs_persistence",0.0) or 0.0) >= 60.0)
                add("Pullback أكبر من 50%", 0.0 < float(v("rs_pullback_pct",99.0) or 99.0) <= 50.0)
                add("لم يحافظ Pullback على Higher Low", bool(v("rs_higher_low",False)))
                add("لم يحدث Trigger", bool(v("rs_pullback",False)))
                common(mom_min=0.03, vol_min=0.90, opening=False, ext_max=6.0 if "m15_state" not in c else 4.5)

        if not out:
            out.append("لم يكتمل trigger الاستراتيجية رغم عدم توفر blocker أدق")
        # Keep the audit readable; the caller already limits the displayed list.
        return out

    # Storage is initialized before the diagnostic-only zero-match audit so
    # _strategy_strength() can safely run for all canonical strategies.
    strategy_component_scores: dict[str, dict[str, float]] = {}

    def _strategy_strength(name: str) -> float:
        """Independent 100-point strategy score: 70% structural Core + 30% Confirmation."""
        def clip(x, lo=0.0, hi=100.0):
            return max(lo, min(hi, float(x)))

        c = strategy_ctx
        _raw_strategy_df = c.get("today_d")
        try:
            _ci_strategy = int(c.get("closed_idx", -1))
            _cp_strategy = len(_raw_strategy_df) + _ci_strategy if _ci_strategy < 0 else _ci_strategy
            strategy_closed_df = _raw_strategy_df.iloc[:_cp_strategy + 1].copy() if _raw_strategy_df is not None and _cp_strategy >= 0 else _raw_strategy_df
        except Exception:
            strategy_closed_df = _raw_strategy_df
        price_v = float(c.get("closed_close", c.get("price", 0.0)) or 0.0)
        vr = float(c.get("vol_session_ratio", c.get("vol_ratio", 1.0)) or 1.0)
        green = bool(c.get("last_green", False))
        mom = float(c.get("mom", 0.0) or 0.0)
        mstate = str(c.get("h4_state", "محايد"))

        # Structural/Core components. These define the setup itself.
        if name == "اختراق مؤكد":
            bq = clip(c.get("breakout_quality", 0.0))
            vol = clip((vr - 0.75) / 0.75 * 100)
            q = bq
            components = {"breakout_quality": bq, "volume": vol, "candle": 100 if green else 0}
        elif name == "اختراق نطاق الافتتاح":
            oq = clip(c.get("orb_quality", 0.0))
            vol = clip((vr - 0.75) / 0.75 * 100)
            q = oq
            components = {"orb_quality": oq, "volume": vol, "candle": 100 if green else 0}
        elif name == "إعادة اختبار":
            prior = 100 if c.get("prior_break", False) else 0
            level = float(c.get("level_high", 0.0) or 0.0)
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 0.7
            near = clip((0.7-dist)/0.7*100)
            reclaim = clip((price_v/max(level,1e-9)-0.997)/0.004*100) if level > 0 else 0
            q = 0.25*prior + 0.35*near + 0.25*reclaim
            components = {"prior_break": prior, "near_level": near, "reclaim": reclaim}
        elif name == "ارتداد VWAP":
            touch = 100 if c.get("vwap_touch", False) else 0
            vwap = float(c.get("vwap_last_closed", c.get("vwap_last", 0.0)) or 0.0)
            reclaim = clip((price_v/max(vwap,1e-9)-0.998)/0.004*100) if vwap > 0 else 0
            q = 0.50*touch + 0.50*reclaim
            components = {"touch": touch, "reclaim": reclaim}
        elif name == "ارتداد EMA20":
            touch = 100 if c.get("ema_touch", False) else 0
            ema = float(c.get("e5_closed", c.get("e5", c.get("e20", 0.0))) or 0.0)
            reclaim = clip((price_v/max(ema,1e-9)-1.001)/0.004*100) if ema > 0 else 0
            q = 0.50*touch + 0.50*reclaim
            components = {"touch": touch, "reclaim": reclaim}
        elif name == "سحب سيولة":
            sweep = 100 if c.get("liquidity_sweep", False) else 0
            q = sweep
            components = {"sweep": sweep}
        elif name == "سحب سيولة مع Displacement":
            # Core = sweep + independent displacement quality. The composite
            # liquidity_displacement flag is never scored as a second trigger.
            sweep = 100 if c.get("liquidity_sweep", False) else 0
            disp = 0.0
            try:
                _df = strategy_closed_df
                if _df is not None and len(_df) >= 8:
                    _cur = _df.iloc[-1]
                    _prev = _df.iloc[-4:-1]
                    _rng = max(float(_cur["High"]) - float(_cur["Low"]), price_v * 0.0001)
                    _body_ratio = abs(float(_cur["Close"]) - float(_cur["Open"])) / _rng
                    _close_pos = (float(_cur["Close"]) - float(_cur["Low"])) / _rng
                    _med_rng = float((_prev["High"].astype(float) - _prev["Low"].astype(float)).median()) if len(_prev) else 0.0
                    _range_exp = (_rng / _med_rng) if _med_rng > 0 else 1.0
                    disp = (0.45 * clip(_body_ratio / 0.55 * 100.0)
                            + 0.25 * clip(_close_pos / 0.75 * 100.0)
                            + 0.30 * clip((_range_exp - 1.0) / 0.35 * 100.0))
            except Exception:
                disp = 0.0
            q = 0.35*sweep + 0.65*disp
            components = {"sweep": sweep, "displacement": disp}
        elif name == "ضغط ثم انفجار":
            match = 100 if c.get("compression_expansion", False) else 0
            q = match
            components = {"match": match}
        elif name == "استمرار الزخم":
            mom_core = clip((mom-0.08)/0.50*100)
            q = mom_core
            components = {"momentum": mom_core}
        elif name == "علم صاعد":
            impulse = clip((float(c.get("impulse_gain",0.0) or 0.0)-1.0)/2.0*100)
            flag = clip((2.0-float(c.get("flag_range",2.0) or 2.0))/1.5*100)
            q = 0.50*impulse + 0.50*flag
            components = {"impulse": impulse, "flag": flag}
        elif name == "استعادة مستوى":
            # Core scores the repeated-test structure and prior loss depth, not
            # the composite resistance_reclaim boolean. Reclaim is separate.
            level = float(c.get("reclaim_level", 0.0) or 0.0)
            match = 0.0
            try:
                _df = strategy_closed_df
                if _df is not None and len(_df) >= 10 and level > 0:
                    _prior = _df.iloc[-10:-2]
                    _touches = int((_prior["High"].astype(float) >= level * 0.995).sum())
                    _prev_close = float(_df["Close"].iloc[-2])
                    _touch_q = 50.0 + min(50.0, max(0.0, _touches - 2) * (50.0 / 3.0))
                    _loss_depth = max(0.0, (level - _prev_close) / max(level, 1e-9) * 100.0)
                    _loss_q = clip(_loss_depth / 0.75 * 100.0)
                    match = 0.70 * _touch_q + 0.30 * _loss_q
            except Exception:
                match = 0.0
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 0.6
            reclaim = clip((0.6-dist)/0.6*100)
            q = 0.60*match + 0.40*reclaim
            components = {"match": match, "reclaim": reclaim}
        elif name == "دخول بعد Opening Drive":
            drive = clip((float(c.get("drive_return",0.0) or 0.0)-2.0)/2.0*100)
            pb = float(c.get("pullback_from_high", 99.0) or 99.0)
            pull = clip((8.0-abs(pb-2.0))/7.5*100)
            q = 0.60*drive + 0.40*pull
            components = {"drive": drive, "pullback": pull}
        elif name == "استعادة قمة اليوم":
            # Core scores prior rejection depth independently from the current
            # reclaim distance; the composite hod_reclaim flag is not scored.
            level = float(c.get("hod_level", 0.0) or 0.0)
            match = 0.0
            try:
                _df = strategy_closed_df
                if _df is not None and len(_df) >= 8 and level > 0:
                    _prev_close = float(_df["Close"].iloc[-2])
                    _pullback = max(0.0, (level - _prev_close) / max(level, 1e-9) * 100.0)
                    match = clip(_pullback / 0.75 * 100.0)
            except Exception:
                match = 0.0
            dist = abs(price_v-level)/max(price_v,1e-9)*100 if level > 0 else 0.6
            reclaim = clip((0.6-dist)/0.6*100)
            q = 0.60*match + 0.40*reclaim
            components = {"match": match, "reclaim": reclaim}
        elif name == "استعادة بعد فشل ORB":
            # Core scores failure depth, ORB quality and reclaim separately; the
            # composite orb_failed_reclaim boolean is not scored as one item.
            failed_reclaim = 0.0
            orbq = clip(float(c.get("orb_quality", 0.0) or 0.0))
            orb_high = float(c.get("orb_high", 0.0) or 0.0)
            try:
                _df = strategy_closed_df
                if _df is not None and len(_df) >= 8 and orb_high > 0:
                    _post = _df.iloc[3:-1]
                    _closes = _post["Close"].astype(float)
                    _failed_closes = _closes[_closes <= orb_high * 0.998]
                    if len(_failed_closes):
                        _depth = max(0.0, (orb_high - float(_failed_closes.min())) / orb_high * 100.0)
                        failed_reclaim = clip((_depth - 0.20) / 0.80 * 100.0)
            except Exception:
                failed_reclaim = 0.0
            reclaim = clip((price_v/max(orb_high,1e-9)-1.001)/0.004*100) if orb_high > 0 else 0
            q = 0.35*failed_reclaim + 0.25*orbq + 0.40*reclaim
            components = {"failed_reclaim": failed_reclaim, "orb_quality": orbq, "reclaim": reclaim}
        elif name == "استمرار ABC":
            a = clip((float(c.get("a_gain",0.0) or 0.0)-0.70)/1.5*100)
            b = clip(100-abs(float(c.get("b_retrace",42.5) or 42.5)-42.5)/22.5*100)
            cb = 100 if c.get("c_break", False) else 0
            q = 0.30*a + 0.30*b + 0.40*cb
            components = {"a": a, "b": b, "c_break": cb}
        elif name == "استمرار/استعادة الفجوة":
            gap = clip((float(c.get("gap_pct",0.0) or 0.0)-2.0)/3.0*100); hold = 100.0 if str(c.get("gap_mode","") or "") in {"continuation","reclaim"} else 0.0; trigger = 100.0 if c.get("gap_setup",False) else 0.0
            q=0.35*gap+0.25*hold+0.40*trigger; components={"gap_quality":gap,"gap_hold":hold,"gap_trigger":trigger}
        elif name == "استعادة بعد فشل كسر دعم":
            touches_n=int(c.get("failed_breakdown_touches",0) or 0); touches=clip(50.0 + (touches_n-2.0)/3.0*50.0) if touches_n >= 2 else 0.0
            depth=float(c.get("failed_breakdown_depth_atr",0.0) or 0.0); breakdown=clip((2.0-depth)/1.8*100.0) if depth > 0 else 0.0; reclaim=100.0 if c.get("failed_breakdown_reclaim",False) else 0.0
            q=0.25*touches+0.35*breakdown+0.40*reclaim; components={"support_quality":touches,"breakdown_quality":breakdown,"breakdown_reclaim":reclaim}
        elif name == "ارتداد بعد تفوق نسبي":
            rsq=clip((min(float(c.get("rs_vs_spy",0.0) or 0.0),float(c.get("rs_vs_qqq",0.0) or 0.0))-1.0)/2.0*100); pb=clip((50.0-float(c.get("rs_pullback_pct",50.0) or 50.0))/50.0*100); hl=100.0 if c.get("rs_higher_low",False) else 0.0; trig=100.0 if c.get("rs_pullback",False) else 0.0
            q=0.30*rsq+0.25*pb+0.20*hl+0.25*trig; components={"rs_strength":rsq,"rs_pullback":pb,"rs_higher_low":hl,"rs_trigger":trig}

        else:  # دخول مبكر
            er = float(c.get("early_range",3.0) or 3.0)
            early_range = clip((3.0-er)/2.0*100)
            near = 100 if c.get("early_near_resistance", False) else 0
            holding = 100 if c.get("early_holding", False) else 0
            q = 0.40*early_range + 0.30*near + 0.30*holding
            components = {"early_range": early_range, "near_resistance": near, "holding": holding}

        # 30% Confirmation block: independent evidence only.
        # A Core component is NEVER reused in Confirmation. Confirmation can
        # validate the quality of the already-matched setup, but it cannot
        # award points again for the event that created the match.
        def _bar_quality(df):
            try:
                if df is None or len(df) < 1:
                    return 0.0, 0.0
                b = df.iloc[-1]
                o, h, l, cc = map(float, (b["Open"], b["High"], b["Low"], b["Close"]))
                rng = max(h - l, 1e-9)
                body = abs(cc - o) / rng
                close_pos = (cc - l) / rng
                return clip(body * 0.55 + close_pos * 0.45), clip(close_pos * 100.0)
            except Exception:
                return 0.0, 0.0

        _df = None
        try:
            _raw_df = strategy_closed_df
            _ci = int(c.get("closed_idx", -1))
            _cp = len(_raw_df) + _ci if _ci < 0 else _ci
            _df = _raw_df.iloc[:_cp + 1].copy() if _raw_df is not None and _cp >= 0 else _raw_df
        except Exception:
            _df = c.get("today_d")
        _bar_q, _close_pos = _bar_quality(_df)
        _vol = clip((vr - 0.90) / 0.60 * 100)
        _trend = 100.0 if bool(c.get("trend_up", False)) else 0.0
        _vwap = 100.0 if bool(c.get("above_vwap", False)) else 0.0
        _mtf = 100.0 if mstate != "معاكس" else 0.0
        _mom = clip((mom - 0.05) / 0.25 * 100)
        _candle = 100.0 if green else 0.0

        # Lower-wick rejection is independent of the structural sweep itself.
        _lower_wick = 0.0
        try:
            b = _df.iloc[-1] if _df is not None and len(_df) else None
            if b is not None:
                o, h, l, cc = map(float, (b["Open"], b["High"], b["Low"], b["Close"]))
                rng = max(h - l, 1e-9)
                lower_wick = max(min(o, cc) - l, 0.0) / rng
                _lower_wick = clip(lower_wick / 0.50 * 100.0)
        except Exception:
            _lower_wick = 0.0

        # None of the confirmation keys below belongs to the strategy Core.
        # The same structural event therefore cannot receive points twice.
        # 30% Confirmation: strategy-specific evidence only.
        # Confirmation never reuses a Core boolean/component. It measures the
        # QUALITY of the already-matched setup using independent price/volume
        # behaviour (stability, efficiency, relative volume, segment structure).
        _df = (c.get("new_setup_df") if name in {"استمرار/استعادة الفجوة", "استعادة بعد فشل كسر دعم", "ارتداد بعد تفوق نسبي"} else (c.get("today_5") if c.get("today_5") is not None else c.get("today_d")))

        def _s(col):
            try:
                return _df[col].astype(float) if _df is not None and col in _df.columns else None
            except Exception:
                return None

        def _clip(x):
            try: return clip(float(x))
            except Exception: return 0.0

        def _bar_quality():
            try:
                if _df is None or len(_df) < 1: return 0.0,0.0,0.0,0.0
                b=_df.iloc[-1]; o,h,l,cc=map(float,(b["Open"],b["High"],b["Low"],b["Close"]))
                r=max(h-l,1e-9); body=abs(cc-o)/r; cp=(cc-l)/r
                lw=max(min(o,cc)-l,0.0)/r; uw=max(h-max(o,cc),0.0)/r
                return _clip(body*100),_clip(cp*100),_clip(lw/0.5*100),_clip(uw/0.5*100)
            except Exception: return 0.0,0.0,0.0,0.0

        def _median_range(n=8):
            try:
                h,l=_s("High"),_s("Low")
                if h is None or l is None or len(h)<n: return 0.0
                return max(float((h-l).iloc[-n:].median()),1e-9)
            except Exception: return 0.0

        def _relative_volume(recent=1, base=6):
            try:
                v=_s("Volume")
                if v is None or len(v)<base+recent: return 50.0
                cur=float(v.iloc[-recent:].mean()); ref=float(v.iloc[-base-recent:-recent].median())
                return _clip((cur/max(ref,1e-9)-0.70)/1.10*100)
            except Exception: return 50.0

        def _segment_volume(a,b):
            try:
                v=_s("Volume")
                if v is None or len(v)<max(abs(a),abs(b),8): return 50.0
                seg=v.iloc[a:b] if b is not None else v.iloc[a:]
                if len(seg)<2: return 50.0
                return float(seg.mean())
            except Exception: return 0.0

        def _efficiency(a,b=None):
            try:
                cl=_s("Close")
                if cl is None: return 0.0
                seg=cl.iloc[a:b] if b is not None else cl.iloc[a:]
                if len(seg)<3: return 0.0
                net=abs(float(seg.iloc[-1])-float(seg.iloc[0])); path=float(seg.diff().abs().sum())
                return _clip(net/max(path,1e-9)*100)
            except Exception: return 0.0

        def _slope(a=3,b=7):
            try:
                cl=_s("Close")
                if cl is None or len(cl)<b+1: return 50.0
                s=(float(cl.iloc[-1])-float(cl.iloc[-a]))/max(abs(float(cl.iloc[-a])),1e-9)*100
                l=(float(cl.iloc[-1])-float(cl.iloc[-b]))/max(abs(float(cl.iloc[-b])),1e-9)*100
                return _clip(50+s*18+(s-l)*10)
            except Exception: return 50.0

        def _range_contraction(recent=3,base=6):
            try:
                h,l=_s("High"),_s("Low")
                if h is None or l is None or len(h)<recent+base: return 50.0
                rr=h-l; cur=float(rr.iloc[-recent:].median()); ref=float(rr.iloc[-recent-base:-recent].median())
                return _clip((1-cur/max(ref,1e-9))*100)
            except Exception: return 50.0

        def _pre_break_contraction(recent=4,base=8):
            try:
                h,l=_s("High"),_s("Low")
                if h is None or l is None or len(h)<recent+base+1: return 50.0
                rr=h-l
                cur=float(rr.iloc[-1-recent:-1].median())
                ref=float(rr.iloc[-1-recent-base:-1-recent].median())
                return _clip((1-cur/max(ref,1e-9))*100)
            except Exception: return 50.0

        def _post_break_hold(level,bars=3):
            try:
                cl=_s("Close")
                if cl is None or level<=0 or len(cl)<bars: return 0.0
                seg=cl.iloc[-bars:]
                above=float((seg>=level).mean())*70.0
                dist=float((seg.iloc[-1]-level)/max(_median_range(8),1e-9))*30.0
                return _clip(above+dist)
            except Exception: return 0.0

        def _level_stability(level,bars=4):
            try:
                cl=_s("Close"); atr=_median_range(8)
                if cl is None or level<=0 or len(cl)<bars or atr<=0: return 0.0
                dev=float((cl.iloc[-bars:]-level).abs().mean())/atr
                return _clip((1.0-dev/1.5)*100)
            except Exception: return 0.0

        def _level_pressure(level,bars=4):
            try:
                h,l,cl=_s("High"),_s("Low"),_s("Close")
                if h is None or l is None or cl is None or level<=0 or len(cl)<bars: return 0.0
                # Measures repeated pressure toward the level, not the breakout/reclaim event.
                dist=((level-cl.iloc[-bars:])/max(level,1e-9))*100
                q=float((dist<=0.8).mean())*70.0 + _clip((float(cl.iloc[-1])-float(cl.iloc[-bars]))/max(level,1e-9)*100*30.0)
                return _clip(q)
            except Exception: return 0.0

        def _relative_to_segment(level, a, b):
            try:
                cl=_s("Close")
                if cl is None or level<=0: return 0.0
                seg=cl.iloc[a:b] if b is not None else cl.iloc[a:]
                if len(seg)<2: return 0.0
                mean_dev=float((seg-level).abs().mean())/max(_median_range(8),1e-9)
                return _clip((1.0-mean_dev/2.0)*100)
            except Exception: return 0.0

        def _segment_recovery(level, start=-5):
            try:
                cl=_s("Close")
                if cl is None or level<=0 or len(cl)<abs(start)+1: return 0.0
                seg=cl.iloc[start:]
                first=float(seg.iloc[0]); last=float(seg.iloc[-1])
                move=(last-first)/max(_median_range(8),1e-9)
                return _clip(50+move*18)
            except Exception: return 50.0

        body,close_pos,lower_wick,upper_wick=_bar_quality()
        # Confirmation metrics used by strategy-specific weights.
        # close_strength is the closed-candle close position within its range.
        vals_close_strength = close_pos
        # trend_alignment uses the higher-timeframe context already selected for
        # this analyzer: supportive=100, neutral=50, contrary=0.
        vals_trend_alignment = 100.0 if mstate == "داعم" else (0.0 if mstate == "معاكس" else 50.0)
        rel_vol=_relative_volume()
        short_slope=_slope(3,7)
        path_eff=_efficiency(-7,None)
        contract=_range_contraction()
        atr=_median_range(8)
        level=float(c.get("level_high",0.0) or 0.0)
        vwap_level=float(c.get("vwap_last",0.0) or 0.0)
        ema_level=float(c.get("e20",c.get("e5",0.0)) or 0.0)
        orb_level=float(c.get("orb_high",0.0) or 0.0)
        hod_level=float(c.get("hod_level",0.0) or 0.0)
        support_level=float(c.get("support_level",0.0) or 0.0)

        # Strategy-specific confirmation map. We intentionally replace only the
        # confirmation layer; Core/Match/Safety logic is untouched.
        confirmation_weights = {
            "اختراق مؤكد": {"volume_expansion":.30,"range_expansion":.20,"pre_break_compression":.25,"post_break_hold":.25},
            "اختراق نطاق الافتتاح": {"opening_volume":.30,"opening_range_quality":.25,"opening_close_strength":.25,"opening_hold":.20},
            "إعادة اختبار": {"retest_stability":.30,"retest_rejection":.25,"retest_volume_vs_break":.20,"retest_duration":.15,"retest_follow_through":.10},
            "ارتداد VWAP": {"vwap_rejection":.30,"bounce_efficiency":.20,"bounce_volume":.15,"vwap_reclaim_persistence":.25,"vwap_post_bounce_slope":.10},
            "ارتداد EMA20": {"ema_rejection":.30,"ema_pullback_efficiency":.20,"ema_volume":.15,"ema_reclaim_persistence":.25,"ema_post_bounce_slope":.10},
            "سحب سيولة": {"sweep_wick":.30,"sweep_depth_quality":.25,"recovery_quality":.20,"sweep_volume":.15,"sweep_reclaim_persistence":.10},
            "سحب سيولة مع Displacement": {"displacement_body":.25,"displacement_range":.25,"displacement_close":.20,"displacement_volume":.20,"follow_through":.10},
            "ضغط ثم انفجار": {"compression_quality":.30,"expansion_efficiency":.25,"volume_shift":.20,"wick_balance":.15,"post_expansion_slope":.10},
            "استمرار الزخم": {"acceleration_quality":.30,"pullback_control":.20,"continuation_volume":.20,"higher_low_sequence":.20,"continuation_efficiency":.10},
            "علم صاعد": {"impulse_efficiency":.25,"flag_contraction":.25,"flag_volume_contraction":.20,"follow_through":.20,"close_strength":.10},
            "استعادة مستوى": {"level_test_quality":.30,"post_reclaim_slope":.25,"post_reclaim_stability":.20,"rejection_quality":.15,"volume_support":.10},
            "دخول بعد Opening Drive": {"drive_efficiency":.25,"pullback_control":.25,"pullback_volume":.15,"drive_recovery":.20,"close_strength":.15},
            "استعادة قمة اليوم": {"post_reclaim_slope":.30,"post_reclaim_stability":.25,"rejection_quality":.15,"volume_support":.15,"trend_alignment":.15},
            "استعادة بعد فشل ORB": {"failure_depth":.25,"time_below_orb":.20,"recovery_quality":.25,"reclaim_volume":.15,"close_strength":.15},
            "استمرار ABC": {"b_structure_quality":.25,"c_acceleration":.25,"c_volume":.20,"c_close_strength":.15,"c_range_expansion":.15},
            "استمرار/استعادة الفجوة": {"gap_volume":.20,"gap_close_strength":.20,"gap_open_hold":.20,"gap_follow_through":.20,"gap_trend_alignment":.20},
            "استعادة بعد فشل كسر دعم": {"breakdown_recovery_speed":.20,"breakdown_reclaim_volume":.20,"breakdown_close_strength":.20,"breakdown_support_stability":.20,"breakdown_trend_alignment":.20},
            "ارتداد بعد تفوق نسبي": {"rs_persistence":.25,"rs_relative_volume":.20,"rs_recovery":.20,"rs_trigger_close":.20,"rs_trend_alignment":.15},
            "دخول مبكر": {"range_contraction":.30,"volume_stability":.20,"resistance_pressure":.20,"holding_quality":.20,"pressure_persistence":.10},
        }.get(name,{})

        # Independent metrics. No metric reads the strategy Core booleans or Core scores.
        vals={}
        vals["close_strength"] = vals_close_strength
        vals["trend_alignment"] = vals_trend_alignment
        vals["volume_expansion"]=_relative_volume()
        vals["range_expansion"]=_clip(((float((_s("High")- _s("Low")).iloc[-1])/max(atr,1e-9))-1.0)/1.0*100) if _s("High") is not None and _s("Low") is not None and atr>0 else 0.0
        vals["pre_break_compression"]=_pre_break_contraction(4,8)
        vals["post_break_hold"]=_post_break_hold(level,3)

        # Opening range: quality of the bars after the opening range, not the ORB trigger itself.
        vals["opening_volume"]=_relative_volume(2,5)
        vals["opening_range_quality"]=_efficiency(-4,None)
        vals["opening_close_strength"]=_clip(float((_s("Close").iloc[-3:].mean()-_s("Low").iloc[-3:].mean())/max(atr,1e-9))*25) if _s("Close") is not None and _s("Low") is not None and atr>0 else 0.0
        vals["opening_hold"]=_level_stability(orb_level,3)

        # Retest: compare stability/volume/duration of the retest segment rather than re-scoring prior_break/near_level/reclaim.
        vals["retest_stability"]=_level_stability(level,4)
        vals["retest_rejection"]=lower_wick*0.75 + close_pos*0.25
        vals["retest_volume_vs_break"]=_relative_volume(2,5)
        vals["retest_duration"]=_clip((float((_s("Close").iloc[-4:]>=level*0.997).sum())/4.0)*100) if _s("Close") is not None and level>0 else 0.0
        vals["retest_follow_through"]=_slope(2,5)

        vals["vwap_proximity_quality"]=_level_stability(vwap_level,3)
        vals["vwap_rejection"]=lower_wick*0.60 + close_pos*0.40
        vals["bounce_efficiency"]=_efficiency(-4,None)
        vals["bounce_volume"]=_relative_volume(2,5)
        vals["vwap_distance_dummy"]=0.0
        try:
            cl=_s("Close")
            vals["vwap_reclaim_persistence"]=_clip(float((cl.iloc[-4:]>=vwap_level).mean())*100) if cl is not None and vwap_level>0 and len(cl)>=4 else 0.0
        except Exception: vals["vwap_reclaim_persistence"]=0.0
        vals["vwap_post_bounce_slope"]=_slope(2,5)

        vals["ema_proximity_quality"]=_level_stability(ema_level,3)
        vals["ema_rejection"]=lower_wick*0.60 + close_pos*0.40
        vals["ema_pullback_efficiency"]=_efficiency(-5,None)
        vals["ema_volume"]=_relative_volume(2,5)
        try:
            cl=_s("Close")
            vals["ema_reclaim_persistence"]=_clip(float((cl.iloc[-4:]>=ema_level).mean())*100) if cl is not None and ema_level>0 and len(cl)>=4 else 0.0
        except Exception: vals["ema_reclaim_persistence"]=0.0
        vals["ema_post_bounce_slope"]=_slope(2,5)

        vals["sweep_wick"]=lower_wick
        vals["sweep_depth_quality"]=_clip(max(0.0,(support_level-float(_s("Low").iloc[-4:].min()))/max(atr,1e-9))/1.5*100) if _s("Low") is not None and support_level>0 and atr>0 else 0.0
        vals["recovery_quality"]=_segment_recovery(support_level,-5)
        vals["sweep_volume"]=_relative_volume(2,5)
        try:
            cl=_s("Close")
            vals["sweep_reclaim_persistence"]=_clip(float((cl.iloc[-4:]>=support_level).mean())*100) if cl is not None and support_level>0 and len(cl)>=4 else 0.0
        except Exception: vals["sweep_reclaim_persistence"]=0.0

        vals["displacement_body"]=body
        vals["displacement_range"]=_clip((float((_s("High")-_s("Low")).iloc[-1])/max(atr,1e-9)-1.0)/1.0*100) if _s("High") is not None and _s("Low") is not None and atr>0 else 0.0
        vals["displacement_close"]=close_pos
        vals["displacement_volume"]=_relative_volume()
        vals["follow_through"]=_slope(2,5)

        vals["compression_quality"]=_range_contraction(4,8)
        vals["expansion_efficiency"]=_efficiency(-4,None)
        vals["volume_shift"]=_relative_volume(1,8)
        vals["wick_balance"]=_clip(100.0-abs(lower_wick-upper_wick))
        vals["post_expansion_slope"]=_slope(2,5)

        vals["acceleration_quality"]=_clip(50.0+(_slope(2,5)-_slope(5,9)))
        vals["pullback_control"]=_efficiency(-5,None)
        vals["continuation_volume"]=_relative_volume(2,5)
        try:
            cl=_s("Close")
            lo=_s("Low")
            if lo is not None and len(lo)>=5:
                diffs=lo.iloc[-5:].diff().dropna()
                vals["higher_low_sequence"]=_clip(float((diffs>0).mean())*100)
            else: vals["higher_low_sequence"]=0.0
        except Exception: vals["higher_low_sequence"]=0.0
        try:
            cl=_s("Close")
            if cl is not None and len(cl)>=7:
                diffs=cl.iloc[-7:].diff().dropna()
                vals["continuation_efficiency"]=_clip(float((diffs>0).mean())*100)
            else: vals["continuation_efficiency"]=50.0
        except Exception: vals["continuation_efficiency"]=50.0

        # Bull flag: compare flag segment with impulse segment, not generic current Volume.
        vals["impulse_efficiency"]=_efficiency(-11,-5)
        vals["flag_contraction"]=_range_contraction(5,5)
        try:
            v=_s("Volume"); flag_v=float(v.iloc[-5:].mean()); imp_v=float(v.iloc[-11:-5].mean()) if v is not None and len(v)>=11 else flag_v
            vals["flag_volume_contraction"]=_clip((1.25-flag_v/max(imp_v,1e-9))/0.75*100)
        except Exception: vals["flag_volume_contraction"]=50.0
        vals["follow_through"] = vals.get("follow_through", _slope(2,5))

        vals["level_test_quality"]=_level_pressure(level,5)
        vals["post_reclaim_slope"]=_slope(3,6)
        vals["post_reclaim_stability"]=_level_stability(level,4)
        vals["rejection_quality"]=lower_wick*0.70 + (100.0-upper_wick)*0.30
        vals["volume_support"]=_relative_volume(2,5)

        vals["drive_efficiency"]=_efficiency(-7,-1)
        vals["pullback_control"]=_efficiency(-5,None)
        vals["pullback_volume"]=_relative_volume(2,5)
        vals["drive_recovery"]=_slope(3,7)

        vals["post_reclaim_slope"]=_slope(3,6)
        vals["post_reclaim_stability"]=_level_stability(hod_level,4)

        vals["failure_depth"]=_clip(max(0.0,(orb_level-float(_s("Low").iloc[-6:].min()))/max(atr,1e-9))/1.5*100) if _s("Low") is not None and orb_level>0 and atr>0 else 0.0
        try:
            cl=_s("Close")
            vals["time_below_orb"]=_clip(float((cl.iloc[-8:]<orb_level*0.998).mean())*100) if cl is not None and orb_level>0 and len(cl)>=8 else 0.0
        except Exception: vals["time_below_orb"]=0.0
        vals["reclaim_volume"]=_relative_volume(2,5)

        # ABC: explicitly measure the middle B segment and final C acceleration.
        try:
            cl=_s("Close"); hi=_s("High"); lo=_s("Low")
            if cl is not None and hi is not None and lo is not None and len(cl)>=12:
                aa=cl.iloc[-12:-8]; bb=cl.iloc[-8:-4]; cc=cl.iloc[-4:]
                a_hi=float(aa.max()); a_lo=float(aa.min()); b_lo=float(bb.min()); b_hi=float(bb.max())
                b_depth=(a_hi-b_lo)/max(a_hi-a_lo,atr,1e-9); b_width=(b_hi-b_lo)/max(atr,1e-9)
                vals["b_structure_quality"]=_clip((1.0-abs(b_depth-0.40)/0.40)*70 + (1.0-min(b_width/3.0,1.0))*30)
                vals["c_acceleration"]=_clip(50.0+(_slope(2,4)-_slope(4,7)))
            else:
                vals["b_structure_quality"]=0.0; vals["c_acceleration"]=0.0
        except Exception:
            vals["b_structure_quality"]=0.0; vals["c_acceleration"]=0.0
        vals["c_volume"]=_relative_volume(2,5)
        vals["c_close_strength"]=close_pos
        try:
            hi=_s("High"); lo=_s("Low")
            if hi is not None and lo is not None and len(hi)>=8 and atr>0:
                c_rng=float((hi.iloc[-4:]-lo.iloc[-4:]).mean())
                b_rng=float((hi.iloc[-8:-4]-lo.iloc[-8:-4]).mean())
                vals["c_range_expansion"]=_clip((c_rng/max(b_rng,1e-9)-1.0)*100)
            else: vals["c_range_expansion"]=0.0
        except Exception: vals["c_range_expansion"]=0.0

        vals["range_contraction"]=_range_contraction(3,6)
        try:
            v=_s("Volume")
            if v is not None and len(v)>=6:
                ratios=v.iloc[-3:]/max(float(v.iloc[-6:-3].median()),1e-9)
                vals["volume_stability"]=_clip(100.0-float(ratios.std())*100.0)
            else: vals["volume_stability"]=50.0
        except Exception: vals["volume_stability"]=50.0
        vals["resistance_pressure"]=_level_pressure(level,4)
        vals["holding_quality"]=_slope(3,6)
        vals["pressure_persistence"]=_level_pressure(level,6)

        vals["gap_volume"]=_relative_volume(2,5); vals["gap_close_strength"]=close_pos; vals["gap_open_hold"]=_level_stability(float(c.get("day_open",0.0) or 0.0),4); vals["gap_follow_through"]=_slope(2,5); vals["gap_trend_alignment"]=vals_trend_alignment
        vals["breakdown_recovery_speed"]=_clip(100.0-(max(1.0,float(c.get("failed_breakdown_bars",0) or 0))-1.0)/4.0*100.0) if float(c.get("failed_breakdown_bars",0) or 0) > 0 else 0.0; vals["breakdown_reclaim_volume"]=_relative_volume(2,5); vals["breakdown_close_strength"]=close_pos; vals["breakdown_support_stability"]=_level_stability(float(c.get("failed_breakdown_support",0.0) or 0.0),4); vals["breakdown_trend_alignment"]=vals_trend_alignment
        vals["rs_persistence"]=_clip(float(c.get("rs_persistence",0.0) or 0.0)); vals["rs_relative_volume"]=_relative_volume(2,5); vals["rs_recovery"]=_slope(2,5); vals["rs_trigger_close"]=close_pos; vals["rs_trend_alignment"]=vals_trend_alignment

        confirmation_score=sum(_clip(vals.get(k,0.0))*float(w) for k,w in confirmation_weights.items())
        confirmation_score=_clip(confirmation_score)


        # Core score uses only structural components. Generic confirmation
        # factors are excluded here so they are counted once, in the 30% block.
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
            "استمرار/استعادة الفجوة": {"gap_quality", "gap_hold", "gap_trigger"},
            "استعادة بعد فشل كسر دعم": {"support_quality", "breakdown_quality", "breakdown_reclaim"},
            "ارتداد بعد تفوق نسبي": {"rs_strength", "rs_pullback", "rs_higher_low", "rs_trigger"},
            "دخول مبكر": {"early_range", "near_resistance", "holding"},
        }.get(name, set())
        all_weights = _strategy_weights_for(policy_for_strategy, name)
        core_weights = {k: w for k, w in all_weights.items() if k in core_keys}
        total_core = sum(core_weights.values()) or 1.0
        core_weights = {k: w / total_core for k, w in core_weights.items()}
        base_q = _weighted_strategy_score(components, core_weights)
        core_score = clip(base_q)
        core_contribution = 0.70 * core_score
        confirmation_contribution = 0.30 * confirmation_score
        final_q = core_contribution + confirmation_contribution

        # SCORE LEDGER (diagnostic/accounting only): records the existing 70/30
        # calculation explicitly. It adds no points and cannot change selection.
        strategy_component_scores[name] = dict(components)
        strategy_component_scores[name]["confirmation_score"] = round(confirmation_score, 4)
        strategy_component_scores[name]["core_score"] = round(core_score, 4)
        strategy_component_scores[name]["core_contribution_70pct"] = round(core_contribution, 4)
        strategy_component_scores[name]["confirmation_contribution_30pct"] = round(confirmation_contribution, 4)
        strategy_component_scores[name]["final_strategy_score"] = round(final_q, 4)
        strategy_component_scores[name]["score_ledger_total"] = round(core_contribution + confirmation_contribution, 4)

        return round(max(0.0, min(100.0, final_q)), 2)

    if not matched_entry_types:
        # DIAGNOSTIC ONLY: when all canonical strategies fail, emit the same
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
            "DAILY STRATEGY COMPETITION AUDIT V3 | %s | primary=NONE | matched=0/%s | "
            "ranking=DIAGNOSTIC_ONLY",
            symbol, len(ENTRY_TYPES),
        )
        for _detail in _zero_details:
            log.info(
                "DAILY STRATEGY COMPETITION DETAIL | %s | %s",
                symbol,
                _detail,
            )
        # Persist only diagnostic blocker counts for this symbol; this does not
        # participate in the trading decision.
        from collections import Counter as _Counter
        _blocker_counts = _Counter()
        for _et in ENTRY_TYPES:
            try:
                _blocker_counts.update(_competition_fail_reasons(_et))
            except Exception as exc:
                log.debug("DAILY non-critical deep fallback exception: %s", exc)
        with _DAILY_NO_SIGNAL_LOCK:
            _DAILY_NO_SIGNAL_AUDIT[str(symbol)] = dict(_blocker_counts)
        log.info(
            "DAILY NO_SIGNAL | %s | reason=no_strategy_match | top_blockers=%s",
            symbol,
            ";".join(f"{r}={c}" for r, c in _blocker_counts.most_common(6)) or "unavailable",
        )
        # Preserve the original trading behavior exactly.
        return None

    def _strategy_identity(name: str) -> float:
        """Structural identity score, separate from generic market quality.

        Common filters (trend/VWAP/open/volume/momentum/extension) are deliberately
        not counted here. This score answers only: "how clearly does the setup
        exhibit the defining structure of this strategy?" It is used as a
        deterministic tie-break for overlapping matches, not as extra trade score.
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
            "استمرار/استعادة الفجوة": 100.0 if gap_setup else 0.0,
            "استعادة بعد فشل كسر دعم": 100.0 if failed_breakdown_reclaim else 0.0,
            "ارتداد بعد تفوق نسبي": 100.0 if rs_pullback else 0.0,
            "دخول بعد Opening Drive": 100.0 if opening_drive_pullback else 0.0,
            "استعادة قمة اليوم": 100.0 if hod_reclaim else 0.0,
            "دخول مبكر": 45.0,
        }
        return round(max(0.0, min(120.0, identity.get(name, 0.0))), 2)

    strategy_identity_scores = {et: _strategy_identity(et) for et in matched_entry_types}
    strategy_component_scores: dict[str, dict[str, float]] = {}
    strategy_scores = {et: _strategy_strength(et) for et in matched_entry_types}
    entry_order = {et: i for i, et in enumerate(ENTRY_TYPES)}
    # When strategies overlap, prefer the more structurally specific setup only
    # when their strength scores are effectively tied. This does not remove
    # matched strategies or affect per-strategy learning; it only prevents a
    # broad setup from winning a near-tie over a setup with a clearer identity.
    strategy_specificity = {
        "سحب سيولة مع Displacement": 16,
        "استعادة بعد فشل ORB": 15,
        "دخول بعد Opening Drive": 14,
        "استمرار ABC": 13,
        "استمرار/استعادة الفجوة": 18,
        "استعادة بعد فشل كسر دعم": 17,
        "ارتداد بعد تفوق نسبي": 16,
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
    entry_emoji = "🟡" if entry_type == "إعادة اختبار" else "🟢"
    if entry_type == "اختراق نطاق الافتتاح":
        breakout_quality = max(breakout_quality, orb_quality)

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
    # This section evaluates ALL canonical strategies for observability and produces a
    # partial competition score even when a strategy did not fully match.
    # It MUST NOT change matched_entry_types, strategy_scores, entry_type,
    # alerts, exits, adaptive learning, market filters, or any trading rule.
    _competition_scores_all: dict[str, float] = {}
    _competition_details: list[str] = []
    for _et in ENTRY_TYPES:
        try:
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
        "DAILY STRATEGY COMPETITION AUDIT V3 | %s | primary=%s | matched=%s/%s | "
        "ranking=DIAGNOSTIC_ONLY",
        symbol,
        entry_type,
        len(matched_entry_types),
        len(ENTRY_TYPES),
    )
    for _detail in _competition_details:
        log.info(
            "DAILY STRATEGY COMPETITION DETAIL | %s | %s",
            symbol,
            _detail,
        )

    log.info(
        "DAILY STRATEGY AUDIT V3 | %s | primary=%s | matched=%s | scores=%s | "
        "tiebreak=%s",
        symbol,
        entry_type,
        list(matched_entry_types or []),
        _audit_scores_text or "none",
        _audit_tiebreak_text or "none",
    )


    reasons: list[str] = []
    warnings: list[str] = []
    # Final score starts from the selected Strategy Score (70% Core + 30% Confirmation).
    # Generic confirmation factors below are diagnostic/context only and are not
    # added again, preventing double counting.
    score = float(strategy_scores.get(entry_type, 0.0))
    factors: list[str] = []

    if trend_up:
        reasons.append("اتجاه الأسبوعي صاعد")
        factors.append("weekly_trend")
    else:
        warnings.append("اتجاه الأسبوعي غير مؤكد")

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
        reasons.append("تأكيد يومي")
        factors.append("daily")
    else:
        warnings.append("لا تأكيد يومي كافٍ")

    if vol_ok:
        reasons.append(f"حجم جلسة {vol_ratio:.2f}x")
        factors.append("vol_session")
    else:
        warnings.append("حجم الجلسة ضعيف نسبياً")

    if market_ok:
        factors.append("market")
        reasons.append(market_state)
    else:
        warnings.append(market_state)

    if chop:
        warnings.append("السوق اليومي متذبذب (Chop)")
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
        # لا نمنع الاستحواذ/الاندماج؛ نرفع المتطلبات بدل ذلك.
        factors.append("news_momentum")
        reasons.append("خبر إيجابي جوهري — وضع NEWS MOMENTUM")
    elif news_state == "positive":
        factors.append("news_positive")

    if h4_state == "داعم":
        reasons.append("4 ساعات داعمة")
        factors.append("h4")
    elif h4_state == "معاكس":
        warnings.append("4 ساعات معاكسة")
    else:
        reasons.append("4 ساعات محايدة")
        factors.append("h4_neutral")

    if 48 <= h_rsi <= 68:
        factors.append("rsi_weekly")

    if dump:
        warnings.append("سقوط من قمة الفترة")

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

    if vwap_weekly_confluence:
        factors.append("vwap_weekly_confluence")
        reasons.append("Confluence: VWAP + اتجاه الأسبوعي + 4س")
    if multi_level_confluence:
        factors.append("multi_level_confluence")
        reasons.append("تجمع مستويات: " + "/".join(confluence_levels[:4]))

    ext = (price - e20) / e20 * 100 if e20 else 0
    if ext > 4.0:
        warnings.append("امتداد عن متوسط الأسبوعي")
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
        and h4_state != "معاكس"
        and vol_ratio >= 1.0
        and not dump
        and ext <= 7.0
    )

    # Market-aware daily execution policy. The market regime is detected first,
    # then the stock must satisfy the rules for that regime before it can be
    # emitted. This prevents mixed/weak markets from using the normal strong-
    # market gate just because market_ok happened to be True.
    strong_market_ok = bool(
        market_condition == "قوي"
        and market_ok
    )

    positive_market_ok = bool(
        market_condition == "إيجابي_تحت_VWAP"
        and market_ok
        and trend_up
        and live_ok
        and above_vwap
        and above_open
        and h4_state != "معاكس"
        and vol_ratio >= 0.95
        and not dump
        and not chop
        and ext <= 5.5
    )

    mixed_market_ok = bool(
        market_condition == "مختلط"
        and market_ok
        and trend_up
        and live_ok
        and above_vwap
        and above_open
        and h4_state != "معاكس"
        and vol_ratio >= 1.0
        and not dump
        and not chop
        and ext <= 6.0
    )

    # Daily strong-stock override: weak market can be bypassed only when the
    # stock itself is exceptionally aligned and materially outperforms SPY/QQQ.
    # Missing/unknown market data can never trigger this override.
    # Preserve the uncapped score for the Daily strong-stock override.
    # Strategy-specific entry caps below are grading limits, not the override's
    # raw-score requirement.
    raw_score = float(score)
    policy = _load_adaptive_policy()
    limits = policy.get("entry_limits", {})
    # IMPORTANT: entry_limits are Score caps only. They NEVER determine alert eligibility.
    # Global eligibility uses raw_score in scan_* after all structural/final gates pass.
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
    elif entry_type == "استمرار/استعادة الفجوة":
        score = min(score, float(limits.get("استمرار/استعادة الفجوة", 98.0)))
    elif entry_type == "استعادة بعد فشل كسر دعم":
        score = min(score, float(limits.get("استعادة بعد فشل كسر دعم", 98.0)))
    elif entry_type == "ارتداد بعد تفوق نسبي":
        score = min(score, float(limits.get("ارتداد بعد تفوق نسبي", 97.0)))
    else:
        score = min(score, 80.0)

    score_i = int(max(0, min(100, round(score))))

    # Weak-market exception candidate. It is deliberately narrow: actual
    # relative strength + strong alignment + high score. Final approval is
    # still recomputed after TP/stop checks below, so market_block is allowed
    # only when it is the sole remaining quality problem.
    strong_stock_market_candidate = bool(
        market_condition == "ضعيف"
        and not market_ok
        and raw_score >= 97.0
        and entry_type != "دخول مبكر"
        and strong_alignment
        and relative_strength_ok
        and vol_ratio >= 1.25
        and ext <= 5.0
        and not chop
    )
    strong_stock_market_override = strong_stock_market_candidate

    market_permission = bool(
        strong_market_ok
        or positive_market_ok
        or mixed_market_ok
        or strong_stock_market_override
    )

    log.info(
        "DAILY MARKET GATE | %s | condition=%s | permission=%s | score=%d | RS=%s | market_avg=%s",
        symbol, market_condition, market_permission, score_i,
        f"{relative_strength_pct:.2f}%" if relative_strength_pct is not None else "n/a",
        f"{market_avg_pct:.2f}%" if market_avg_pct is not None else "n/a",
    )

    strong_for_grade = (
        score_i >= 95
        and strong_alignment
        and entry_type in (set(ENTRY_TYPES) - {"دخول مبكر"})
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
        and not (h4_state == "معاكس" and raw_score < 92)
        and market_permission
    )

    recent_low = float(today_d["Low"].tail(12).min())

    # Structure-aware daily stop. The stop is placed behind the structure
    # that actually justifies the entry, then constrained to a practical
    # daily risk band of 0.60%–4.50%.
    # Strategy-specific structural stop is primary. Generic ATR/recent-low
    # references are fallback-only and never compete with the strategy stop.
    strategy_stop = 0.0
    if entry_type == "ارتداد VWAP":
        strategy_stop = float(today_d["Low"].iloc[max(0, closed_idx - 4):closed_idx].min()) * 0.997 if closed_idx > 0 else vwap_last * 0.997
    elif entry_type == "ارتداد EMA20":
        strategy_stop = float(today_d["Low"].iloc[max(0, closed_idx - 4):closed_idx].min()) * 0.997 if closed_idx > 0 else e5 * 0.997
    elif entry_type in {"سحب سيولة", "سحب سيولة مع Displacement"}:
        sweep_extreme = float(today_d["Low"].iloc[max(0, closed_idx - 3):closed_idx].min()) if closed_idx > 0 else 0.0
        strategy_stop = sweep_extreme * 0.997 if sweep_extreme > 0 else (support_level * 0.997 if support_level > 0 else recent_low * 0.997)
    elif entry_type in {"اختراق مؤكد", "اختراق نطاق الافتتاح"}:
        level = orb_high if entry_type == "اختراق نطاق الافتتاح" else level_high
        if level and level > 0:
            strategy_stop = level * 0.997
    elif entry_type == "إعادة اختبار":
        strategy_stop = level_high * 0.997 if level_high > 0 else 0.0
    elif entry_type == "استمرار/استعادة الفجوة":
        _setup_lows = new_setup_df["Low"].astype(float) if new_setup_df is not None and len(new_setup_df) else today_d["Low"].astype(float)
        strategy_stop = gap_mid * 0.997 if gap_mid > 0 else float(_setup_lows.tail(4).min()) * 0.997
    elif entry_type == "استعادة بعد فشل كسر دعم":
        strategy_stop = failed_breakdown_low * 0.997 if failed_breakdown_low > 0 else (failed_breakdown_support * 0.997 if failed_breakdown_support > 0 else recent_low * 0.997)
    elif entry_type == "ارتداد بعد تفوق نسبي":
        _setup_lows = new_setup_df["Low"].astype(float) if new_setup_df is not None and len(new_setup_df) else today_d["Low"].astype(float)
        strategy_stop = float(_setup_lows.tail(3).min()) * 0.997
    elif entry_type == "دخول مبكر":
        try:
            strategy_stop = float(today_d["Low"].tail(3).min()) * 0.997
        except Exception:
            strategy_stop = recent_low * 0.997
    elif entry_type == "علم صاعد":
        strategy_stop = float(flag["Low"].min()) * 0.997 if "flag" in locals() and len(flag) else float(today_d["Low"].tail(5).min()) * 0.997
    elif entry_type == "استعادة مستوى":
        strategy_stop = reclaim_level * 0.997 if reclaim_level > 0 else recent_low * 0.997
    elif entry_type == "استعادة بعد فشل ORB":
        strategy_stop = float(post_orb["Low"].min()) * 0.997 if "post_orb" in locals() and len(post_orb) else (orb_high * 0.997 if orb_high > 0 else recent_low * 0.997)
    elif entry_type == "استمرار ABC":
        strategy_stop = float(b["Low"].min()) * 0.997 if "b" in locals() and len(b) else float(today_d["Low"].tail(4).min()) * 0.997
    elif entry_type == "دخول بعد Opening Drive":
        strategy_stop = drive_level * 0.997 if drive_level > 0 else recent_low * 0.997
    elif entry_type == "استعادة قمة اليوم":
        strategy_stop = float(prior["Low"].min()) * 0.997 if "prior" in locals() and len(prior) else (hod_level * 0.997 if hod_level > 0 else recent_low * 0.997)
    elif entry_type in {"ضغط ثم انفجار", "استمرار الزخم"}:
        strategy_stop = float(today_d["Low"].tail(5).min()) * 0.997

    if np.isfinite(float(strategy_stop)) and 0 < float(strategy_stop) < price:
        structural_stop = float(strategy_stop)
    else:
        fallback_stops = [float(price - 1.5 * atr), float(recent_low * 0.997)]
        valid_fallback_stops = [x for x in fallback_stops if np.isfinite(x) and 0 < x < price]
        structural_stop = max(valid_fallback_stops) if valid_fallback_stops else price * 0.985
    stop = structural_stop

    risk = price - stop
    min_risk = price * (DAILY_MIN_RISK_PCT / 100.0)
    max_risk = price * (DAILY_MAX_RISK_PCT / 100.0)
    if risk < min_risk:
        stop = price - min_risk
        risk = min_risk
    elif risk > max_risk:
        # Too-wide structures are rejected rather than hiding the risk.
        quality_ok = False
        warnings.append("وقف هيكلي واسع جدًا")

    # This is the final validated structural stop for THIS strategy.
    # Adaptive Exit must always start from this exact strategy stop.
    structural_stop = stop

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
    # Keep the already-validated structural stop so a failed adaptive safety
    # check can revert exactly to the pre-adaptive value.
    # `stop` above is the validated structural stop for this strategy.
    # Adaptive Exit may adjust it only within the existing risk band.
    policy_now = _load_adaptive_policy()
    adaptive_stop, adaptive_tp1, adaptive_tp1_r = _apply_adaptive_exit(
        price, structural_stop, market_regime, entry_type, policy_now, atr
    )
    if policy_now.get("exit_active"):
        stop = adaptive_stop
        tp1 = adaptive_tp1
        risk = price - stop
        risk_pct_check = risk / price * 100 if price else 0.0
        if DAILY_MIN_RISK_PCT <= risk_pct_check <= DAILY_MAX_RISK_PCT:
            warnings.append(f"Adaptive Exit: TP1={adaptive_tp1_r:.2f}R")
        else:
            # Safety: revert to the original structural stop if adaptive scaling
            # somehow leaves the allowed daily risk band.
            stop = structural_stop
            risk = price - stop
            tp1 = price + risk * 1.20

    # Keep targets strictly ordered even when Adaptive Exit raises TP1 above 2R.
    # This does not change strategy selection or the Adaptive TP1 rule.
    risk_pct = risk / price * 100 if price else 0.0
    reward_r = (tp1 - price) / risk if risk else 0.0
    tp2_r = max(2.0, reward_r + 1e-6)
    tp3_r = max(3.0, tp2_r + 1e-6)
    tp2 = price + risk * tp2_r
    tp3 = price + risk * tp3_r
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

    # فحص Spread/السيولة يُجرى في scan_daily للمرشحين فقط، حتى لا يبطئ تحليل كل الأسهم.
    liquidity = {"ok": True, "spread_pct": 0.0, "slippage_pct": 0.0}
    liquidity_ok = True

    interaction_keys = _interaction_keys(
        entry_type, market_regime, h4_state, vol_ratio, breakout_quality
    )

    # Recompute the exception after all quality/TP/stop checks. This enforces
    # the policy: in a weak market, only a genuinely strong stock whose sole
    # blocker is market_block may pass.
    strong_stock_market_override = bool(
        strong_stock_market_candidate
        and entry_type != "دخول مبكر"
        and raw_score >= 97.0
        and not dump
        and not failed
        and ext <= 5.0
        and atr_pct <= 8.0
        and vol_ratio >= max(1.25, float(policy.get("min_volume_ratio", 0.85)))
        and not chop
        and news_momentum_ok
        and not (h4_state == "معاكس" and raw_score < 92)
        and risk <= price * (DAILY_MAX_RISK_PCT / 100.0)
        and tp1 > price
        and tp1_distance_pct >= 0.8
        and reward_r >= float(policy.get("min_tp1_r", 1.2))
    )
    # The weak-market override may bypass ONLY the market blocker. It must
    # never erase another quality failure (wide stop, volume, chop, failed
    # breakout, news, HTF contradiction, etc.).
    non_market_quality_ok = bool(
        (not dump)
        and (not failed)
        and ext <= 8.0
        and atr_pct <= 8.0
        and vol_ratio >= float(policy.get("min_volume_ratio", 0.85))
        and not chop
        and news_momentum_ok
        and not (h4_state == "معاكس" and raw_score < 92)
    )
    strong_stock_market_override = bool(
        strong_stock_market_override and non_market_quality_ok
    )
    market_permission = bool(
        strong_market_ok
        or positive_market_ok
        or mixed_market_ok
        or strong_stock_market_override
    )
    if strong_stock_market_override:
        # Only market_block is waived; all other quality gates remain intact.
        quality_ok = non_market_quality_ok

    if resistance_source != "هدف مخاطر 1.20R":
        reasons.append(f"TP1 مقاومة: {tp1:.2f}")
    else:
        warnings.append("لم توجد مقاومة قريبة مناسبة؛ TP1 احتياطي")

    # Diagnostics only: explain exactly why a signal failed the final quality gate.
    # This list is observational and does not change quality_ok or any selection rule.
    quality_reasons = []
    if dump:
        quality_reasons.append("dump")
    if failed:
        quality_reasons.append("failed_breakout")
    if ext > 8.0:
        quality_reasons.append(f"extension>{8.0:.0f}%")
    if atr_pct > 8.0:
        quality_reasons.append(f"atr>{8.0:.0f}%")
    if vol_ratio < float(policy.get("min_volume_ratio", 0.85)):
        quality_reasons.append("volume")
    if chop:
        quality_reasons.append("chop")
    if not news_momentum_ok:
        quality_reasons.append("news_momentum")
    if h4_state == "معاكس" and raw_score < 92:
        quality_reasons.append("h4_contrary")
    if not market_permission:
        quality_reasons.append("market_block")
        if market_condition == "مختلط":
            quality_reasons.append("mixed_conditions")
        elif market_condition == "ضعيف":
            if not relative_strength_ok:
                quality_reasons.append("weak_relative_strength")
            if raw_score < 97:
                quality_reasons.append("weak_score<97")
        elif market_condition == "غير مؤكد":
            quality_reasons.append("market_unknown")
    if "وقف هيكلي واسع جدًا" in warnings:
        quality_reasons.append("wide_stop")
    if tp1 <= price:
        quality_reasons.append("invalid_tp1")
    if tp1_distance_pct < 0.8:
        quality_reasons.append("tp1_too_close")
    if reward_r < float(policy.get("min_tp1_r", 1.2)):
        quality_reasons.append("weak_tp1_r")

    return DailySignal(
        symbol=symbol,
        name=name or symbol,
        price=round(price, 4),
        change_pct=round(change_pct, 2),
        score=score_i,
        grade=_grade(score_i, strong=strong_for_grade),
        raw_score=round(float(raw_score), 2),
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
        live_gate_reasons=live_gate_reasons,
        volume_ratio=round(vol_ratio, 2),
        factor_keys=factors,
        sma20=round(e20, 4),
        atr_pct=round(atr_pct, 2),
        ext_sma20=round(ext, 2),
        alert_entry_price=0.0,
        entry_type=entry_type,
        entry_emoji=entry_emoji,
        matched_entry_types=matched_entry_types,
        strategy_scores=strategy_scores,
        strategy_component_scores=strategy_component_scores,
        h4_state=h4_state,
        learning_adjustment=round(total_learning_adj, 2),
        resistance_tp1=round(tp1, 4),
        news_state=news_state,
        news_title=news_title,
        news_source=news_source,
        breakout_quality=round(breakout_quality, 1),
        market_state=market_state,
        market_condition=market_condition,
        market_relative_strength=round(relative_strength_pct or 0.0, 2),
        market_avg_change=round(market_avg_pct or 0.0, 2),
        chop=chop,
        market_regime=market_regime,
        interaction_keys=interaction_keys,
        spread_pct=round(float(liquidity.get("spread_pct", 0) or 0), 3),
        expected_slippage_pct=round(float(liquidity.get("slippage_pct", 0) or 0), 3),
        liquidity_ok=liquidity_ok,
        quality_reasons=quality_reasons,
    )


def format_daily_ar(sig: DailySignal, min_score: int = DAILY_MIN_SCORE) -> str:
    arrow = "▲" if sig.change_pct >= 0 else "▼"
    market_map = {
        "قوي": "🟢 قوي",
        "إيجابي_تحت_VWAP": "🟠 إيجابي تحت VWAP",
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
            market_label = "🟡 إيجابي"
        elif "مختلطان" in state:
            market_label = "🟠 مختلط"
        elif "ضعيفان" in state:
            market_label = "🔴 ضعيف"
        else:
            market_label = "⚪️ غير مؤكد"

    lines = [
        f"⚡ يومي | {sig.symbol} | {sig.score}/100 | {sig.grade} | ساعة+4س+يومي",
        f"{market_label} | نظام السوق",
        f"{sig.entry_emoji} الدخول: {sig.entry_type}",
        f"{sig.name}",
        "—————————————",
        f"السعر: {sig.price:.2f} $  ({arrow} {sig.change_pct:+.2f}%)",
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



def _daily_strategy_route_scores(*, price: float, trend: bool, above_vwap: bool, above_open: bool, vol_ratio: float, mom: float, wrsi: float, we20: float, we50: float, daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, float]:
    """Cheap Stage-1 routing proxies for all canonical daily strategies. Exact gates remain in analyze_daily."""
    dc=daily["Close"].astype(float); op=daily["Open"].astype(float); hi=daily["High"].astype(float); lo=daily["Low"].astype(float)
    prev_close=float(dc.iloc[-2]) if len(dc)>=2 else float(op.iloc[-1]); last_high=float(hi.iloc[-1]); last_low=float(lo.iloc[-1])
    vw_s=_vwap(daily.tail(60)); vw=float(vw_s.iloc[-1]) if pd.notna(vw_s.iloc[-1]) else price
    e20=float(_ema(dc,20).iloc[-1]); recent_high=float(hi.tail(20).max()); recent_low=float(lo.tail(20).min())
    near_high=abs(price-recent_high)/max(price,1e-9)*100<=1.0; near_vwap=abs(price-vw)/max(price,1e-9)*100<=1.0; near_ema=abs(price-e20)/max(price,1e-9)*100<=1.0
    breakout=price>=max(float(hi.iloc[-2]) if len(hi)>=2 else recent_high,recent_high*0.995); reclaim=price>=recent_high*0.998 and prev_close<recent_high*0.998
    pullback=trend and price>=e20*0.995 and price<=e20*1.015; drive=trend and mom>1.0 and pullback
    sweep=last_low <= recent_low*1.002 and price>float(op.iloc[-1]); compression=((float(hi.tail(5).max())-float(lo.tail(5).min()))/max(price,1e-9)*100<=3.0 and mom>0)
    abc=trend and mom>0.5 and pullback; flag=trend and compression and breakout; orb=breakout and above_open; failed_orb=prev_close<recent_high and reclaim
    scores={et:0.0 for et in ENTRY_TYPES}
    scores["اختراق مؤكد"]=(35 if breakout else 0)+20*(1 if above_vwap else 0)+min(20,max(0,mom)*4)+min(15,max(0,vol_ratio-1)*10)
    scores["إعادة اختبار"]=(35 if pullback else 0)+(20 if near_high else 0)+(15 if above_vwap else 0)+(10 if trend else 0)
    scores["دخول مبكر"]=(30 if above_vwap and above_open else 0)+(20 if trend else 0)+min(20,max(0,2.5-mom)*8)
    scores["ارتداد VWAP"]=(35 if near_vwap else 0)+(20 if above_vwap else 0)+(15 if trend else 0)
    scores["ارتداد EMA20"]=(40 if near_ema else 0)+(20 if trend else 0)+(10 if above_vwap else 0)
    scores["سحب سيولة"]=(45 if sweep else 0)+(15 if above_vwap else 0)+min(20,max(0,vol_ratio-1)*10)
    scores["اختراق نطاق الافتتاح"]=(45 if orb else 0)+(15 if above_vwap else 0)+min(20,max(0,mom)*4)
    scores["استمرار الزخم"]=(30 if trend else 0)+min(35,max(0,mom)*7)+min(20,max(0,vol_ratio-1)*10)
    scores["ضغط ثم انفجار"]=(45 if compression else 0)+(20 if breakout else 0)+min(15,max(0,mom)*3)
    scores["علم صاعد"]=(40 if flag else 0)+(15 if trend else 0)+(15 if above_vwap else 0)
    scores["استعادة مستوى"]=(40 if reclaim or near_high else 0)+(20 if above_vwap else 0)+(15 if trend else 0)
    scores["دخول بعد Opening Drive"]=(45 if drive else 0)+(20 if above_vwap else 0)+min(15,max(0,mom)*3)
    scores["استعادة قمة اليوم"]=(45 if near_high and price>=last_high*0.995 else 0)+(20 if above_vwap else 0)+(10 if trend else 0)
    scores["استعادة بعد فشل ORB"]=(45 if failed_orb else 0)+(20 if above_vwap else 0)+(10 if trend else 0)
    scores["استمرار ABC"]=(40 if abc else 0)+(20 if trend else 0)+min(15,max(0,mom)*3)
    scores["سحب سيولة مع Displacement"]=(45 if sweep and mom>0.5 else 0)+(20 if vol_ratio>=1.2 else 0)+(10 if trend else 0)
    gap_pct=((float(op.iloc[-1])-prev_close)/max(prev_close,1e-9))*100 if prev_close else 0.0
    gap_proxy=gap_pct>=2.0 and price>=float(op.iloc[-1])*0.997
    failed_breakdown_proxy=bool(recent_low>0 and prev_close<=recent_low*0.998 and price>=recent_low*1.001)
    rs_proxy=bool(trend and mom>=0.75 and pullback)
    scores["استمرار/استعادة الفجوة"]=(50 if gap_proxy else 0)+min(20,max(0,gap_pct-2.0)*10)+min(15,max(0,mom)*3)
    scores["استعادة بعد فشل كسر دعم"]=(50 if failed_breakdown_proxy else 0)+(15 if trend else 0)+min(15,max(0,vol_ratio-0.9)*10)
    scores["ارتداد بعد تفوق نسبي"]=(45 if rs_proxy else 0)+min(25,max(0,mom)*5)+(15 if above_vwap else 0)
    return {k:round(float(v),2) for k,v in scores.items()}


def _prefilter_daily(symbol: str, audit_counts: dict[str, int] | None = None, audit_lock: Lock | None = None,
                     data_audit_counts: dict[str, int] | None = None) -> tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame] | None:
    """Stage 1: weekly + daily routing for the full universe."""
    def _audit_stage1(reason: str) -> None:
        if audit_counts is None:
            return
        if audit_lock is not None:
            with audit_lock:
                _daily_bump(audit_counts, reason)
        else:
            _daily_bump(audit_counts, reason)
    def _audit_data(reason: str) -> None:
        if data_audit_counts is None:
            return
        if audit_lock is not None:
            with audit_lock:
                _daily_bump(data_audit_counts, reason)
        else:
            _daily_bump(data_audit_counts, reason)
    try:
        from market_data import fetch_intraday, intraday_data_fresh
        weekly = fetch_intraday(symbol, interval="1wk", period="5y")
        daily = fetch_intraday(symbol, interval="1d", period="2y")
        if weekly is None or daily is None or len(weekly) < 60 or len(daily) < 80:
            if weekly is None:
                _audit_stage1("weekly_missing"); _audit_data("weekly_missing")
            elif len(weekly) < 60:
                _audit_stage1("weekly_bars<60"); _audit_data("weekly_bars<60")
            if daily is None:
                _audit_stage1("daily_missing"); _audit_data("daily_missing")
            elif len(daily) < 80:
                _audit_stage1("daily_bars<80"); _audit_data("daily_bars<80")
            _audit_stage1("data_missing_or_short"); _audit_data("data_missing_or_short")
            return None
        price = float(daily["Close"].iloc[-1])
        if price <= 0 or price > float(MAX_AUTO_PRICE):
            _audit_stage1("invalid_price")
            _audit_data("invalid_price")
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
        vol_ratio = _daily_volume_ratio_time_of_day(symbol, fetch_intraday, now_ny())
        mom = (price - float(dc.iloc[-6])) / max(float(dc.iloc[-6]),1e-9) * 100 if len(dc)>=6 else 0.0
        route = 0.0
        route += 3.0 if trend else 0.0
        route += 2.0 if above_vwap else 0.0
        route += 1.5 if above_open else 0.0
        route += min(2.5,max(0.0,mom))
        route += min(2.0,max(0.0,vol_ratio-0.75)*2.0)
        route += 1.0 if we20 > we50 else 0.0
        route -= 1.0 if wrsi >= 80 else 0.0
        if vol_ratio < 0.55:
            _audit_stage1("low_volume_ratio<0.55x")
            return None
        routes=_daily_strategy_route_scores(price=price, trend=trend, above_vwap=above_vwap, above_open=above_open, vol_ratio=vol_ratio, mom=mom, wrsi=wrsi, we20=we20, we50=we50, daily=daily, weekly=weekly)
        return route, routes, weekly, daily
    except Exception as exc:
        _audit_stage1("exception")
        log.info("DAILY STAGE 1 REJECT | %s | exception=%s", symbol, str(exc))
        return None



def _prefilter_daily_from_frames(symbol: str, weekly: pd.DataFrame, daily: pd.DataFrame, audit_counts: dict[str, int] | None = None, audit_lock: Lock | None = None,
                                   data_audit_counts: dict[str, int] | None = None) -> tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame] | None:
    def _audit_stage1(reason: str) -> None:
        if audit_counts is None:
            return
        if audit_lock is not None:
            with audit_lock:
                _daily_bump(audit_counts, reason)
        else:
            _daily_bump(audit_counts, reason)
    def _audit_data(reason: str) -> None:
        if data_audit_counts is None:
            return
        if audit_lock is not None:
            with audit_lock:
                _daily_bump(data_audit_counts, reason)
        else:
            _daily_bump(data_audit_counts, reason)
    try:
        from market_data import fetch_intraday
        if weekly is None or daily is None or len(weekly) < 60 or len(daily) < 80:
            if weekly is None:
                _audit_stage1("weekly_missing"); _audit_data("weekly_missing")
            elif len(weekly) < 60:
                _audit_stage1("weekly_bars<60"); _audit_data("weekly_bars<60")
            if daily is None:
                _audit_stage1("daily_missing"); _audit_data("daily_missing")
            elif len(daily) < 80:
                _audit_stage1("daily_bars<80"); _audit_data("daily_bars<80")
            _audit_stage1("data_missing_or_short"); _audit_data("data_missing_or_short")
            return None
        price = float(daily["Close"].iloc[-1])
        if price <= 0 or price > float(MAX_AUTO_PRICE):
            _audit_stage1("invalid_price")
            _audit_data("invalid_price")
            return None
        wc = weekly["Close"].astype(float); dc = daily["Close"].astype(float)
        we20 = float(_ema(wc,20).iloc[-1]); we50 = float(_ema(wc,50).iloc[-1])
        wrsi = float(_rsi(wc,14).iloc[-1]); drsi = float(_rsi(dc,14).iloc[-1])
        trend = price >= we20 * 0.99 and we20 >= we50 * 0.995 and drsi >= 42
        v = _vwap(daily.tail(60)); vw = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else price
        above_vwap = price >= vw * 0.995
        above_open = price >= float(daily["Open"].iloc[-1]) * 0.995
        vol_ratio = _daily_volume_ratio_time_of_day(symbol, fetch_intraday, now_ny())
        mom = (price - float(dc.iloc[-6])) / max(float(dc.iloc[-6]),1e-9) * 100 if len(dc)>=6 else 0.0
        route = (3.0 if trend else 0.0) + (2.0 if above_vwap else 0.0) + (1.5 if above_open else 0.0)
        route += min(2.5,max(0.0,mom)) + min(2.0,max(0.0,vol_ratio-0.75)*2.0) + (1.0 if we20 > we50 else 0.0)
        route -= 1.0 if wrsi >= 80 else 0.0
        if vol_ratio < 0.55:
            _audit_stage1("low_volume_ratio<0.55x")
            return None
        routes=_daily_strategy_route_scores(price=price, trend=trend, above_vwap=above_vwap, above_open=above_open, vol_ratio=vol_ratio, mom=mom, wrsi=wrsi, we20=we20, we50=we50, daily=daily, weekly=weekly)
        return route, routes, weekly, daily
    except Exception as exc:
        _audit_stage1("exception")
        log.info("DAILY STAGE 1 REJECT | %s | exception=%s", symbol, str(exc))
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
        market_condition = _daily_market_condition(market_context[1])
        # Centralized state thresholds; unknown remains non-supportive via
        # the market permission gate even though it has a conservative score floor.
        min_score = max(min_score, DAILY_MARKET_SCORE_BY_STATE.get(market_condition, DAILY_MARKET_SCORE_BY_STATE["غير مؤكد"]))
        log.info(
            "DAILY MARKET REGIME POLICY | condition=%s | state=%s | min_score=%d",
            market_condition, market_context[1], min_score,
        )
    except Exception as exc:
        log.warning("Daily market context unavailable after retries: %s", exc)
        market_context = (False, "بيانات SPY/QQQ غير متاحة")

    workers = min(8, max(2, len(symbols)))
    stage1=[]
    stage1_audit_counts: dict[str, int] = {}
    stage1_data_audit_counts: dict[str, int] = {}
    stage1_audit_passed = 0
    stage1_audit_rejected = 0
    stage1_audit_lock = Lock()
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
                return _prefilter_daily_from_frames(sym, weekly, daily, stage1_audit_counts, stage1_audit_lock, stage1_data_audit_counts)
            for sym in symbols:
                item = _route_from_frames(sym)
                if item is None:
                    # Partial-batch hardening: a missing symbol must get its own
                    # normal fetch path instead of being silently discarded.
                    try:
                        item = _prefilter_daily(sym, stage1_audit_counts, stage1_audit_lock, stage1_data_audit_counts)
                        if item:
                            log.info("DAILY INDIVIDUAL FALLBACK | %s | recovered after batch miss", sym)
                    except Exception as exc:
                        log.warning("DAILY INDIVIDUAL FALLBACK FAILED | %s | %s", sym, exc)
                        item = None
                if item:
                    route, routes, weekly, daily = item; stage1.append((route, routes, sym, weekly, daily))
                    with stage1_audit_lock: stage1_audit_passed += 1
                else:
                    with stage1_audit_lock: stage1_audit_rejected += 1
        else:
            raise RuntimeError("Alpaca not configured")
    except Exception as exc:
        log.warning("Daily batch scan unavailable; using per-symbol fallback: %s", exc)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_prefilter_daily,sym,stage1_audit_counts,stage1_audit_lock,stage1_data_audit_counts):sym for sym in symbols}
            for fut in as_completed(futures):
                sym=futures[fut]
                try: item=fut.result()
                except Exception as exc:
                    log.warning("DAILY STAGE 1 FUTURE EXCEPTION | %s | %s", sym, str(exc))
                    item=None
                if item:
                    route,routes,weekly,daily=item; stage1.append((route,routes,sym,weekly,daily))
                    with stage1_audit_lock: stage1_audit_passed += 1
                else:
                    with stage1_audit_lock: stage1_audit_rejected += 1
    stage1.sort(key=lambda x:(x[0], max(x[1].values()) if x[1] else 0.0), reverse=True)
    log.info("STAGE 1 DAILY: %d/%d passed | rejected=%d | audit_reasons=%s", len(stage1), len(symbols), len(symbols)-len(stage1), stage1_audit_counts)
    base_n=max(PREFILTER_MAX_CANDIDATES, limit*5)
    selected={item[2]:item for item in stage1[:base_n]}
    for et in ENTRY_TYPES:
        candidates=sorted((item for item in stage1 if float(item[1].get(et,0.0))>0.0), key=lambda item:float(item[1].get(et,0.0)), reverse=True)[:PREFILTER_STRATEGY_TOP_K]
        for item in candidates:
            selected[item[2]]=item
    # Protected strategy lanes: once a symbol enters the Top-K lane of any
    # strategy, it remains in the Stage-2 candidate pool. The union is
    # de-duplicated by symbol, so overlap never creates duplicate work.
    # Do not re-rank this union back through a smaller generic cap: doing so
    # would silently evict protected strategy lanes. The absolute cap is only
    # a safety ceiling above the mathematical maximum of 50 + (len(ENTRY_TYPES)*4).
    finalists=sorted(selected.values(), key=lambda item:(float(item[0]), max(item[1].values()) if item[1] else 0.0), reverse=True)[:min(PREFILTER_STRATEGY_CAP,base_n+len(ENTRY_TYPES)*PREFILTER_STRATEGY_TOP_K)]
    log.info("STAGE 2 DAILY: top %d", len(finalists))
    results=[]
    stage2_scores = []
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
        "final_execution": 0,
    }
    stage2_gate_reasons: dict[str, dict[str, int]] = {}
    stage2_gate_diag: dict[str, dict[str, int]] = {} 
    data_stage2_audit: dict[str, int] = {}
    strategy_matched: dict[str, int] = {}
    strategy_failed: dict[str, int] = {}
    strategy_blockers: dict[str, dict[str, int]] = {}
    def _stage2_reason(gate: str, reason: str, diagnostic: bool = False) -> None:
        target = stage2_gate_diag if diagnostic else stage2_gate_reasons
        _daily_bump(target.setdefault(gate, {}), reason)
    def _strategy_audit_from_signal(sig) -> None:
        matched = set(getattr(sig, "matched_entry_types", []) or [])
        scores_map = dict(getattr(sig, "strategy_scores", {}) or {})
        for _et in ENTRY_TYPES:
            if _et in matched:
                _daily_bump(strategy_matched, _et)
            else:
                _daily_bump(strategy_failed, _et)
                _daily_bump(strategy_blockers.setdefault(_et, {}), "not_matched")
    def one(item):
        _,_,sym,weekly,daily=item
        try:
            return analyze_daily(sym,names.get(sym,sym),True,market_context,(weekly,daily))
        except Exception as exc:
            log.warning("DAILY STAGE 2 EXCEPTION | %s | %s", sym, str(exc))
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,x): x[2] for x in finalists}
        for fut in as_completed(futures):
            _stage2_symbol = str(futures.get(fut, ""))
            try:
                sig=fut.result()
            except Exception as exc:
                stage2_rejects["exception"] += 1
                for _dr, _dc in _daily_data_audit_pop(_stage2_symbol).items():
                    _daily_bump(data_stage2_audit, _dr, _dc)
                _stage2_reason("exception", "future_exception")
                _stage2_reason("exception", str(exc)[:120], diagnostic=True)
                log.warning("DAILY STAGE 2 FUTURE EXCEPTION | %s", str(exc))
                continue
            if not sig:
                stage2_rejects["no_signal"] += 1
                _stage2_reason("no_signal", "analyze_daily_returned_none")
                with _DAILY_NO_SIGNAL_LOCK:
                    _ns_blockers = dict(_DAILY_NO_SIGNAL_AUDIT.pop(_stage2_symbol, {}) or {})
                for _reason, _count in _ns_blockers.items():
                    _stage2_reason("no_signal", _reason, diagnostic=True)
                for _dr, _dc in _daily_data_audit_pop(_stage2_symbol).items():
                    _daily_bump(data_stage2_audit, _dr, _dc)
                log.info(
                    "DAILY NO_SIGNAL AUDIT | %s | blockers=%s",
                    _stage2_symbol,
                    ";".join(f"{r}={c}" for r, c in sorted(_ns_blockers.items(), key=lambda kv: (-kv[1], kv[0]))[:8]) or "unavailable",
                )
                continue
            for _dr, _dc in _daily_data_audit_pop(_stage2_symbol).items():
                _daily_bump(data_stage2_audit, _dr, _dc)
            stage2_scores.append((str(getattr(sig, "symbol", "?")), float(getattr(sig, "score", 0) or 0)))
            _strategy_audit_from_signal(sig)
            raw_score = float(getattr(sig, "raw_score", sig.score) or 0.0)
            if raw_score < min_score:
                stage2_rejects["score"] += 1
                _stage2_reason("score", "raw_score_below_min")
                _stage2_reason("score", f"raw_score<{min_score}", diagnostic=True)
                continue
            if not sig.live_ok:
                stage2_rejects["live_ok"] += 1
                # Exact sub-conditions are calculated beside the live gate itself.
                live_reasons = list(getattr(sig, "live_gate_reasons", []) or [])
                for rr in (live_reasons or ["live_ok_unexplained"]):
                    _stage2_reason("live_ok", rr)
                continue
            if not sig.quality_ok:
                stage2_rejects["quality"] += 1
                qreasons = getattr(sig, "quality_reasons", None) or []
                log.info(
                    "DAILY QUALITY REJECT | %s | score=%s | reasons=%s | ext=%.2f%% | atr=%.2f%% | vol=%.2fx | h4=%s | market=%s | tp1R=%.2f",
                    sig.symbol, sig.score, qreasons or ["unspecified"],
                    float(getattr(sig, "ext_sma20", 0) or 0),
                    float(getattr(sig, "atr_pct", 0) or 0),
                    float(getattr(sig, "volume_ratio", 0) or 0),
                    getattr(sig, "h4_state", "?"),
                    getattr(sig, "market_state", "?"),
                    float(getattr(sig, "reward_r", 0) or 0),
                )
                for rr in (qreasons or ["quality_unexplained"]):
                    _stage2_reason("quality", rr)
                if not qreasons:
                    _stage2_reason("quality", "quality=False_unexplained", diagnostic=True)
                continue
            if sig.news_state == "negative":
                stage2_rejects["negative_news"] += 1
                _stage2_reason("negative_news", "negative_news")
                continue
            # Validate current bid/ask only for the Stage-2 finalists.
            # This keeps the full-universe scan fast while preventing Daily V2
            # from reporting a false "liquidity suitable" result.
            try:
                liq = _quote_liquidity(sig.symbol, sig.price)
                sig.spread_pct = round(float(liq.get("spread_pct", 0) or 0), 3)
                sig.expected_slippage_pct = round(float(liq.get("slippage_pct", 0) or 0), 3)
                sig.liquidity_ok = bool(liq.get("ok", False))
                if not sig.liquidity_ok:
                    stage2_rejects["liquidity"] += 1
                    _stage2_reason("liquidity", "liquidity_ok_false")
                    _stage2_reason("liquidity", f"spread={sig.spread_pct:.3f}%", diagnostic=True)
                    _stage2_reason("liquidity", f"slippage={sig.expected_slippage_pct:.3f}%", diagnostic=True)
                    continue
                if sig.spread_pct > MAX_SPREAD_PCT:
                    log.warning(
                        "DAILY LIQUIDITY WARNING | %s | spread=%.3f%% > %.2f%%",
                        sig.symbol, sig.spread_pct, MAX_SPREAD_PCT,
                    )
                if liq.get("quote_source") == "none" or float(liq.get("quote_age_min", 999) or 999) > QUOTE_MAX_AGE_MIN:
                    stage2_rejects["stale_or_no_quote"] += 1
                    _stage2_reason("stale_or_no_quote", "quote_source_none" if liq.get("quote_source") == "none" else "quote_stale")
                    _stage2_reason("stale_or_no_quote", f"age>{QUOTE_MAX_AGE_MIN}m", diagnostic=True)
                    continue
            except Exception as exc:
                stage2_rejects["liquidity"] += 1
                _stage2_reason("liquidity", "liquidity_exception")
                _stage2_reason("liquidity", str(exc)[:120], diagnostic=True)
                log.warning("DAILY LIQUIDITY CHECK FAILED | %s | %s", sig.symbol, str(exc))
                continue

            # Final execution gate: آخر Quote مستقل قبل قبول التنبيه.
            # لا يغيّر سعر التحليل أو Stop/TP.
            execution = _final_execution_snapshot(sig.symbol, sig.price)
            if not execution.get("ok"):
                stage2_rejects["final_execution"] += 1
                _stage2_reason("final_execution", "execution_ok_false")
                _exec_reason = str(execution.get("reason", "unspecified") or "unspecified")
                _stage2_reason("final_execution", f"reason:{_exec_reason}")
                if execution.get("quote_source") == "none":
                    _stage2_reason("final_execution", "quote_source_none")
                if float(execution.get("quote_age_min", 999) or 999) > QUOTE_MAX_AGE_MIN:
                    _stage2_reason("final_execution", f"quote_age>{QUOTE_MAX_AGE_MIN}m")
                if float(execution.get("spread_pct", 999) or 999) > MAX_SPREAD_PCT:
                    _stage2_reason("final_execution", f"spread>{MAX_SPREAD_PCT:.2f}%")
                _stage2_reason("final_execution", str(execution.get("reason", "unspecified")), diagnostic=True)
                log.info(
                    "DAILY FINAL EXECUTION REJECT | %s | source=%s age=%.2fm spread=%.3f%%",
                    sig.symbol, execution.get("quote_source"),
                    float(execution.get("quote_age_min", 999) or 999),
                    float(execution.get("spread_pct", 999) or 999),
                )
                continue
            sig.alert_entry_price = float(execution["entry_price"])
            results.append(sig)

    stage1_passed = int(stage1_audit_passed)
    stage1_rejected = int(stage1_audit_rejected)
    cumulative_audit = _commit_daily_audit(
        stage1_counts=stage1_audit_counts,
        stage1_passed=stage1_passed,
        stage1_rejected=stage1_rejected,
        stage2_deep=len(finalists),
        stage2_qualified=len(results),
        stage2_gate_counts=stage2_rejects,
        stage2_gate_reasons=stage2_gate_reasons,
        stage2_gate_diag=stage2_gate_diag,
        data_stage1=stage1_data_audit_counts,
        data_stage2=data_stage2_audit,
        strategy_matched=strategy_matched,
        strategy_failed=strategy_failed,
        strategy_blockers=strategy_blockers,
    )
    top_stage1 = sorted(cumulative_audit.get("stage1", {}).get("reasons", {}).items(), key=lambda x: x[1], reverse=True)[:8]
    top_stage2 = {gate: sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5] for gate, reasons in cumulative_audit.get("stage2", {}).get("reason_counts_by_gate", {}).items()}
    log.info("DAILY AUDIT CUMULATIVE | scans=%d | stage1_top=%s | stage2_top=%s", cumulative_audit.get("scans", 0), top_stage1, top_stage2)

    # Score distribution diagnostics only. These values are observational and
    # do not alter any selection rule. They show whether the zero-qualified
    # result is caused by scores being far below the threshold or merely just
    # below it.
    scores = [score for _, score in stage2_scores]
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        ge_min = sum(1 for score in scores if score >= float(min_score))
        b85_87 = sum(1 for score in scores if 85 <= score < 88)
        b80_84 = sum(1 for score in scores if 80 <= score < 85)
        lt80 = sum(1 for score in scores if score < 80)
        top10 = sorted(stage2_scores, key=lambda x: (-x[1], x[0]))[:10]
        top10_text = ", ".join(f"{sym}:{score:.1f}" for sym, score in top10)
        log.info(
            "STAGE 2 DAILY SCORE DISTRIBUTION: analyzed=%d | max=%.1f | avg=%.1f | >=%d=%d | 85-87=%d | 80-84=%d | <80=%d | top10=%s",
            len(scores), max_score, avg_score, int(min_score), ge_min,
            b85_87, b80_84, lt80, top10_text,
        )
    else:
        log.info("STAGE 2 DAILY SCORE DISTRIBUTION: analyzed=0 | no completed signals")

    log.info(
        "STAGE 2 DAILY RESULT: finalists=%d | qualified=%d | rejects=%s",
        len(finalists),
        len(results),
        stage2_rejects,
    )
    records_for_dedup = _read_learning_records()
    before_dedup = len(results)
    results = [sig for sig in results if not _signal_duplicate_recent(sig, records_for_dedup)]
    log.info("SIGNAL DEDUP | removed=%d | remaining=%d", before_dedup - len(results), len(results))
    rank={et:i for i,et in enumerate(ENTRY_TYPES)}
    results.sort(key=lambda x:(
        -(float(x.score)+1.5*min(float(getattr(x,'reward_r',0) or 0),3.0)
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

