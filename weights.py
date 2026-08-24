"""
أوزان التسجيل التكيفية.
تُحدَّث تلقائياً بناءً على نتائج الإشارات السابقة (win-rate لكل عامل).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger("halal-bot.weights")

# الأوزان الافتراضية (نقاط)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "uptrend_200": 20.0,      # السعر + SMA50 فوق SMA200
    "price_above_200": 10.0,
    "stacked_ma": 12.0,       # 20 > 50 > 200
    "sma20_above_50": 6.0,
    "rsi_healthy": 14.0,      # 50-64
    "rsi_near": 8.0,          # 45-50
    "rsi_strong": 5.0,        # 64-70
    "macd_expand": 14.0,
    "macd_positive": 8.0,
    "vol_high": 10.0,
    "vol_ok": 5.0,
    "near_sma20": 10.0,
    "bounce_sma20": 8.0,
    "structure_hold": 6.0,
    # إطار 15 دقيقة
    "m15_confirm": 10.0,
    "m15_partial": 5.0,
    # إطار 5 دقائق
    "m5_vwap_ema": 8.0,
    "m5_partial": 4.0,
    "m5_green": 3.0,
    "m5_rsi": 4.0,
    "m5_vol": 3.0,
    "m5_momentum": 2.0,
}

# حدود التعديل (±30% من القيمة الافتراضية)
MIN_MULT = 0.70
MAX_MULT = 1.30


class AdaptiveWeights:
    def __init__(self, path: Path | str = "weights_state.json"):
        self.path = Path(path)
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        self.stats: Dict[str, dict] = {}  # factor -> {wins, total}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.weights = {**DEFAULT_WEIGHTS, **data.get("weights", {})}
            self.stats = data.get("stats", {})
        except Exception as exc:
            log.warning("weights load: %s", exc)

    def save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(
                    {"weights": self.weights, "stats": self.stats},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("weights save: %s", exc)

    def get(self, key: str) -> float:
        return float(self.weights.get(key, DEFAULT_WEIGHTS.get(key, 5.0)))

    def record_outcome(self, factors: list[str], won: bool) -> None:
        """تسجيل نتيجة إشارة لعوامل معينة."""
        for f in factors:
            if f not in self.stats:
                self.stats[f] = {"wins": 0, "total": 0}
            self.stats[f]["total"] += 1
            if won:
                self.stats[f]["wins"] += 1
        self._recompute()
        self.save()

    def _recompute(self) -> None:
        """إعادة حساب الأوزان بناءً على win-rate."""
        for key, base in DEFAULT_WEIGHTS.items():
            st = self.stats.get(key)
            if not st or st["total"] < 8:  # نحتاج عينات كافية
                self.weights[key] = base
                continue
            wr = st["wins"] / st["total"]
            # win-rate 50% → مضاعف 1.0
            # 70% → ~1.20 ، 30% → ~0.80
            mult = 0.6 + (wr * 0.8)
            mult = max(MIN_MULT, min(MAX_MULT, mult))
            self.weights[key] = round(base * mult, 2)

    def summary_text(self) -> str:
        lines = ["⚖️ الأوزان التكيفية (بعد التعلم):"]
        changed = []
        for k, v in sorted(self.weights.items()):
            base = DEFAULT_WEIGHTS.get(k, v)
            if abs(v - base) > 0.3:
                changed.append(f"• {k}: {base:.0f} → {v:.1f}")
        if changed:
            lines.extend(changed[:12])
        else:
            lines.append("لا تغييرات كبيرة بعد (تحتاج مزيداً من النتائج).")
        return "\n".join(lines)


# نسخة عامة مشتركة
WEIGHTS = AdaptiveWeights()
