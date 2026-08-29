"""
تعديل تلقائي محدود لشروط الدخول بناءً على نتائج السجل.
لا يغيّر النظام جذرياً — خطوة صغيرة بعد عدد كافٍ من الصفقات.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("halal-bot.autotune")

MIN_CLOSED = 20
SCORE_MIN = 84
SCORE_MAX = 88
VOL_MIN = 0.95
VOL_MAX = 1.10


class AutoTune:
    def __init__(self, path: Path | str = "auto_tune.json"):
        self.path = Path(path)
        self.min_score_boost = 0  # 0..4 تضاف فوق الحد الأساسي 84
        self.vol_gate = 1.00
        self.require_above_vwap = False
        self.require_vol_at_level = False
        self.last_note = "بانتظار 20 صفقة مغلقة للتعديل التلقائي"
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.min_score_boost = int(data.get("min_score_boost", 0))
            self.vol_gate = float(data.get("vol_gate", 1.00))
            self.require_above_vwap = bool(data.get("require_above_vwap", False))
            self.require_vol_at_level = bool(data.get("require_vol_at_level", False))
            self.last_note = str(data.get("last_note") or self.last_note)
        except Exception as exc:
            log.warning("auto_tune load: %s", exc)

    def save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(
                    {
                        "min_score_boost": self.min_score_boost,
                        "vol_gate": round(self.vol_gate, 2),
                        "require_above_vwap": self.require_above_vwap,
                        "require_vol_at_level": self.require_vol_at_level,
                        "last_note": self.last_note,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("auto_tune save: %s", exc)

    def effective_floor(self, base: int) -> int:
        return int(base) + int(self.min_score_boost)

    def volume_gate(self) -> float:
        return float(self.vol_gate)

    def recompute_from_rows(self, rows: list[dict[str, Any]]) -> bool:
        closed = [r for r in rows if r.get("status") == "closed"]
        if len(closed) < MIN_CLOSED:
            self.last_note = f"تعلم تلقائي متوقف — {len(closed)}/{MIN_CLOSED} صفقة مغلقة"
            self.save()
            return False

        def wr(subset: list[dict[str, Any]]) -> float | None:
            if len(subset) < 8:
                return None
            wins = [r for r in subset if (r.get("pnl_pct") or 0) > 0]
            return len(wins) / len(subset)

        low = [r for r in closed if 84 <= int(r.get("score") or 0) <= 89]
        high = [r for r in closed if int(r.get("score") or 0) >= 90]
        wr_low, wr_high = wr(low), wr(high)

        losses = [r for r in closed if (r.get("pnl_pct") or 0) <= 0]
        weak_vol_losses = sum(1 for r in losses if float(r.get("volume_ratio") or 1) < 1.05)
        weak_ratio = (weak_vol_losses / len(losses)) if losses else 0

        changed = False
        notes = []

        # إذا شريحة 84-89 أضعف بوضوح من 90+ → ارفع الحد نقطة
        if wr_low is not None and wr_high is not None and wr_high - wr_low >= 0.15:
            if self.min_score_boost < 4:
                self.min_score_boost += 1
                changed = True
                notes.append("رفع حد الدرجة نقطة لأن 84-89 أضعف من 90+")
        # إذا الشريحتان متقاربتان والنجاح العام ضعيف جداً → ارفع نقطة
        elif wr_low is not None and wr_low < 0.40 and self.min_score_boost < 2:
            self.min_score_boost += 1
            changed = True
            notes.append("رفع حد الدرجة نقطة لأن نجاح 84-89 ضعيف")
        # إذا النجاح جيد والحد مرتفع → نزول نقطة بحذر
        elif wr_low is not None and wr_low >= 0.55 and self.min_score_boost > 0:
            self.min_score_boost -= 1
            changed = True
            notes.append("خفض حد الدرجة نقطة لأن 84-89 نتائجها مقبولة")

        self.min_score_boost = max(0, min(4, self.min_score_boost))

        # حجم: خسائر كثيرة بحجم ضعيف → ارفع البوابة
        if weak_ratio >= 0.45 and self.vol_gate < VOL_MAX:
            self.vol_gate = min(VOL_MAX, round(self.vol_gate + 0.05, 2))
            changed = True
            notes.append("رفع شرط الحجم لأن كثيراً من الخسائر كانت بحجم ضعيف")
        elif weak_ratio <= 0.20 and self.vol_gate > VOL_MIN:
            # نجاح بدون ارتباط واضح بالحجم الضعيف → لا نخفض إلا بحذر شديد
            pass

        self.vol_gate = max(VOL_MIN, min(VOL_MAX, self.vol_gate))

        above = [r for r in closed if "فوق" in str(r.get("vwap_day_note") or "")]
        below = [r for r in closed if "تحت" in str(r.get("vwap_day_note") or "")]
        wr_up, wr_dn = wr(above), wr(below)
        if wr_up is not None and wr_dn is not None and wr_up - wr_dn >= 0.15:
            if not self.require_above_vwap:
                self.require_above_vwap = True
                changed = True
                notes.append("تفعيل شرط فوق VWAP لأن الإشارات تحته أضعف")
        elif wr_dn is not None and wr_dn >= 0.50 and self.require_above_vwap:
            self.require_above_vwap = False
            changed = True
            notes.append("إلغاء شرط VWAP لأن الإشارات تحته ليست ضعيفة")

        at = [r for r in closed if r.get("vol_at_level") is True]
        no = [r for r in closed if r.get("vol_at_level") is False]
        wr_at, wr_no = wr(at), wr(no)
        if wr_at is not None and wr_no is not None and wr_at - wr_no >= 0.15:
            if not self.require_vol_at_level:
                self.require_vol_at_level = True
                changed = True
                notes.append("تفعيل شرط الحجم عند المستوى لأن بدونه أضعف")
        elif wr_no is not None and wr_no >= 0.50 and self.require_vol_at_level:
            self.require_vol_at_level = False
            changed = True
            notes.append("إلغاء شرط الحجم عند المستوى لأن النتائج بدونه مقبولة")

        self.last_note = " | ".join(notes) if notes else "لا تعديل — النتائج ضمن النطاق"
        self.save()
        return changed

    def summary_text(self) -> str:
        return (
            "🤖 التعديل التلقائي:\n"
            f"• رفع الحد الحالي: +{self.min_score_boost} (حد فعلي من {SCORE_MIN + self.min_score_boost})\n"
            f"• بوابة الحجم: {self.vol_gate:.2f}x\n"
            f"• شرط فوق VWAP: {'نعم' if self.require_above_vwap else 'لا'}\n"
            f"• شرط الحجم عند المستوى: {'نعم' if self.require_vol_at_level else 'لا'}\n"
            f"• {self.last_note}"
        )


AUTO = AutoTune()
