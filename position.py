"""حاسبة حجم الصفقة حسب رأس المال ونسبة المخاطرة."""

from __future__ import annotations


def calc_position(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
) -> dict:
    if capital <= 0 or risk_pct <= 0 or entry <= 0:
        raise ValueError("قيم غير صالحة")
    risk_amount = capital * (risk_pct / 100.0)
    per_share_risk = entry - stop_loss
    if per_share_risk <= 0:
        raise ValueError("الوقف يجب أن يكون تحت سعر الدخول")
    shares = int(risk_amount // per_share_risk)
    if shares < 1:
        shares = 0
    cost = shares * entry
    actual_risk = shares * per_share_risk
    return {
        "capital": capital,
        "risk_pct": risk_pct,
        "risk_amount": round(risk_amount, 2),
        "entry": entry,
        "stop_loss": stop_loss,
        "per_share_risk": round(per_share_risk, 4),
        "shares": shares,
        "cost": round(cost, 2),
        "actual_risk": round(actual_risk, 2),
        "actual_risk_pct": round(actual_risk / capital * 100, 3) if capital else 0,
    }


def format_position_ar(p: dict) -> str:
    if p["shares"] < 1:
        return (
            "⚠️ رأس المال أو نسبة المخاطرة صغيرة جداً لهذه المسافة.\n"
            f"المخاطرة المسموحة: {p['risk_amount']:.2f} $\n"
            f"مخاطرة السهم الواحد: {p['per_share_risk']:.4f} $\n"
            "زد رأس المال أو نسبة المخاطرة، أو اختر صفقة بوقف أقرب."
        )
    return "\n".join(
        [
            "💼 حاسبة حجم الصفقة",
            f"رأس المال: {p['capital']:,.2f} $",
            f"نسبة المخاطرة: {p['risk_pct']}%  →  {p['risk_amount']:,.2f} $",
            f"الدخول: {p['entry']:.2f} $",
            f"الوقف: {p['stop_loss']:.2f} $",
            f"مخاطرة السهم: {p['per_share_risk']:.4f} $",
            "",
            f"✅ عدد الأسهم المقترح: {p['shares']}",
            f"تكلفة الصفقة تقريباً: {p['cost']:,.2f} $",
            f"المخاطرة الفعلية: {p['actual_risk']:,.2f} $ ({p['actual_risk_pct']}%)",
            "",
            "قاعدة: لا تخاطر بأكثر مما حددت، ولا تدخل إذا العدد صفر.",
        ]
    )
