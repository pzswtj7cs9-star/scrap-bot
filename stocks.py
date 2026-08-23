"""
قائمة أسهم أمريكية شائعة التصنيف كحلال وفق معايير AAOIFI
(نشاط مباح + نسب الدين/الفائدة ضمن الحدود المتعارف عليها).

تنبيه مهم:
- التصنيف الشرعي يتغير مع القوائم المالية الفصلية.
- هذه القائمة نقطة بداية تعليمية وليست فتوى دائمة.
- راجع Zoya / Musaffa / HalalScreener / HalalWallet قبل الشراء.
"""

HALAL_STOCKS = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet Class C",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "AMZN": "Amazon",
    "TSM": "TSMC",
    "ASML": "ASML",
    "AMAT": "Applied Materials",
    "QCOM": "Qualcomm",
    "MU": "Micron",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "INTU": "Intuit",
    "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks",
    "KLAC": "KLA",
    "LRCX": "Lam Research",
    "SNPS": "Synopsys",
    "CDNS": "Cadence",
    "PEP": "PepsiCo",
    "ABBV": "AbbVie",
    "ABT": "Abbott",
    "LLY": "Eli Lilly",
    "AMGN": "Amgen",
    "ACN": "Accenture",
    "ADI": "Analog Devices",
    "ANET": "Arista Networks",
    "APH": "Amphenol",
    "TDG": "TransDigm",
    "CTAS": "Cintas",
}

# الأسهم الأساسية للمسح السريع (أقل ضغط على yfinance)
CORE_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "META",
    "AMD", "AVGO", "AMZN", "TSM", "AMAT", "QCOM",
    "ADBE", "CRM", "LLY", "ABBV", "PEP", "MU",
]


def is_known_halal(symbol: str) -> bool:
    return symbol.upper() in HALAL_STOCKS


def display_name(symbol: str) -> str:
    return HALAL_STOCKS.get(symbol.upper(), symbol.upper())
