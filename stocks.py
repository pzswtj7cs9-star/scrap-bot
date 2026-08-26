"""
قائمة أسهم أمريكية شائعة التصنيف كحلال وفق معايير AAOIFI.
المسح التلقائي: 35 سهماً مختارة لتكون عادة تحت 200$ + فلتر سعر وقت المسح.

تنبيه:
- التصنيف الشرعي يتغير فصلياً. راجع Zoya / Musaffa قبل الشراء.
- الأسعار تتحرك؛ أي سهم يتجاوز MAX_AUTO_PRICE يُستبعد من التنبيه التلقائي.
"""

MAX_AUTO_PRICE = 200.0

HALAL_STOCKS = {
    "INTC": "Intel",
    "ON": "ON Semiconductor",
    "SWKS": "Skyworks",
    "MCHP": "Microchip",
    "QCOM": "Qualcomm",
    "MRVL": "Marvell",
    "QRVO": "Qorvo",
    "SMTC": "Semtech",
    "CRUS": "Cirrus Logic",
    "STM": "STMicroelectronics",
    "AMKR": "Amkor",
    "UMC": "United Microelectronics",
    "ASX": "ASE Technology",
    "GLW": "Corning",
    "CSCO": "Cisco",
    "HPQ": "HP",
    "FTNT": "Fortinet",
    "PYPL": "PayPal",
    "PEP": "PepsiCo",
    "ORCL": "AbbVie",
    "ABT": "Abbott",
    "CRWD": "Amgen",
    "MDT": "Medtronic",
    "GILD": "Gilead",
    "BMY": "Bristol-Myers",
    "PFE": "Pfizer",
    "FAST": "Fastenal",
    "PAYX": "Paychex",
    "MRK": "Ross Stores",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "TGT": "Target",
    "DIOD": "Diodes",
    "KLIC": "Kulicke & Soffa",
    "PG": "Silicon Motion",
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMD": "AMD",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "MU": "Micron",
    "TSM": "TSMC",
    "AMAT": "Applied Materials",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "LLY": "Eli Lilly",
}

CORE_WATCHLIST = [
    "INTC", "ON", "SWKS", "MCHP", "QCOM", "MRVL",
    "QRVO", "SMTC", "CRUS", "STM", "AMKR", "UMC",
    "ASX", "GLW", "CSCO", "HPQ", "FTNT", "PYPL",
    "PEP", "ORCL", "ABT", "CRWD", "MDT", "GILD",
    "BMY", "PFE", "FAST", "PAYX", "MRK", "NKE",
    "SBUX", "TGT", "DIOD", "KLIC", "PG",
]


def is_known_halal(symbol: str) -> bool:
    return symbol.upper() in HALAL_STOCKS


def display_name(symbol: str) -> str:
    return HALAL_STOCKS.get(symbol.upper(), symbol.upper())
