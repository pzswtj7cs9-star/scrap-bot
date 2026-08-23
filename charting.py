"""رسم شموع مع منطقة الشراء والوقف والأهداف."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf

from analyzer import SignalResult


def build_signal_chart(sig: SignalResult, out_dir: Path) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = yf.Ticker(sig.symbol).history(period="3mo", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return None
        df = df.tail(60).copy()
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        dates = pd.to_datetime(df[date_col])

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
        # شموع مبسطة
        for i, row in df.iterrows():
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            color = "#16a34a" if c >= o else "#dc2626"
            ax.plot([dates[i], dates[i]], [l, h], color=color, linewidth=1)
            ax.plot([dates[i], dates[i]], [o, c], color=color, linewidth=3.2)

        # مستويات
        ax.axhspan(sig.buy_low, sig.buy_high, color="#22c55e", alpha=0.18, label="منطقة الشراء")
        ax.axhline(sig.stop_loss, color="#ef4444", linestyle="--", linewidth=1.4, label=f"وقف {sig.stop_loss:.2f}")
        ax.axhline(sig.tp1, color="#3b82f6", linestyle=":", linewidth=1.2, label=f"هدف1 {sig.tp1:.2f}")
        ax.axhline(sig.tp2, color="#2563eb", linestyle=":", linewidth=1.2, label=f"هدف2 {sig.tp2:.2f}")
        ax.axhline(sig.tp3, color="#1d4ed8", linestyle=":", linewidth=1.2, label=f"هدف3 {sig.tp3:.2f}")
        ax.axhline(sig.price, color="#111827", linestyle="-", linewidth=1.0, alpha=0.7, label=f"السعر {sig.price:.2f}")

        ax.set_title(f"{sig.name} ({sig.symbol}) — درجة {sig.score}/100", fontsize=13)
        ax.set_ylabel("السعر $")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.autofmt_xdate()
        fig.tight_layout()

        path = out_dir / f"chart_{sig.symbol}_{dates.iloc[-1].strftime('%Y%m%d')}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        plt.close("all")
        return None
