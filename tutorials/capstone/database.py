"""Synthetic sales dataset for the Insight Engine capstone.

Generates a seeded `orders` table in SQLite so the tutorial is fully reproducible.
The data carries two deliberate signals for the engine to discover: a gentle upward
revenue trend with a Q4 seasonal bump, and an injected anomaly (a viral
Electronics/Phone spike in *South* during *2025-11*).

Run standalone to (re)build the database::

    python database.py [path/to/insight.db]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(db_path: str | Path) -> None:
    """Write ~2,000 synthetic sales orders to an `orders` table (seeded, one anomaly)."""
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East", "West"]
    cats = {"Electronics": ["Laptop", "Phone", "Headset"],
            "Apparel": ["Jacket", "Shoes", "Tshirt"],
            "Home": ["Lamp", "Chair", "Rug"],
            "Sports": ["Bike", "Ball", "Racket"],
            "Beauty": ["Serum", "Lipstick", "Perfume"]}
    channels = ["Online", "Retail", "Partner"]
    months = pd.date_range("2024-07-01", "2025-12-01", freq="MS")
    rows, oid = [], 1000
    for i, m in enumerate(months):
        base = 80 + i * 6                            # upward trend
        season = 40 if m.month in (11, 12) else 0    # Q4 bump
        for _ in range(int(rng.integers(90, 130))):
            cat = rng.choice(list(cats))
            units = int(rng.integers(1, 6))
            price = round(float(rng.uniform(20, 400)), 2)
            rev = round(units * price * (1 + (base + season) / 400), 2)
            day = int(rng.integers(1, 28))
            rows.append((oid, m.replace(day=day).strftime("%Y-%m-%d"),
                         rng.choice(regions), cat, rng.choice(cats[cat]),
                         rng.choice(channels), units, price, rev))
            oid += 1
    # Injected anomaly: a viral Electronics spike in South during 2025-11.
    for _ in range(60):
        price, units = round(float(rng.uniform(300, 600)), 2), int(rng.integers(3, 9))
        rows.append((oid, "2025-11-%02d" % rng.integers(1, 28), "South", "Electronics",
                     "Phone", "Online", units, price, round(units * price * 1.8, 2)))
        oid += 1

    df = pd.DataFrame(rows, columns=["order_id", "order_date", "region", "category",
                                     "product", "channel", "units", "unit_price", "revenue"])
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS orders")
    df.to_sql("orders", con, index=False)
    con.commit(); con.close()
    print(f"dataset: {len(df):,} rows -> {db_path}")


if __name__ == "__main__":
    build_dataset(sys.argv[1] if len(sys.argv) > 1 else "insight.db")
