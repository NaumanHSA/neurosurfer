"""Custom tools for the Insight Engine capstone, exposed as graph `function` nodes.

Every graph `function` node calls one of these with ``fn(**{**graph_inputs, **dep_outputs})``
— so each function accepts ``**kwargs`` and pulls out only the keys it needs. Node
*ids* become kwarg names for their outputs (e.g. the ``synthesis`` node's text arrives
as ``synthesis=...``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Object-oriented matplotlib (no global pyplot state) so charts are thread-safe
# when the graph runs nodes in parallel.
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

BLUE, GREY = "#3b7dd8", "#9aa7b5"


def _orders(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM orders", con, parse_dates=["order_date"])
    finally:
        con.close()
    df["month"] = df["order_date"].dt.strftime("%Y-%m")
    return df


# ── loader ───────────────────────────────────────────────────────────────────
def profile_dataset(db_path: str, **_) -> str:
    """Profile the dataset: shape, date range, columns, head. (loader node)"""
    df = _orders(db_path)
    lines = [
        f"Rows: {len(df):,}   Columns: {len(df.columns)}",
        f"Date range: {df['order_date'].min():%Y-%m-%d} → {df['order_date'].max():%Y-%m-%d}",
        f"Total revenue: ${df['revenue'].sum():,.0f}",
        "",
        "Columns: " + ", ".join(df.columns),
        "",
        "First 5 rows:",
        df.head(5).to_string(index=False),
    ]
    return "\n".join(lines)


# ── stats ────────────────────────────────────────────────────────────────────
def compute_stats(db_path: str, **_) -> str:
    """Deterministic pandas analytics: breakdowns + trend + MoM growth. (stats node)"""
    df = _orders(db_path)

    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    by_month = df.groupby("month")["revenue"].sum()
    mom = by_month.pct_change().mul(100).round(1)
    top_products = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(5)

    def fmt(s: pd.Series, money=True) -> str:
        return "\n".join(
            f"  {k:<14} {'$' + format(v, ',.0f') if money else format(v, '.1f') + '%'}"
            for k, v in s.items()
        )

    return "\n".join([
        "REVENUE BY REGION", fmt(by_region), "",
        "REVENUE BY CATEGORY", fmt(by_cat), "",
        "MONTHLY REVENUE (last 6)", fmt(by_month.tail(6)), "",
        "MONTH-OVER-MONTH GROWTH % (last 6)", fmt(mom.tail(6), money=False), "",
        "TOP 5 PRODUCTS", fmt(top_products),
    ])


# ── charts ───────────────────────────────────────────────────────────────────
def make_charts(db_path: str, artifacts_dir: str, **_) -> str:
    """Render 3 PNG charts into artifacts_dir, return their paths. (charts node)

    Uses the OO matplotlib API (Figure + FigureCanvasAgg) — no global pyplot
    state — so it is safe to run in parallel with other nodes.
    """
    df = _orders(db_path)
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = out / name
        FigureCanvasAgg(fig).print_png(p)
        paths.append(str(p.resolve()))

    # 1. Monthly revenue trend (line)
    by_month = df.groupby("month")["revenue"].sum()
    fig = Figure(figsize=(8, 3.5)); ax = fig.subplots()
    ax.plot(by_month.index, by_month.values, marker="o", color=BLUE)
    ax.set_title("Monthly Revenue Trend"); ax.set_ylabel("Revenue ($)")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    save(fig, "monthly_revenue.png")

    # 2. Revenue by region (bar)
    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    fig = Figure(figsize=(6, 3.5)); ax = fig.subplots()
    ax.bar(by_region.index, by_region.values, color=BLUE)
    ax.set_title("Revenue by Region"); ax.set_ylabel("Revenue ($)"); fig.tight_layout()
    save(fig, "revenue_by_region.png")

    # 3. Revenue by category (bar)
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    fig = Figure(figsize=(6, 3.5)); ax = fig.subplots()
    ax.bar(by_cat.index, by_cat.values, color=GREY)
    ax.set_title("Revenue by Category"); ax.set_ylabel("Revenue ($)"); fig.tight_layout()
    save(fig, "revenue_by_category.png")

    # 4. A single combined dashboard (all three panels) — this is the ONE image the
    #    vision node reads, keeping the multimodal context small and fast.
    fig = Figure(figsize=(15, 4)); axes = fig.subplots(1, 3)
    axes[0].plot(by_month.index, by_month.values, marker="o", color=BLUE)
    axes[0].set_title("Monthly Revenue Trend"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(by_region.index, by_region.values, color=BLUE)
    axes[1].set_title("Revenue by Region")
    axes[2].bar(by_cat.index, by_cat.values, color=GREY)
    axes[2].set_title("Revenue by Category"); fig.tight_layout()
    dash = out / "dashboard.png"
    FigureCanvasAgg(fig).print_png(dash)

    return (f"Dashboard image (read THIS one with vision): {dash.resolve()}\n"
            "Individual charts:\n" + "\n".join(paths))


# ── report ───────────────────────────────────────────────────────────────────
def write_report(artifacts_dir: str, **kw) -> str:
    """Assemble REPORT.md from upstream node outputs. (report node)

    Pulls dependency outputs by node id from kwargs: synthesis, stats, vision,
    db_analyst.
    """
    out = Path(artifacts_dir)
    charts = ["monthly_revenue.png", "revenue_by_region.png", "revenue_by_category.png"]
    md = [
        "# Sales Insight Report",
        "_Generated by the Insight Engine graph (function + react + vision + MCP nodes)._\n",
        str(kw.get("synthesis", "(none)")), "",
        "## Supporting Detail\n",
        "### Database Analysis (SQL via MCP)\n", str(kw.get("db_analyst", "(none)")), "",
        "### Chart Observations (vision)\n", str(kw.get("vision", "(none)")), "",
        "### Key Metrics\n", "```\n" + str(kw.get("stats", "(none)")) + "\n```", "",
        "### Charts\n", *[f"![{c}]({c})" for c in charts],
    ]
    report = out / "REPORT.md"
    report.write_text("\n".join(md), encoding="utf-8")
    return f"Report written to {report.resolve()} ({len(report.read_text())} chars)."
