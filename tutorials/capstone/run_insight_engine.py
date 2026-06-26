"""Insight Engine capstone — validated end-to-end runner (headless).

A DAG that profiles a synthetic sales dataset, analyses it three ways IN PARALLEL
(deterministic pandas stats, charts, and an LLM SQL-analyst over a sqlite MCP server),
reads the rendered dashboard with a VISION node, synthesises an executive summary, and
writes REPORT.md.

Run (conda env `LLMs`, LM Studio up with a vision-capable model loaded):

    MODEL="qwen/qwen3.5-9b" NS_PAR=3 python run_insight_engine.py

See NOTES.md for the full findings, the model comparison, and the integration gotchas
this script encodes (MCP loop bridge, matplotlib warmup, react-without-finish, etc.).
"""
import asyncio, os, sys, sqlite3, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = str(HERE.parent.parent)               # repo root (for source checkouts)
sys.path.insert(0, REPO)
sys.path.insert(0, str(HERE))                # so import_string finds insight_nodes

import numpy as np
import pandas as pd

from neurosurfer.llm.providers.openai import OpenAICompatProvider
from neurosurfer.config.mcp import McpServerConfig
from neurosurfer.mcp import McpManager
from neurosurfer.graph import Graph, GraphNode, GraphExecutor
from neurosurfer.graph.engine.schema import NodePolicy
from neurosurfer.tools import default_pool, ToolPool
from neurosurfer.tools.base import ToolContext

ART = HERE / "artifacts"; ART.mkdir(exist_ok=True)
DB = HERE / "insight.db"


# ── 1. synthetic dataset (seeded, with an injected anomaly) ──────────────────
def build_dataset(db_path: Path) -> None:
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East", "West"]
    cats = {"Electronics": ["Laptop", "Phone", "Headset"],
            "Apparel": ["Jacket", "Shoes", "Tshirt"],
            "Home": ["Lamp", "Chair", "Rug"],
            "Sports": ["Bike", "Ball", "Racket"],
            "Beauty": ["Serum", "Lipstick", "Perfume"]}
    channels = ["Online", "Retail", "Partner"]
    months = pd.date_range("2024-07-01", "2025-12-01", freq="MS")
    rows = []
    oid = 1000
    for i, m in enumerate(months):
        base = 80 + i * 6                       # upward trend
        season = 40 if m.month in (11, 12) else 0   # Q4 bump
        n = rng.integers(90, 130)
        for _ in range(n):
            region = rng.choice(regions)
            cat = rng.choice(list(cats))
            product = rng.choice(cats[cat])
            units = int(rng.integers(1, 6))
            price = round(float(rng.uniform(20, 400)), 2)
            rev = round(units * price * (1 + (base + season) / 400), 2)
            day = int(rng.integers(1, 28))
            rows.append((oid, m.replace(day=day).strftime("%Y-%m-%d"),
                         region, cat, product, rng.choice(channels), units, price, rev))
            oid += 1
    # Injected anomaly: a viral Electronics spike in South during 2025-11.
    for _ in range(60):
        price = round(float(rng.uniform(300, 600)), 2)
        units = int(rng.integers(3, 9))
        rows.append((oid, "2025-11-%02d" % rng.integers(1, 28), "South",
                     "Electronics", "Phone", "Online", units, price,
                     round(units * price * 1.8, 2)))
        oid += 1

    df = pd.DataFrame(rows, columns=["order_id", "order_date", "region", "category",
                                     "product", "channel", "units", "unit_price", "revenue"])
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS orders")
    df.to_sql("orders", con, index=False)
    con.commit(); con.close()
    print(f"dataset: {len(df):,} rows → {db_path}")


# ── 2. MCP loop-thread bridge ────────────────────────────────────────────────
# An MCP ClientSession is bound to the event loop that opened it. The synchronous
# GraphExecutor drives every react node in a *fresh* loop, so a node calling the
# session directly cross-loop times out. Fix: keep the session on ONE persistent
# background loop and marshal every call onto it via run_coroutine_threadsafe.
class McpLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        self.mgr = None
    def _run(self):
        asyncio.set_event_loop(self.loop); self.loop.run_forever()
    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()
    def connect(self, servers):
        self.mgr = McpManager(servers); return self._submit(self.mgr.connect_all())
    def bridged_tools(self):
        return [_BridgedTool(t, self.loop) for t in self.mgr.tools()]
    def close(self):
        if self.mgr: self._submit(self.mgr.aclose())
        self.loop.call_soon_threadsafe(self.loop.stop); self._t.join(timeout=5)


class _BridgedTool:
    """Delegate to the inner McpTool, but marshal the async `run`/`call` onto the
    MCP loop so the loop-bound session works from the executor's fresh node loops.
    NB: the agent calls tool.run(), so bridging `run` is the one that matters."""
    def __init__(self, inner, loop): self._inner, self._loop = inner, loop
    def __getattr__(self, n): return getattr(self._inner, n)
    async def run(self, raw, ctx):
        fut = asyncio.run_coroutine_threadsafe(self._inner.run(raw, ctx), self._loop)
        return await asyncio.wrap_future(fut)
    async def call(self, args, ctx):
        fut = asyncio.run_coroutine_threadsafe(self._inner.call(args, ctx), self._loop)
        return await asyncio.wrap_future(fut)


# ── 3. JupyterIO (auto-approve) ──────────────────────────────────────────────
class JupyterIO:
    async def ask(self, q, options=None): return "yes"
    async def request_plan_approval(self, plan): return True, ""
    async def request_shell_approval(self, cmd, reason): return True
    async def request_write_approval(self, path, summary): return "once"
    def notify(self, m): pass


def build_graph() -> Graph:
    return Graph(
        name="insight_engine",
        description="Profile a sales dataset, analyse it (stats + SQL + charts), read the charts with vision, and write a report.",
        nodes=[
            GraphNode(id="loader", kind="function", callable="insight_nodes:profile_dataset",
                      description="Profile the dataset."),
            GraphNode(id="stats", kind="function", callable="insight_nodes:compute_stats",
                      depends_on=["loader"], description="Deterministic pandas analytics."),
            GraphNode(id="charts", kind="function", callable="insight_nodes:make_charts",
                      depends_on=["loader"], description="Render PNG charts."),
            GraphNode(id="db_analyst", kind="react",
                      depends_on=["loader"],
                      tools=["list_tables", "describe_table", "run_sql"],
                      policy=NodePolicy(max_new_tokens=1200, temperature=0.2, timeout_s=180),
                      goal=("Use the SQL tools to answer the user's question with concrete numbers. "
                            "Run a few SELECT queries (totals by region, by category, and the monthly "
                            "trend). When you have the numbers, stop calling tools and write your findings "
                            "as your final reply — a short bulleted summary of the key figures.")),
            GraphNode(id="vision", kind="react",
                      depends_on=["charts"],
                      tools=["read_file"],
                      policy=NodePolicy(max_new_tokens=900, temperature=0.3, timeout_s=180),
                      goal=("The charts node printed a 'Dashboard image' path above. Call read_file ONCE on "
                            "that dashboard path to view it. Then stop calling tools and write your final "
                            "reply: describe the trend and any unusual spike you see across the three panels.")),
            GraphNode(id="synthesis", kind="react", tools=[],
                      depends_on=["stats", "db_analyst", "vision"],
                      policy=NodePolicy(max_new_tokens=1200, temperature=0.3, timeout_s=180),
                      goal=("Write a concise executive summary (≤ 6 short paragraphs) of sales performance "
                            "from the stats, the SQL findings, and the chart observations above. Call out the "
                            "biggest drivers and any anomaly, and recommend where to focus next quarter. "
                            "Write the summary as your reply; do not call any tool.")),
            GraphNode(id="report", kind="function", callable="insight_nodes:write_report",
                      depends_on=["synthesis", "stats", "vision", "db_analyst", "charts"],
                      description="Assemble REPORT.md."),
        ],
        outputs=["report"],
    )


def main():
    build_dataset(DB)

    # qwen/qwen3.5-9b is the validated model (fast + clean + vision). gemma-4-12b-qat
    # also works for vision but is slow and stalls on base nodes — see NOTES.md.
    model = os.environ.get("MODEL", "qwen/qwen3.5-9b")
    print("model:", model)
    provider = OpenAICompatProvider(model=model,
                                    base_url="http://localhost:1234/v1",
                                    api_key="lm-studio", context_window=36_000)
    # Local OpenAI-compatible models aren't auto-detected as vision-capable; force it on.
    provider.capabilities.supports_vision = True
    print("supports_vision:", provider.capabilities.supports_vision)

    # Warm up matplotlib on the MAIN thread so the parallel charts node doesn't
    # trip matplotlib's IPython backend hook when first touched in a worker thread.
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(Figure())

    bg = McpLoopThread()
    status = bg.connect([McpServerConfig(name="db", transport="stdio",
                                         command=sys.executable,
                                         args=[str(HERE / "insight_mcp_server.py"), str(DB)])])
    print("MCP:", [(s.name, s.connected, s.tools) for s in status])

    # Live MCP tools are deliberately NOT in workflow_node_tools(), so WorkflowRunner
    # can't see them — drive GraphExecutor directly with an explicit pool instead.
    builtins = default_pool().select(["read_file"]).all()
    pool = ToolPool([*bg.bridged_tools(), *builtins])
    print("pool:", pool.names())

    ctx = ToolContext(cwd=HERE, io=JupyterIO())
    graph = build_graph()
    par = int(os.environ.get("NS_PAR", "3"))
    print("parallelism:", par)
    executor = GraphExecutor(graph, provider=provider, native_tools=pool,
                             tool_ctx=ctx, parallelism=par)

    question = ("Analyse our sales performance over the period. Which regions and product "
                "categories drive revenue, what is the overall trend, and are there any unusual "
                "spikes or anomalies worth investigating?")

    t0 = time.time()
    def node_event(nid, status, *a):
        print(f"  [node] {nid:12} {status}  (+{time.time()-t0:.0f}s)", flush=True)
    result = executor.run({"user_intent": question, "db_path": str(DB),
                           "artifacts_dir": str(ART)}, node_event=node_event)
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(result.execution_summary(), f"  ({dt:.0f}s)")
    for nid, nr in result.nodes.items():
        st = "ok" if nr.ok else ("skip" if nr.skipped else "ERR")
        print(f"  {nid:12} [{st}] {nr.duration_ms} ms")
        if not nr.ok and nr.error:
            print(f"       error: {nr.error}")
    print("=" * 70)
    for nid in ("db_analyst", "vision", "synthesis"):
        print(f"\n----- {nid} -----\n{str(result.nodes[nid].raw_output)[:700]}")
    print("\n----- report -----\n", result.final.get("report"))
    bg.close()


if __name__ == "__main__":
    main()
