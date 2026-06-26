# Capstone Tutorial 05 — The Insight Engine (resume notes)

> Working notes for tutorial `05_capstone_insight_engine.ipynb` (NOT yet built).
> The **pipeline is fully validated end-to-end** via `run_insight_engine.py`. The
> remaining work is wrapping this validated code into the teaching notebook.
> Date: 2026-06-26.

## What this capstone is

Project **D — Insight Engine (Data Analyst with Vision)**, chosen from a shortlist.
It ties together *everything* the tutorials have covered into one graph that solves a
real problem: **point it at a sales dataset → it produces a professional Markdown insight
report**, using function nodes (custom tools), a react node over an **MCP** server,
a **vision** node that reads a chart, a base/react synthesis node, and **parallelism**.

It also uniquely exercises the just-landed **Phase-3 vision** support, which the
`INTEGRATIONS_PLAN.md` flags as the remaining missing tutorial.

## How to run (validated)

```bash
conda activate LLMs                       # only env with pytest+anthropic+neurosurfer deps
# LM Studio up on :1234 with a vision-capable model loaded (qwen/qwen3.5-9b)
cd tutorials/capstone
MODEL="qwen/qwen3.5-9b" NS_PAR=3 python run_insight_engine.py
```

Result: **7 nodes ok, 0 failed, ~45s**, writes `artifacts/REPORT.md` (sample committed as
`SAMPLE_REPORT.md`) + `artifacts/dashboard.png` (sample as `sample_dashboard.png`).

## Graph topology

```
inputs {user_intent, db_path, artifacts_dir}
            │
            ▼
        loader (function: profile_dataset)
            │
   ┌────────┼─────────────┐                ← PARALLEL layer (parallelism=3)
   ▼        ▼             ▼
 stats    charts      db_analyst           stats/charts = function (custom tools)
 (func)   (func)      (react: sqlite MCP)  db_analyst = react + MCP tools
   │        │             │
   │        ▼             │
   │      vision          │                vision = react + read_file (VISION on dashboard.png)
   │     (react)          │
   └────────┼─────────────┘
            ▼
        synthesis (react, tools=[])         executive summary as plain text
            │
            ▼
        report (function: write_report)     assembles REPORT.md
```

Node kinds: **function ×4** (loader, stats, charts, report), **react ×3** (db_analyst,
vision, synthesis). MCP in db_analyst. Vision in vision. Parallelism in the stats/charts/
db_analyst layer.

## Model decision (important)

- **qwen/qwen3.5-9b → USE THIS.** Vision-capable, fast (~45s whole graph), clean output,
  no stalls. Default in `run_insight_engine.py`.
- **google/gemma-4-12b-qat** → works for vision but is SLOW (300s+) and **stalls the
  provider stream on a base node that emits reasoning-only/empty content**. Avoid for the
  synthesis/base path. (Vision reads charts correctly; tool calls work.)
- Both are local OpenAI-compatible models, so `supports_vision` is False by name → we
  force `provider.capabilities.supports_vision = True`.

## The 5 integration gotchas this code encodes (the real teaching value)

1. **MCP sessions are event-loop-bound.** The synchronous `GraphExecutor` runs every react
   node in a *fresh* event loop (`run_coro_blocking` → `asyncio.run`), so calling a
   persistent MCP session cross-loop **times out (30s)**. Fix: keep the session on ONE
   persistent background loop thread (`McpLoopThread`) and marshal calls onto it with
   `asyncio.run_coroutine_threadsafe` + `asyncio.wrap_future`. **Bridge `run()`, not just
   `call()`** — the agent loop invokes `tool.run()` (loop.py:105), so wrapping only `call`
   silently falls through `__getattr__` to the unbridged inner method and still hangs.
2. **`python_exec` runs in a throwaway temp sandbox** — files it writes don't reach `cwd`,
   so a chart written there can't be read by a later node. Use a **`function` node** doing
   in-process matplotlib → a known `artifacts/` path instead.
3. **matplotlib in a worker thread** (parallelism) trips an IPython backend hook
   (`partially initialized module 'IPython'`). Fix: **warm up matplotlib on the main
   thread** once (`FigureCanvasAgg(Figure())`) and use the OO API (`Figure` +
   `FigureCanvasAgg`, no global pyplot). In a live Jupyter kernel IPython is already loaded
   so it's usually a no-op, but keep the warmup for safety.
4. **Local "thinking" models + clean text out of react nodes.** A react node that ends by
   calling `finish` puts its answer in `result.report`, but `run_react_node` returns
   `result.final_text` (the assistant *text*) → empty. Fix: tell the analysis nodes to
   **answer in plain text (no `finish`)**; the loop ends on a no-tool-call response and
   `final_text` is populated. Also cap `max_new_tokens` (via `NodePolicy`) so the model
   doesn't burn minutes reasoning. (gemma/qwen both dump chain-of-thought; in-context react
   answers come out clean, cold base-node answers get a "Thinking Process:" preamble.)
5. **Force vision on for local models** and read **one combined dashboard image** (not 3
   separate) — far less multimodal context = faster + more reliable on small VL models.

Also: **drive `GraphExecutor` directly** (not `WorkflowRunner`) for live MCP tools — MCP
tools are deliberately excluded from `workflow_node_tools()`, so the executor must be handed
an explicit `ToolPool([*bridged_mcp_tools, read_file])` + `ToolContext`.

## Files here

- `insight_nodes.py` — the 4 custom tools as `function`-node callables (loader/stats/charts/report).
- `insight_mcp_server.py` — tiny read-only sqlite MCP server (list_tables/describe_table/run_sql).
- `run_insight_engine.py` — validated headless runner (dataset gen + bridge + graph + run).
- `SAMPLE_REPORT.md` / `sample_dashboard.png` — a real generated report + dashboard (reference).

## Remaining work (resume here)

1. Build `tutorials/05_capstone_insight_engine.ipynb` from this code, in the series style
   (banner, Contents, incremental sections). Suggested sections: Setup → architecture →
   dataset → custom tools (function nodes) → MCP server + the loop bridge → build the graph
   → assemble pool + GraphExecutor → run → inspect outputs + render dashboard + REPORT.md →
   Summary (teach the 5 gotchas above) → What's next.
2. The notebook can write/import these modules (add `tutorials/capstone` to `sys.path`).
3. Execute the notebook with LM Studio up (qwen3.5-9b) to capture outputs.
4. Update `INTEGRATIONS_PLAN.md` TODO(X2) and link 05 from `04`'s What's Next.
