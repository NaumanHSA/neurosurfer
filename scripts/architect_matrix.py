#!/usr/bin/env python
"""Run one Architect build across several models and compare the outcomes.

**The check that would have caught the fault this exists because of.** The live
suite pins one model per run through `NEUROSURFER_TEST_*`, so "works on the model
I happened to try" was indistinguishable from "works". It was not: `gpt-5.1`
declared a two-node summarise-and-title workflow infeasible while `gpt-5-mini`
and a local 9B built it, and nothing in the suite compared them.

What matters here is not which model is best. It is that the **outcome kind** is
the same everywhere — a build either registers or names a missing capability, and
"the model gave up" is neither.

Usage:

    python scripts/architect_matrix.py
    python scripts/architect_matrix.py --models gpt-5-mini,gpt-5.1
    python scripts/architect_matrix.py --intent "Read a CSV and chart the totals"

Hosted models need `OPENAI_API_KEY`; local ones need the server up. A model that
cannot be reached is reported as `skipped`, never as a pass.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: One hosted mid, one hosted strong, one local. The point of the row is the
#: *class* of model, so replace freely — the comparison is what matters.
DEFAULT_MODELS = ("gpt-5-mini", "gpt-5.1", "qwen/qwen3.5-9b")

DEFAULT_INTENT = (
    "Take a short article text as input, summarise it in three sentences, "
    "and then write a catchy title for the summary."
)

LOCAL_BASE_URL = "http://localhost:1234/v1"


def _provider(model: str):
    """Hosted OpenAI for a bare model id, LM Studio for anything with a slash."""
    from neurosurfer.llm.providers.openai import OpenAICompatProvider, OpenAIProvider

    if "/" in model:
        return OpenAICompatProvider(
            base_url=LOCAL_BASE_URL, api_key="lm-studio",
            model=model, context_window=32768,
        )
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAIProvider(api_key=key, model=model)


async def run_one(model: str, intent: str, root: Path) -> dict:
    """Build *intent* on *model*. Never raises — a failure is a result."""
    from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible
    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.registry import WorkflowRegistry
    from neurosurfer.graph.workflow.validation import validate_package

    out: dict = {"model": model, "outcome": "?", "detail": "", "seconds": 0}
    started = time.time()
    try:
        provider = _provider(model)
    except Exception as e:  # noqa: BLE001 - an unreachable model is not a verdict
        out["outcome"] = "skipped"
        out["detail"] = f"{type(e).__name__}: {e}"
        return out

    work = root / model.replace("/", "_")
    agent = ArchitectAgent(
        provider,
        registry=WorkflowRegistry(workflows_dir=work / "registry"),
        staging_root=work / "staging",
        notify=lambda m: print(f"    [{model}] {m}", flush=True),
        max_turns=30,
    )
    try:
        path = await agent.build(intent)
        pkg = load_package(Path(path))
        report = validate_package(pkg)
        out["outcome"] = "registered" if report.ok else "registered-invalid"
        out["detail"] = (
            f"{len(pkg.graph.nodes)} nodes"
            + (f", {len(report.errors)} validation errors" if not report.ok else "")
            + (f", {agent.session.failed_verifications} failed verifications"
               if agent.session.failed_verifications else "")
        )
    except WorkflowInfeasible as e:
        # The outcome this script exists to make visible. Legitimate for a
        # request needing something nobody has — and a bug for one that does not.
        out["outcome"] = "BLOCKED"
        out["detail"] = str(e)[:300]
    except Exception as e:  # noqa: BLE001 - report, never abort the matrix
        out["outcome"] = "ERROR"
        out["detail"] = f"{type(e).__name__}: {str(e)[:300]}"
    out["seconds"] = int(time.time() - started)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--intent", default=DEFAULT_INTENT)
    ap.add_argument("--out", default=str(REPO / "tutorials" / "tmp" / "matrix"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    print(f"\nintent: {args.intent}\nmodels: {', '.join(models)}\n")
    results = []
    for model in models:
        print(f"── {model} " + "─" * max(0, 60 - len(model)))
        results.append(await run_one(model, args.intent, root))

    print("\n" + "=" * 78)
    print(f"{'model':22} {'outcome':20} {'s':>5}  detail")
    print("-" * 78)
    for r in results:
        print(f"{r['model']:22} {r['outcome']:20} {r['seconds']:>5}  {r['detail'][:60]}")

    # A build that gave up on a request nothing is missing from is the fault this
    # script watches for. Non-zero so it can gate CI when it is worth doing so.
    bad = [r for r in results if r["outcome"] in {"BLOCKED", "ERROR", "registered-invalid"}]
    ran = [r for r in results if r["outcome"] != "skipped"]
    print("=" * 78)
    if not ran:
        print("nothing ran — no model was reachable.")
        return 1
    if bad:
        print(f"\n{len(bad)}/{len(ran)} model(s) did not produce a workflow:")
        for r in bad:
            print(f"  {r['model']}: {r['outcome']} — {r['detail'][:200]}")
        return 1
    print(f"\nall {len(ran)} reachable model(s) registered a valid workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
