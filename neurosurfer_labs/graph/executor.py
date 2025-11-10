# neurosurfer/agents/graph/executor.py
from __future__ import annotations
import asyncio, inspect, math
import json, re
from typing import Any, Dict, Tuple, List, Callable, Optional, Iterable
import contextlib, traceback, time

from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.models.chat_models.base import BaseModel

from .types import Graph, Node, Ref, GraphResult
from .errors import NodeError
from .tracing import Tracer, NullTracer
from .model_pool import ModelPool
from .artifacts import ArtifactStore, LocalArtifactStore


_TMPL_RE = re.compile(r"\$\{([^}]+)\}")

def _interpolate_template(text: str, ctx: dict, *, item=None) -> str:
    def _repl(m: re.Match) -> str:
        path = m.group(1).strip()
        if path == "item" and item is not None:
            return str(item)
        try:
            return str(_get_from_ctx(ctx, path))
        except Exception:
            return m.group(0)  # leave unresolved
    return _TMPL_RE.sub(_repl, text)

def _preview_value(v: Any, max_chars: int) -> str:
    try:
        if isinstance(v, (dict, list, tuple)):
            s = json.dumps(v, ensure_ascii=False, default=str)
        else:
            s = str(v)
    except Exception:
        s = f"<unserializable:{type(v).__name__}>"
    return s if len(s) <= max_chars else s[:max_chars] + "…"

def _apply_redact(v: Any, redactor: Optional[Callable[[str], str]], max_chars: int) -> str:
    s = _preview_value(v, max_chars)
    return redactor(s) if redactor else s

def _is_ref(x: Any) -> bool:
    return isinstance(x, Ref)

def _get_from_ctx(ctx: Dict[str, Any], path: str) -> Any:
    # path like "plan.subtopics" or "inputs.topic"
    cur: Any = ctx
    for part in path.split("."):
        if part.endswith("[]"):
            part = part[:-2]
        cur = cur[part]
    return cur

def _set_ctx(ctx: Dict[str, Any], node_id: str, outputs: Dict[str, Any]):
    if node_id not in ctx:
        ctx[node_id] = {}
    ctx[node_id].update(outputs)

def _sleep_backoff(kind: str, base: float, attempt: int):
    if kind == "fixed":
        return base
    # exponential with jitter-lite
    return base * (2 ** max(0, attempt - 1))

class GraphExecutor:
    """
    Async executor with safe parallelism and retries.
    """
    def __init__(
        self,
        llm: BaseModel,
        toolkit: Toolkit,
        model_pool: Optional[ModelPool] = None,
        tracer: Optional[Tracer] = None,
        artifact_store: Optional[ArtifactStore] = None,
        max_concurrency: int = 4,
        *,
        trace_payloads: bool = False,               # enable IO tracing
        trace_max_chars: int = 600,                 # per-field preview limit
        trace_redactor: Optional[Callable[[str], str]] = None,  # redact fn
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.model_pool = model_pool or ModelPool.from_llms([llm])
        self.tracer = tracer or NullTracer()
        self.artifacts = artifact_store or LocalArtifactStore()
        self.sema = asyncio.Semaphore(max_concurrency)

        # tracing config
        self._trace_payloads = trace_payloads
        self._trace_max_chars = int(trace_max_chars)
        self._trace_redactor = trace_redactor

    async def run(
        self,
        graph: Graph,
        inputs: Dict[str, Any],
        stream: bool = True,
    ) -> GraphResult:
        ctx: Dict[str, Any] = {"inputs": inputs}
        errors: Dict[str, str] = {}

        # Build dependency index
        deps: Dict[str, set] = {}
        for n in graph.nodes:
            dep_ids = set()
            for v in n.inputs.values():
                if _is_ref(v):
                    p = v.path
                    head = p.split(".", 1)[0]
                    if head not in ("inputs", n.id):
                        dep_ids.add(head)
            if n.kind == "map" and n.map_over:
                head = n.map_over.split(".", 1)[0]
                if head not in ("inputs", n.id):
                    dep_ids.add(head)
            deps[n.id] = dep_ids

        # Main loop
        pending = {n.id: n for n in graph.nodes}
        in_progress: set[str] = set()
        pending_tasks: set[asyncio.Task] = set()

        async def ready_nodes() -> List[Node]:
            ready: List[Node] = []
            for nid, node in list(pending.items()):
                if all(d not in pending and d not in in_progress for d in deps[nid]):
                    ready.append(node)
            return ready

        async def launch_node(n: Node):
            in_progress.add(n.id)
            async with self.sema:
                try:
                    with self.tracer.span(f"node:{n.id}", {"kind": n.kind, "fn": str(n.fn)}):
                        if n.kind == "map":
                            await self._run_map_node(n, ctx)
                        else:
                            out = await self._run_single_node(n, ctx)
                            _set_ctx(ctx, n.id, out)
                except Exception as e:
                    tb = traceback.format_exc()
                    errors[n.id] = tb
                    raise NodeError(n.id, f"Node failed: {e}", cause=e)
                finally:
                    in_progress.discard(n.id)
                    pending.pop(n.id, None)

        while pending or in_progress or pending_tasks:
            # 1) Launch all nodes that are now ready
            for node in await ready_nodes():
                t = asyncio.create_task(launch_node(node))
                pending_tasks.add(t)

            if not pending_tasks:
                # No runnable tasks yet; likely waiting on something to finish
                await asyncio.sleep(0.01)
                continue

            # 2) Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            # 3) Surface errors immediately (and optionally cancel the rest)
            for t in done:
                exc = t.exception()
                if exc:
                    # If you prefer to keep others running, remove this cancel loop
                    for p in pending_tasks:
                        p.cancel()
                    raise exc
                    
        # Compute outputs
        outs: Dict[str, Any] = {}
        for k, ref_path in (graph.outputs or {}).items():
            outs[k] = _get_from_ctx(ctx, ref_path[2:-1] if ref_path.startswith("${") else ref_path)

        return GraphResult(ok=len(errors) == 0, context=ctx, outputs=outs, errors=errors)

    async def _run_map_node(self, node: Node, ctx: Dict[str, Any]):
        if not node.map_over:
            raise NodeError(node.id, "map node requires map_over")
        seq = _get_from_ctx(ctx, node.map_over)
        if not isinstance(seq, (list, tuple)):
            raise NodeError(node.id, f"map_over must be list/tuple; got {type(seq)}")
        results = []
        for i, item in enumerate(seq):
            bound_inputs = self._materialize_inputs(node.inputs, ctx, item=item)
            # TRACE: map-item-inputs
            self._trace_node_io(node_id=node.id, phase="map-item-inputs", data=bound_inputs, extra={"index": i})
            out = await self._call_fn(node, bound_inputs)
            # TRACE: map-item-outputs
            self._trace_node_io(node_id=node.id, phase="map-item-outputs", data=out, extra={"index": i})
            results.append(out)
        collated: Dict[str, List[Any]] = {}
        for o in node.outputs:
            collated[o] = [res.get(o) for res in results]
        _set_ctx(ctx, node.id, collated)

    async def _run_single_node(self, node: Node, ctx: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._materialize_inputs(node.inputs, ctx)
        # TRACE: inputs
        self._trace_node_io(node_id=node.id, phase="inputs", data=payload)

        attempts = 0
        while True:
            attempts += 1
            try:
                out = await self._call_fn(node, payload)
                # TRACE: outputs
                self._trace_node_io(node_id=node.id, phase="outputs", data=out)
                return out
            except Exception as e:
                if attempts > max(1, node.policy.retries):
                    raise
                delay = _sleep_backoff(node.policy.backoff, node.policy.backoff_base, attempts)
                await asyncio.sleep(delay)

    def _materialize_inputs(self, inputs: Dict[str, Any], ctx: Dict[str, Any], *, item: Any = None) -> Dict[str, Any]:
        bound: Dict[str, Any] = {}
        for k, v in inputs.items():
            if _is_ref(v):
                bound[k] = _get_from_ctx(ctx, v.path)
            else:
                bound[k] = v
            if isinstance(bound[k], str):
                bound[k] = _interpolate_template(bound[k], ctx, item=item)
        if item is not None:
            bound.setdefault("item", item)
        return bound


    async def _call_fn(self, node: Node, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = node.fn
        # Resolve string function via toolkit registry (tools) or built-in llm.* helpers
        if isinstance(fn, str):
            # tool lookup
            tool = self.toolkit.registry.get(fn) if fn in self.toolkit.registry else None
            if tool:
                return await self._run_tool(node, tool, payload)
            # simple llm helper dispatch
            if fn.startswith("llm."):
                return await self._run_llm_program(node, fn, payload)
            raise NodeError(node.id, f"Unknown function: {fn}")

        # Direct callable
        if callable(fn):
            if inspect.iscoroutinefunction(fn):
                res = await fn(**payload)
            else:
                res = fn(**payload)
            return self._normalize_outputs(node, res)
        raise NodeError(node.id, f"Invalid node.fn: {type(fn)}")

    async def _run_tool(self, node: Node, tool: BaseTool, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Tools may stream; convert to text if needed
        try:
            resp = tool(**payload)
        except Exception as e:
            raise NodeError(node.id, f"Tool call error: {e}", cause=e)

        if not isinstance(resp, ToolResponse):
            # allow raw dict returns
            return self._normalize_outputs(node, resp)

        # Handle extras → artifact store
        for k, v in (resp.extras or {}).items():
            key = self.artifacts.put(v)
            payload[f"artifact_{k}"] = key  # pass handle forward if referenced later

        # observation may be str or a generator
        if hasattr(resp.observation, "__iter__") and not isinstance(resp.observation, (str, bytes)):
            # consume stream
            text = ""
            for chunk in resp.observation:
                if hasattr(chunk, "choices"):  # ChatCompletionChunk
                    text += chunk.choices[0].delta.content or ""
                else:
                    text += str(chunk)
            res = {"text": text}
        else:
            res = {"text": resp.observation}

        return self._normalize_outputs(node, res)

    async def _run_llm_program(self, node: Node, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal helpers:
        - llm.echo(text)
        - llm.compose_report(outline, body)
        - llm.summarize_pages(pages)
        Extend as needed; or switch to a general prompt-template loader.
        """
        model_name = node.policy.model_hint or getattr(self.llm, "model_name", "unknown")
        req_max_new = int(node.policy.budget.get("max_new_tokens", 512))
        # respect ModelPool metering
        await self.model_pool.acquire(model_name)
        try:
            max_new = self.model_pool.recommend_max_new_tokens(model_name, req_max_new)
            if name == "llm.echo":
                text = str(payload.get("text", ""))
                return self._normalize_outputs(node, {"text": text})

            if name == "llm.compose_report":
                outline = payload.get("outline", "")
                body = payload.get("body", "")
                prompt = f"Compose a clear report.\nOutline:\n{outline}\n\nBody:\n{body}\n"
                resp = self.llm.ask(
                    user_prompt=prompt,
                    system_prompt="You are a helpful writer.",
                    temperature=0.2,
                    max_new_tokens=max_new,
                    stream=False
                )
                return self._normalize_outputs(node, {"text": resp.choices[0].message.content})

            if name == "llm.summarize_pages":
                pages = payload.get("pages", [])
                joined = "\n\n---\n\n".join(pages if isinstance(pages, list) else [str(pages)])
                prompt = f"Summarize the key points from the following sources:\n{joined}"
                resp = self.llm.ask(
                    user_prompt=prompt,
                    system_prompt="Be concise and factual.",
                    temperature=0.2,
                    max_new_tokens=max_new,
                    stream=False
                )
                return self._normalize_outputs(node, {"summary": resp.choices[0].message.content})

            raise NodeError(node.id, f"Unknown llm program: {name}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e).lower():
                self.model_pool.notify_oom(model_name)
            raise
        finally:
            self.model_pool.release(model_name)

    def _trace_node_io(self, *, node_id: str, phase: str, data: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        """
        phase: 'inputs' | 'outputs' | 'map-item-inputs' | 'map-item-outputs'
        Emits a span with a pretty, truncated, optionally redacted preview.
        """
        if not self._trace_payloads:
            return
        preview = {k: _apply_redact(v, self._trace_redactor, self._trace_max_chars) for k, v in (data or {}).items()}
        attrs = {"phase": phase, "node": node_id, "io_preview": preview}
        if extra:
            attrs.update(extra)
        with self.tracer.span(f"node:{node_id}:{phase}", attrs):
            # no body; start/end timestamps + attrs do the job
            pass

    def _normalize_outputs(self, node: Node, res: Any) -> Dict[str, Any]:
        """
        If node.outputs is empty, assume {"text": ...}.
        If node.outputs is provided, try to pick those keys from res.
        If res is a string, map to {"text": res}.
        """
        if isinstance(res, str):
            res = {"text": res}
        if not isinstance(res, dict):
            res = {"value": res}

        if not node.outputs:
            # default
            return res
        out: Dict[str, Any] = {}
        for k in node.outputs:
            out[k] = res.get(k)
        return out
