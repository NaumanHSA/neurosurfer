/**
 * Pure, immutable graph-edit helpers (S4). Each returns a new Graph; callers
 * hold the working draft in state and re-render. Deletion/rename clean up every
 * reference (depends_on, routes/cases/default, on_error, outputs) so the graph
 * stays valid-shaped.
 */
import type { Graph, GraphNode, NodeKind } from "@/api/types";

export function cloneGraph(g: Graph): Graph {
  return structuredClone(g);
}

export function updateNode(
  g: Graph,
  id: string,
  patch: Partial<GraphNode>,
): Graph {
  return {
    ...g,
    nodes: g.nodes.map((n) => (n.id === id ? { ...n, ...patch } : n)),
  };
}

/** Unique id like `base_2` given existing ids. */
export function freshId(g: Graph, kind: NodeKind): string {
  const base = kind;
  const ids = new Set(g.nodes.map((n) => n.id));
  if (!ids.has(base)) return base;
  let i = 2;
  while (ids.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

const DEFAULTS: Record<string, Partial<GraphNode>> = {
  base: { goal: "" },
  react: { goal: "", tools: [] },
  function: {},
  python: {},
  tool: { tools: [] },
  router: { routes: {}, repair: true },
  loop: { max_iterations: 3, until: "", body: [] },
  map: { over: "", concurrency: 1, body: [] },
  subgraph: { body: [] },
  input: { purpose: "" },
};

export function addNode(g: Graph, kind: NodeKind, id?: string): { graph: Graph; id: string } {
  const nodeId = id ?? freshId(g, kind);
  const node: GraphNode = { id: nodeId, kind, depends_on: [], ...DEFAULTS[kind] };
  return { graph: { ...g, nodes: [...g.nodes, node] }, id: nodeId };
}

export function duplicateNode(g: Graph, id: string): { graph: Graph; id: string } {
  const src = g.nodes.find((n) => n.id === id);
  if (!src) return { graph: g, id };
  const newId = freshId(g, src.kind);
  // Copy but drop wiring so the clone is a clean, unconnected starting point.
  const copy: GraphNode = { ...structuredClone(src), id: newId, depends_on: [] };
  return { graph: { ...g, nodes: [...g.nodes, copy] }, id: newId };
}

/** Remove a node and scrub every reference to it. */
export function deleteNode(g: Graph, id: string): Graph {
  const nodes = g.nodes
    .filter((n) => n.id !== id)
    .map((n) => {
      const next: GraphNode = { ...n };
      if (next.depends_on) next.depends_on = next.depends_on.filter((d) => d !== id);
      if (next.on_error === id) next.on_error = null;
      if (next.default === id) next.default = null;
      if (next.routes) {
        const routes = Object.fromEntries(
          Object.entries(next.routes).filter(([, t]) => t !== id),
        );
        next.routes = routes;
      }
      if (next.cases) next.cases = next.cases.filter((c) => c.to !== id);
      return next;
    });
  return { ...g, nodes, outputs: g.outputs.filter((o) => o !== id) };
}

/** Toggle a dependency edge target→source (source depends_on target). */
export function setDependsOn(g: Graph, nodeId: string, deps: string[]): Graph {
  return updateNode(g, nodeId, { depends_on: deps });
}

/** Set/replace a router route label→target (empty target removes it). */
export function setRoute(
  g: Graph,
  routerId: string,
  label: string,
  target: string,
): Graph {
  const node = g.nodes.find((n) => n.id === routerId);
  const routes = { ...(node?.routes ?? {}) };
  if (target) routes[label] = target;
  else delete routes[label];
  return updateNode(g, routerId, { routes });
}

export function renameRoute(
  g: Graph,
  routerId: string,
  oldLabel: string,
  newLabel: string,
): Graph {
  const node = g.nodes.find((n) => n.id === routerId);
  if (!node?.routes) return g;
  const entries = Object.entries(node.routes).map(([k, v]) =>
    k === oldLabel ? [newLabel, v] : [k, v],
  );
  return updateNode(g, routerId, { routes: Object.fromEntries(entries) });
}

export function setOutputs(g: Graph, outputs: string[]): Graph {
  return { ...g, outputs };
}

export function setInputs(g: Graph, inputs: Graph["inputs"]): Graph {
  return { ...g, inputs };
}

/**
 * Wire source → target (data flows source→target). For a router source, `label`
 * creates a route AND the required depends_on (targets must depend on the
 * router); otherwise it just adds `target.depends_on = [...+source]`.
 */
export function connect(
  g: Graph,
  source: string,
  target: string,
  label?: string,
): Graph {
  if (source === target) return g;
  let next = g;
  const t = next.nodes.find((n) => n.id === target);
  const deps = new Set(t?.depends_on ?? []);
  if (!deps.has(source)) next = setDependsOn(next, target, [...deps, source]);
  if (label !== undefined) next = setRoute(next, source, label, target);
  return next;
}

/** Remove a wiring edge (data / route / error) between two nodes. */
export function disconnect(
  g: Graph,
  source: string,
  target: string,
  edgeKind: "data" | "route" | "error",
  label?: string,
): Graph {
  let next = g;
  if (edgeKind === "error") {
    return updateNode(next, source, { on_error: null });
  }
  if (edgeKind === "route") {
    const node = next.nodes.find((n) => n.id === source);
    const routes = { ...(node?.routes ?? {}) };
    if (label && routes[label] === target) delete routes[label];
    else
      for (const [k, v] of Object.entries(routes)) if (v === target) delete routes[k];
    next = updateNode(next, source, { routes });
    if (node?.default === target) next = updateNode(next, source, { default: null });
  }
  // Drop the data dependency unless another control edge still needs it.
  const src = next.nodes.find((n) => n.id === source);
  const stillRouted =
    !!src?.routes && Object.values(src.routes).includes(target);
  const stillDefault = src?.default === target;
  if (!stillRouted && !stillDefault) {
    const t = next.nodes.find((n) => n.id === target);
    next = setDependsOn(
      next,
      target,
      (t?.depends_on ?? []).filter((d) => d !== source),
    );
  }
  return next;
}

/** Names usable as `{template}` vars in a node's prompts, from the node's scope. */
export function availableVars(g: Graph, nodeId: string): string[] {
  const names = new Set<string>();
  for (const i of g.inputs) names.add(i.name);
  for (const n of g.nodes) {
    if (n.writes) names.add(n.writes);
    names.add(n.id); // dependency_results are keyed by node id
  }
  names.delete(nodeId);
  return [...names].sort();
}

/** `{tokens}` referenced in a node's prompts that aren't in the available set. */
export function unresolvedVars(g: Graph, nodeId: string): string[] {
  const node = g.nodes.find((n) => n.id === nodeId);
  if (!node) return [];
  const avail = new Set(availableVars(g, nodeId));
  const text = [node.goal, node.purpose, node.expected_result].filter(Boolean).join(" ");
  const refs = [...text.matchAll(/\{([A-Za-z_]\w*)\}/g)].map((m) => m[1]);
  return [...new Set(refs.filter((r) => !avail.has(r)))];
}
