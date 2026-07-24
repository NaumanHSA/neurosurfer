/**
 * Transforms a neurosurfer graph (Phase-1 JSON) into React Flow nodes + edges.
 *
 * This is the core M0 adapter and the single place that reshapes server data
 * for the canvas. It derives:
 *
 *   - data-flow edges     from each node's `depends_on`
 *   - route edges         from a router's `routes` / `cases` / `default`
 *   - error edges         from any node's `on_error`
 *
 * Control edges (route/error) take precedence: when a router->target pair is
 * already expressed as a route, we don't also draw the plain data edge (the
 * engine's "targets must depend on the router" rule would otherwise double it).
 *
 * Layout is a simple longest-path layering over `depends_on` (left→right),
 * which reads well for the mostly-DAG workflows the engine produces. Nested
 * loop/map bodies are summarized on the container node for M0 (a body-node
 * count + ids); full nested rendering is a later milestone.
 */
import type { Edge, Node } from "@xyflow/react";
import type { Graph, GraphNode } from "@/api/types";

export type FlowNodeData = {
  node: GraphNode;
  /** ids of nested body nodes, for loop/map/subgraph summaries. */
  bodyIds: string[];
  /** live run status, injected by the canvas during/after a run (S3). */
  runStatus?: "pending" | "running" | "succeeded" | "failed" | "skipped";
  /** validation severity, injected by the canvas in edit mode (S4). */
  issue?: "error" | "warning";
};

export type EdgeKind = "data" | "route" | "error";

const X_GAP = 340;
const Y_GAP = 150;

/** Longest-path layer index per node id, over top-level depends_on edges. */
function computeLayers(nodes: GraphNode[]): Map<string, number> {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const layer = new Map<string, number>();
  const visiting = new Set<string>();

  const resolve = (id: string): number => {
    const cached = layer.get(id);
    if (cached !== undefined) return cached;
    const node = byId.get(id);
    if (!node || visiting.has(id)) return 0; // missing dep or cycle guard
    visiting.add(id);
    const deps = (node.depends_on ?? []).filter((d) => byId.has(d));
    const value = deps.length ? Math.max(...deps.map(resolve)) + 1 : 0;
    visiting.delete(id);
    layer.set(id, value);
    return value;
  };

  for (const n of nodes) resolve(n.id);
  return layer;
}

function edgeStyle(kind: EdgeKind, accent: string) {
  switch (kind) {
    case "route":
      return { stroke: accent, strokeWidth: 2 };
    case "error":
      return { stroke: "#ff5c5c", strokeWidth: 1.5, strokeDasharray: "6 4" };
    default:
      return { stroke: "#4a4f5a", strokeWidth: 1.5 };
  }
}

export function graphToFlow(graph: Graph): {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
} {
  const topLevel = graph.nodes;
  const layers = computeLayers(topLevel);

  // Position: group by layer, stack vertically within a layer.
  const byLayer = new Map<number, GraphNode[]>();
  for (const n of topLevel) {
    const l = layers.get(n.id) ?? 0;
    (byLayer.get(l) ?? byLayer.set(l, []).get(l)!).push(n);
  }

  const nodes: Node<FlowNodeData>[] = [];
  for (const [l, group] of [...byLayer.entries()].sort((a, b) => a[0] - b[0])) {
    const offset = ((group.length - 1) * Y_GAP) / 2;
    group.forEach((n, i) => {
      nodes.push({
        id: n.id,
        type: "workflow",
        position: { x: l * X_GAP, y: i * Y_GAP - offset },
        data: { node: n, bodyIds: (n.body ?? []).map((b) => b.id) },
      });
    });
  }

  const edges: Edge[] = [];
  const seen = new Set<string>(); // "source->target" pairs already drawn
  const add = (
    source: string,
    target: string,
    kind: EdgeKind,
    accent: string,
    label?: string,
  ) => {
    const key = `${source}->${target}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({
      id: `${key}:${kind}`,
      source,
      target,
      label,
      animated: kind === "route",
      style: edgeStyle(kind, accent),
      labelStyle: { fill: "#c7ccd6", fontSize: 11 },
      labelBgStyle: { fill: "#1e222b", fillOpacity: 0.9 },
      data: { kind },
    });
  };

  const ids = new Set(topLevel.map((n) => n.id));

  // Control edges first so they win over the plain data edge for the same pair.
  for (const n of topLevel) {
    if (n.kind === "router") {
      const accent = "#ff8a3d";
      if (n.routes) {
        for (const [label, target] of Object.entries(n.routes)) {
          if (ids.has(target)) add(n.id, target, "route", accent, label);
        }
      }
      if (n.cases) {
        for (const c of n.cases) {
          if (ids.has(c.to)) add(n.id, c.to, "route", accent, c.when ?? "case");
        }
      }
      if (n.default && ids.has(n.default)) {
        add(n.id, n.default, "route", accent, "default");
      }
    }
  }
  for (const n of topLevel) {
    if (n.on_error && ids.has(n.on_error)) {
      add(n.id, n.on_error, "error", "#ff5c5c", "on_error");
    }
  }
  // Data-flow edges last (skipped where a control edge already exists).
  for (const n of topLevel) {
    for (const dep of n.depends_on ?? []) {
      if (ids.has(dep)) add(dep, n.id, "data", "#4a4f5a");
    }
  }

  return { nodes, edges };
}
