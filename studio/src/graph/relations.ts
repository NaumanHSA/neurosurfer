/** Graph relationship helpers for the inspector. */
import type { Graph, GraphNode } from "@/api/types";

export function nodeById(graph: Graph, id: string): GraphNode | undefined {
  return graph.nodes.find((n) => n.id === id);
}

/** ids of nodes this node points at via routes/cases/default/on_error. */
export function targetsOf(node: GraphNode): string[] {
  const out = new Set<string>();
  if (node.routes) for (const t of Object.values(node.routes)) out.add(t);
  if (node.cases) for (const c of node.cases) out.add(c.to);
  if (node.default) out.add(node.default);
  if (node.on_error) out.add(node.on_error);
  return [...out];
}

/** ids of nodes that depend on / route to / backstop `id` (downstream). */
export function downstreamOf(graph: Graph, id: string): string[] {
  const out = new Set<string>();
  for (const n of graph.nodes) {
    if (n.id === id) continue;
    if ((n.depends_on ?? []).includes(id)) out.add(n.id);
    if (targetsOf(n).includes(id)) out.add(n.id);
  }
  return [...out];
}
