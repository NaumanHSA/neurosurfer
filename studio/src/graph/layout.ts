/**
 * Layout: auto-arrange (dagre) + per-workflow position persistence.
 *
 * The adapter produces a first-pass layered layout; this module gives a nicer
 * dagre-based auto layout and lets the user's manual drags survive reloads by
 * saving positions to localStorage keyed by workflow name + version.
 */
import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "./adapter";

const NODE_W = 220;
const NODE_H = 128;

export type XY = { x: number; y: number };

/** Dagre left→right layered layout. Returns new nodes with fresh positions. */
export function autoLayout(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  rankdir: "LR" | "TB" = "LR",
): Node<FlowNodeData>[] {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir, nodesep: 48, ranksep: 96, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of edges) if (e.source !== e.target) g.setEdge(e.source, e.target);

  dagre.layout(g);

  return nodes.map((n) => {
    const p = g.node(n.id);
    return p
      ? { ...n, position: { x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 } }
      : n;
  });
}

// ── Persistence ─────────────────────────────────────────────────────────────

const key = (wf: string, version?: string) =>
  `ns-studio:layout:${wf}@${version ?? ""}`;

export function saveLayout(
  wf: string,
  version: string | undefined,
  nodes: Node<FlowNodeData>[],
): void {
  try {
    const map: Record<string, XY> = {};
    for (const n of nodes) map[n.id] = n.position;
    localStorage.setItem(key(wf, version), JSON.stringify(map));
  } catch {
    /* storage unavailable / quota — non-fatal */
  }
}

export function loadLayout(
  wf: string,
  version?: string,
): Record<string, XY> | null {
  try {
    const raw = localStorage.getItem(key(wf, version));
    return raw ? (JSON.parse(raw) as Record<string, XY>) : null;
  } catch {
    return null;
  }
}

export function clearLayout(wf: string, version?: string): void {
  try {
    localStorage.removeItem(key(wf, version));
  } catch {
    /* non-fatal */
  }
}

/** Apply a saved position map onto nodes; nodes without a saved pos keep theirs. */
export function applySaved(
  nodes: Node<FlowNodeData>[],
  saved: Record<string, XY> | null,
): Node<FlowNodeData>[] {
  if (!saved) return nodes;
  return nodes.map((n) => (saved[n.id] ? { ...n, position: saved[n.id] } : n));
}
