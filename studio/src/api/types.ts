/**
 * Types mirroring the neurosurfer Phase-2 workflow API and the Phase-1 graph
 * schema. These are the exact JSON shapes returned by the gateway:
 *
 *   GET /v1/workflows            -> { workflows: WorkflowSummary[] }
 *   GET /v1/workflows/{name}     -> WorkflowDetail  (graph JSON for the canvas)
 *   POST /v1/workflows/{name}/runs -> RunRecord
 *   GET /v1/runs                 -> { runs: RunRecord[] }
 *   GET /v1/runs/{id}            -> RunRecord  (?events=true adds `events`)
 *   GET /v1/runs/{id}/events     -> SSE stream of RunEvent (then `[DONE]`)
 *   GET /v1/runs/{id}/nodes/{id} -> RunNodeDetail
 *
 * Kept deliberately close to the server models (see graph/engine/schema.py and
 * app/server/workflow_runs/store.py) so the adapter layer is the only place that
 * reshapes anything.
 */

/** The ten engine node kinds (schema.py `_VALID_NODE_KINDS`). */
export type NodeKind =
  | "base"
  | "react"
  | "function"
  | "python"
  | "tool"
  | "router"
  | "loop"
  | "map"
  | "subgraph"
  | "input";

/** One branch of an expression router: `when` truthy -> `to`. */
export interface RouterCase {
  when?: string | null;
  to: string;
}

/**
 * A single graph node. Mirrors GraphNode in graph/engine/schema.py. Only the
 * fields the studio reads are typed explicitly; the index signature keeps any
 * additional/less-common fields (policy, tool, etc.) accessible without loss.
 */
export interface GraphNode {
  id: string;
  kind: NodeKind;
  purpose?: string | null;
  goal?: string | null;
  expected_result?: string | null;
  tools?: string[];
  depends_on?: string[];
  // Control flow
  when?: string | null;
  writes?: string | null;
  on_error?: string | null;
  // Router
  routes?: Record<string, string> | null;
  cases?: RouterCase[] | null;
  default?: string | null;
  // Loop / map
  body?: GraphNode[] | null;
  max_iterations?: number | null;
  until?: string | null;
  break_when?: string | null;
  accumulate?: string | null;
  over?: string | null;
  concurrency?: number;
  [key: string]: unknown;
}

export interface GraphInput {
  name: string;
  [key: string]: unknown;
}

/** The `graph` object nested in a WorkflowDetail (Graph in schema.py). */
export interface Graph {
  name: string;
  description?: string | null;
  fail_fast?: boolean;
  strict_inputs?: boolean;
  inputs: GraphInput[];
  nodes: GraphNode[];
  outputs: string[];
}

export interface WorkflowSummary {
  name: string;
  description?: string | null;
  version?: string;
  tags?: string[];
  error?: string;
}

export interface WorkflowDetail {
  name: string;
  description?: string | null;
  version?: string;
  graph: Graph;
}

/** One issue from the validate endpoint (mirrors ValidationIssue). */
export interface ValidationIssue {
  kind: string;
  message: string;
  node_id?: string | null;
  suggestion?: string | null;
  subject?: string | null;
}

export interface ValidationReport {
  ok: boolean;
  errors: ValidationIssue[];
  gaps: ValidationIssue[];
  warnings: ValidationIssue[];
}

// ── Runs ────────────────────────────────────────────────────────────────────

export type RunStatus =
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "awaiting_input";

/** Per-node status as reported in run events / node records. */
export type NodeRunStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped"
  | string;

export interface RunNodeResult {
  status?: NodeRunStatus;
  output?: unknown;
  error?: string | null;
  skip_reason?: string | null;
  duration_s?: number;
  [key: string]: unknown;
}

/** An append-only event from the SSE stream / run log. */
export interface RunEvent {
  seq: number;
  ts: number;
  type: "run" | "node" | "input" | string;
  node_id?: string;
  status?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface RunRecord {
  id: string;
  workflow: string;
  inputs: Record<string, unknown>;
  status: RunStatus;
  created_at: number;
  updated_at: number;
  nodes: Record<string, RunNodeResult>;
  final?: Record<string, unknown> | null;
  error?: string | null;
  trace_path?: string | null;
  events?: RunEvent[];
}

export interface RunNodeDetail extends RunNodeResult {
  run_id: string;
  node_id: string;
}

// ── Architect builds (S5) ─────────────────────────────────────────────────

export type BuildStatus =
  | "running"
  | "succeeded"
  | "blocked"
  | "failed"
  | "cancelled";

export interface BuildEvent {
  seq: number;
  ts: number;
  type: "build" | "log" | "graph" | string;
  message?: string;
  status?: string;
  workflow?: string;
  path?: string;
  error?: string | null;
  graph?: Graph;
  [key: string]: unknown;
}

export interface BuildRecord {
  id: string;
  intent: string;
  status: BuildStatus;
  created_at: number;
  updated_at: number;
  graph?: Graph | null;
  workflow?: string | null;
  path?: string | null;
  error?: string | null;
  events?: BuildEvent[];
}
