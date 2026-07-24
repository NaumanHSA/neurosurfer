/**
 * Typed client for the neurosurfer Phase-2 workflow API.
 *
 * All requests go to `/v1/*` — in dev, Vite proxies that to the gateway
 * (see vite.config.ts); in a build, serve the studio behind the same origin as
 * the gateway or set a proxy. The bearer token (if the gateway's auth
 * middleware is on) comes from `VITE_NEUROSURFER_TOKEN`.
 *
 * M0 uses the read paths (list/get workflows). The run/resume/cancel/stream
 * methods are here so M1/M2 build straight on top without touching this file.
 */

import type {
  BuildEvent,
  BuildRecord,
  Graph,
  RunEvent,
  RunNodeDetail,
  RunRecord,
  ValidationReport,
  WorkflowDetail,
  WorkflowSummary,
} from "./types";

const TOKEN = import.meta.env.VITE_NEUROSURFER_TOKEN as string | undefined;

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(`API ${status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    this.name = "ApiError";
  }
}

function authHeaders(): Record<string, string> {
  return TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

// ── Workflows ─────────────────────────────────────────────────────────────

export function listWorkflows(): Promise<WorkflowSummary[]> {
  return request<{ workflows: WorkflowSummary[] }>("/v1/workflows").then(
    (r) => r.workflows,
  );
}

export function getWorkflow(name: string): Promise<WorkflowDetail> {
  return request<WorkflowDetail>(`/v1/workflows/${encodeURIComponent(name)}`);
}

// ── Authoring (S4) ──────────────────────────────────────────────────────────

export function validateWorkflow(
  graph: Graph,
  name?: string,
): Promise<ValidationReport> {
  return request<ValidationReport>("/v1/workflows/validate", {
    method: "POST",
    body: JSON.stringify({ graph, name }),
  });
}

export interface SaveMeta {
  description?: string;
  version?: string;
  tags?: string[];
}

export function createWorkflow(
  name: string,
  graph: Graph,
  meta: SaveMeta = {},
): Promise<WorkflowDetail> {
  return request<WorkflowDetail>("/v1/workflows", {
    method: "POST",
    body: JSON.stringify({ name, graph, ...meta }),
  });
}

export function updateWorkflow(
  name: string,
  graph: Graph,
  meta: SaveMeta = {},
): Promise<WorkflowDetail> {
  return request<WorkflowDetail>(`/v1/workflows/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify({ graph, ...meta }),
  });
}

export function deleteWorkflow(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/v1/workflows/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// ── Runs ──────────────────────────────────────────────────────────────────

export function startRun(
  name: string,
  inputs: Record<string, unknown>,
): Promise<RunRecord> {
  return request<RunRecord>(`/v1/workflows/${encodeURIComponent(name)}/runs`, {
    method: "POST",
    body: JSON.stringify({ inputs }),
  });
}

export function listRuns(): Promise<RunRecord[]> {
  return request<{ runs: RunRecord[] }>("/v1/runs").then((r) => r.runs);
}

export function getRun(id: string, events = false): Promise<RunRecord> {
  return request<RunRecord>(`/v1/runs/${id}${events ? "?events=true" : ""}`);
}

export function getRunNode(
  id: string,
  nodeId: string,
): Promise<RunNodeDetail> {
  return request<RunNodeDetail>(`/v1/runs/${id}/nodes/${encodeURIComponent(nodeId)}`);
}

export function resumeRun(
  id: string,
  values: Record<string, unknown>,
): Promise<RunRecord> {
  return request<RunRecord>(`/v1/runs/${id}/resume`, {
    method: "POST",
    body: JSON.stringify({ values }),
  });
}

export function cancelRun(id: string): Promise<RunRecord> {
  return request<RunRecord>(`/v1/runs/${id}`, { method: "DELETE" });
}

// ── Architect builds (S5) ─────────────────────────────────────────────────

export function startArchitectBuild(
  intent: string,
  verify?: string,
): Promise<BuildRecord> {
  return request<BuildRecord>("/v1/architect/builds", {
    method: "POST",
    body: JSON.stringify({ intent, verify }),
  });
}

export function getArchitectBuild(id: string, events = false): Promise<BuildRecord> {
  return request<BuildRecord>(
    `/v1/architect/builds/${id}${events ? "?events=true" : ""}`,
  );
}

export function streamArchitectBuild(
  id: string,
  onEvent: (evt: BuildEvent) => void,
  onDone?: () => void,
): () => void {
  const es = new EventSource(`/v1/architect/builds/${id}/events`);
  es.onmessage = (msg) => {
    if (msg.data === "[DONE]") {
      es.close();
      onDone?.();
      return;
    }
    try {
      onEvent(JSON.parse(msg.data) as BuildEvent);
    } catch {
      /* ignore keep-alive frames */
    }
  };
  return () => es.close();
}

/**
 * Subscribe to a run's SSE event stream. Replays from seq 1 then tails live.
 * Returns an unsubscribe fn. Used by M2's live run view.
 *
 * Note: EventSource can't send an Authorization header. When the gateway
 * requires a token, run the studio behind a proxy that injects it, or add a
 * query-param token path server-side. For the open local gateway this is fine.
 */
export function streamRunEvents(
  id: string,
  onEvent: (evt: RunEvent) => void,
  onDone?: () => void,
  onError?: (err: Event) => void,
): () => void {
  const es = new EventSource(`/v1/runs/${id}/events`);
  es.onmessage = (msg) => {
    if (msg.data === "[DONE]") {
      es.close();
      onDone?.();
      return;
    }
    try {
      onEvent(JSON.parse(msg.data) as RunEvent);
    } catch {
      /* ignore keep-alive / non-JSON frames */
    }
  };
  es.onerror = (err) => onError?.(err);
  return () => es.close();
}
