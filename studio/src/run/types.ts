/** Run-view types for the studio (S3). */
import type { RunNodeResult, RunStatus } from "@/api/types";

/** Per-node live status derived from SSE events + the final run record. */
export type NodeLiveStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped";

/** The studio's overall view of the current/loaded run. */
export interface RunView {
  runId: string | null;
  /** "idle" before any run this session; otherwise the server RunStatus. */
  status: RunStatus | "idle";
  nodeStatus: Record<string, NodeLiveStatus>;
  /** Per-node results from the run record (output/error/duration), once known. */
  nodeResults: Record<string, RunNodeResult>;
  startedAt: number | null;
  finishedAt: number | null;
  awaitingNode: string | null;
  final: Record<string, unknown> | null;
  error: string | null;
}

export const IDLE_RUN: RunView = {
  runId: null,
  status: "idle",
  nodeStatus: {},
  nodeResults: {},
  startedAt: null,
  finishedAt: null,
  awaitingNode: null,
  final: null,
  error: null,
};

/** Map a live SSE node status (start|ok|error|skipped) → NodeLiveStatus. */
export function liveStatus(s: string): NodeLiveStatus {
  switch (s) {
    case "start":
      return "running";
    case "ok":
      return "succeeded";
    case "error":
      return "failed";
    case "skipped":
      return "skipped";
    default:
      return "pending";
  }
}

/** Map a final run-record node status (ok|error|skipped) → NodeLiveStatus. */
export function recordStatus(s?: string): NodeLiveStatus {
  switch (s) {
    case "ok":
    case "succeeded":
      return "succeeded";
    case "error":
    case "failed":
      return "failed";
    case "skipped":
      return "skipped";
    case "running":
      return "running";
    default:
      return "pending";
  }
}
