/**
 * useRun — drives a single workflow run for the studio (S3).
 *
 * Starts a run, subscribes to its SSE event stream, and maintains live node
 * statuses; on completion it fetches the full run record for per-node outputs.
 * Also supports cancel, resume (human-in-the-loop), and loading a past run for
 * replay. All I/O goes through the typed Phase-2 client.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelRun as apiCancel,
  getRun,
  resumeRun as apiResume,
  startRun,
  streamRunEvents,
} from "@/api/client";
import type { RunEvent, RunRecord } from "@/api/types";
import { IDLE_RUN, liveStatus, recordStatus, type RunView } from "./types";

const TERMINAL = new Set(["succeeded", "failed", "cancelled", "awaiting_input"]);

export function useRun() {
  const [run, setRun] = useState<RunView>(IDLE_RUN);
  const unsub = useRef<null | (() => void)>(null);

  const closeStream = useCallback(() => {
    unsub.current?.();
    unsub.current = null;
  }, []);

  useEffect(() => closeStream, [closeStream]);

  /** Merge a finished run record into the view (statuses + outputs). */
  const applyRecord = useCallback((rec: RunRecord) => {
    setRun((prev) => {
      const nodeStatus = { ...prev.nodeStatus };
      for (const [id, nr] of Object.entries(rec.nodes ?? {})) {
        nodeStatus[id] = recordStatus(nr.status);
      }
      return {
        ...prev,
        runId: rec.id,
        status: rec.status,
        nodeStatus,
        nodeResults: rec.nodes ?? {},
        final: rec.final ?? null,
        error: rec.error ?? null,
        finishedAt: TERMINAL.has(rec.status) ? Date.now() : prev.finishedAt,
      };
    });
  }, []);

  const subscribe = useCallback(
    (runId: string) => {
      closeStream();
      unsub.current = streamRunEvents(
        runId,
        (evt: RunEvent) => {
          if (evt.type === "node" && evt.node_id && evt.status) {
            const id = evt.node_id;
            const st = liveStatus(evt.status);
            setRun((prev) => ({
              ...prev,
              nodeStatus: { ...prev.nodeStatus, [id]: st },
            }));
          } else if (evt.type === "input_required" && evt.node_id) {
            const nid = evt.node_id;
            setRun((prev) => ({ ...prev, awaitingNode: nid }));
          } else if (evt.type === "run" && evt.status) {
            const s = evt.status;
            setRun((prev) => ({ ...prev, status: s as RunView["status"] }));
          }
        },
        () => {
          // [DONE]: fetch the full record for outputs/errors.
          getRun(runId).then(applyRecord).catch(() => {});
        },
      );
    },
    [applyRecord, closeStream],
  );

  const start = useCallback(
    async (workflow: string, inputs: Record<string, unknown>) => {
      const rec = await startRun(workflow, inputs);
      setRun({
        ...IDLE_RUN,
        runId: rec.id,
        status: rec.status,
        startedAt: Date.now(),
      });
      subscribe(rec.id);
      return rec;
    },
    [subscribe],
  );

  const cancel = useCallback(async () => {
    if (!run.runId) return;
    const rec = await apiCancel(run.runId);
    applyRecord(rec);
    closeStream();
  }, [run.runId, applyRecord, closeStream]);

  const resume = useCallback(
    async (values: Record<string, unknown>) => {
      if (!run.runId) return;
      const rec = await apiResume(run.runId, values);
      setRun({
        ...IDLE_RUN,
        runId: rec.id,
        status: rec.status,
        startedAt: Date.now(),
      });
      subscribe(rec.id);
      return rec;
    },
    [run.runId, subscribe],
  );

  /** Load a past run (no live stream) for replay/inspection. */
  const loadRun = useCallback(
    async (runId: string) => {
      closeStream();
      const rec = await getRun(runId, true);
      setRun({
        ...IDLE_RUN,
        startedAt: rec.created_at * 1000,
        finishedAt: rec.updated_at * 1000,
      });
      applyRecord(rec);
      // If somehow still live, subscribe to catch the tail.
      if (!TERMINAL.has(rec.status)) subscribe(runId);
    },
    [applyRecord, closeStream, subscribe],
  );

  const reset = useCallback(() => {
    closeStream();
    setRun(IDLE_RUN);
  }, [closeStream]);

  return { run, start, cancel, resume, loadRun, reset };
}
