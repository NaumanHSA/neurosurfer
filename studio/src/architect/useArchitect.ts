/**
 * useArchitect (S5) — drives one Architect build and streams its steps.
 *
 * Starts a build, subscribes to its SSE stream, and maintains the live step log,
 * the evolving staged-graph snapshot (to animate the canvas), and the terminal
 * outcome (registered workflow name / blocked reason / error).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  getArchitectBuild,
  startArchitectBuild,
  streamArchitectBuild,
} from "@/api/client";
import type { BuildEvent, BuildStatus, Graph } from "@/api/types";

export interface LogLine {
  ts: number;
  message: string;
}

export interface BuildView {
  buildId: string | null;
  intent: string;
  status: BuildStatus | "idle";
  log: LogLine[];
  graph: Graph | null;
  workflow: string | null;
  error: string | null;
}

const IDLE: BuildView = {
  buildId: null,
  intent: "",
  status: "idle",
  log: [],
  graph: null,
  workflow: null,
  error: null,
};

export function useArchitect() {
  const [build, setBuild] = useState<BuildView>(IDLE);
  const unsub = useRef<null | (() => void)>(null);

  const close = useCallback(() => {
    unsub.current?.();
    unsub.current = null;
  }, []);
  useEffect(() => close, [close]);

  const subscribe = useCallback(
    (id: string) => {
      close();
      unsub.current = streamArchitectBuild(
        id,
        (evt: BuildEvent) => {
          if (evt.type === "log" && evt.message) {
            const line: LogLine = { ts: evt.ts, message: evt.message };
            setBuild((b) => ({ ...b, log: [...b.log, line] }));
          } else if (evt.type === "graph" && evt.graph) {
            const g = evt.graph;
            setBuild((b) => ({ ...b, graph: g }));
          } else if (evt.type === "build" && evt.status) {
            const s = evt.status as BuildStatus;
            setBuild((b) => ({
              ...b,
              status: s,
              workflow: evt.workflow ?? b.workflow,
              error: evt.error ?? b.error,
            }));
          }
        },
        () => {
          getArchitectBuild(id).then((rec) =>
            setBuild((b) => ({
              ...b,
              status: rec.status,
              graph: rec.graph ?? b.graph,
              workflow: rec.workflow ?? b.workflow,
              error: rec.error ?? b.error,
            })),
          ).catch(() => {});
        },
      );
    },
    [close],
  );

  const start = useCallback(
    async (intent: string, verify?: string) => {
      const rec = await startArchitectBuild(intent, verify);
      setBuild({ ...IDLE, buildId: rec.id, intent, status: rec.status });
      subscribe(rec.id);
      return rec;
    },
    [subscribe],
  );

  const clear = useCallback(() => {
    close();
    setBuild(IDLE);
  }, [close]);

  return { build, start, clear };
}
