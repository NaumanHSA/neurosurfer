/**
 * RunStatusBar (S3) — a slim bar over the canvas showing the active/loaded run:
 * status, elapsed time, per-node counts, and cancel/resume/clear actions.
 */
import { useEffect, useState } from "react";
import type { RunView } from "@/run/types";

const STATUS_COLOR: Record<string, string> = {
  running: "#5b9dff",
  succeeded: "#25c2a0",
  failed: "#ff5c5c",
  cancelled: "#9aa0aa",
  awaiting_input: "#f2c94c",
};

interface Props {
  run: RunView;
  totalNodes: number;
  onCancel: () => void;
  onResume: () => void;
  onClear: () => void;
}

export default function RunStatusBar({
  run,
  totalNodes,
  onCancel,
  onResume,
  onClear,
}: Props) {
  const [, tick] = useState(0);
  const live = run.status === "running";

  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => tick((n) => n + 1), 200);
    return () => clearInterval(t);
  }, [live]);

  if (run.status === "idle") return null;

  const counts = Object.values(run.nodeStatus).reduce<Record<string, number>>(
    (acc, s) => ((acc[s] = (acc[s] ?? 0) + 1), acc),
    {},
  );
  const end = run.finishedAt ?? Date.now();
  const elapsed = run.startedAt ? Math.max(0, (end - run.startedAt) / 1000) : 0;

  return (
    <div className="run-bar">
      <span
        className={`run-status${live ? " pulsing" : ""}`}
        style={{ color: STATUS_COLOR[run.status] ?? "#e6e8ec" }}
      >
        ● {run.status.replace("_", " ")}
      </span>
      <span className="run-elapsed">{elapsed.toFixed(1)}s</span>
      <span className="run-counts">
        {counts.running ? <span className="rc rc-running">▶ {counts.running}</span> : null}
        {counts.succeeded ? <span className="rc rc-ok">✓ {counts.succeeded}</span> : null}
        {counts.failed ? <span className="rc rc-fail">✕ {counts.failed}</span> : null}
        {counts.skipped ? <span className="rc rc-skip">⤼ {counts.skipped}</span> : null}
        <span className="rc rc-total">
          {Object.values(run.nodeStatus).filter((s) => s !== "pending").length}/{totalNodes}
        </span>
      </span>
      {run.error && <span className="run-error" title={run.error}>{run.error}</span>}
      <span className="spacer" />
      {live && (
        <button className="btn-ghost sm" onClick={onCancel}>
          Cancel
        </button>
      )}
      {run.status === "awaiting_input" && (
        <button className="btn-primary sm" onClick={onResume}>
          Resume{run.awaitingNode ? ` (${run.awaitingNode})` : ""}
        </button>
      )}
      {!live && (
        <button className="btn-ghost sm" onClick={onClear}>
          Clear
        </button>
      )}
    </div>
  );
}
