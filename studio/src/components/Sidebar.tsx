/** Left rail: tabs for registered Workflows and run history (Runs). */
import { useState } from "react";
import type { RunRecord, WorkflowSummary } from "@/api/types";

const RUN_DOT: Record<string, string> = {
  running: "#5b9dff",
  succeeded: "#25c2a0",
  failed: "#ff5c5c",
  cancelled: "#9aa0aa",
  awaiting_input: "#f2c94c",
};

function ago(tsSeconds: number): string {
  const s = Math.max(0, Date.now() / 1000 - tsSeconds);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

interface Props {
  workflows: WorkflowSummary[];
  selected: string | null;
  onSelect: (name: string) => void;
  onNew: () => void;
  runs: RunRecord[];
  activeRunId: string | null;
  onSelectRun: (id: string) => void;
  onRefreshRuns: () => void;
}

export default function Sidebar({
  workflows,
  selected,
  onSelect,
  onNew,
  runs,
  activeRunId,
  onSelectRun,
  onRefreshRuns,
}: Props) {
  const [tab, setTab] = useState<"workflows" | "runs">("workflows");

  return (
    <aside className="sidebar">
      <div className="side-tabs">
        <button
          className={tab === "workflows" ? "on" : ""}
          onClick={() => setTab("workflows")}
        >
          Workflows
        </button>
        <button
          className={tab === "runs" ? "on" : ""}
          onClick={() => {
            setTab("runs");
            onRefreshRuns();
          }}
        >
          Runs
        </button>
      </div>

      {tab === "workflows" && (
        <>
          <div className="side-tools">
            <button className="btn-ghost sm" onClick={onNew}>
              ＋ New workflow
            </button>
          </div>
          {workflows.length === 0 && (
            <div className="wf-item">
              <div className="desc">No registered workflows.</div>
            </div>
          )}
          {workflows.map((wf) => (
            <div
              key={wf.name}
              className={`wf-item${selected === wf.name ? " active" : ""}`}
              onClick={() => onSelect(wf.name)}
            >
              <div className="name">{wf.name}</div>
              {wf.description && <div className="desc">{wf.description}</div>}
              {wf.error && (
                <div className="desc" style={{ color: "var(--danger)" }}>
                  {wf.error}
                </div>
              )}
            </div>
          ))}
        </>
      )}

      {tab === "runs" && (
        <>
          <div className="side-tools">
            <button className="btn-ghost sm" onClick={onRefreshRuns}>
              ↻ Refresh
            </button>
          </div>
          {runs.length === 0 && (
            <div className="wf-item">
              <div className="desc">No runs yet. Run a workflow to see history.</div>
            </div>
          )}
          {runs.map((r) => (
            <div
              key={r.id}
              className={`wf-item${activeRunId === r.id ? " active" : ""}`}
              onClick={() => onSelectRun(r.id)}
            >
              <div className="name">
                <span
                  className="run-dot"
                  style={{ background: RUN_DOT[r.status] ?? "#9aa0aa" }}
                />
                {r.workflow}
              </div>
              <div className="desc">
                {r.status.replace("_", " ")} · {ago(r.created_at)}
              </div>
            </div>
          ))}
        </>
      )}
    </aside>
  );
}
