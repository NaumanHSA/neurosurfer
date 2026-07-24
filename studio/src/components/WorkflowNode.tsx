/**
 * Custom React Flow node — the ComfyUI-style card for a neurosurfer graph node.
 *
 * Header carries a kind-colored accent bar, glyph, node id, and kind label.
 * The body shows the node's prompt (goal/purpose) and small chips summarizing
 * its tools and control-flow features (router routes, loop bounds, map fan-out,
 * `when`/`on_error` guards). Left/right handles are the typed sockets.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { kindMeta } from "@/graph/nodeKinds";
import type { FlowNodeData } from "@/graph/adapter";

function flowChips(data: FlowNodeData): string[] {
  const { node } = data;
  const chips: string[] = [];
  if (node.when) chips.push("when");
  if (node.on_error) chips.push("on_error");
  if (node.writes) chips.push(`→ ${node.writes}`);
  if (node.kind === "router") {
    const n = node.routes
      ? Object.keys(node.routes).length
      : node.cases?.length ?? 0;
    chips.push(`${n}-way`);
  }
  if (node.kind === "loop") {
    if (node.until) chips.push("until");
    else if (node.break_when) chips.push("break_when");
    if (node.max_iterations) chips.push(`≤${node.max_iterations}×`);
  }
  if (node.kind === "map") {
    if (node.over) chips.push(`over ${node.over}`);
    if (node.concurrency) chips.push(`∥${node.concurrency}`);
  }
  if (data.bodyIds.length) chips.push(`body: ${data.bodyIds.length}`);
  return chips;
}

const STATUS_DOT: Record<string, string> = {
  running: "#5b9dff",
  succeeded: "#25c2a0",
  failed: "#ff5c5c",
  skipped: "#6b7280",
};

export default function WorkflowNode({
  data,
  selected,
}: NodeProps & { data: FlowNodeData }) {
  const { node, runStatus, issue } = data;
  const meta = kindMeta(node.kind);
  const prompt = node.goal || node.purpose || node.expected_result || "";
  const chips = flowChips(data);
  const statusClass = runStatus ? ` run-${runStatus}` : "";
  const issueClass = issue ? ` issue-${issue}` : "";

  return (
    <div
      className={`wf-node${selected ? " selected" : ""}${statusClass}${issueClass}`}
      style={{ ["--kind-accent" as string]: meta.accent }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="head">
        <span className="glyph">{meta.glyph}</span>
        <div className="titles">
          <div className="id" title={node.id}>
            {node.id}
          </div>
          <div className="kind">{meta.label}</div>
        </div>
        {issue && (
          <span className={`issue-badge ${issue}`} title={issue}>
            {issue === "error" ? "!" : "?"}
          </span>
        )}
        {runStatus && runStatus !== "pending" && (
          <span
            className={`status-dot${runStatus === "running" ? " pulsing" : ""}`}
            style={{ background: STATUS_DOT[runStatus] }}
            title={runStatus}
          />
        )}
      </div>
      <div className="body">
        {prompt && <div className="prompt">{prompt}</div>}
        {(node.tools?.length || chips.length) > 0 && (
          <div className="chips">
            {node.tools?.map((t) => (
              <span className="chip tool" key={`t-${t}`}>
                {t}
              </span>
            ))}
            {chips.map((c) => (
              <span className="chip flow" key={`c-${c}`}>
                {c}
              </span>
            ))}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
