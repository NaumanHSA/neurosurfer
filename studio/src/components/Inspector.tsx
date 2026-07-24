/**
 * Inspector — read-only detail (S2) + inline editing (S4).
 *
 * When an `edit` API is passed, node fields become editable widgets
 * (goal/purpose/config, tools, depends_on wiring, router routes, loop/map
 * config) and node actions (duplicate/delete) appear. Without it, the panel is
 * the read-only view. Tabs: Config · Last Run · Trace.
 */
import { useState } from "react";
import { kindMeta } from "@/graph/nodeKinds";
import { downstreamOf, nodeById } from "@/graph/relations";
import { availableVars, unresolvedVars } from "@/graph/edit";
import type { GraphInput, GraphNode, WorkflowDetail } from "@/api/types";
import type { FlowSelection } from "@/selection";
import type { RunView } from "@/run/types";

export interface EditApi {
  onPatch: (patch: Partial<GraphNode>) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  /** all node ids in the graph (for wiring selects). */
  nodeIds: string[];
}

export interface GraphEditApi {
  setInputs: (inputs: GraphInput[]) => void;
  setOutputs: (outputs: string[]) => void;
}

interface Props {
  detail: WorkflowDetail;
  selection: FlowSelection;
  onSelect: (sel: FlowSelection) => void;
  onClose: () => void;
  run?: RunView;
  edit?: EditApi | null;
  graphEdit?: GraphEditApi | null;
  onRemoveEdge?: () => void;
}

function copy(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {});
}
function asText(v: unknown): string {
  if (v == null) return "";
  return typeof v === "string" ? v : JSON.stringify(v, null, 2);
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="insp-row">
      <span className="insp-key">{label}</span>
      <span className="insp-val">{children}</span>
    </div>
  );
}

function NodeChip({ id, onGo }: { id: string; onGo: (id: string) => void }) {
  return (
    <button className="node-chip" onClick={() => onGo(id)} title={`Go to ${id}`}>
      {id}
    </button>
  );
}

// ── editable field widgets ───────────────────────────────────────────────────

function EditText({
  label,
  value,
  rows = 3,
  onChange,
}: {
  label: string;
  value?: string | null;
  rows?: number;
  onChange: (v: string) => void;
}) {
  return (
    <div className="insp-block">
      <div className="insp-block-label">{label}</div>
      <textarea
        className="edit-area"
        rows={rows}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function EditLine({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value?: string | number | null;
  placeholder?: string;
  onChange: (v: string) => void;
}) {
  return (
    <Row label={label}>
      <input
        className="edit-line"
        value={value ?? ""}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </Row>
  );
}

function TargetSelect({
  value,
  options,
  onChange,
  allowNone = true,
}: {
  value?: string | null;
  options: string[];
  onChange: (v: string) => void;
  allowNone?: boolean;
}) {
  return (
    <select
      className="edit-line"
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value)}
    >
      {allowNone && <option value="">— none —</option>}
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function VarHints({ graph, nodeId }: { graph: WorkflowDetail["graph"]; nodeId: string }) {
  const vars = availableVars(graph, nodeId);
  const unresolved = unresolvedVars(graph, nodeId);
  return (
    <div className="var-hints">
      <span className="var-hints-label">Template vars:</span>
      {vars.length ? (
        vars.map((v) => (
          <span className="var-chip" key={v}>
            {"{" + v + "}"}
          </span>
        ))
      ) : (
        <span className="dim">none in scope</span>
      )}
      {unresolved.length > 0 && (
        <div className="var-warn">
          ⚠ unknown: {unresolved.map((v) => `{${v}}`).join(", ")}
        </div>
      )}
    </div>
  );
}

function DependsEditor({
  node,
  nodeIds,
  onChange,
}: {
  node: GraphNode;
  nodeIds: string[];
  onChange: (deps: string[]) => void;
}) {
  const deps = new Set(node.depends_on ?? []);
  const candidates = nodeIds.filter((id) => id !== node.id);
  return (
    <div className="dep-grid">
      {candidates.length === 0 && <span className="dim">No other nodes.</span>}
      {candidates.map((id) => (
        <label key={id} className="dep-check">
          <input
            type="checkbox"
            checked={deps.has(id)}
            onChange={(e) => {
              const next = new Set(deps);
              if (e.target.checked) next.add(id);
              else next.delete(id);
              onChange([...next]);
            }}
          />
          {id}
        </label>
      ))}
    </div>
  );
}

function RoutesEditor({
  node,
  nodeIds,
  onChange,
}: {
  node: GraphNode;
  nodeIds: string[];
  onChange: (routes: Record<string, string>) => void;
}) {
  const entries = Object.entries(node.routes ?? {});
  const targets = nodeIds.filter((id) => id !== node.id);
  const update = (list: [string, string][]) => onChange(Object.fromEntries(list));

  return (
    <div className="routes-editor">
      {entries.map(([label, target], i) => (
        <div className="route-row" key={i}>
          <input
            className="edit-line"
            value={label}
            placeholder="label"
            onChange={(e) => {
              const next = [...entries] as [string, string][];
              next[i] = [e.target.value, target];
              update(next);
            }}
          />
          <span className="route-arrow">→</span>
          <TargetSelect
            value={target}
            options={targets}
            allowNone={false}
            onChange={(v) => {
              const next = [...entries] as [string, string][];
              next[i] = [label, v];
              update(next);
            }}
          />
          <button
            className="mini-copy"
            title="Remove route"
            onClick={() => update(entries.filter((_, j) => j !== i) as [string, string][])}
          >
            ✕
          </button>
        </div>
      ))}
      <button
        className="btn-ghost sm"
        onClick={() =>
          update([...entries, ["", targets[0] ?? ""]] as [string, string][])
        }
      >
        + Add route
      </button>
    </div>
  );
}

// ── node view ────────────────────────────────────────────────────────────────

function NodeView({
  node,
  detail,
  go,
  run,
  edit,
}: {
  node: GraphNode;
  detail: WorkflowDetail;
  go: (id: string) => void;
  run?: RunView;
  edit?: EditApi | null;
}) {
  const [tab, setTab] = useState<"config" | "run" | "trace">("config");
  const meta = kindMeta(node.kind);
  const upstream = node.depends_on ?? [];
  const downstream = downstreamOf(detail.graph, node.id);
  const result = run?.nodeResults?.[node.id];
  const liveStatus = run?.nodeStatus?.[node.id];
  const P = edit?.onPatch;

  return (
    <>
      <div className="insp-head" style={{ ["--kind-accent" as string]: meta.accent }}>
        <span className="insp-glyph">{meta.glyph}</span>
        <div className="insp-titles">
          <div className="insp-id">{node.id}</div>
          <div className="insp-kind">{meta.label}</div>
        </div>
      </div>
      <div className="insp-blurb">{meta.blurb}</div>

      <div className="insp-tabs">
        {(["config", "run", "trace"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {t === "config" ? "Config" : t === "run" ? "Last Run" : "Trace"}
          </button>
        ))}
      </div>

      {tab === "config" && (
        <div className="insp-body">
          {/* Prompts */}
          {edit && P ? (
            <>
              <EditText label="Goal" value={node.goal} rows={4} onChange={(v) => P({ goal: v })} />
              <EditText label="Purpose" value={node.purpose} rows={3} onChange={(v) => P({ purpose: v })} />
              <EditText
                label="Expected result"
                value={node.expected_result}
                rows={2}
                onChange={(v) => P({ expected_result: v })}
              />
              <VarHints graph={detail.graph} nodeId={node.id} />
            </>
          ) : (
            <>
              {node.goal && <PromptBlock label="Goal" text={node.goal} />}
              {node.purpose && <PromptBlock label="Purpose" text={node.purpose} />}
              {node.expected_result && (
                <PromptBlock label="Expected result" text={node.expected_result} />
              )}
            </>
          )}

          <div className="insp-section">Configuration</div>
          <Row label="kind">{node.kind}</Row>
          {edit && P ? (
            <>
              <EditLine label="writes" value={node.writes} placeholder="var name" onChange={(v) => P({ writes: v || null })} />
              <EditLine label="model" value={node.model as string} placeholder="(provider default)" onChange={(v) => P({ model: v || null })} />
              <EditLine label="when" value={node.when} placeholder="expression" onChange={(v) => P({ when: v || null })} />
              <Row label="on_error">
                <TargetSelect
                  value={node.on_error}
                  options={edit.nodeIds.filter((id) => id !== node.id)}
                  onChange={(v) => P({ on_error: v || null })}
                />
              </Row>
              <EditLine
                label="tools"
                value={(node.tools ?? []).join(", ")}
                placeholder="comma,separated"
                onChange={(v) => P({ tools: v.split(",").map((s) => s.trim()).filter(Boolean) })}
              />
            </>
          ) : (
            <>
              {node.writes && <Row label="writes">→ {node.writes}</Row>}
              {(node.model as string) && <Row label="model">{String(node.model)}</Row>}
              {node.when && <Row label="when">{node.when}</Row>}
              {node.on_error && (
                <Row label="on_error">
                  <NodeChip id={node.on_error} onGo={go} />
                </Row>
              )}
              {node.tools && node.tools.length > 0 && (
                <Row label="tools">
                  <span className="chips">
                    {node.tools.map((t) => (
                      <span className="chip tool" key={t}>
                        {t}
                      </span>
                    ))}
                  </span>
                </Row>
              )}
            </>
          )}

          {/* Router */}
          {node.kind === "router" && (
            <>
              <div className="insp-section">Routes</div>
              {edit && P ? (
                <>
                  <RoutesEditor node={node} nodeIds={edit.nodeIds} onChange={(routes) => P({ routes })} />
                  <Row label="default">
                    <TargetSelect
                      value={node.default}
                      options={edit.nodeIds.filter((id) => id !== node.id)}
                      onChange={(v) => P({ default: v || null })}
                    />
                  </Row>
                </>
              ) : (
                <>
                  {node.routes &&
                    Object.entries(node.routes).map(([label, target]) => (
                      <Row key={label} label={label}>
                        <NodeChip id={target} onGo={go} />
                      </Row>
                    ))}
                  {node.cases?.map((c, i) => (
                    <Row key={i} label={c.when ?? "case"}>
                      <NodeChip id={c.to} onGo={go} />
                    </Row>
                  ))}
                  {node.default && (
                    <Row label="default">
                      <NodeChip id={node.default} onGo={go} />
                    </Row>
                  )}
                </>
              )}
            </>
          )}

          {/* Loop */}
          {node.kind === "loop" && (
            <>
              <div className="insp-section">Loop</div>
              {edit && P ? (
                <>
                  <EditLine label="until" value={node.until} placeholder="plain-English stop" onChange={(v) => P({ until: v || null })} />
                  <EditLine label="break_when" value={node.break_when} placeholder="expression" onChange={(v) => P({ break_when: v || null })} />
                  <EditLine label="max_iter" value={node.max_iterations ?? ""} onChange={(v) => P({ max_iterations: v ? parseInt(v, 10) : null })} />
                  <EditLine label="accumulate" value={node.accumulate} onChange={(v) => P({ accumulate: v || null })} />
                </>
              ) : (
                <>
                  {node.until && <Row label="until">{node.until}</Row>}
                  {node.break_when && <Row label="break_when">{node.break_when}</Row>}
                  {node.max_iterations != null && <Row label="max_iter">{node.max_iterations}</Row>}
                  {node.accumulate && <Row label="accumulate">{node.accumulate}</Row>}
                </>
              )}
              {node.body && <Row label="body">{node.body.map((b) => b.id).join(" → ")}</Row>}
            </>
          )}

          {/* Map */}
          {node.kind === "map" && (
            <>
              <div className="insp-section">Map</div>
              {edit && P ? (
                <>
                  <EditLine label="over" value={node.over} placeholder="collection expr" onChange={(v) => P({ over: v || null })} />
                  <EditLine label="concurrency" value={node.concurrency ?? ""} onChange={(v) => P({ concurrency: v ? parseInt(v, 10) : 1 })} />
                </>
              ) : (
                <>
                  {node.over && <Row label="over">{node.over}</Row>}
                  {node.concurrency != null && <Row label="concurrency">{node.concurrency}</Row>}
                </>
              )}
              {node.body && <Row label="body">{node.body.map((b) => b.id).join(" → ")}</Row>}
            </>
          )}

          {/* Relationships / wiring */}
          <div className="insp-section">{edit ? "Depends on" : "Relationships"}</div>
          {edit && P ? (
            <DependsEditor node={node} nodeIds={edit.nodeIds} onChange={(deps) => P({ depends_on: deps })} />
          ) : (
            <>
              <Row label="upstream">
                {upstream.length ? (
                  <span className="chips">
                    {upstream.map((id) => (
                      <NodeChip key={id} id={id} onGo={go} />
                    ))}
                  </span>
                ) : (
                  <span className="dim">— entry node</span>
                )}
              </Row>
              <Row label="downstream">
                {downstream.length ? (
                  <span className="chips">
                    {downstream.map((id) => (
                      <NodeChip key={id} id={id} onGo={go} />
                    ))}
                  </span>
                ) : (
                  <span className="dim">— leaf node</span>
                )}
              </Row>
            </>
          )}

          <div className="insp-actions">
            {edit ? (
              <>
                <button onClick={edit.onDuplicate}>Duplicate</button>
                <button className="danger" onClick={edit.onDelete}>
                  Delete node
                </button>
              </>
            ) : (
              <>
                <button onClick={() => copy(node.id)}>Copy id</button>
                <button onClick={() => copy(JSON.stringify(node, null, 2))}>Copy JSON</button>
              </>
            )}
          </div>
        </div>
      )}

      {tab === "run" && (
        <div className="insp-body">
          {!result && !liveStatus && (
            <div className="insp-empty">Run this workflow (▶) to see this node's result.</div>
          )}
          {(result || liveStatus) && (
            <>
              <Row label="status">{(result?.status as string) ?? liveStatus ?? "—"}</Row>
              {result?.duration_ms != null && (
                <Row label="duration">{(Number(result.duration_ms) / 1000).toFixed(2)}s</Row>
              )}
              {result?.skip_reason && <Row label="skipped">{String(result.skip_reason)}</Row>}
              {result?.error && (
                <div className="insp-block">
                  <div className="insp-block-label">Error</div>
                  <div className="insp-prompt insp-error">{String(result.error)}</div>
                </div>
              )}
              {result?.output != null && (
                <div className="insp-block">
                  <div className="insp-block-label">
                    Output
                    <button className="mini-copy" onClick={() => copy(asText(result.output))}>
                      ⧉
                    </button>
                  </div>
                  <div className="insp-prompt">{asText(result.output)}</div>
                </div>
              )}
            </>
          )}
        </div>
      )}
      {tab === "trace" && (
        <div className="insp-body">
          <div className="insp-empty">
            Per-node trace — inputs, tokens, tool calls, sub-spans. Needs a
            trace-detail API endpoint. <em>(planned — S3 [BE])</em>
          </div>
        </div>
      )}
    </>
  );
}

function PromptBlock({ label, text }: { label: string; text?: string | null }) {
  if (!text) return null;
  return (
    <div className="insp-block">
      <div className="insp-block-label">
        {label}
        <button className="mini-copy" onClick={() => copy(text)} title="Copy">
          ⧉
        </button>
      </div>
      <div className="insp-prompt">{text}</div>
    </div>
  );
}

function EdgeView({
  selection,
  go,
  onRemove,
}: {
  selection: Extract<FlowSelection, { type: "edge" }>;
  go: (id: string) => void;
  onRemove?: () => void;
}) {
  const kindLabel = { data: "Data flow", route: "Route", error: "Error / fallback" }[
    selection.edgeKind
  ];
  return (
    <>
      <div className="insp-head">
        <span className="insp-glyph">→</span>
        <div className="insp-titles">
          <div className="insp-id">{kindLabel}</div>
          <div className="insp-kind">edge</div>
        </div>
      </div>
      <div className="insp-body">
        <Row label="from">
          <NodeChip id={selection.source} onGo={go} />
        </Row>
        <Row label="to">
          <NodeChip id={selection.target} onGo={go} />
        </Row>
        <Row label="kind">{selection.edgeKind}</Row>
        {selection.label && <Row label="label">{selection.label}</Row>}
        {onRemove && (
          <div className="insp-actions">
            <button className="danger" onClick={onRemove}>
              Remove edge
            </button>
          </div>
        )}
      </div>
    </>
  );
}

const INPUT_TYPES = ["string", "number", "integer", "boolean", "object", "array"];

function InputsEditor({
  inputs,
  onChange,
}: {
  inputs: GraphInput[];
  onChange: (inputs: GraphInput[]) => void;
}) {
  const patch = (i: number, p: Partial<GraphInput>) =>
    onChange(inputs.map((inp, j) => (j === i ? { ...inp, ...p } : inp)));
  return (
    <div className="routes-editor">
      {inputs.map((inp, i) => (
        <div className="input-row" key={i}>
          <input
            className="edit-line"
            value={inp.name}
            placeholder="name"
            onChange={(e) => patch(i, { name: e.target.value })}
          />
          <select
            className="edit-line"
            value={String((inp as Record<string, unknown>).type ?? "string")}
            onChange={(e) => patch(i, { type: e.target.value } as Partial<GraphInput>)}
          >
            {INPUT_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <label className="req-check" title="required">
            <input
              type="checkbox"
              checked={Boolean((inp as Record<string, unknown>).required)}
              onChange={(e) => patch(i, { required: e.target.checked } as Partial<GraphInput>)}
            />
            req
          </label>
          <button
            className="mini-copy"
            title="Remove input"
            onClick={() => onChange(inputs.filter((_, j) => j !== i))}
          >
            ✕
          </button>
        </div>
      ))}
      <button
        className="btn-ghost sm"
        onClick={() => onChange([...inputs, { name: "", type: "string", required: false }])}
      >
        + Add input
      </button>
    </div>
  );
}

function GraphView({
  detail,
  go,
  graphEdit,
}: {
  detail: WorkflowDetail;
  go: (id: string) => void;
  graphEdit?: GraphEditApi | null;
}) {
  const g = detail.graph;
  const kinds = g.nodes.reduce<Record<string, number>>((acc, n) => {
    acc[n.kind] = (acc[n.kind] ?? 0) + 1;
    return acc;
  }, {});
  const outSet = new Set(g.outputs);
  return (
    <>
      <div className="insp-head">
        <span className="insp-glyph">▤</span>
        <div className="insp-titles">
          <div className="insp-id">{g.name}</div>
          <div className="insp-kind">workflow{detail.version ? ` · v${detail.version}` : ""}</div>
        </div>
      </div>
      {g.description && <div className="insp-blurb">{g.description}</div>}
      <div className="insp-body">
        <div className="insp-section">Inputs</div>
        {graphEdit ? (
          <InputsEditor inputs={g.inputs} onChange={graphEdit.setInputs} />
        ) : g.inputs.length ? (
          g.inputs.map((i) => (
            <Row key={i.name} label={i.name}>
              {String((i as Record<string, unknown>).type ?? "any")}
              {(i as Record<string, unknown>).required ? " · required" : ""}
            </Row>
          ))
        ) : (
          <div className="dim">No declared inputs.</div>
        )}

        <div className="insp-section">Outputs</div>
        {graphEdit ? (
          <div className="dep-grid">
            {g.nodes.map((n) => (
              <label key={n.id} className="dep-check">
                <input
                  type="checkbox"
                  checked={outSet.has(n.id)}
                  onChange={(e) => {
                    const next = new Set(outSet);
                    if (e.target.checked) next.add(n.id);
                    else next.delete(n.id);
                    graphEdit.setOutputs([...next]);
                  }}
                />
                {n.id}
              </label>
            ))}
          </div>
        ) : g.outputs.length ? (
          <span className="chips">
            {g.outputs.map((o) => (
              <NodeChip key={o} id={o} onGo={go} />
            ))}
          </span>
        ) : (
          <div className="dim">No declared outputs.</div>
        )}

        <div className="insp-section">Composition</div>
        <Row label="nodes">{g.nodes.length}</Row>
        <Row label="kinds">
          <span className="chips">
            {Object.entries(kinds).map(([k, n]) => (
              <span className="chip flow" key={k}>
                {k} ×{n}
              </span>
            ))}
          </span>
        </Row>
        <Row label="fail_fast">{String(g.fail_fast ?? false)}</Row>
        <Row label="strict_inputs">{String(g.strict_inputs ?? false)}</Row>
      </div>
    </>
  );
}

export default function Inspector({
  detail,
  selection,
  onSelect,
  onClose,
  run,
  edit,
  graphEdit,
  onRemoveEdge,
}: Props) {
  const go = (id: string) => onSelect({ type: "node", id });

  return (
    <aside className="inspector">
      <div className="insp-topbar">
        <span className="insp-topbar-title">Inspector{edit || graphEdit ? " · editing" : ""}</span>
        <button className="insp-close" onClick={onClose} title="Close">
          ✕
        </button>
      </div>
      {selection.type === "node" &&
        (() => {
          const node = nodeById(detail.graph, selection.id);
          return node ? (
            <NodeView node={node} detail={detail} go={go} run={run} edit={edit} />
          ) : (
            <div className="insp-empty">Node not found.</div>
          );
        })()}
      {selection.type === "edge" && (
        <EdgeView selection={selection} go={go} onRemove={onRemoveEdge} />
      )}
      {selection.type === "graph" && (
        <GraphView detail={detail} go={go} graphEdit={graphEdit} />
      )}
    </aside>
  );
}
