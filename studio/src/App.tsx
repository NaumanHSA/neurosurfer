/**
 * neurosurfer studio — shell (S1–S4).
 *
 * View, run/trace, and fully edit workflows: draft-based editing with drag-to-
 * wire, inputs/outputs editors, add/delete/duplicate nodes, undo/redo, inline
 * validation, import/export, new-from-blank, and save via the authoring API.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import yaml from "js-yaml";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import Inspector, { type EditApi, type GraphEditApi } from "./components/Inspector";
import RunLauncher from "./components/RunLauncher";
import RunStatusBar from "./components/RunStatusBar";
import ProblemsPanel from "./components/ProblemsPanel";
import ArchitectDock from "./components/ArchitectDock";
import { useArchitect } from "./architect/useArchitect";
import {
  ApiError,
  createWorkflow,
  getWorkflow,
  listRuns,
  listWorkflows,
  updateWorkflow,
  validateWorkflow,
} from "./api/client";
import {
  addNode as addNodeOp,
  cloneGraph,
  connect as connectOp,
  deleteNode as deleteNodeOp,
  disconnect as disconnectOp,
  duplicateNode as duplicateNodeOp,
  setInputs as setInputsOp,
  setOutputs as setOutputsOp,
  updateNode as updateNodeOp,
} from "./graph/edit";
import { useRun } from "./run/useRun";
import type {
  Graph,
  GraphNode,
  NodeKind,
  RunRecord,
  ValidationReport,
  WorkflowDetail,
  WorkflowSummary,
} from "./api/types";
import type { FlowSelection } from "./selection";

const KINDS: NodeKind[] = [
  "base", "react", "function", "tool", "python",
  "router", "loop", "map", "subgraph", "input",
];

export default function App() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<WorkflowDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<FlowSelection>({ type: "graph" });
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [launcher, setLauncher] = useState<null | "run" | "resume">(null);

  // Edit state (S4)
  const [editMode, setEditMode] = useState(false);
  const [draft, setDraft] = useState<Graph | null>(null);
  const [dirty, setDirty] = useState(false);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [showProblems, setShowProblems] = useState(false);
  const [toast, setToast] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  // Undo/redo history (refs to avoid updater side effects / StrictMode double-run)
  const undoRef = useRef<Graph[]>([]);
  const redoRef = useRef<Graph[]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Architect (S5)
  const [architectOpen, setArchitectOpen] = useState(false);
  const { build: archBuild, start: startBuild, clear: clearBuild } = useArchitect();

  const { run, start, cancel, resume, loadRun, reset } = useRun();

  const flash = useCallback((kind: "ok" | "err", msg: string) => {
    setToast({ kind, msg });
    setTimeout(() => setToast(null), 3500);
  }, []);

  const syncHistFlags = () => {
    setCanUndo(undoRef.current.length > 0);
    setCanRedo(redoRef.current.length > 0);
  };
  const resetHistory = () => {
    undoRef.current = [];
    redoRef.current = [];
    syncHistFlags();
  };

  useEffect(() => {
    listWorkflows()
      .then((wfs) => {
        setWorkflows(wfs);
        if (wfs.length) setSelected((s) => s ?? wfs[0].name);
      })
      .catch((e) => setError(String(e?.message ?? e)));
  }, []);

  const loadDetail = useCallback((name: string) => {
    setError(null);
    setDetail(null);
    setSelection({ type: "graph" });
    getWorkflow(name)
      .then(setDetail)
      .catch((e) => setError(String(e?.message ?? e)));
  }, []);

  useEffect(() => {
    if (selected) loadDetail(selected);
  }, [selected, loadDetail]);

  useEffect(() => {
    if (!dirty) return;
    const h = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [dirty]);

  const confirmDiscard = useCallback(() => {
    if (!dirty) return true;
    return window.confirm("Discard unsaved changes?");
  }, [dirty]);

  const refreshWorkflows = () => listWorkflows().then(setWorkflows).catch(() => {});
  const refreshRuns = useCallback(() => {
    listRuns()
      .then((r) => setRuns(r.sort((a, b) => b.created_at - a.created_at)))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (["succeeded", "failed", "cancelled", "awaiting_input"].includes(run.status)) {
      refreshRuns();
    }
  }, [run.status, refreshRuns]);

  const chooseWorkflow = (name: string) => {
    if (name === selected) return;
    if (!confirmDiscard()) return;
    setEditMode(false);
    setDraft(null);
    setDirty(false);
    setValidation(null);
    resetHistory();
    setSelected(name);
  };

  const onSelectRun = useCallback(
    (id: string) => {
      if (!confirmDiscard()) return;
      const rec = runs.find((r) => r.id === id);
      if (rec && rec.workflow !== selected) setSelected(rec.workflow);
      loadRun(id);
    },
    [runs, selected, loadRun, confirmDiscard],
  );

  // ── edit lifecycle ──────────────────────────────────────────────────────
  const enterEdit = () => {
    if (!detail) return;
    setDraft(cloneGraph(detail.graph));
    setEditMode(true);
    setDirty(false);
    setValidation(null);
    resetHistory();
    reset();
  };
  const cancelEdit = () => {
    if (!confirmDiscard()) return;
    setEditMode(false);
    setDraft(null);
    setDirty(false);
    setValidation(null);
    setShowProblems(false);
    resetHistory();
  };

  // mutate reads the current draft directly (fresh per event) and records history.
  const mutate = (fn: (g: Graph) => Graph) => {
    if (!draft) return;
    undoRef.current.push(draft);
    if (undoRef.current.length > 60) undoRef.current.shift();
    redoRef.current = [];
    setDraft(fn(draft));
    setDirty(true);
    setValidation(null);
    syncHistFlags();
  };
  const undo = () => {
    if (!undoRef.current.length || !draft) return;
    redoRef.current.push(draft);
    setDraft(undoRef.current.pop()!);
    setDirty(true);
    setValidation(null);
    syncHistFlags();
  };
  const redo = () => {
    if (!redoRef.current.length || !draft) return;
    undoRef.current.push(draft);
    setDraft(redoRef.current.pop()!);
    setDirty(true);
    setValidation(null);
    syncHistFlags();
  };

  // Ctrl/Cmd+Z / +Shift+Z for graph undo/redo (skip when typing in a field).
  useEffect(() => {
    if (!editMode) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editMode, draft]);

  const doValidate = async () => {
    if (!draft || !selected) return;
    try {
      const report = await validateWorkflow(draft, selected);
      setValidation(report);
      setShowProblems(true);
      flash(report.ok ? "ok" : "err", report.ok ? "Valid ✓" : "Validation found problems");
    } catch (e) {
      flash("err", String((e as Error)?.message ?? e));
    }
  };

  const doSave = async () => {
    if (!draft || !selected) return;
    try {
      const saved = await updateWorkflow(selected, draft, { version: detail?.version });
      setDetail(saved);
      setEditMode(false);
      setDraft(null);
      setDirty(false);
      setValidation(null);
      setShowProblems(false);
      resetHistory();
      flash("ok", "Saved ✓");
      refreshWorkflows();
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) {
        const v = (e.detail as { validation?: ValidationReport })?.validation;
        if (v) {
          setValidation(v);
          setShowProblems(true);
        }
        flash("err", "Can't save — fix validation problems");
      } else {
        flash("err", String((e as Error)?.message ?? e));
      }
    }
  };

  // ── node / edge ops ───────────────────────────────────────────────────────
  const addNode = (kind: NodeKind) => {
    setAddOpen(false);
    if (!draft) return;
    const { graph, id } = addNodeOp(draft, kind);
    undoRef.current.push(draft);
    redoRef.current = [];
    setDraft(graph);
    setDirty(true);
    setValidation(null);
    syncHistFlags();
    setSelection({ type: "node", id });
  };
  const onConnect = (source: string, target: string) => {
    const src = draft?.nodes.find((n) => n.id === source);
    if (src?.kind === "router") {
      const label = window.prompt(`Route label for ${source} → ${target}:`, target);
      if (label === null) return;
      mutate((g) => connectOp(g, source, target, label || target));
    } else {
      mutate((g) => connectOp(g, source, target));
    }
  };
  const onRemoveEdge = () => {
    if (selection.type !== "edge") return;
    const { source, target, edgeKind, label } = selection;
    mutate((g) => disconnectOp(g, source, target, edgeKind, label));
    setSelection({ type: "graph" });
  };

  // ── new / import / export ────────────────────────────────────────────────
  const onNewWorkflow = async () => {
    if (!confirmDiscard()) return;
    const name = window.prompt("New workflow name:")?.trim();
    if (!name) return;
    const graph: Graph = {
      name,
      description: "",
      fail_fast: false,
      strict_inputs: false,
      inputs: [],
      nodes: [{ id: "start", kind: "base", goal: "", depends_on: [] }],
      outputs: ["start"],
    };
    try {
      const saved = await createWorkflow(name, graph);
      await refreshWorkflows();
      setSelected(name);
      setDetail(saved);
      setDraft(cloneGraph(saved.graph));
      setEditMode(true);
      setDirty(false);
      setValidation(null);
      resetHistory();
      flash("ok", `Created ${name}`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) flash("err", `'${name}' already exists`);
      else flash("err", String((e as Error)?.message ?? e));
    }
  };

  const onExport = () => {
    if (!activeGraph) return;
    const text = yaml.dump(activeGraph, { sortKeys: false });
    const blob = new Blob([text], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeGraph.name || "workflow"}.graph.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const onImportFile = (file: File) => {
    file.text().then((text) => {
      try {
        const parsed = (text.trim().startsWith("{")
          ? JSON.parse(text)
          : yaml.load(text)) as Graph;
        if (!parsed || !Array.isArray(parsed.nodes)) throw new Error("no nodes[] in file");
        if (!editMode) {
          undoRef.current = [];
          redoRef.current = [];
        } else if (draft) {
          undoRef.current.push(draft);
        }
        setDraft(parsed);
        setEditMode(true);
        setDirty(true);
        setValidation(null);
        syncHistFlags();
        flash("ok", "Imported — review & Save to persist");
      } catch (e) {
        flash("err", `Import failed: ${(e as Error).message}`);
      }
    });
  };

  // ── architect handlers ───────────────────────────────────────────────────
  const openArchitect = () => {
    if (!confirmDiscard()) return;
    setEditMode(false);
    setDraft(null);
    setDirty(false);
    setArchitectOpen(true);
  };
  const openBuiltWorkflow = (name: string) => {
    setArchitectOpen(false);
    clearBuild();
    refreshWorkflows().then(() => setSelected(name));
    // if already selected, force a reload
    if (selected === name) loadDetail(name);
  };

  // ── derived ────────────────────────────────────────────────────────────
  const architectPreview =
    architectOpen && archBuild.status !== "idle" && !!archBuild.graph;

  const activeDetail: WorkflowDetail | null = architectPreview
    ? { name: archBuild.graph!.name, version: undefined, graph: archBuild.graph! }
    : (() => {
        const g = editMode && draft ? draft : detail?.graph ?? null;
        return detail && g ? { ...detail, graph: g } : null;
      })();
  const activeGraph = activeDetail?.graph ?? null;

  const issues = useMemo<Record<string, "error" | "warning">>(() => {
    if (!validation) return {};
    const m: Record<string, "error" | "warning"> = {};
    for (const w of validation.warnings) if (w.node_id) m[w.node_id] = "warning";
    for (const g of validation.gaps) if (g.node_id) m[g.node_id] = "error";
    for (const e of validation.errors) if (e.node_id) m[e.node_id] = "error";
    return m;
  }, [validation]);

  const editApi: EditApi | null =
    editMode && selection.type === "node" && activeGraph
      ? {
          onPatch: (patch: Partial<GraphNode>) =>
            mutate((g) => updateNodeOp(g, selection.id, patch)),
          onDelete: () => {
            mutate((g) => deleteNodeOp(g, selection.id));
            setSelection({ type: "graph" });
          },
          onDuplicate: () => {
            if (!draft) return;
            const { graph, id } = duplicateNodeOp(draft, selection.id);
            undoRef.current.push(draft);
            redoRef.current = [];
            setDraft(graph);
            setDirty(true);
            setValidation(null);
            syncHistFlags();
            setSelection({ type: "node", id });
          },
          nodeIds: activeGraph.nodes.map((n) => n.id),
        }
      : null;

  const graphEdit: GraphEditApi | null = editMode
    ? {
        setInputs: (inputs) => mutate((g) => setInputsOp(g, inputs)),
        setOutputs: (outputs) => mutate((g) => setOutputsOp(g, outputs)),
      }
    : null;

  const showInspector = inspectorOpen && !!activeDetail;

  return (
    <div className={`app${showInspector ? " with-inspector" : ""}`}>
      <header className="topbar">
        <span className="brand">
          neuro<span className="accent">surfer</span> studio
        </span>
        <span className="spacer" />
        {architectPreview ? (
          <span className="meta">
            ✨ building: {activeGraph?.name} · {activeGraph?.nodes.length ?? 0} nodes
          </span>
        ) : (
          detail && (
            <span className="meta">
              {detail.name}
              {detail.version ? ` · v${detail.version}` : ""} · {activeGraph?.nodes.length ?? 0} nodes
              {dirty && <span className="dirty-dot" title="Unsaved changes"> ●</span>}
            </span>
          )
        )}

        {!editMode && !architectOpen && (
          <button
            className="btn-arch sm"
            style={{ marginLeft: 12 }}
            onClick={openArchitect}
          >
            ✨ Architect
          </button>
        )}

        {detail && !editMode && !architectOpen && (
          <>
            <button
              className="btn-primary sm"
              onClick={() => setLauncher("run")}
              disabled={run.status === "running"}
            >
              ▶ Run
            </button>
            <button className="topbar-btn" onClick={enterEdit}>
              ✎ Edit
            </button>
          </>
        )}

        {editMode && (
          <span className="edit-toolbar">
            <span className="add-wrap">
              <button className="btn-ghost sm" onClick={() => setAddOpen((o) => !o)}>
                ＋ Add ▾
              </button>
              {addOpen && (
                <div className="add-menu">
                  {KINDS.map((k) => (
                    <button key={k} onClick={() => addNode(k)}>
                      {k}
                    </button>
                  ))}
                </div>
              )}
            </span>
            <button className="btn-ghost sm" onClick={undo} disabled={!canUndo} title="Undo (⌘Z)">
              ↶
            </button>
            <button className="btn-ghost sm" onClick={redo} disabled={!canRedo} title="Redo (⇧⌘Z)">
              ↷
            </button>
            <button className="btn-ghost sm" onClick={onExport} title="Export YAML">
              ⭳
            </button>
            <button className="btn-ghost sm" onClick={() => fileRef.current?.click()} title="Import YAML/JSON">
              ⭱
            </button>
            <button className="btn-ghost sm" onClick={doValidate}>
              Validate
            </button>
            <button className="btn-primary sm" onClick={doSave} disabled={!dirty}>
              Save
            </button>
            <button className="btn-ghost sm" onClick={cancelEdit}>
              Cancel
            </button>
          </span>
        )}

        {detail && !inspectorOpen && (
          <button className="topbar-btn" onClick={() => setInspectorOpen(true)}>
            Inspector
          </button>
        )}
      </header>

      <Sidebar
        workflows={workflows}
        selected={selected}
        onSelect={chooseWorkflow}
        onNew={onNewWorkflow}
        runs={runs}
        activeRunId={run.runId}
        onSelectRun={onSelectRun}
        onRefreshRuns={refreshRuns}
      />

      <main className="canvas-area">
        {error && <div className="hint error">{error}</div>}
        {!error && !detail && selected && !architectPreview && (
          <div className="hint">Loading {selected}…</div>
        )}
        {!error && !selected && !architectOpen && (
          <div className="hint">Select a workflow, + New, or ✨ Architect.</div>
        )}
        {activeDetail && (
          <>
            {!editMode && !architectPreview && (
              <RunStatusBar
                run={run}
                totalNodes={activeDetail.graph.nodes.length}
                onCancel={cancel}
                onResume={() => setLauncher("resume")}
                onClear={reset}
              />
            )}
            {editMode && (
              <div className="edit-hint">
                Editing · drag between node edges to wire · click nodes/edges to edit
              </div>
            )}
            {architectPreview && (
              <div className="edit-hint arch">✨ Architect building — live preview</div>
            )}
            <Canvas
              graph={activeDetail.graph}
              workflowName={activeDetail.name}
              version={activeDetail.version}
              selection={selection}
              onSelect={setSelection}
              nodeStatus={editMode ? undefined : run.nodeStatus}
              issues={editMode ? issues : undefined}
              editable={editMode}
              onConnect={onConnect}
            />
            {showProblems && validation && (
              <ProblemsPanel
                report={validation}
                onGoNode={(id) => setSelection({ type: "node", id })}
                onClose={() => setShowProblems(false)}
              />
            )}
          </>
        )}
        {architectOpen && (
          <ArchitectDock
            build={archBuild}
            onStart={(intent, verify) => startBuild(intent, verify)}
            onOpen={openBuiltWorkflow}
            onNewBuild={clearBuild}
            onClose={() => {
              setArchitectOpen(false);
              clearBuild();
            }}
          />
        )}
        {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
      </main>

      {showInspector && activeDetail && (
        <Inspector
          detail={activeDetail}
          selection={selection}
          onSelect={setSelection}
          onClose={() => setInspectorOpen(false)}
          run={editMode ? undefined : run}
          edit={editApi}
          graphEdit={graphEdit}
          onRemoveEdge={editMode ? onRemoveEdge : undefined}
        />
      )}

      {launcher && detail && (
        <RunLauncher
          title={launcher === "resume" ? `Resume ${detail.name}` : `Run ${detail.name}`}
          inputs={detail.graph.inputs}
          submitLabel={launcher === "resume" ? "Resume" : "Run"}
          busy={run.status === "running"}
          onSubmit={
            launcher === "resume"
              ? (v) => resume(v).then(() => setLauncher(null))
              : (v) => start(detail.name, v).then(() => setLauncher(null))
          }
          onClose={() => setLauncher(null)}
        />
      )}

      <input
        ref={fileRef}
        type="file"
        accept=".yaml,.yml,.json"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onImportFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
