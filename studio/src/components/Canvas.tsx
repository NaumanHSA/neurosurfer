/**
 * The graph canvas — an interactive, dark React Flow surface (S1).
 *
 * Nodes are draggable; positions persist to localStorage per workflow. A dagre
 * "Auto arrange" and "Reset layout" live in a toolbar, plus fit-view and a node
 * search. Selecting a node highlights its connected edges and syncs the shared
 * selection that drives the inspector. Read-only: no wiring/editing yet (S4).
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import WorkflowNode from "./WorkflowNode";
import { graphToFlow, type EdgeKind, type FlowNodeData } from "@/graph/adapter";
import {
  applySaved,
  autoLayout,
  clearLayout,
  loadLayout,
  saveLayout,
} from "@/graph/layout";
import { kindMeta } from "@/graph/nodeKinds";
import type { Graph } from "@/api/types";
import type { FlowSelection } from "@/selection";
import type { NodeLiveStatus } from "@/run/types";

const nodeTypes = { workflow: WorkflowNode };
const HIGHLIGHT = "#8ab4ff";

interface Props {
  graph: Graph;
  workflowName: string;
  version?: string;
  selection: FlowSelection;
  onSelect: (sel: FlowSelection) => void;
  nodeStatus?: Record<string, NodeLiveStatus>;
  issues?: Record<string, "error" | "warning">;
  editable?: boolean;
  onConnect?: (source: string, target: string) => void;
}

function CanvasInner({
  graph,
  workflowName,
  version,
  selection,
  onSelect,
  nodeStatus,
  issues,
  editable,
  onConnect,
}: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const rf = useReactFlow();
  const wrapRef = useRef<HTMLDivElement>(null);

  const baseEdges = useMemo<Edge[]>(() => graphToFlow(graph).edges, [graph]);

  // Structural signature: changes only when nodes are added/removed (not when a
  // field is edited). Positions are rebuilt on structural change only, so typing
  // in the inspector never resets the layout or refits the view.
  const structuralKey = useMemo(
    () => graph.nodes.map((n) => n.id).sort().join(","),
    [graph],
  );

  // (Re)build node positions on structural change: saved layout if any, else dagre.
  useEffect(() => {
    const { nodes: fresh, edges } = graphToFlow(graph);
    const saved = loadLayout(workflowName, version);
    const positioned = saved
      ? applySaved(fresh, saved)
      : autoLayout(fresh as Node<FlowNodeData>[], edges);
    setNodes(positioned);
    requestAnimationFrame(() => rf.fitView({ padding: 0.2, duration: 300 }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey, workflowName, version]);

  // Refresh node config (prompts/fields) in place when the graph changes, without
  // moving nodes — keeps the canvas cards in sync with inspector edits.
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => {
        const g = graph.nodes.find((x) => x.id === n.id);
        if (!g) return n;
        const data = n.data as FlowNodeData;
        return { ...n, data: { ...data, node: g, bodyIds: (g.body ?? []).map((b) => b.id) } };
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  // Reflect external selection onto node highlight + center the selected node.
  useEffect(() => {
    const selId = selection.type === "node" ? selection.id : null;
    setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === selId })));
    if (selId) {
      rf.fitView({ nodes: [{ id: selId }], maxZoom: 1.4, padding: 0.6, duration: 400 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection]);

  // Inject live run status + validation severity into node data.
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => {
        const st = nodeStatus?.[n.id];
        const iss = issues?.[n.id];
        const data = n.data as FlowNodeData;
        if (data.runStatus === st && data.issue === iss) return n;
        return { ...n, data: { ...data, runStatus: st, issue: iss } };
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeStatus, issues]);

  // Edges: accent + animate those touching the selected node.
  const edges = useMemo<Edge[]>(() => {
    const selId = selection.type === "node" ? selection.id : null;
    if (!selId) return baseEdges;
    return baseEdges.map((e) =>
      e.source === selId || e.target === selId
        ? { ...e, animated: true, style: { ...e.style, stroke: HIGHLIGHT, strokeWidth: 2.5 } }
        : { ...e, style: { ...e.style, opacity: 0.5 } },
    );
  }, [baseEdges, selection]);

  const persist = useCallback(() => {
    setNodes((cur) => {
      saveLayout(workflowName, version, cur as Node<FlowNodeData>[]);
      return cur;
    });
  }, [setNodes, workflowName, version]);

  const doAutoLayout = useCallback(() => {
    setNodes((cur) => {
      const laid = autoLayout(cur as Node<FlowNodeData>[], baseEdges);
      saveLayout(workflowName, version, laid);
      requestAnimationFrame(() => rf.fitView({ padding: 0.2, duration: 300 }));
      return laid;
    });
  }, [setNodes, baseEdges, workflowName, version, rf]);

  const resetLayout = useCallback(() => {
    clearLayout(workflowName, version);
    doAutoLayout();
  }, [workflowName, version, doAutoLayout]);

  // Keyboard: f = fit, Escape = show graph overview. Ignore when typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "f") rf.fitView({ padding: 0.2, duration: 300 });
      if (e.key === "Escape") onSelect({ type: "graph" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rf, onSelect]);

  const search = useCallback(
    (q: string) => {
      const query = q.trim().toLowerCase();
      if (!query) return;
      const hit = graph.nodes.find(
        (n) =>
          n.id.toLowerCase().includes(query) ||
          (n.goal ?? n.purpose ?? "").toLowerCase().includes(query),
      );
      if (hit) onSelect({ type: "node", id: hit.id });
    },
    [graph, onSelect],
  );

  return (
    <div ref={wrapRef} style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, n) => onSelect({ type: "node", id: n.id })}
        onNodeDragStop={persist}
        onEdgeClick={(_, e) =>
          onSelect({
            type: "edge",
            source: e.source,
            target: e.target,
            edgeKind: ((e.data?.kind as EdgeKind) ?? "data"),
            label: typeof e.label === "string" ? e.label : undefined,
          })
        }
        onPaneClick={() => onSelect({ type: "graph" })}
        onConnect={(c) => {
          if (c.source && c.target) onConnect?.(c.source, c.target);
        }}
        minZoom={0.15}
        maxZoom={2}
        nodesConnectable={!!editable}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#2a2f3a" />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeStrokeWidth={2}
          nodeBorderRadius={6}
          nodeColor={(n: Node) =>
            kindMeta((n.data as FlowNodeData)?.node?.kind ?? "base").accent
          }
          nodeStrokeColor="#0d0f13"
          maskColor="rgba(10,12,16,0.6)"
        />
        <Panel position="top-left">
          <div className="canvas-toolbar">
            <button onClick={doAutoLayout} title="Auto arrange (dagre)">
              ⧉ Auto arrange
            </button>
            <button onClick={resetLayout} title="Reset saved layout">
              ↺ Reset
            </button>
            <button onClick={() => rf.fitView({ padding: 0.2, duration: 300 })} title="Fit view (f)">
              ⤢ Fit
            </button>
            <input
              className="canvas-search"
              placeholder="Find node…"
              onKeyDown={(e) => {
                if (e.key === "Enter") search((e.target as HTMLInputElement).value);
              }}
            />
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}

export default function Canvas(props: Props) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
