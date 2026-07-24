/** What the inspector is currently showing. */
import type { EdgeKind } from "./graph/adapter";

export type FlowSelection =
  | { type: "node"; id: string }
  | { type: "edge"; source: string; target: string; edgeKind: EdgeKind; label?: string }
  | { type: "graph" };
