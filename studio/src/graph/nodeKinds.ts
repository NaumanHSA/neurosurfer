/**
 * Visual metadata per engine node kind — the studio's "node vocabulary".
 *
 * Each kind maps to ComfyUI's node-card feel: a colored header/accent so kinds
 * are distinguishable at a glance, a short glyph, and a human label. The accent
 * doubles as the output-socket color, the way ComfyUI colors sockets by type.
 */
import type { NodeKind } from "@/api/types";

export interface KindMeta {
  label: string;
  /** Accent / header / output-socket color. */
  accent: string;
  /** Short glyph shown in the node header. */
  glyph: string;
  /** One-line description shown in tooltips / the (future) inspector. */
  blurb: string;
}

export const KIND_META: Record<NodeKind, KindMeta> = {
  base: {
    label: "Agent",
    accent: "#5b9dff",
    glyph: "◆",
    blurb: "A single LLM step — prompted by its goal/purpose.",
  },
  react: {
    label: "ReAct Agent",
    accent: "#7c6cff",
    glyph: "◈",
    blurb: "A reasoning+acting agent that can call tools in a loop.",
  },
  function: {
    label: "Function",
    accent: "#25c2a0",
    glyph: "ƒ",
    blurb: "A registered function/tool invoked with typed inputs.",
  },
  python: {
    label: "Python",
    accent: "#f2c94c",
    glyph: "λ",
    blurb: "Inline Python transformation over state.",
  },
  tool: {
    label: "Tool",
    accent: "#25c2a0",
    glyph: "⚙",
    blurb: "A single tool call (native, generated, or MCP).",
  },
  router: {
    label: "Router",
    accent: "#ff8a3d",
    glyph: "◇",
    blurb: "Classifies input and selects one downstream branch.",
  },
  loop: {
    label: "Loop",
    accent: "#ff5c8a",
    glyph: "↻",
    blurb: "Repeats its body until a stop condition or iteration cap.",
  },
  map: {
    label: "Map",
    accent: "#c07cff",
    glyph: "⧉",
    blurb: "Runs its body once per item of a collection (fan-out).",
  },
  subgraph: {
    label: "Subgraph",
    accent: "#9aa0aa",
    glyph: "▤",
    blurb: "Runs a nested sub-workflow once (composition).",
  },
  input: {
    label: "Human Input",
    accent: "#eaeaea",
    glyph: "⏸",
    blurb: "Human-in-the-loop: pauses for a supplied value.",
  },
};

export function kindMeta(kind: string): KindMeta {
  return (
    KIND_META[kind as NodeKind] ?? {
      label: kind,
      accent: "#9aa0aa",
      glyph: "•",
      blurb: "Unknown node kind.",
    }
  );
}
