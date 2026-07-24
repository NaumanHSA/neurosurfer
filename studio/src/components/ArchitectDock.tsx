/**
 * ArchitectDock (S5) — describe an intent and watch the ReAct Architect design,
 * test, and register a workflow live. Streams the agent's step log; the canvas
 * shows the graph assembling (via the build's graph snapshots). Terminal states:
 * succeeded (Open the workflow), blocked (reason), or failed (error).
 */
import { useEffect, useRef, useState } from "react";
import type { BuildView } from "@/architect/useArchitect";

const EXAMPLES = [
  "Summarise an article and write a title for it",
  "Classify a support ticket and route it to the right team",
  "Draft a product slogan and refine it until it's punchy",
];

interface Props {
  build: BuildView;
  onStart: (intent: string, verify: string) => void;
  onOpen: (workflow: string) => void;
  onNewBuild: () => void;
  onClose: () => void;
}

const STATUS_COLOR: Record<string, string> = {
  running: "#7c6cff",
  succeeded: "#25c2a0",
  blocked: "#f2c94c",
  failed: "#ff5c5c",
};

export default function ArchitectDock({
  build,
  onStart,
  onOpen,
  onNewBuild,
  onClose,
}: Props) {
  const [intent, setIntent] = useState("");
  const [verify, setVerify] = useState("encouraged");
  const logRef = useRef<HTMLDivElement>(null);
  const idle = build.status === "idle";
  const running = build.status === "running";

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [build.log.length]);

  return (
    <div className="arch-dock">
      <div className="arch-head">
        <span className="arch-title">✨ Architect</span>
        {!idle && (
          <span className="arch-status" style={{ color: STATUS_COLOR[build.status] }}>
            <span className={running ? "pulsing" : ""}>●</span> {build.status}
          </span>
        )}
        <span className="spacer" />
        {!idle && !running && (
          <button className="btn-ghost sm" onClick={onNewBuild}>
            New build
          </button>
        )}
        <button className="insp-close" onClick={onClose}>
          ✕
        </button>
      </div>

      {idle ? (
        <div className="arch-body">
          <div className="arch-label">Describe the workflow you want:</div>
          <textarea
            className="arch-intent"
            rows={3}
            placeholder="e.g. Summarise an article and write a title…"
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
          />
          <div className="arch-examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip" onClick={() => setIntent(ex)}>
                {ex}
              </button>
            ))}
          </div>
          <div className="arch-controls">
            <label className="arch-verify">
              verify
              <select value={verify} onChange={(e) => setVerify(e.target.value)}>
                <option value="off">off</option>
                <option value="encouraged">encouraged</option>
                <option value="required">required</option>
              </select>
            </label>
            <span className="spacer" />
            <button
              className="btn-primary sm"
              disabled={!intent.trim()}
              onClick={() => onStart(intent.trim(), verify)}
            >
              ✨ Build
            </button>
          </div>
        </div>
      ) : (
        <div className="arch-body">
          <div className="arch-intent-echo">“{build.intent || intentFrom(build)}”</div>
          <div className="arch-log" ref={logRef}>
            {build.log.map((l, i) => (
              <div className="arch-log-line" key={i}>
                <span className="arch-log-dot">›</span>
                {l.message}
              </div>
            ))}
            {running && <div className="arch-log-line pulsing">working…</div>}
          </div>

          {build.status === "succeeded" && build.workflow && (
            <div className="arch-outcome ok">
              <span>✓ Registered <b>{build.workflow}</b></span>
              <button className="btn-primary sm" onClick={() => onOpen(build.workflow!)}>
                Open workflow
              </button>
            </div>
          )}
          {build.status === "blocked" && (
            <div className="arch-outcome warn">⚠ Blocked: {build.error}</div>
          )}
          {build.status === "failed" && (
            <div className="arch-outcome err">✕ Failed: {build.error}</div>
          )}
        </div>
      )}
    </div>
  );
}

function intentFrom(build: BuildView): string {
  return build.graph?.name ?? "building…";
}
