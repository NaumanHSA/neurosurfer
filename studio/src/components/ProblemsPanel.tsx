/** Problems panel (S4) — lists validation errors/gaps/warnings; click to select. */
import type { ValidationReport } from "@/api/types";

interface Props {
  report: ValidationReport;
  onGoNode: (id: string) => void;
  onClose: () => void;
}

const SECTIONS: { key: keyof ValidationReport; label: string; cls: string }[] = [
  { key: "errors", label: "Errors", cls: "err" },
  { key: "gaps", label: "Capability gaps", cls: "err" },
  { key: "warnings", label: "Warnings", cls: "warn" },
];

export default function ProblemsPanel({ report, onGoNode, onClose }: Props) {
  const total = report.errors.length + report.gaps.length + report.warnings.length;
  return (
    <div className="problems">
      <div className="problems-head">
        <span>
          {report.ok ? "✓ Valid" : "Problems"}
          {total > 0 ? ` · ${total}` : ""}
        </span>
        <button className="insp-close" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="problems-body">
        {total === 0 && <div className="dim">No problems — the graph is valid.</div>}
        {SECTIONS.map(({ key, label, cls }) => {
          const items = report[key] as ValidationReport["errors"];
          if (!Array.isArray(items) || items.length === 0) return null;
          return (
            <div key={key}>
              <div className={`problems-section ${cls}`}>{label}</div>
              {items.map((it, i) => (
                <div
                  key={i}
                  className={`problem ${cls}${it.node_id ? " clickable" : ""}`}
                  onClick={() => it.node_id && onGoNode(it.node_id)}
                >
                  {it.node_id && <span className="problem-node">{it.node_id}</span>}
                  <span>{it.message}</span>
                  {it.suggestion && <span className="problem-hint"> → {it.suggestion}</span>}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
