/**
 * RunLauncher (S3) — a modal that builds an inputs form from a workflow's
 * declared `inputs` and starts (or resumes) a run. Field widget is chosen by the
 * input's declared type; object/array inputs take JSON. Required fields are
 * validated before submit.
 */
import { useState } from "react";
import type { GraphInput } from "@/api/types";

interface Props {
  title: string;
  inputs: GraphInput[];
  initial?: Record<string, unknown>;
  submitLabel?: string;
  busy?: boolean;
  onSubmit: (values: Record<string, unknown>) => void;
  onClose: () => void;
}

function typeOf(input: GraphInput): string {
  return String((input as Record<string, unknown>).type ?? "string").toLowerCase();
}
function isRequired(input: GraphInput): boolean {
  return Boolean((input as Record<string, unknown>).required);
}
function descOf(input: GraphInput): string | undefined {
  const d = (input as Record<string, unknown>).description;
  return typeof d === "string" ? d : undefined;
}

export default function RunLauncher({
  title,
  inputs,
  initial,
  submitLabel = "Run",
  busy,
  onSubmit,
  onClose,
}: Props) {
  const [raw, setRaw] = useState<Record<string, string>>(() => {
    const seed: Record<string, string> = {};
    for (const i of inputs) {
      const v = initial?.[i.name];
      seed[i.name] =
        v == null ? "" : typeof v === "string" ? v : JSON.stringify(v, null, 2);
    }
    return seed;
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const setVal = (name: string, v: string) =>
    setRaw((r) => ({ ...r, [name]: v }));

  const submit = () => {
    const out: Record<string, unknown> = {};
    const errs: Record<string, string> = {};
    for (const input of inputs) {
      const t = typeOf(input);
      const s = raw[input.name] ?? "";
      if (!s.trim()) {
        if (isRequired(input)) errs[input.name] = "required";
        continue;
      }
      try {
        if (t === "number" || t === "float") out[input.name] = parseFloat(s);
        else if (t === "integer" || t === "int") out[input.name] = parseInt(s, 10);
        else if (t === "boolean" || t === "bool")
          out[input.name] = s === "true" || s === "1";
        else if (t === "object" || t === "array" || t === "json")
          out[input.name] = JSON.parse(s);
        else out[input.name] = s;
      } catch {
        errs[input.name] = "invalid JSON";
      }
    }
    setErrors(errs);
    if (Object.keys(errs).length === 0) onSubmit(out);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>{title}</span>
          <button className="insp-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="modal-body">
          {inputs.length === 0 && (
            <div className="dim">This workflow declares no inputs — just run it.</div>
          )}
          {inputs.map((input) => {
            const t = typeOf(input);
            const desc = descOf(input);
            const err = errors[input.name];
            const isBool = t === "boolean" || t === "bool";
            return (
              <div className="field" key={input.name}>
                <label className="field-label">
                  {input.name}
                  <span className="field-type">{t}</span>
                  {isRequired(input) && <span className="field-req">required</span>}
                </label>
                {desc && <div className="field-desc">{desc}</div>}
                {isBool ? (
                  <select
                    value={raw[input.name] || "false"}
                    onChange={(e) => setVal(input.name, e.target.value)}
                  >
                    <option value="false">false</option>
                    <option value="true">true</option>
                  </select>
                ) : t === "number" || t === "float" || t === "integer" || t === "int" ? (
                  <input
                    type="number"
                    value={raw[input.name] ?? ""}
                    onChange={(e) => setVal(input.name, e.target.value)}
                  />
                ) : (
                  <textarea
                    rows={t === "object" || t === "array" || t === "json" ? 4 : 3}
                    placeholder={
                      t === "object" || t === "array" || t === "json"
                        ? "JSON…"
                        : "value…"
                    }
                    value={raw[input.name] ?? ""}
                    onChange={(e) => setVal(input.name, e.target.value)}
                  />
                )}
                {err && <div className="field-err">{err}</div>}
              </div>
            );
          })}
        </div>
        <div className="modal-foot">
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-primary" onClick={submit} disabled={busy}>
            {busy ? "Starting…" : `▶ ${submitLabel}`}
          </button>
        </div>
      </div>
    </div>
  );
}
