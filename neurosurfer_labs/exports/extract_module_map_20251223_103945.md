# Node `extract_module_map` output

- Mode: `structured`
- Started at: `2025-12-23T10:39:45.809782`
- Duration: `10161` ms
- Error: `None`

---

{
  "overview": "The 'neurosurger_tracing_docs' module provides a structured tracing system for tracking steps in workflows, with support for logging, rendering, and configuration. It includes classes for trace logs, steps, contexts, and tracers, along with configuration options for enabling tracing and controlling log output.",
  "components": "Main components include TraceLog, TraceStep, TraceStepContext, Tracer, TracerConfig, and various span tracers (e.g., ConsoleTracer, LoggerTracer). These components work together to record, log, and render tracing data.",
  "flows": "Tracing starts with Tracer.step(), which creates a TraceStepContext. This context records start and end times, logs messages, and stores inputs/outputs. The Tracer renders the trace using configured span tracers, and results can be viewed as structured data or formatted text/markdown.",
  "config": "TracerConfig controls tracing behavior, including whether tracing is enabled, whether steps are logged, and formatting options like indentation and log types. It also allows setting a maximum preview length for outputs.",
  "examples": "Example usage includes creating a traced step with Tracer.step(), logging messages with .log(), and rendering the trace with .render(). Tracing can be disabled by setting config.enabled=False.",
  "pitfalls": "Tracing is disabled by default, so ensure config.enabled=True if needed. Logs may be truncated if outputs exceed max_output_preview_chars. Ensure proper import of TracerConfig and span tracers for correct behavior."
}