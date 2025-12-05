from typing import Dict, Any, List, Optional
import json
import textwrap

from .templates import PYTHON_EXEC_USER_PROMPT_TEMPLATE
from .config import MemoryStyle, PythonExecToolConfig

def build_user_prompt(
    task: str,
    files_listing: str,
    context: Optional[Dict[str, Any]],
    *,
    max_snippet_len: int = 1200,
) -> str:
    """
    Build the user prompt for the code-generation LLM.

    - `files_listing` is already a JSON string of available files.
    - `context` is a dict of memory slots (e.g. python_last_result_summary).
      For strings, we print them as-is (no json.dumps to avoid \"\\n\" spam).
      For non-strings, we pretty-print JSON.
    """
    ctx_text = "(none)"
    if context:
        parts: List[str] = []
        for key, value in context.items():
            # 1) Choose a human-friendly representation
            if isinstance(value, str):
                # Already a nice multi-line description (from extras builder)
                snippet = value
            else:
                # Fallback: JSON pretty-print for structures
                try:
                    snippet = json.dumps(
                        value,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                except Exception:
                    snippet = str(value)

            # 2) Truncate very large snippets to avoid blowing up the prompt
            if len(snippet) > max_snippet_len:
                snippet = snippet[:max_snippet_len] + "\n... (truncated)"

            # 3) Indent snippet for readability under the key
            snippet_indented = textwrap.indent(snippet, "  ")

            parts.append(f"{key}:\n{snippet_indented}")

        ctx_text = "\n\n".join(parts)

    tpl = PYTHON_EXEC_USER_PROMPT_TEMPLATE.format(
        task=task,
        files_listing=files_listing,
        context=ctx_text,
    )
    return textwrap.dedent(tpl).strip()
    

 # ------------- Internals -------------
def format_files_listing(
    files_context: Dict[str, Dict[str, Any]],
    file_names: List[str],
) -> str:
    """
    Render the available files as JSON so the LLM clearly sees
    the *exact* dictionary keys it can use in `files[...]`.

    Example output:

    Available files (from chat session):
    {
    "archive.zip/Student Degree College Data.csv": {
        "mime": "text/csv",
        "size": 411545,
        "path": "/abs/path/Student Degree College Data.csv"
    },
    ...
    }

    The prompt around this should explicitly say:
    - 'Use the *keys* of this JSON object as dictionary keys for `files`.'
    """
    if not files_context:
        return "(no files available)"

    filtered: Dict[str, Dict[str, Any]] = {}
    for name in file_names:
        meta = files_context.get(name)
        if not meta:
            continue
        # Keep only useful fields (optional)
        filtered[name] = {
            "mime": meta.get("mime"),
            "size": meta.get("size"),
            "path": meta.get("path"),
        }
    if not filtered:
        return "(no matching files in context)"
    return json.dumps(filtered, indent=2)
    
def format_result(result: Any, max_table_rows: int = 15) -> str:
    # Import here to avoid hard dependency at module import time
    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd is not None:
        import pandas as _pd
        if isinstance(result, _pd.DataFrame):
            if len(result) > max_table_rows:
                result = result.head(max_table_rows)
            return result.to_markdown(index=False)
        if isinstance(result, _pd.Series):
            if len(result) > max_table_rows:
                result = result.head(max_table_rows)
            return result.to_markdown()
    if isinstance(result, (list, dict)):
        return json.dumps(result, indent=2, default=str)
    return str(result)


def build_memory_extras_for_result(
    result: Any,
    *,
    result_limit: int = 10,
    char_limit: int = 1200,
    style: MemoryStyle = "text",
) -> Dict[str, Any]:
    """
    Build JSON-safe memory extras summarizing `result`.

    `style` controls what goes in `python_last_result_summary.value`:
      - "text": a compact, human-readable string (best for LLM prompts).
      - "structured": the old JSON-structured summary.
      - "both": a dict with {"as_text": ..., "as_structured": ...}.
    """
    extras: Dict[str, Any] = {}

    def _clip(s: str) -> str:
        s = s.strip()
        if len(s) <= char_limit:
            return s
        return s[: char_limit] + "... [truncated]"

    def _wrap_value(text_value: Any, description: str) -> Dict[str, Any]:
        return {
            "value": text_value,
            "description": description,
            "visible_to_llm": True,
        }

    # ----------------- pandas specialization -----------------
    try:
        import pandas as _pd  # type: ignore

        # DataFrame
        if isinstance(result, _pd.DataFrame):
            description = (
                "Summary of the last DataFrame computed by python_execute "
                "(columns, dtypes, shape, and small head sample)."
            )

            # TEXT summary
            df_head = result.head(result_limit)
            try:
                preview = df_head.to_string(index=False)
            except Exception:
                # Fallback if to_string fails for some reason
                preview = df_head.to_csv(index=False)

            text_summary = _clip(
                textwrap.dedent(
                    f"""
                    DataFrame summary:
                    - shape: {result.shape}
                    - columns: {list(result.columns)}

                    Head sample:
                    {preview}
                    """
                )
            )

            # STRUCTURED summary (old style)
            structured_summary = {
                "kind": "dataframe",
                "columns": list(result.columns),
                "dtypes": {c: str(t) for c, t in result.dtypes.items()},
                "shape": list(result.shape),
                "sample_head": df_head.to_dict(orient="list"),
            }

            if style == "text":
                extras["python_last_result_summary"] = _wrap_value(
                    text_summary, description
                )
            elif style == "structured":
                extras["python_last_result_summary"] = _wrap_value(
                    structured_summary, description
                )
            else:  # "both"
                extras["python_last_result_summary"] = _wrap_value(
                    {
                        "as_text": text_summary,
                        "as_structured": structured_summary,
                    },
                    description,
                )
            return extras

        # Series
        if isinstance(result, _pd.Series):
            description = (
                "Summary of the last Series computed by python_execute "
                "(name, dtype, length, and a small sample)."
            )

            # TEXT summary
            try:
                preview = result.head(result_limit).to_string()
            except Exception:
                preview = str(result.head(result_limit).tolist())

            text_summary = _clip(
                textwrap.dedent(
                    f"""
                    Series summary:
                    - name: {result.name}
                    - dtype: {result.dtype}
                    - length: {len(result)}

                    Head sample:
                    {preview}
                    """
                )
            )

            # STRUCTURED summary
            structured_summary = {
                "kind": "series",
                "name": result.name,
                "dtype": str(result.dtype),
                "length": int(len(result)),
                "index_sample": list(result.index[:result_limit]),
                "values_sample": result.head(result_limit).tolist(),
            }

            if style == "text":
                extras["python_last_result_summary"] = _wrap_value(
                    text_summary, description
                )
            elif style == "structured":
                extras["python_last_result_summary"] = _wrap_value(
                    structured_summary, description
                )
            else:  # "both"
                extras["python_last_result_summary"] = _wrap_value(
                    {
                        "as_text": text_summary,
                        "as_structured": structured_summary,
                    },
                    description,
                )
            return extras

    except Exception:
        # pandas not installed or error; fall through to generic paths
        pass

    # ----------------- Generic fallbacks -----------------

    # Simple primitives
    if isinstance(result, (str, int, float, bool, type(None))):
        text_summary = _clip(repr(result))
        structured = {
            "kind": type(result).__name__,
            "value": result,
        }

        if style == "text":
            extras["python_last_result_summary"] = _wrap_value(
                text_summary, "Primitive result from the last python_execute call."
            )
        elif style == "structured":
            extras["python_last_result_summary"] = _wrap_value(
                structured, "Primitive result from the last python_execute call."
            )
        else:
            extras["python_last_result_summary"] = _wrap_value(
                {
                    "as_text": text_summary,
                    "as_structured": structured,
                },
                "Primitive result from the last python_execute call.",
            )
        return extras

    # Lists / tuples
    if isinstance(result, (list, tuple)):
        preview = list(result[:result_limit])
        text_summary = _clip(
            f"Sequence of length {len(result)}. Sample: {preview!r}"
        )
        structured = {
            "kind": "sequence",
            "type": type(result).__name__,
            "length": len(result),
            "sample": preview,
        }

        if style == "text":
            extras["python_last_result_summary"] = _wrap_value(
                text_summary, "Sequence result from the last python_execute call."
            )
        elif style == "structured":
            extras["python_last_result_summary"] = _wrap_value(
                structured, "Sequence result from the last python_execute call."
            )
        else:
            extras["python_last_result_summary"] = _wrap_value(
                {
                    "as_text": text_summary,
                    "as_structured": structured,
                },
                "Sequence result from the last python_execute call.",
            )
        return extras

    # Dict
    if isinstance(result, dict):
        keys = list(result.keys())
        preview_keys = keys[:result_limit]
        preview = {k: result[k] for k in preview_keys}
        text_summary = _clip(
            f"Mapping with {len(keys)} keys. Sample keys: {preview_keys!r}. "
            f"Sample values: {preview!r}"
        )
        structured = {
            "kind": "mapping",
            "type": type(result).__name__,
            "keys_sample": preview_keys,
            "values_sample": preview,
        }

        if style == "text":
            extras["python_last_result_summary"] = _wrap_value(
                text_summary, "Mapping/dict-like result from the last python_execute call."
            )
        elif style == "structured":
            extras["python_last_result_summary"] = _wrap_value(
                structured, "Mapping/dict-like result from the last python_execute call."
            )
        else:
            extras["python_last_result_summary"] = _wrap_value(
                {
                    "as_text": text_summary,
                    "as_structured": structured,
                },
                "Mapping/dict-like result from the last python_execute call.",
            )
        return extras

    # Arbitrary objects
    rep = _clip(repr(result))
    text_summary = f"Object of type {type(result).__name__}. Repr snippet:\n{rep}"
    structured = {
        "kind": "object",
        "type": type(result).__name__,
        "repr": rep,
    }

    if style == "text":
        extras["python_last_result_summary"] = _wrap_value(
            text_summary,
            "Opaque object result from the last python_execute call "
            "(type name and repr snippet only).",
        )
    elif style == "structured":
        extras["python_last_result_summary"] = _wrap_value(
            structured,
            "Opaque object result from the last python_execute call "
            "(type name and repr snippet only).",
        )
    else:
        extras["python_last_result_summary"] = _wrap_value(
            {
                "as_text": text_summary,
                "as_structured": structured,
            },
            "Opaque object result from the last python_execute call "
            "(type name and repr snippet only).",
        )
    return extras

def build_error_extras(e: Exception, tb: str, char_limit: int = 2000) -> Dict[str, Any]:
    """
    Generic, type-agnostic metadata about the last Python error.
    """
    extras: Dict[str, Any] = {}
    exc_type = type(e).__name__
    msg = str(e)

    # Default suggestion: let the LLM try adjusting code once or twice.
    suggestion = "change_code"
    # Lightweight heuristics, not tied to pandas:
    if isinstance(e, KeyError):
        suggestion = "inspect_data_or_keys"
    elif isinstance(e, FileNotFoundError):
        suggestion = "check_files_or_paths"
    elif isinstance(e, ImportError):
        suggestion = "forbidden_or_missing_library"
    elif isinstance(e, MemoryError):
        suggestion = "reduce_memory_usage"
    elif isinstance(e, TimeoutError):
        suggestion = "reduce_computation_time"
    extras["python_last_error"] = {
        "value": {
            "exc_type": exc_type,
            "message": msg,
            "suggested_next_step": suggestion,
        },
        "description": (
            "Structured metadata about the last Python error raised by python_execute."
        ),
        "visible_to_llm": True,
    }

    # Optionally keep a short traceback snippet (not full 10k chars)
    extras["python_last_error_traceback"] = {
        "value": tb[-char_limit:],  # last char_limit chars
        "description": f"Truncated traceback for the last Python error (limit: {char_limit} chars).",
        "visible_to_llm": False,  # or True, depending on how much you want to expose
    }
    return extras
    
def extract_code_block(text: str) -> str:
    import re
    pattern = r"```(?:python)?\s*(.*?)```"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()
