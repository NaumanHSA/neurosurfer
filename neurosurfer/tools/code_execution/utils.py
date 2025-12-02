from typing import Dict, Any, List, Optional
import json
import textwrap

from .templates import PYTHON_EXEC_USER_PROMPT_TEMPLATE


def build_user_prompt(task: str, files_listing: str, context: Optional[Dict[str, Any]]) -> str:
    # Pretty-print context (memory) for the code-LLM, but keep it bounded.
    ctx_text = "(none)"
    if context:
        parts: List[str] = []
        for key, value in context.items():
            # Try to turn the value into a JSON-like string
            try:
                snippet = json.dumps(value, indent=2, default=str)
            except Exception:
                snippet = str(value)

            # Truncate very large snippets to avoid blowing up the prompt
            max_len = 1000
            if len(snippet) > max_len:
                snippet = snippet[:max_len] + "\n... (truncated)"

            parts.append(f"{key}:\n{snippet}")
        ctx_text = "\n\n".join(parts)
    tpl = PYTHON_EXEC_USER_PROMPT_TEMPLATE.format(
        task=task,
        files_listing=files_listing,
        context=ctx_text,
    )
    return textwrap.dedent(tpl).strip()
    

 # ------------- Internals -------------
def format_files_listing(files_context: Dict[str, Dict[str, Any]], file_names: List[str]) -> str:
    if not files_context:
        return "(no files available)"
    lines: List[str] = []
    for name in file_names:
        meta = files_context.get(name)
        if not meta:
            continue
        size = meta.get("size")
        mime = meta.get("mime")
        path = meta.get("path")
        lines.append(f"- {name} (mime={mime}, size={size} bytes, path='{path}')")
    if not lines:
        return "(no matching files in context)"
    return "\n".join(lines)
    
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


def build_memory_extras_for_result(result: Any, result_limit: int = 10, char_limit: int = 1000) -> Dict[str, Any]:
        """
        Build generic, JSON-safe memory extras summarizing `result`.

        - Works for primitives, lists, dicts, and arbitrary objects.
        - Optionally specializes for pandas DataFrame/Series if pandas is installed.
        """
        extras: Dict[str, Any] = {}
        # Try pandas specialization, but do NOT depend on it
        try:
            import pandas as _pd  # type: ignore
            if isinstance(result, _pd.DataFrame):
                extras["python_last_result_summary"] = {
                    "value": {
                        "kind": "dataframe",
                        "columns": list(result.columns),
                        "dtypes": {c: str(t) for c, t in result.dtypes.items()},
                        "shape": list(result.shape),
                        "sample_head": result.head(result_limit).to_dict(orient="list"),
                    },
                    "description": (
                        "Summary of the last DataFrame computed by python_execute "
                        "(columns, dtypes, shape, and a small head sample)."
                    ),
                    "visible_to_llm": True,
                }
                return extras

            if isinstance(result, _pd.Series):
                extras["python_last_result_summary"] = {
                    "value": {
                        "kind": "series",
                        "name": result.name,
                        "dtype": str(result.dtype),
                        "length": int(len(result)),
                        "index_sample": list(result.index[:result_limit]),
                        "values_sample": result.head(result_limit).tolist(),
                    },
                    "description": (
                        "Summary of the last Series computed by python_execute "
                        "(name, dtype, length, and a small sample)."
                    ),
                    "visible_to_llm": True,
                }
                return extras
        except Exception:
            # pandas not installed or something weird; we just fall through
            pass

        # --- Generic fallback path ---
        # Simple primitives – safe to store directly
        if isinstance(result, (str, int, float, bool, type(None))):
            extras["python_last_result_summary"] = {
                "value": {
                    "kind": type(result).__name__,
                    "value": result,
                },
                "description": "Primitive result from the last python_execute call.",
                "visible_to_llm": True,
            }
            return extras

        # Lists / tuples – store length and a short prefix
        if isinstance(result, (list, tuple)):
            preview = list(result[:result_limit])
            extras["python_last_result_summary"] = {
                "value": {
                    "kind": "sequence",
                    "type": type(result).__name__,
                    "length": len(result),
                    "sample": preview,
                },
                "description": "Sequence result from the last python_execute call.",
                "visible_to_llm": True,
            }
            return extras

        # Dict – store top-level keys and a small preview
        if isinstance(result, dict):
            keys = list(result.keys())
            preview_keys = keys[:result_limit]
            preview = {k: result[k] for k in preview_keys}
            extras["python_last_result_summary"] = {
                "value": {
                    "kind": "mapping",
                    "type": type(result).__name__,
                    "keys_sample": preview_keys,
                    "values_sample": preview,
                },
                "description": "Mapping/dict-like result from the last python_execute call.",
                "visible_to_llm": True,
            }
            return extras

        # Fallback for arbitrary objects – just type name + repr snippet
        extras["python_last_result_summary"] = {
            "value": {
                "kind": "object",
                "type": type(result).__name__,
                "repr": repr(result)[:char_limit],
            },
            "description": (
                "Opaque object result from the last python_execute call "
                "(only type name and repr snippet are exposed)."
            ),
            "visible_to_llm": True,
        }
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
