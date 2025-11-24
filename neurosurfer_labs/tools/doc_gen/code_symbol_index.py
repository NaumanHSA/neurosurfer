from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class CodeSymbolIndexConfig:
    """
    Configuration for CodeSymbolIndexTool.

    Attributes
    ----------
    project_root:
        Optional root directory for the project. If provided, and `python_files`
        are not passed at call time, the tool will discover .py files under this root.

    max_files:
        Optional hard cap on the number of files to index (after filtering).
        Helps avoid huge scans in very large repos.

    max_file_bytes:
        Optional maximum size per file (in bytes). Files larger than this are skipped.

    include_private:
        Whether to include private names (starting with '_') in the symbol index.

    include_methods:
        Whether to include class methods in the index.

    include_function_args:
        If True, captures a simple list of argument names for functions/methods.
    """
    project_root: Optional[Union[str, Path]] = None
    max_files: Optional[int] = None
    max_file_bytes: Optional[int] = 512_000  # ~500 KB
    include_private: bool = False
    include_methods: bool = True
    include_function_args: bool = True


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

class CodeSymbolIndexTool(BaseTool):
    """
    Build a high-level symbol index (modules, classes, functions) over a set
    of Python files.

    Typical usage:
      - Use DirectoryScanTool first to get `python_files` (with module + package).
      - Feed that list into this tool to get a structured symbol index for docs.
    """

    spec = ToolSpec(
        name="code_symbol_index",
        description=(
            "Analyze Python files to build a high-level symbol index of modules, "
            "classes, methods, and functions (with docstrings). "
            "Useful for generating API documentation and navigation."
        ),
        when_to_use=(
            "Use this tool when you need a structured view of the codebase: "
            "which modules exist, which classes and functions they define, and "
            "their docstrings. Typically called after a directory scan."
        ),
        inputs=[
            ToolParam(
                name="python_files",
                type="array",
                description=(
                    "Optional list of Python file descriptors, typically the "
                    "`python_files` output from `directory_scan`. Each item should "
                    "be an object with at least `abs_path` and `module` keys. "
                    "If omitted, the tool will scan its configured `project_root` "
                    "for .py files."
                ),
                required=False,
                llm=False,  # comes from another tool / graph, not the LLM
            ),
            ToolParam(
                name="module_paths",
                type="array",
                description=(
                    "Optional list of module or path patterns to focus on. "
                    "Patterns are glob-like and applied to both module names "
                    "and relative paths, e.g. ['neurosurfer.agents.*', "
                    "'neurosurfer/rag/**']. If omitted, all discovered files "
                    "are considered."
                ),
                required=False,
                llm=True,  # LLM can decide what subset to focus on
            ),
        ],
        returns=ToolReturn(
            type="object",
            description=(
                "JSON object with keys:\n"
                "- modules: mapping module_name -> { path, package, classes, functions }\n"
                "- summary: basic counts (module_count, class_count, function_count)\n"
                "- errors: list of {path, error} for files that failed to parse"
            ),
        ),
    )

    def __init__(
        self,
        config: CodeSymbolIndexConfig = CodeSymbolIndexConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self._project_root: Optional[Path] = (
            Path(config.project_root).expanduser().resolve()
            if config.project_root is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        python_files: Optional[List[Dict[str, Any]]] = None,
        module_paths: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Build a symbol index either from an explicit list of python_files
        (recommended) or by scanning the configured project_root.
        """
        if python_files is None:
            if self._project_root is None:
                msg = (
                    "CodeSymbolIndexTool requires either `python_files` argument "
                    "or `project_root` configured in CodeSymbolIndexConfig."
                )
                logger.warning(msg)
                results = {
                    "modules": {},
                    "summary": {
                        "module_count": 0,
                        "class_count": 0,
                        "function_count": 0,
                    },
                    "errors": [{"path": None, "error": msg}],
                }
                return ToolResponse(final_answer=False, results=results, extras={})

            python_files = self._discover_python_files(self._project_root)

        # Filter python_files by module_paths (if provided)
        filtered_files = self._filter_python_files(python_files, module_paths)

        if self.config.max_files is not None and len(filtered_files) > self.config.max_files:
            logger.info(
                "Limiting symbol index to max_files=%s (down from %s)",
                self.config.max_files,
                len(filtered_files),
            )
            filtered_files = filtered_files[: self.config.max_files]

        modules_index: Dict[str, Any] = {}
        errors: List[Dict[str, str]] = []

        total_classes = 0
        total_functions = 0

        for pf in filtered_files:
            abs_path = pf.get("abs_path") or pf.get("path")
            if not abs_path:
                continue

            path = Path(abs_path)
            try:
                if self.config.max_file_bytes is not None and path.stat().st_size > self.config.max_file_bytes:
                    logger.debug("Skipping %s (size > %s bytes)", path, self.config.max_file_bytes)
                    continue
            except FileNotFoundError:
                errors.append({"path": str(path), "error": "File not found"})
                continue
            except OSError as e:
                errors.append({"path": str(path), "error": f"OS error: {e}"})
                continue

            module_name = pf.get("module") or self._guess_module_name(path)
            package = pf.get("package") or self._guess_package_name(module_name)

            try:
                with path.open("r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as e:
                logger.warning("SyntaxError while parsing %s: %s", path, e)
                errors.append({"path": str(path), "error": f"SyntaxError: {e}"})
                continue
            except UnicodeDecodeError as e:
                logger.warning("UnicodeDecodeError while reading %s: %s", path, e)
                errors.append({"path": str(path), "error": f"UnicodeDecodeError: {e}"})
                continue
            except Exception as e:
                logger.exception("Unexpected error while parsing %s: %s", path, e)
                errors.append({"path": str(path), "error": f"Exception: {e}"})
                continue

            module_info = self._extract_module_symbols(
                tree,
                module_name=module_name,
                include_private=self.config.include_private,
                include_methods=self.config.include_methods,
                include_function_args=self.config.include_function_args,
            )

            total_classes += len(module_info["classes"])
            total_functions += len(module_info["functions"])

            modules_index[module_name] = {
                "path": pf.get("path") or str(path),
                "abs_path": str(path),
                "package": package,
                "classes": module_info["classes"],
                "functions": module_info["functions"],
            }

        results: Dict[str, Any] = {
            "modules": modules_index,
            "summary": {
                "module_count": len(modules_index),
                "class_count": total_classes,
                "function_count": total_functions,
            },
            "errors": errors,
        }

        return ToolResponse(final_answer=False, results=results, extras={})

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _discover_python_files(self, root: Path) -> List[Dict[str, Any]]:
        """
        Fallback: discover .py files under the configured project_root.
        Only used when `python_files` argument is not supplied.
        """
        python_files: List[Dict[str, Any]] = []
        root_name = root.name

        for dirpath, dirnames, filenames in os.walk(root):
            dir_path = Path(dirpath)
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                file_path = dir_path / filename
                rel_path = file_path.relative_to(root)
                module_name = self._make_module_name(root_name, rel_path)
                package = self._guess_package_name(module_name)

                python_files.append(
                    {
                        "path": rel_path.as_posix(),
                        "abs_path": str(file_path),
                        "module": module_name,
                        "package": package,
                    }
                )
        return python_files

    def _filter_python_files(
        self,
        python_files: List[Dict[str, Any]],
        module_paths: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Filter python_files by module_paths patterns if provided.
        Patterns are matched against both 'module' and 'path' fields.
        """
        if not module_paths:
            return python_files

        filtered: List[Dict[str, Any]] = []
        for pf in python_files:
            module_name = pf.get("module") or ""
            rel_path = pf.get("path") or ""

            if any(self._match_module_or_path(module_name, rel_path, pat) for pat in module_paths):
                filtered.append(pf)

        return filtered

    @staticmethod
    def _match_module_or_path(module_name: str, rel_path: str, pattern: str) -> bool:
        """
        Match pattern against module name or relative path using fnmatch.
        """
        return fnmatch(module_name, pattern) or fnmatch(rel_path, pattern)

    @staticmethod
    def _guess_module_name(path: Path) -> str:
        """
        Best-effort guess of a module name from a file path, without knowing
        the project root. Falls back to stem.
        """
        return ".".join(path.with_suffix("").parts)

    @staticmethod
    def _guess_package_name(module_name: str) -> str:
        """
        Best-effort guess of a package name from a module name.
        """
        parts = module_name.split(".")
        if len(parts) > 1:
            return ".".join(parts[:-1])
        return ""

    @staticmethod
    def _make_module_name(root_name: str, rel_path: Path) -> str:
        """
        Build a dotted module name from a python file relative to project_root.

        Example:
          root_name='neurosurfer', rel_path='agents/graph/agent.py'
          -> 'neurosurfer.agents.graph.agent'
        """
        parts = list(rel_path.with_suffix("").parts)
        return ".".join([root_name, *parts])

    def _extract_module_symbols(
        self,
        tree: ast.AST,
        *,
        module_name: str,
        include_private: bool,
        include_methods: bool,
        include_function_args: bool,
    ) -> Dict[str, Any]:
        """
        Extract top-level classes and functions (and optionally methods)
        from an AST module tree.
        """
        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if not include_private and node.name.startswith("_"):
                    continue
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                }
                if include_methods:
                    class_info["methods"] = self._extract_methods(
                        node,
                        include_private=include_private,
                        include_function_args=include_function_args,
                    )
                classes.append(class_info)

            elif isinstance(node, ast.FunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                func_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                }
                if include_function_args:
                    func_info["args"] = self._extract_args(node.args)
                functions.append(func_info)

        return {"classes": classes, "functions": functions}

    def _extract_methods(
        self,
        class_node: ast.ClassDef,
        *,
        include_private: bool,
        include_function_args: bool,
    ) -> List[Dict[str, Any]]:
        """
        Extract methods from a class definition.
        """
        methods: List[Dict[str, Any]] = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                method_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                }
                if include_function_args:
                    method_info["args"] = self._extract_args(node.args)
                methods.append(method_info)
        return methods

    @staticmethod
    def _extract_args(args: ast.arguments) -> List[str]:
        """
        Extract a simple list of argument names from an ast.arguments node.
        Does not attempt full signature reconstruction (defaults, kwonly, etc.).
        """
        names: List[str] = []
        for a in list(args.posonlyargs) + list(args.args):
            names.append(a.arg)
        if args.vararg is not None:
            names.append("*" + args.vararg.arg)
        for a in args.kwonlyargs:
            names.append(a.arg)
        if args.kwarg is not None:
            names.append("**" + args.kwarg.arg)
        return names
