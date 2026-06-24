from __future__ import annotations

import logging
import os
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class DirectoryScanConfig:
    """
    Configuration for DirectoryScanTool.

    default_include_patterns:
        Glob patterns (relative to project_root) that define which files
        are considered "interesting". If None, all files are considered,
        then we classify by extension.

    default_exclude_patterns:
        Glob patterns (relative to project_root) that will always be excluded
        (e.g. "*/.git/*", "*/__pycache__/*", "*/.venv/*").

    max_depth:
        Optional maximum depth (relative to project_root). If None, no limit.
        Depth is measured in number of path components (0 = project_root).

    doc_extensions:
        File extensions considered as documentation files (e.g. ".md", ".rst").

    python_extension:
        File extension considered as Python source files (typically ".py").
    """
    default_include_patterns: Optional[Sequence[str]] = None
    default_exclude_patterns: Sequence[str] = field(default_factory=lambda: (
        "*/.git/*",
        "*/.hg/*",
        "*/.svn/*",
        "*/__pycache__/*",
        "*/.venv/*",
        "*/env/*",
        "*/venv/*",
        "*/.mypy_cache/*",
        "*/.pytest_cache/*",
    ))
    max_depth: Optional[int] = None
    doc_extensions: Sequence[str] = (".md", ".rst")
    python_extension: str = ".py"


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

class DirectoryScanTool(BaseTool):
    """
    Scan a project directory (and optional docs directory) and return a
    structured index of packages, modules, python files, and documentation
    files.

    Typical usage in docs workflows:
      - project_root: path to the neurosurfer package or repo root
      - docs_root: path to the MkDocs docs directory
      - include_patterns / exclude_patterns: optional glob filters
      - max_depth: limit recursion depth if desired
    """

    spec = ToolSpec(
        name="directory_scan",
        description=(
            "Scan a project directory (and optional docs directory) to build a "
            "structured index of Python modules, packages, and documentation files. "
            "Useful as a first step for automated documentation generation."
        ),
        when_to_use=(
            "Use this tool when you need an overview of the codebase and docs "
            "structure: which packages, modules, Python files, and markdown files "
            "are present, along with basic counts."
        ),
        inputs=[
            ToolParam(name="project_root", type="string", description="Absolute or relative path to the project root directory.", required=True, llm=False),
            ToolParam(name="docs_root", type="string", description="Optional path to the docs root directory (e.g. 'docs').", required=False, llm=False),
            ToolParam(name="include_patterns", type="array", description="Optional list of glob patterns to include, e.g. ['neurosurfer/models/**'].", required=False),
            ToolParam(name="exclude_patterns", type="array", description="Optional list of glob patterns to exclude, e.g. ['neurosurfer/tests/**'].", required=False),
            ToolParam(name="max_depth", type="integer", description="Optional maximum depth to scan. ", required=False),
        ],
        returns=ToolReturn(
            type="object", 
            description=(
                "JSON object describing the project structure, with keys like:\n"
                "- project_root: normalized absolute path\n"
                "- docs_root: normalized absolute path (if provided & exists)\n"
                "- python_files: list of objects {path, abs_path, module, package}\n"
                "- doc_files: list of objects {path, abs_path}\n"
                "- packages: list of dotted package names\n"
                "- modules: list of dotted module names\n"
                "- summary: basic counts\n"
            ),
        ),
    )

    def __init__(
        self,
        config: DirectoryScanConfig = DirectoryScanConfig(),
    ) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        project_root: str,
        docs_root: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        **kwargs: Any,  
    ) -> ToolResponse:
        """
        Scan the given project_root (and optional docs_root) and build a
        structured index of Python and docs files.
        """
        # Resolve roots
        project_root_path = Path(project_root).expanduser().resolve()
        docs_root_path: Optional[Path] = (
            Path(docs_root).expanduser().resolve() if docs_root else None
        )

        if not project_root_path.exists() or not project_root_path.is_dir():
            msg = f"Project root does not exist or is not a directory: {project_root_path}"
            logger.warning(msg)
            results: Dict[str, Any] = {
                "project_root": str(project_root_path),
                "docs_root": str(docs_root_path) if docs_root_path else None,
                "python_files": [],
                "doc_files": [],
                "packages": [],
                "modules": [],
                "summary": {
                    "python_file_count": 0,
                    "doc_file_count": 0,
                    "package_count": 0,
                    "module_count": 0,
                },
                "error": msg,
            }
            return ToolResponse(final_answer=False, results=results, extras={})

        # Determine effective config
        inc_patterns = include_patterns or self.config.default_include_patterns
        exc_patterns = list(self.config.default_exclude_patterns)
        if exclude_patterns:
            exc_patterns.extend(exclude_patterns)
        depth_limit = max_depth if max_depth is not None else self.config.max_depth

        logger.info(
            "DirectoryScanTool scanning project_root=%s docs_root=%s max_depth=%s",
            project_root_path,
            docs_root_path,
            depth_limit,
        )

        # Scan project_root
        python_files, packages, modules = self._scan_project_root(
            project_root_path,
            include_patterns=inc_patterns,
            exclude_patterns=exc_patterns,
            max_depth=depth_limit,
        )

        # Scan docs_root (if provided)
        doc_files: List[Dict[str, Any]] = []
        if docs_root_path and docs_root_path.exists() and docs_root_path.is_dir():
            doc_files = self._scan_docs_root(docs_root_path)

        results: Dict[str, Any] = {
            "project_root": str(project_root_path),
            "docs_root": str(docs_root_path) if docs_root_path else None,
            "python_files": python_files,
            "doc_files": doc_files,
            "packages": sorted(list(packages)),
            "modules": sorted(list(modules)),
            "summary": {
                "python_file_count": len(python_files),
                "doc_file_count": len(doc_files),
                "package_count": len(packages),
                "module_count": len(modules),
            },
        }
        return ToolResponse(final_answer=False, results=results, extras={})

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _scan_project_root(
        self,
        project_root: Path,
        *,
        include_patterns: Optional[Sequence[str]],
        exclude_patterns: Sequence[str],
        max_depth: Optional[int],
    ) -> tuple[List[Dict[str, Any]], set[str], set[str]]:
        """
        Walk project_root and collect:
          - Python files (with module & package info)
          - Package names (directories with __init__.py)
          - Module names (one per Python file)
        """
        python_files: List[Dict[str, Any]] = []
        packages: set[str] = set()
        modules: set[str] = set()

        root_name = project_root.name

        for dirpath, dirnames, filenames in os.walk(project_root):
            dir_path = Path(dirpath)

            # Enforce depth limit if provided
            rel_dir = dir_path.relative_to(project_root)
            depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)
            if max_depth is not None and depth > max_depth:
                # Prune deeper traversal
                dirnames[:] = []
                continue

            rel_dir_posix = "." if str(rel_dir) == "." else rel_dir.as_posix()

            # Skip excluded directories quickly
            if self._is_excluded(rel_dir_posix + "/", exclude_patterns):
                dirnames[:] = []
                continue

            # Detect packages (directory with __init__.py)
            if "__init__.py" in filenames:
                pkg_name = self._make_package_name(root_name, rel_dir)
                packages.add(pkg_name)

            for filename in filenames:
                file_path = dir_path / filename
                rel_path = file_path.relative_to(project_root)
                rel_posix = rel_path.as_posix()

                # Skip excluded files
                if self._is_excluded(rel_posix, exclude_patterns):
                    continue

                # If include_patterns are set, enforce them
                if include_patterns and not self._matches_any(rel_posix, include_patterns):
                    continue

                # Only classify python files here; docs handled separately
                if file_path.suffix == self.config.python_extension:
                    module_name = self._make_module_name(root_name, rel_path)
                    pkg_name = self._make_package_name(root_name, rel_path.parent)
                    modules.add(module_name)
                    if pkg_name:
                        packages.add(pkg_name)

                    python_files.append(
                        {
                            "path": rel_posix,               # relative to project_root
                            "abs_path": str(file_path),      # absolute path
                            "module": module_name,
                            "package": pkg_name,
                        }
                    )

        return python_files, packages, modules

    def _scan_docs_root(self, docs_root: Path) -> List[Dict[str, Any]]:
        """
        Walk docs_root and collect files that look like docs (e.g. .md, .rst).
        """
        doc_files: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(docs_root):
            dir_path = Path(dirpath)
            for filename in filenames:
                file_path = dir_path / filename
                if file_path.suffix.lower() in self.config.doc_extensions:
                    rel_path = file_path.relative_to(docs_root)
                    doc_files.append(
                        {
                            "path": rel_path.as_posix(),   # relative to docs_root
                            "abs_path": str(file_path),
                        }
                    )
        return doc_files

    @staticmethod
    def _matches_any(path: str, patterns: Sequence[str]) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in patterns)

    @staticmethod
    def _is_excluded(path: str, exclude_patterns: Sequence[str]) -> bool:
        """
        Return True if the given path (relative posix) matches any exclude pattern.
        """
        return any(fnmatch.fnmatch(path, pat) for pat in exclude_patterns)

    @staticmethod
    def _make_package_name(root_name: str, rel_dir: Path) -> str:
        """
        Build a dotted package name from a directory relative to project_root.

        Examples:
          root_name='neurosurfer', rel_dir='.'          -> 'neurosurfer'
          root_name='neurosurfer', rel_dir='agents'     -> 'neurosurfer.agents'
          root_name='neurosurfer', rel_dir='agents/graph' -> 'neurosurfer.agents.graph'
        """
        if str(rel_dir) == ".":
            return root_name
        return ".".join([root_name, *rel_dir.parts])

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
