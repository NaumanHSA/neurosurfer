import os
from pathlib import Path
import re
import logging
import tempfile
import zipfile
import shutil
from typing import Callable, Dict, Optional, Generator, Union, Literal, List, Any, Tuple, Set
from datetime import datetime


def generate_folder_structure(
    root_path: Union[str, os.PathLike],
    max_depth: int = 5,
    exclude_dirs: Optional[List[str]] = None,
    supported_files: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Generate a tree-like folder structure.
    Supports both directory paths and .zip files.
    If a zip is given, it is extracted to a temporary directory,
    processed, and then deleted.
    """
    exclude_dirs = set(exclude_dirs or [])
    supported_files = set(supported_files or [])
    tmpdir = None
    is_zip = str(root_path).lower().endswith(".zip")
    # --- Handle zip extraction ---
    if is_zip:
        if not os.path.isfile(root_path):
            raise ValueError(f"Invalid zip file: {root_path}")
        tmpdir = tempfile.mkdtemp(prefix="tree_zip_")
        with zipfile.ZipFile(root_path, "r") as zf:
            zf.extractall(tmpdir)
        root_path = tmpdir  # work on extracted dir

    # --- Recursive tree builder ---
    def _tree(dir_path: str, prefix: str = "", depth: int = 0) -> Optional[str]:
        if depth > max_depth:
            return None
        try:
            entries = sorted(os.listdir(dir_path))
        except Exception:
            return None

        output = ""
        for i, entry in enumerate(entries):
            path = os.path.join(dir_path, entry)
            is_dir = os.path.isdir(path)

            # Exclude hidden/system entries and user-specified dirs
            if entry.startswith((".", "%", "_")) or (is_dir and entry in exclude_dirs):
                continue
            
            # # Exclude empty files
            # if not is_dir and os.path.splitext(entry)[1] == "":
            #     continue

            # Only show dirs and supported files
            if is_dir or (not supported_files or os.path.splitext(entry)[1] in supported_files) or os.path.splitext(entry)[1] == "":
                connector = "└── " if i == len(entries) - 1 else "├── "
                output += f"{prefix}{connector}{entry}\n"
                if is_dir:
                    extension = "    " if i == len(entries) - 1 else "│   "
                    sub_tree = _tree(path, prefix + extension, depth + 1) or ""
                    output += sub_tree
        return output

    # --- Build the tree ---
    tree__ = _tree(root_path)
    result = f"/\n{tree__}" if tree__ else None

    # --- Cleanup zip extraction ---
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return result
