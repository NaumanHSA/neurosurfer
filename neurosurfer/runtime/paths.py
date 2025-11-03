# src/neurosurfer/runtime/paths.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys
import os

try:
    from platformdirs import user_cache_path  # returns a pathlib.Path
except Exception:  # optional fallback if platformdirs not installed
    user_cache_path = None  # type: ignore

APP_NAME = "Neurosurfer"
APP_SLUG = "neurosurfer"  # used for env var overrides

def get_cache_dir(create: bool = True, suffix: Optional[str] = None) -> Path:
    """
    Return the per-user cache directory for Neurosurfer, creating it if needed.
    Respects NEUROSURF_CACHE_DIR if set; otherwise uses platform-specific defaults.
    Linux:  ~/.cache/Neurosurfer
    macOS:  ~/Library/Caches/Neurosurfer
    Windows: %LOCALAPPDATA%\\Neurosurfer\\Cache (per platformdirs)
    """
    # Highest priority: explicit override via env
    override = os.environ.get("NEUROSURF_CACHE_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        if user_cache_path is None:
            # ultra-minimal fallback to XDG pattern if platformdirs missing
            home = Path.home()
            if os.name == "nt":
                # Use LOCALAPPDATA if available, otherwise fallback to home
                base = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))) / "Neurosurfer" / "Cache"
            elif sys.platform == "darwin":
                base = home / "Library" / "Caches" / "Neurosurfer"
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", str(home / ".cache"))) / "Neurosurfer"
        else:
            base = user_cache_path(appname=APP_NAME, ensure_exists=False)

    path = base if suffix is None else (Path(base) / suffix)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return Path(path)
