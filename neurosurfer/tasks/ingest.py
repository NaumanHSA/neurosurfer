"""Repository ingestion: resolve a local path or git URL to a working directory.

Used by the runner when a Task declares a ``path_or_url`` input (e.g. a ``repo``
input). A local path is used in place (no cleanup). A git URL is shallow-cloned
into a temp directory that is removed when the run finishes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..observability.logging import get_logger

log = get_logger("tasks.ingest")

_GIT_URL_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")


def looks_like_git_url(value: str) -> bool:
    v = value.strip()
    return v.startswith(_GIT_URL_PREFIXES) or v.endswith(".git")


@dataclass
class IngestedRepo:
    path: Path
    is_temp: bool

    def cleanup(self) -> None:
        if self.is_temp and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            log.debug("cleaned up temp clone at %s", self.path)


def ingest_repo(value: str) -> IngestedRepo:
    """Resolve ``value`` (local path or git URL) to a local working directory."""
    value = value.strip()
    if looks_like_git_url(value):
        return _clone(value)

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {value}")
    if not path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {value}")
    return IngestedRepo(path=path, is_temp=False)


def _clone(url: str) -> IngestedRepo:
    tmp = Path(tempfile.mkdtemp(prefix="neurosurfer-repo-"))
    log.info("cloning %s → %s", url, tmp)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(tmp)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as e:  # git not installed
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("git is not available to clone the repository.") from e
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {e.stderr.strip() or e}") from e
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("git clone timed out after 300s.") from e
    return IngestedRepo(path=tmp, is_temp=True)
