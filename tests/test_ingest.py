"""Repository ingestion — local-path resolution and git-URL detection.

`ingest.py` backs any Task that declares a `path_or_url` input, resolving a local
path or cloning a git URL into a working directory for the run.
"""

from __future__ import annotations

import pytest

from neurosurfer.tasks.ingest import ingest_repo, looks_like_git_url


def test_git_url_detection():
    assert looks_like_git_url("https://github.com/x/y.git")
    assert looks_like_git_url("git@github.com:x/y.git")
    assert looks_like_git_url("./local/path.git")
    assert not looks_like_git_url("/home/user/project")
    assert not looks_like_git_url("relative/dir")


def test_ingest_local_path(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    got = ingest_repo(str(tmp_path))
    assert got.path == tmp_path.resolve()
    assert got.is_temp is False
    got.cleanup()  # no-op for local paths
    assert tmp_path.exists()


def test_ingest_missing_path():
    with pytest.raises(FileNotFoundError):
        ingest_repo("/no/such/repo/anywhere")
