"""Enable ``python -m neurosurfer`` as an alias for the ``neurosurfer`` console script."""

from __future__ import annotations

from .app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
