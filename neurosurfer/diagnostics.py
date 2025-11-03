# src/neurosurfer/diagnostics.py
from __future__ import annotations
import sys
from .runtime.checks import banner, diagnostic_report

def main(argv: list[str] | None = None) -> int:
    print(banner())
    print()
    print(diagnostic_report())
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
