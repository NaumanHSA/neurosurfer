from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from neurosurfer.version import __version__
from neurosurfer.config import config
from .serve import ServeOptions, serve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("neurosurfer", description="Neurosurfer CLI")
    # add --version
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="Start the Neurosurfer backend")
    s.add_argument("--backend-app", type=str, default=os.environ.get("NEUROSURFER_BACKEND_APP", None), help="Backend application: file.py or module[:attr] (defaults to example)")
    s.add_argument("--backend-host", type=str, default=os.environ.get("NEUROSURFER_BACKEND_HOST", config.app.host_ip))
    s.add_argument("--backend-port", type=int, default=int(os.environ.get("NEUROSURFER_BACKEND_PORT", config.app.host_port)))
    s.add_argument("--backend-log-level", type=str, default=os.environ.get("NEUROSURFER_BACKEND_LOG", config.app.logs_level))
    s.add_argument("--backend-reload", action="store_true", help="Enable auto-reload for backend")
    s.add_argument("--backend-workers", type=int, default=int(os.environ.get("NEUROSURFER_BACKEND_WORKERS", config.app.workers)))
    s.add_argument("--backend-worker-timeout", type=int, default=int(os.environ.get("NEUROSURFER_BACKEND_WORKER_TIMEOUT", config.app.worker_timeout)))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "serve":
        parser.print_help()
        return 2

    opts = ServeOptions(
        backend_app=args.backend_app,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        backend_log_level=args.backend_log_level,
        backend_reload=bool(args.backend_reload),
        backend_workers=args.backend_workers,
        backend_worker_timeout=args.backend_worker_timeout,
    )

    return asyncio.run(serve(opts))


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
