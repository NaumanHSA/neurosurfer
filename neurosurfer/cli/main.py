from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional
import sys
from neurosurfer.version import __version__
from neurosurfer.config import config
from .serve import ServeOptions, serve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("neurosurfer", description="Neurosurfer CLI")
    # add --version
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="Start Neurosurfer backend and NeurowebUI")
    s.add_argument("--backend-app", type=str, default=None, help="Backend application: file.py or module[:attr] (defaults to example)")
    s.add_argument("--backend-host", type=str, default=os.environ.get("NEUROSURF_BACKEND_HOST", config.app.host_ip))
    s.add_argument("--backend-port", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_PORT", config.app.host_port)))
    s.add_argument("--backend-log-level", type=str, default=os.environ.get("NEUROSURF_BACKEND_LOG", config.app.logs_level))
    s.add_argument("--backend-reload", action="store_true", help="Enable auto-reload for backend")
    s.add_argument("--backend-workers", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_WORKERS", config.app.workers)))
    s.add_argument("--backend-worker-timeout", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_WORKER_TIMEOUT", config.app.worker_timeout)))

    s.add_argument("--ui-root", type=Path, default=None, help="Path to NeurowebUI root (package.json) or build dir")
    s.add_argument("--ui-host", type=str, default=os.environ.get("NEUROSURF_UI_HOST", config.app.ui_host))
    s.add_argument("--ui-port", type=int, default=int(os.environ.get("NEUROSURF_UI_PORT", config.app.ui_port)))
    s.add_argument("--ui-open", default=os.environ.get("NEUROSURF_UI_OPEN", True), help="Open browser for UI on start")
    s.add_argument("--ui-strict-port", action="store_true", help="Fail if UI port is already in use")

    s.add_argument("--npm-install", choices=["auto", "always", "never"], default=os.environ.get("NEUROSURF_NPM_INSTALL", "auto"), help="First-run npm install behavior")
    s.add_argument("--only-backend", action="store_true", help="Start only backend")
    s.add_argument("--only-ui", action="store_true", help="Start only UI")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "serve":
        parser.print_help()
        return 2

    if args.only_backend and args.only_ui:
        print("Cannot use --only-backend and --only-ui together.", file=sys.stderr)
        return 2

    if isinstance(args.ui_open, str) and args.ui_open.lower() in ("false", "0"):
        args.ui_open = False
    
    opts = ServeOptions(
        backend_app=args.backend_app,
        backend_host=args.backend_host, 
        backend_port=args.backend_port,
        backend_log_level=args.backend_log_level,
        backend_reload=bool(args.backend_reload),
        backend_workers=args.backend_workers,
        backend_worker_timeout=args.backend_worker_timeout,
        ui_root=args.ui_root,
        ui_host=args.ui_host,
        ui_port=args.ui_port,
        ui_strict_port=bool(args.ui_strict_port),
        ui_open=bool(args.ui_open),
        npm_install_mode=args.npm_install,
        only_backend=bool(args.only_backend),
        only_ui=bool(args.only_ui),
    )

    return asyncio.run(serve(opts))


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
