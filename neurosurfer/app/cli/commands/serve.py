"""``neurosurfer serve`` — start the OpenAI-compatible gateway."""

from __future__ import annotations

import argparse
from typing import Any


def handle_serve(args: argparse.Namespace) -> int:
    try:
        from neurosurfer.server import NeurosurferServer, UpstreamBackend
    except ImportError as exc:
        print(
            f"[error] {exc}\n"
            "Install the serve extra: pip install 'neurosurfer[serve]'"
        )
        return 1

    kwargs: dict[str, Any] = {}
    if args.host:
        kwargs["host"] = args.host
    if args.port:
        kwargs["port"] = args.port
    if args.log_level:
        kwargs["log_level"] = args.log_level
    if args.workers:
        kwargs["workers"] = args.workers
    if args.reload:
        kwargs["reload"] = True
    if args.no_docs:
        kwargs["enable_docs"] = False

    server = NeurosurferServer(**kwargs)

    if args.upstream_url:
        backend = UpstreamBackend(
            name="upstream",
            base_url=args.upstream_url,
            api_key=args.upstream_api_key or "",
        )
        server.register_backend(backend)

    host = args.host or server.settings.host
    port = args.port or server.settings.port
    print(f"Starting Neurosurfer Gateway on http://{host}:{port}")
    if args.upstream_url:
        print(f"  Proxying to: {args.upstream_url}")
    if server.settings.enable_docs and not args.no_docs:
        print(f"  Docs: http://{host}:{port}/docs")

    server.run()
    return 0


def add_serve_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "serve",
        help="Start the OpenAI-compatible gateway (requires neurosurfer[serve])",
    )
    p.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    p.add_argument(
        "--upstream-url",
        default=None,
        metavar="URL",
        help="Upstream OpenAI-compatible base URL (e.g. http://localhost:8001/v1)",
    )
    p.add_argument("--upstream-api-key", default=None, metavar="KEY", help="Upstream API key")
    p.add_argument("--log-level", default=None, help="Uvicorn log level")
    p.add_argument("--workers", type=int, default=None, help="Number of uvicorn workers")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    p.add_argument("--no-docs", action="store_true", help="Disable /docs UI")
