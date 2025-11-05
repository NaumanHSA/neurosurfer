from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neurosurfer.config import config

from .processes import (
    check_and_install_serve,
    pipe_output,
    run_npm_install,
    start_backend_proc,
    start_static_serve,
)
from .utils import (
    detect_ui_root,
    effective_public_host,
    env_truthy,
    find_packaged_ui_dir,
    has_package_json,
    looks_like_build_dir,
    open_browser_safe,
    print_ready_banner,
    wait_for_http_ok,
    wait_for_port,
    which,
)

logger = logging.getLogger("neurosurfer")


@dataclass
class ServeOptions:
    backend_app: Optional[str]
    backend_host: str
    backend_port: int
    backend_log_level: str
    backend_reload: bool
    backend_workers: int
    backend_worker_timeout: int
    ui_root: Optional[Path]
    ui_host: str
    ui_port: int
    ui_strict_port: bool
    ui_open: bool
    npm_install_mode: str  # "auto" | "always" | "never"
    only_backend: bool
    only_ui: bool


def needs_npm_install(ui_root: Path, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    node_modules = ui_root / "node_modules"
    return not node_modules.exists()


async def serve(opts: ServeOptions) -> int:
    # Decide UI run mode
    ui_root: Optional[Path] = None
    run_mode: str = "none"  # "static-packaged" | "vite-dev" | "static-path" | "none"
    static_dir: Optional[Path] = None

    if not opts.only_backend and not opts.only_ui:
        if opts.ui_root is None:
            packaged = find_packaged_ui_dir()
            if packaged:
                run_mode, static_dir = "static-packaged", packaged
            else:
                logger.warning("No UI requested and no packaged UI found -> only backend")
        else:
            ui_root = detect_ui_root(opts.ui_root) or opts.ui_root
            if ui_root and has_package_json(ui_root):
                run_mode = "vite-dev"
            elif ui_root and looks_like_build_dir(ui_root):
                run_mode, static_dir = "static-path", ui_root
            else:
                raise SystemExit(f"--ui-root is not a Vite project nor a build folder: {ui_root}")

    # Start backend (unless only_ui)
    backend_proc = None
    backend_pipe_task = None
    if not opts.only_ui:
        logger.info(f"Starting Neurosurfer backend at http://{opts.backend_host}:{opts.backend_port}")
        os.environ["NEUROSURF_SILENCE"] = "1"
        backend_proc = await start_backend_proc(
            backend_app=opts.backend_app,
            backend_host=opts.backend_host,
            backend_port=opts.backend_port,
            backend_log_level=opts.backend_log_level,
            backend_reload=opts.backend_reload,
            backend_workers=opts.backend_workers,
            backend_worker_timeout=opts.backend_worker_timeout,
        )
        backend_pipe_task = asyncio.create_task(pipe_output("api", backend_proc))

    # Start UI according to run mode (unless only_backend)
    ui_proc = None
    ui_pipe_task = None
    ui_url_for_banner: Optional[str] = None

    if not opts.only_backend:
        if run_mode == "vite-dev":
            if not which("npm"):
                raise SystemExit("npm not found in PATH; required for Vite dev mode (--ui-root with package.json).")
            if ui_root is None:
                raise SystemExit("Internal error: ui_root not resolved.")
            if needs_npm_install(ui_root, opts.npm_install_mode):
                await run_npm_install(ui_root)

            logger.info(f"Starting NeurowebUI dev server at http://{opts.ui_host}:{opts.ui_port} (root={ui_root})")

            env = os.environ.copy()
            if "VITE_BACKEND_URL" not in env:
                backend_host_for_url = opts.backend_host
                if backend_host_for_url in ("0.0.0.0", "::"):
                    backend_host_for_url = os.environ.get("NEUROSURF_PUBLIC_HOST", "127.0.0.1")
                env["VITE_BACKEND_URL"] = f"http://{backend_host_for_url}:{opts.backend_port}"

            ui_proc = await asyncio.create_subprocess_exec(
                "npm", "run", "dev", "--", "--host", opts.ui_host, "--port", str(opts.ui_port),
                cwd=str(ui_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            ui_pipe_task = asyncio.create_task(pipe_output("ui", ui_proc))
            ui_url_for_banner = f"http://{effective_public_host(opts.ui_host)}:{opts.ui_port}"

        elif run_mode in ("static-packaged", "static-path"):
            logger.info(f"Serving static UI from {static_dir} at http://{opts.ui_host}:{opts.ui_port}")
            if static_dir is None:
                raise SystemExit("Internal error: static_dir is None.")
            ui_proc = await start_static_serve(static_dir, opts.ui_port)
            ui_pipe_task = asyncio.create_task(pipe_output("ui", ui_proc))
            ui_url_for_banner = f"http://{effective_public_host(opts.ui_host)}:{opts.ui_port}"

        else:
            logger.warning("No UI in this run. Only backend is running.")

    # Signals
    stop_event = asyncio.Event()

    def _on_signal(signame: str):
        eprint(f"Received {signame}, stopping...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _on_signal, s.name)

    # Readiness + banner + optional auto-open
    bh = effective_public_host(opts.backend_host)
    backend_url_for_banner = f"http://{bh}:{opts.backend_port}"

    # backend ready (unless only_ui)
    if not opts.only_ui:
        backend_ready = await wait_for_http_ok(bh, opts.backend_port, "/health", timeout=45.0)
        if not backend_ready:
            backend_ready = await wait_for_port(bh, opts.backend_port, timeout=10.0)

    # ui ready
    ui_ready = True
    if ui_url_for_banner:
        uh = effective_public_host(opts.ui_host)
        ui_ready = await wait_for_port(uh, opts.ui_port, timeout=45.0)

    print_ready_banner(backend_url_for_banner, ui_url_for_banner if ui_ready else None)
    if opts.ui_open and ui_ready and ui_url_for_banner:
        open_browser_safe(ui_url_for_banner)

    # Wait for one child to exit or Ctrl+C
    async def _wait_children_once() -> int:
        tasks = []
        if backend_pipe_task:
            tasks.append(backend_pipe_task)
        if ui_pipe_task:
            tasks.append(ui_pipe_task)
        if not tasks:
            return 0
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            with contextlib.suppress(BaseException):
                return d.result()
        return 0

    stop_task = asyncio.create_task(stop_event.wait(), name="stop_event.wait")
    children_task = asyncio.create_task(_wait_children_once(), name="wait_children_once")
    done, pending = await asyncio.wait({stop_task, children_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*pending)

    # Terminate children
    async def _term_kill(proc: Optional[asyncio.subprocess.Process], name: str):
        if not proc:
            return
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1.0)

    await asyncio.gather(
        _term_kill(backend_proc, "api"),
        _term_kill(ui_proc, "ui"),
    )
    return 0
