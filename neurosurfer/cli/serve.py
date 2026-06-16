from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from dataclasses import dataclass
from typing import Optional

from .processes import pipe_output, start_backend_proc
from .utils import (
    eprint,
    effective_public_host,
    print_ready_banner,
    wait_for_http_ok,
    wait_for_port,
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


async def serve(opts: ServeOptions) -> int:
    logger.info(f"Starting Neurosurfer backend at http://{opts.backend_host}:{opts.backend_port}")
    os.environ["NEUROSURFER_SILENCE"] = "1"
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

    # Signals
    stop_event = asyncio.Event()

    def _on_signal(signame: str):
        eprint(f"Received {signame}, stopping...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _on_signal, s.name)

    # Readiness + banner
    bh = effective_public_host(opts.backend_host)
    backend_url_for_banner = f"http://{bh}:{opts.backend_port}"

    backend_ready = await wait_for_http_ok(bh, opts.backend_port, "/health", timeout=45.0)
    if not backend_ready:
        await wait_for_port(bh, opts.backend_port, timeout=10.0)

    print_ready_banner(backend_url_for_banner)

    # Wait for the backend to exit or Ctrl+C
    stop_task = asyncio.create_task(stop_event.wait(), name="stop_event.wait")
    done, pending = await asyncio.wait({stop_task, backend_pipe_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*pending)

    # Terminate backend
    if backend_proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            backend_proc.terminate()
        try:
            await asyncio.wait_for(backend_proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                backend_proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(backend_proc.wait(), timeout=1.0)

    return 0
