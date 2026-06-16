from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from typing import Any


# ----------------------- Printing -----------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


# ----------------------- Readiness probes -----------------------

async def wait_for_port(host: str, port: int, timeout: float = 30.0, interval: float = 0.25) -> bool:
    start = time.monotonic()
    while time.monotonic() - start <= timeout:
        try:
            fut = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(fut, timeout=interval)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return True
        except Exception:
            await asyncio.sleep(interval)
    return False


async def wait_for_http_ok(host: str, port: int, path: str = "/health",
                           timeout: float = 30.0, interval: float = 0.4) -> bool:
    req = f"GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: neurosurfer/cli\r\n\r\n".encode("ascii", "ignore")
    start = time.monotonic()
    while time.monotonic() - start <= timeout:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=interval)
            writer.write(req)
            await writer.drain()
            head = await asyncio.wait_for(reader.read(256), timeout=interval)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if head.startswith(b"HTTP/1.") and any(code in head for code in
                                                   (b" 200 ", b" 201 ", b" 204 ", b" 301 ", b" 302 ", b" 303 ", b" 307 ", b" 308 ")):
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


def effective_public_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return os.environ.get("NEUROSURFER_PUBLIC_HOST", "127.0.0.1")
    return host


# ----------------------- Banner -----------------------

def print_ready_banner(backend_url: str) -> None:
    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════════╗")
    lines.append("║                      🚀 Neurosurfer Gateway is running!              ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  API     : {backend_url:<58}║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append("║  Press Ctrl+C to stop.                                               ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()
