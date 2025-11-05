from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional


# ----------------------- Printing -----------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


# ----------------------- PATH / WHICH -----------------------

def which(cmd: str, verbose: bool = False) -> Optional[str]:
    """
    Robust 'which': return resolved path if the command is in PATH.
    """
    path = shutil.which(cmd)
    if path and verbose:
        real = os.path.realpath(path)
        print(f"[which] {cmd} -> {path} (real: {real})")
    return path


# ----------------------- ENV -----------------------

def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ----------------------- UI discovery -----------------------

def find_packaged_ui_dir() -> Optional[Path]:
    """
    Returns neurosurfer/ui_build if bundled and contains index.html.
    """
    try:
        from importlib.resources import files
        p = files("neurosurfer") / "ui_build"
        pp = Path(str(p))
        return pp if pp.exists() and (pp / "index.html").exists() else None
    except Exception:
        return None


def has_package_json(path: Optional[Path]) -> bool:
    return bool(path and path.is_dir() and (path / "package.json").exists())


def looks_like_build_dir(path: Optional[Path]) -> bool:
    return bool(path and path.exists() and (path / "index.html").exists())


def detect_ui_root(arg: Optional[Path]) -> Optional[Path]:
    if arg:
        return arg
    env = os.environ.get("NEUROSURF_UI_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent / "neurosurferui",  # repo layout
        here.parent / "neurosurferui",         # package-local
    ]:
        if candidate.exists():
            return candidate
    return None


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
        return os.environ.get("NEUROSURF_PUBLIC_HOST", "127.0.0.1")
    return host


# ----------------------- Browser -----------------------

def open_browser_safe(url: str) -> None:
    try:
        webbrowser.open_new_tab(url)
        return
    except Exception:
        pass
    # fallbacks
    with contextlib.suppress(Exception):
        if sys.platform.startswith("linux"):
            os.spawnlp(os.P_NOWAIT, "xdg-open", "xdg-open", url)
        elif sys.platform == "darwin":
            os.spawnlp(os.P_NOWAIT, "open", "open", url)
        elif os.name == "nt":
            os.startfile(url)  # type: ignore[attr-defined]


# ----------------------- Banner -----------------------

def print_ready_banner(backend_url: str, ui_url: Optional[str]) -> None:
    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════════╗")
    lines.append("║                      🚀 Neurosurfer Server is running!               ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  API     : {backend_url:<58}║")
    if ui_url:
        lines.append(f"║  UI      : {ui_url:<58}║")
    else:
        lines.append("║  UI      : (not enabled)                                             ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append("║  Press Ctrl+C to stop.                                               ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()
