from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from .utils import which, eprint


async def pipe_output(prefix: str, proc: asyncio.subprocess.Process) -> int:
    if proc.stdout is None:
        return await proc.wait()
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.write(f"[{prefix}] {line.decode(errors='ignore')}")
    return await proc.wait()


async def run_npm_install(ui_root: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        "npm", "install", "--force",
        cwd=str(ui_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.write("[ui/install] " + line.decode(errors="ignore"))
    return await proc.wait()


async def check_and_install_serve() -> int:
    if which("serve"):
        return 0
    # try global install
    proc = await asyncio.create_subprocess_exec(
        "npm", "install", "serve@14.2.5", "-g",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.write("[ui/install] " + line.decode(errors="ignore"))
    return await proc.wait()


async def start_static_serve(src_dir: Path, port: int) -> asyncio.subprocess.Process:
    """
    Start a static server for a build folder using 'npx serve -s'.
    Falls back to global 'serve' if npx not available.
    """
    await check_and_install_serve()

    def mk_args(bin_name: str) -> list[str]:
        return [bin_name, "serve", "-s", "-p", str(port), str(src_dir)]

    npx = which("npx")
    if npx:
        args = mk_args(npx)
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    serve_bin = which("serve")
    if not serve_bin:
        raise SystemExit(
            "Static UI requested but neither 'npx' nor global 'serve' found.\n"
            "Install one of:\n"
            "  - npm i -g serve\n"
            "  - or ensure npx is available"
        )

    return await asyncio.create_subprocess_exec(
        "serve", "-s", "-p", str(port), str(src_dir),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )


async def start_backend_proc(
    backend_app: Optional[str],
    backend_host: str,
    backend_port: int,
    backend_log_level: str,
    backend_reload: bool,
    backend_workers: int,
    backend_worker_timeout: int,
) -> asyncio.subprocess.Process:
    """
    Launch a NeurosurferApp via the built-in example or a user module/file.
    """
    if backend_app is None:
        module, app = "neurosurfer.examples.quickstart_app", "ns"
        args = [
            sys.executable, "-c",
            (
                f"from {module} import {app}; "
                f"{app}.run(host='{backend_host}', port={backend_port}, "
                f"reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
            ),
        ]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    if ":" in backend_app and not backend_app.endswith(".py"):
        module, app = backend_app.split(":")
        args = [
            sys.executable, "-c",
            (
                f"from {module} import {app}; "
                f"{app}.run(host='{backend_host}', port={backend_port}, "
                f"reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
            ),
        ]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    # Path case
    p = Path(backend_app)
    if p.suffix == ".py" and p.exists():
        from neurosurfer.server.app import NeurosurferApp  # type: ignore
        import runpy
        ns = runpy.run_path(str(p), run_name="__neurosurfer_user_app__")
        app_attr = None
        for k, v in ns.items():
            try:
                if isinstance(v, NeurosurferApp):
                    app_attr = k
                    break
            except Exception:
                continue
        if not app_attr:
            raise SystemExit(
                f"Could not locate instance of NeurosurferApp in {p}.\n"
                f"Define: app = NeurosurferApp(...)\n"
                f"Do not call app.run() directly; CLI will run it."
            )
        module = p.parent / p.stem  # importable path if package layout
        args = [
            sys.executable, "-c",
            (
                f"import importlib.util,sys; "
                f"spec=importlib.util.spec_from_file_location('{p.stem}','{p.as_posix()}'); "
                f"m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                f"getattr(m,'{app_attr}').run(host='{backend_host}', port={backend_port}, "
                f"reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
            ),
        ]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    raise SystemExit(
        f"Invalid backend app reference: {backend_app}\n"
        f"Use module:attr, or /path/to/app.py containing a NeurosurferApp instance."
    )
