from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import runpy
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from importlib.resources import files
from neurosurfer.config import config
import logging

logger = logging.getLogger("neurosurfer")

# ---------- Utilities ----------
def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)

def _which(cmd: str) -> Optional[str]:
    # Use shutil.which to detect executables in PATH (cross-platform)
    return shutil.which(cmd)

def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _find_packaged_ui_dir() -> Optional[Path]:
    """Return neurosurfer/ui_build if packaged (index.html present)."""
    try:
        p = files("neurosurfer") / "ui_build"
        pp = Path(str(p))
        return pp if pp.exists() and (pp / "index.html").exists() else None
    except Exception:
        return None

def _has_package_json(path: Path) -> bool:
    return path is not None and path.is_dir() and (path / "package.json").exists()

def _looks_like_build_dir(path: Path) -> bool:
    """Heuristic: a build/dist dir with index.html at root."""
    if path is None or not path.exists():
        return False
    return (path / "index.html").exists()

async def _start_static_serve(src_dir: Path, host: str, port: int, open_browser: bool) -> asyncio.subprocess.Process:
    """
    Start a static server for a build folder using 'npx serve -s'.
    Falls back to global 'serve' if npx not available.
    We pass the folder LAST and use --listen HOST:PORT (no tcp://).
    """
    # bind = f"{host}:{port}"

    def _mk_args(bin_name: str) -> list[str]:
        # Folder LAST. Some serve versions mis-parse when folder isn't last or when using tcp://
        # return [bin_name, "serve", "-s", "--listen", bind, str(src_dir), "--open" if open_browser else ""]
        args = [bin_name, "serve", "-s", "-p", str(port), str(src_dir)]
        if open_browser:
            args.append("--open")
        return args

    npx = _which("npx")
    if npx:
        args = _mk_args(npx)
        logger.info(f"running {' '.join(args)}")
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    serve_bin = _which("serve")
    if not serve_bin:
        raise SystemExit(
            "Static UI requested but neither 'npx' nor global 'serve' found.\n"
            "Install one of:\n"
            "  - npm i -g serve\n"
            "  - or ensure npx is available"
        )

    # Global 'serve' binary doesn’t need the extra 'serve' subcommand
    args = ["serve", "-s", "-p", str(port), str(src_dir)]
    if open_browser:
        args.append("--open")
    return await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

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

# ---------- Frontend (Vite) helpers ----------

def _detect_ui_root(arg: Optional[Path]) -> Optional[Path]:
    # Priority: explicit --ui-root, env var, typical in-repo path, sibling install paths
    if arg: return arg
    env = os.environ.get("NEUROSURF_UI_ROOT")
    if env: return Path(env)
    # Try repo-style path: neurowebui at project root
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent / "neurosurferui",              # src/neurosurfer/cli.py -> repo/neurowebui
        here.parent / "neurosurferui",                            # package-local
    ]:
        if candidate.exists():
            return candidate
    return None

def _needs_npm_install(ui_root: Path, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    # auto: install if node_modules missing or package-lock changed scenario
    node_modules = ui_root / "node_modules"
    return not node_modules.exists()

async def _run_npm_install(ui_root: Path) -> int:
    # Use `npm install --force` to be resilient to peer-dep mismatches in template setups
    # This is occasionally used to bypass peer dep conflicts; use judiciously
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

async def _pipe_output(prefix: str, proc: asyncio.subprocess.Process) -> int:
    if proc.stdout is None:
        return await proc.wait()
    async for line in proc.stdout:
        sys.stdout.write(f"[{prefix}] {line.decode(errors='ignore')}")
    return await proc.wait()

# Flexible resolver for file.py or module[ :attr ] with instance scanning
async def _start_backend_proc(
    backend_app: Optional[str],
    backend_host: str,
    backend_port: int,
    backend_log_level: str,
    backend_reload: bool,
    backend_workers: int,
    backend_worker_timeout: int = config.app.worker_timeout,
    **kwargs,
) -> asyncio.subprocess.Process:
    """
    Returns gunicorn/uvicorn subprocess.
    Accepts:
      - backend_app: None -> default 'neurosurfer.examples.quickstart_app:ns'
      -    'pkg.mod' or 'pkg.mod:attr' or 'pkg.mod:factory()'
      -    '/path/to/app.py' -> executes in isolated namespace and probes:
      - backend_host: str -> host to bind the server to
      - backend_port: int -> port to bind the server to
      - backend_log_level: str -> logging level for the application
      - backend_reload: bool -> whether to enable auto-reload during development
      - backend_workers: int -> number of worker processes for the server
      - backend_worker_timeout: int -> worker timeout in seconds
    """
    import runpy
    from pathlib import Path
    # 1) Default example shipped in the wheel
    if backend_app is None: 
        module, app = "neurosurfer.examples.quickstart_app", "ns"
        args = [
            sys.executable, "-c", 
            f"from {module} import {app}; {app}.run(host='{backend_host}', port={backend_port}, reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
        ]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    # 2) If explicit "module:attr" or "module:factory()" provided, trust it
    if ":" in backend_app and not backend_app.endswith(".py"):
        module, app = backend_app.split(":")
        args = [
            sys.executable, "-c", 
            f"from {module} import {app}; {app}.run(host='{backend_host}', port={backend_port}, reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
        ]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

    p = Path(backend_app)
    # 3) Filesystem path case
    if p.suffix == ".py" and p.exists():
        from neurosurfer.server.app import NeurosurferApp  # import class to test instances
        # Execute in isolated namespace without __main__ side-effects
        ns = runpy.run_path(str(p), run_name="__neurosurfer_user_app__")
        app = None
        for k, v in ns.items():
            try:
                if isinstance(v, NeurosurferApp):
                    app = k
            except Exception:
                continue
        if app:
            # return (f"{p.stem}:{app}", str(p.parent))
            module = p.parent / p.stem
            args = [
                sys.executable, "-c", 
                f"from {module} import {app}; {app}.run(host='{backend_host}', port={backend_port}, reload={backend_reload}, workers={backend_workers}, log_level='{backend_log_level}')"
            ]
            return await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        raise SystemExit(
            f"Could not locate instance of NeurosurferApp in {p}.\n"
            f"Make sure you initialize a NeurosurferApp instance in your app.\n"
            f"For example: from neurosurfer.server.app import NeurosurferApp; app = NeurosurferApp(...);\n"
            f"Do not call app.run() in your app. CLI will do it for you.\n"
            f"Prefered to put app.run() in the __main__ block of your app.\n"
        )
    raise SystemExit(f"Could not locate an instance of NeurosurferApp in {backend_app}. Please make sure it is a valid path to a Python file or a valid module name.")

# ---------- Orchestration ----------
async def _serve(opts: ServeOptions) -> int:

    # ------------ FRONTEND DECISION LOGIC ------------
    ui_root: Optional[Path] = None
    run_mode: str = "none"      # "static-packaged" | "vite-dev" | "static-path" | "none"
    static_dir: Optional[Path] = None

    if not opts.only_backend:
        if opts.ui_root is None:
            # Default: look for packaged static UI
            packaged = _find_packaged_ui_dir()
            if packaged:
                run_mode = "static-packaged"
                static_dir = packaged
            else:
                # No UI requested and no packaged UI found -> only backend
                logger.warning("No UI requested and no packaged UI found -> only backend")
                run_mode = "none"
        else:
            # --ui-root provided
            ui_root = _detect_ui_root(opts.ui_root) or opts.ui_root  # keep user path if detector returns None
            if ui_root and _has_package_json(ui_root):
                run_mode = "vite-dev"
            elif ui_root and _looks_like_build_dir(ui_root):
                run_mode = "static-path"
                static_dir = ui_root
            else:
                raise SystemExit(
                    f"--ui-root provided but does not look like a Vite project or build folder:\n{ui_root}"
                )

    # ------------ START BACKEND FIRST ------------
    logger.info(f"Starting Neurosurfer backend at http://{opts.backend_host}:{opts.backend_port}")
    os.environ["NEUROSURF_SILENCE"] = "1"
    backend_proc = await _start_backend_proc(**opts.__dict__)
    backend_pipe_task = asyncio.create_task(_pipe_output("api", backend_proc))

    # ------------ START UI ACCORDING TO MODE ------------
    ui_proc = None
    ui_pipe_task = None

    if run_mode == "vite-dev":
        npm_path = _which("npm")
        if npm_path is None:
            raise SystemExit("npm not found in PATH; required for Vite dev mode (--ui-root with package.json).")
        if _needs_npm_install(ui_root, opts.npm_install_mode):
            await _run_npm_install(ui_root)

        logger.info(f"Starting NeurowebUI dev server at http://{opts.ui_host}:{opts.ui_port} (root={ui_root})")
        # Set VITE_BACKEND_URL if not provided to ensure the UI talks to the backend port
        env = os.environ.copy()
        if "VITE_BACKEND_URL" not in env:
            backend_host_for_url = opts.backend_host
            if backend_host_for_url in ("0.0.0.0", "::"):
                backend_host_for_url = os.environ.get("NEUROSURF_PUBLIC_HOST", "127.0.0.1")
            env["VITE_BACKEND_URL"] = f"http://{backend_host_for_url}:{opts.backend_port}"

        # Prefer calling dev with our env (so Vite sees VITE_BACKEND_URL)
        ui_proc = await asyncio.create_subprocess_exec(
            "npm", "run", "dev", "--", "--host", opts.ui_host, "--port", str(opts.ui_port),
            cwd=str(ui_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        ui_pipe_task = asyncio.create_task(_pipe_output("ui", ui_proc))

    elif run_mode in ("static-packaged", "static-path"):
        # IMPORTANT: per your request, bind the static server on the BACKEND PORT.
        # This will conflict if backend is already on that port. If you meant ui_port, swap the variable below.
        logger.info(f"Serving static UI from {static_dir} at http://{opts.ui_host}:{opts.ui_port}")
        ui_proc = await _start_static_serve(static_dir, opts.ui_host, opts.ui_port, opts.ui_open)
        ui_pipe_task = asyncio.create_task(_pipe_output("ui", ui_proc))
    else:
        logger.warning("No UI in this run. Only backend is running.")

    # ------------ SIGNALS & SHUTDOWN ------------
    stop_event = asyncio.Event()

    def _on_signal(signame: str):
        _eprint(f"Received {signame}, stopping...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _on_signal, s.name)

    # Log backend and frontend URLs when both processes are up
    if run_mode in ('static-packaged', 'static-path'):
        logger.info(f'\nFrontend dev server is running on: http://{opts.ui_host}:{opts.ui_port}\n')
        
    async def _wait_children_once() -> int:
        tasks = [backend_pipe_task] + ([ui_pipe_task] if ui_pipe_task else [])
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

    # terminate children
    async def _term_kill(proc: asyncio.subprocess.Process, name: str):
        if proc and proc.returncode is None:
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
        _term_kill(ui_proc, "ui") if ui_proc else asyncio.sleep(0),
    )
    return 0


# ---------- CLI ----------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("neurosurfer", description="Neurosurfer CLI")
    sub = p.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Start Neurosurfer backend and NeurowebUI")
    serve.add_argument("--backend-app", type=str, default=None, help="Backend application: file.py or module[:attr] (defaults to built-in neurosurfer.server.app)")
    serve.add_argument("--backend-host", type=str, default=os.environ.get("NEUROSURF_BACKEND_HOST", config.app.host_ip))
    serve.add_argument("--backend-port", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_PORT", config.app.host_port)))
    serve.add_argument("--backend-log-level", type=str, default=os.environ.get("NEUROSURF_BACKEND_LOG", config.app.logs_level))
    serve.add_argument("--backend-reload", action="store_true", help="Enable auto-reload for backend")
    serve.add_argument("--backend-workers", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_WORKERS", config.app.workers)))
    serve.add_argument("--backend-worker-timeout", type=int, default=int(os.environ.get("NEUROSURF_BACKEND_WORKER_TIMEOUT", config.app.worker_timeout)))
    serve.add_argument("--ui-root", type=Path, default=None, help="Path to NeurowebUI root (package.json)")
    serve.add_argument("--ui-host", type=str, default=os.environ.get("NEUROSURF_UI_HOST", config.app.ui_host))
    serve.add_argument("--ui-port", type=int, default=int(os.environ.get("NEUROSURF_UI_PORT", config.app.ui_port)))
    serve.add_argument("--ui-strict-port", action="store_true", help="Fail if UI port is already in use")
    serve.add_argument("--ui-open", action="store_true", help="Open browser for UI on start")
    serve.add_argument("--npm-install", choices=["auto", "always", "never"], default=os.environ.get("NEUROSURF_NPM_INSTALL", "auto"), help="First-run npm install behavior")
    serve.add_argument("--only-backend", action="store_true", help="Start only backend")
    serve.add_argument("--only-ui", action="store_true", help="Start only UI")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "serve":
        parser.print_help()
        return 2

    if args.only_backend and args.only_ui:
        _eprint("Cannot use --only-backend and --only-ui together.")
        return 2

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

    if opts.only_ui:
        # Provide a no-op minimal ASGI app for backend if needed in future; for now, skip backend entirely
        pass
    
    rc = asyncio.run(_serve(opts))
    # try:
    #     rc = asyncio.run(_serve(opts))
    #     return int(rc or 0)
    # except KeyboardInterrupt:
    #     return 130
    # except SystemExit as e:
    #     return int(e.code) if e.code is not None else 1
    # except Exception as e:
    #     _eprint(f"neurosurfer serve failed: {e}")
    #     return 1

if __name__ == "__main__":
    raise SystemExit(main())
