"""Import-time startup banner printed once on ``import neurosurfer``.

Suppressed by setting either env var to 1/true/yes:
  NEUROSURF_SILENCE=1   or   NEUROSURFER_NO_BANNER=1
"""
from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import List, Optional, Tuple


# ── silencing ─────────────────────────────────────────────────────────────────

def _silenced() -> bool:
    for key in ("NEUROSURF_SILENCE", "NEUROSURFER_NO_BANNER"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes"):
            return True
    ci_keys = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILD_NUMBER")
    return any(os.environ.get(k, "") for k in ci_keys)


def _in_jupyter() -> bool:
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        return ip is not None and "ZMQ" in type(ip).__name__
    except NameError:
        return False


# ── package version helpers ───────────────────────────────────────────────────

def _v(pkg: str) -> Optional[str]:
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return None


# ── torch / CUDA / MPS detection ─────────────────────────────────────────────

def _torch_info() -> Tuple[Optional[str], Optional[str], bool, bool, Optional[bool], List[str]]:
    try:
        import torch  # type: ignore

        torch_ver = torch.__version__
        cuda_ver: Optional[str] = getattr(torch.version, "cuda", None)
        cuda_avail = bool(torch.cuda.is_available())
        gpu_names: List[str] = []
        if cuda_avail:
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_names.append(torch.cuda.get_device_name(i))
                except Exception:
                    gpu_names.append(f"cuda:{i}")

        mps_avail = False
        mps_built: Optional[bool] = None
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None:
                mps_avail = bool(mps.is_available())
                mps_built = bool(mps.is_built())
        except Exception:
            pass

        return torch_ver, cuda_ver, cuda_avail, mps_avail, mps_built, gpu_names
    except ImportError:
        return None, None, False, False, None, []


# ── logo ──────────────────────────────────────────────────────────────────────

_LOGO = [
    r"  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗██╗   ██╗██████╗ ███████╗███████╗██████╗ ",
    r"  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██║   ██║██╔══██╗██╔════╝██╔════╝██╔══██╗",
    r"  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║╚█████╗ ██║   ██║██████╔╝█████╗  █████╗  ██████╔╝",
    r"  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║ ╚═══██╗██║   ██║██╔══██╗██╔══╝  ██╔══╝  ██╔══██╗",
    r"  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██████╔╝╚██████╔╝██║  ██║██║     ███████╗██║  ██║",
    r"  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝",
]
_TAGLINE = "Agents · RAG · Tools · Multi-LLM · FastAPI Ready"


# ── Jupyter HTML banner ───────────────────────────────────────────────────────

def _html_row(label: str, value: str) -> str:
    return (
        f'<tr>'
        f'<td style="text-align:right;color:#22d3ee;opacity:.7;padding:1px 12px 1px 0;'
        f'font-size:13px;white-space:nowrap">{label}</td>'
        f'<td style="color:#e2e8f0;font-size:13px">{value}</td>'
        f'</tr>'
    )


def _jupyter_banner(version: str) -> str:
    import html as _html

    py_ver = sys.version.split()[0]
    os_str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    torch_ver, cuda_ver, cuda_avail, mps_avail, mps_built, gpu_names = _torch_info()

    def yn(x: bool) -> str:
        return "<span style='color:#22d3ee'>yes</span>" if x else "<span style='opacity:.5'>no</span>"

    def cv(s: Optional[str]) -> str:
        return f"<span style='color:#22d3ee'>{s}</span>" if s else "<span style='opacity:.4'>-</span>"

    logo_escaped = _html.escape("\n".join(_LOGO))

    rows = [
        _html_row("version", f"<span style='color:#67e8f9;font-weight:600'>{version}</span>"
                  f"<span style='opacity:.4'> | </span>python {cv(py_ver)}"),
        _html_row("os", f"<span style='opacity:.6'>{os_str}</span>"),
    ]

    if torch_ver is not None:
        cuda_str = (
            f"{cv(torch_ver)}&nbsp;&nbsp;&nbsp;CUDA: {yn(cuda_avail)}"
            + (f"<span style='opacity:.5'> ({cuda_ver})</span>" if cuda_ver else "")
        )
        rows.append(_html_row("torch", cuda_str))
        rows.append(_html_row("mps", f"{yn(mps_avail)} <span style='opacity:.4'>(built: {mps_built})</span>"))
    else:
        rows.append(_html_row("torch", "<span style='opacity:.4'>not installed</span>"))

    tf_v  = _v("transformers")
    se_v  = _v("sentence-transformers")
    acc_v = _v("accelerate")
    bnb_v = _v("bitsandbytes")

    if tf_v or se_v:
        rows.append(_html_row("transformers",
            f"{cv(tf_v)}&nbsp;&nbsp;&nbsp;<span style='opacity:.5'>sentEmb</span> {cv(se_v)}"))
    if acc_v or bnb_v:
        rows.append(_html_row("accelerate",
            f"{cv(acc_v)}&nbsp;&nbsp;&nbsp;<span style='opacity:.5'>bnb</span> {cv(bnb_v)}"))
    if cuda_avail and gpu_names:
        rows.append(_html_row("gpu",
            f"<span style='color:#67e8f9'>{', '.join(gpu_names)}</span>"))

    return f"""
<div style="padding:8px 0;font-family:'DejaVu Sans Mono','Noto Sans Mono',monospace;">
  <pre style="color:#22d3ee;font-size:13px;line-height:1.1;
              margin:0;background:transparent;border:none;padding:0">{logo_escaped}</pre>
  <div style="color:#22d3ee;opacity:.45;font-size:11px;letter-spacing:.15em;
              margin:6px 0 10px 20px">{_TAGLINE}</div>
  <hr style="border:none;border-top:1px solid rgba(34,211,238,.2);margin:8px 0 10px">
  <table style="border-collapse:collapse;margin-left:4px">
    {''.join(rows)}
  </table>
</div>
"""


# ── terminal Rich banner ──────────────────────────────────────────────────────

def _terminal_banner(version: str) -> None:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table

    console = Console()
    py_ver = sys.version.split()[0]
    os_str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    torch_ver, cuda_ver, cuda_avail, mps_avail, mps_built, gpu_names = _torch_info()

    def yn(x: bool) -> str:
        return "yes" if x else "no"

    console.print()
    for line in _LOGO:
        console.print(f"[bright_cyan]{line}[/bright_cyan]")
    console.print(f"     [dim]{_TAGLINE}[/dim]")
    console.print()
    console.print(Rule(style="cyan", characters="─"))
    console.print()

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan dim", justify="right", min_width=16)
    grid.add_column()

    grid.add_row("version",
        f"[bright_cyan]{version}[/bright_cyan]  [dim]|[/dim]  python [cyan]{py_ver}[/cyan]")
    grid.add_row("os", f"[dim]{os_str}[/dim]")

    if torch_ver is not None:
        grid.add_row("torch",
            f"[cyan]{torch_ver}[/cyan]   CUDA: [bright_cyan]{yn(cuda_avail)}[/bright_cyan]"
            + (f" [dim]({cuda_ver})[/dim]" if cuda_ver else ""))
        grid.add_row("mps",
            f"[cyan]{yn(mps_avail)}[/cyan] [dim](built: {mps_built})[/dim]")
    else:
        grid.add_row("torch", "[dim]not installed[/dim]")

    tf_v  = _v("transformers")
    se_v  = _v("sentence-transformers")
    acc_v = _v("accelerate")
    bnb_v = _v("bitsandbytes")

    if tf_v or se_v:
        grid.add_row("transformers",
            f"[cyan]{tf_v or '-'}[/cyan]   [dim]sentEmb[/dim] [cyan]{se_v or '-'}[/cyan]")
    if acc_v or bnb_v:
        grid.add_row("accelerate",
            f"[cyan]{acc_v or '-'}[/cyan]   [dim]bnb[/dim] [cyan]{bnb_v or '-'}[/cyan]")

    if cuda_avail and gpu_names:
        grid.add_row("gpu", f"[bright_cyan]{', '.join(gpu_names)}[/bright_cyan]")

    console.print(grid)
    console.print()


# ── public ────────────────────────────────────────────────────────────────────

def print_startup_banner(version: str) -> None:
    """Print the neurosurfer startup banner. Uses HTML in Jupyter, Rich in terminal."""
    if _silenced():
        return

    if _in_jupyter():
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        ip.display_pub.publish(
            data={
                "text/plain": "\n".join(_LOGO),
                "text/html": _jupyter_banner(version),
            },
            metadata={},
        )
    else:
        _terminal_banner(version)


__all__ = ["print_startup_banner"]
