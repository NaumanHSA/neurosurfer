# src/neurosurfer/runtime/checks.py
from __future__ import annotations

import importlib
import os
import platform
import sys
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

try:
    # Python 3.8+: importlib.metadata is stdlib; use backport if needed
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore


# ------------- Public datatypes ------------------------------------------------

@dataclass
class DeviceInfo:
    python: str
    os: str
    neurosurfer: Optional[str]
    torch: Optional[str]
    transformers: Optional[str]
    sentence_transformers: Optional[str]
    accelerate: Optional[str]
    bitsandbytes: Optional[str]
    unsloth: Optional[str]
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: List[str]
    cuda_version: Optional[str]
    mps_available: bool
    mps_built: Optional[bool]


# ------------- Internal helpers ----------------------------------------------

def _get_version_safe(dist_name: str) -> Optional[str]:
    if version is None:
        return None
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _import_optional(mod_name: str):
    try:
        return importlib.import_module(mod_name)
    except Exception:
        return None


def _torch_module():
    return _import_optional("torch")


def _platform_triplet() -> Tuple[str, str, str]:
    return platform.system(), platform.release(), platform.machine()


def _is_ci_like() -> bool:
    # Don’t be too chatty in CI
    return any(
        os.environ.get(k, "")
        for k in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILD_NUMBER")
    )


def _silenced() -> bool:
    val = os.environ.get("NEUROSURF_SILENCE", "") or os.environ.get("NEUROSURF_NO_BANNER", "")
    return str(val).lower() in ("1", "true", "yes")


# ------------- Device / runtime detection -------------------------------------

def detect_devices() -> DeviceInfo:
    torch = _torch_module()
    cuda_available = False
    cuda_count = 0
    cuda_names: List[str] = []
    cuda_ver: Optional[str] = None
    mps_available = False
    mps_built: Optional[bool] = None

    if torch is not None:
        try:
            # CUDA checks
            cuda_available = bool(getattr(torch.cuda, "is_available")())
            cuda_count = int(getattr(torch.cuda, "device_count")())
            for i in range(cuda_count):
                try:
                    name = getattr(torch.cuda, "get_device_name")(i)
                except Exception:
                    name = f"cuda:{i}"
                cuda_names.append(str(name))
            cuda_ver = getattr(getattr(torch, "version"), "cuda", None)
        except Exception:
            pass

        try:
            # MPS (Apple Metal) checks
            mps_mod = getattr(torch, "backends", None)
            if mps_mod and hasattr(mps_mod, "mps"):
                mps = mps_mod.mps
                mps_available = bool(mps.is_available())
                mps_built = bool(mps.is_built())
        except Exception:
            pass

    sys_name, sys_rel, sys_arch = _platform_triplet()
    return DeviceInfo(
        python=sys.version.split()[0],
        os=f"{sys_name} {sys_rel} ({sys_arch})",
        neurosurfer=_get_version_safe("neurosurfer"),
        torch=_get_version_safe("torch"),
        transformers=_get_version_safe("transformers"),
        sentence_transformers=_get_version_safe("sentence-transformers"),
        accelerate=_get_version_safe("accelerate"),
        bitsandbytes=_get_version_safe("bitsandbytes"),
        unsloth=_get_version_safe("unsloth"),
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
        cuda_version=cuda_ver,
        mps_available=mps_available,
        mps_built=mps_built,
    )


# ------------- Pretty banner ---------------------------------------------------


# Optional: better width calc for Unicode; falls back to len()
try:
    from wcwidth import wcwidth
    def _wlen(s: str) -> int:
        return sum(max(wcwidth(ch), 0) for ch in s)
except Exception:
    def _wlen(s: str) -> int:
        return len(s)

def _wrap_line(s: str, width: int) -> List[str]:
    # wrap without breaking words; preserve spacing in art lines
    return textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False,
                         drop_whitespace=False) or [""]

def _draw_box(lines: Iterable[str], margin: int = 1, min_inner: int = 0, max_inner: int = 120) -> str:
    # wrap none; first compute longest visible width
    raw = list(lines)

    # pre-wrap any lines that exceed max_inner (minus margins)
    prewrap_width = max_inner
    wrapped: List[str] = []
    for ln in raw:
        # we'll decide inner width after seeing all wrapped segments
        wrapped.extend(_wrap_line(ln, prewrap_width) if ln else [""])

    longest = max((_wlen(ln) for ln in wrapped), default=0)
    inner = max(min_inner, longest + 2 * margin)
    # borders
    top    = "╔" + "═" * inner + "╗"
    bottom = "╚" + "═" * inner + "╝"

    out = [top]
    padL = " " * margin
    padR = " " * margin
    for ln in wrapped:
        vis = _wlen(ln)
        spaces = inner - vis - (2 * margin)
        if spaces < 0:
            # safety: truncate visually if wrap couldn't (rare)
            ln = ln[:max(0, inner - 2 * margin)]
            spaces = 0
        out.append("║" + padL + ln + (" " * spaces) + padR + "║")
    out.append(bottom)
    return "\n".join(out)

# ----- your banner, now using dynamic box -----

def banner(info: Optional["DeviceInfo"] = None) -> str:
    info = info or detect_devices()

    def yn(x: bool) -> str:
        return "yes" if x else "no"

    # Build content WITHOUT borders; the box will size & pad automatically
    art = [
        "▓▓▓▓▓   ▓▓▓▓                                   ▓▓▓",
        " ▓▓ ▓▓   ▓▓  ▓▓▓▓▓ ▓  ▓ ▓ ▓ ▓▓▓▓ ▓▓▓ ▓  ▓ ▓ ▓  ▓",
        " ▓▓  ▓▓  ▓▓  ▓▁▁▁▓ ▓  ▓ ▓▓▏ ▓  ▓ ▓▁  ▓  ▓ ▓▓▏ ▓▓▓",
        " ▓▓   ▓▓ ▓▓  ▓     ▓  ▓ ▓   ▓  ▓   ▓ ▓  ▓ ▓    ▓",
        "▓▓▓▓   ▓▓▓▓▓ ▓▓▓▓▓ ▓▓▓▓ ▓   ▓▓▓▓ ▓▓▓ ▓▓▓▓ ▓    ▓",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "Orchestrate Agents - RAG - SQL Tools - Multi-LLM - FastAPI Ready",
        "Faster builds, clearer flows, production-first",
    ]

    meta = [
        f"Version: {info.neurosurfer or 'unknown'} | Python: {info.python}",
        f"OS: {info.os}",
        f"Torch: {info.torch or 'not installed'}   CUDA: {yn(info.cuda_available)} ({info.cuda_version or '-'})",
        f"MPS: {yn(info.mps_available)} (built: {info.mps_built})",
        f"Transformers: {info.transformers or '-'}   SentEmb: {info.sentence_transformers or '-'}",
        f"Accelerate: {info.accelerate or '-'}   bnb: {info.bitsandbytes or '-'}",
        f"Unsloth: {info.unsloth or '-'}",
    ]

    tail: List[str] = []
    if info.cuda_available and info.cuda_device_count:
        tail.append("Detected CUDA devices: " + ", ".join(info.cuda_device_names))

    # Compose with some empty lines for breathing room
    content = [""] + art + [""] + meta + [""] + tail
    # Draw the dynamic box; margins = 1 space, inner width capped to 120 columns
    return _draw_box(content, margin=1, min_inner=0, max_inner=120)


def print_banner_once():
    if _silenced() or _is_ci_like():
        return
    if getattr(print_banner_once, "_printed", False):
        return
    print(banner())
    print_banner_once._printed = True  # type: ignore


# ------------- Install hints & warnings ---------------------------------------

# Modules that define the "LLM stack"
LLM_STACK: Dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "unsloth": "unsloth",
}

# For import name -> PyPI name where they differ
IMPORT_TO_PYPI = {
    "sentence_transformers": "sentence-transformers",
}


def _missing_llm_modules() -> List[str]:
    missing = []
    for pypi_name, import_name in LLM_STACK.items():
        if _import_optional(import_name) is None:
            missing.append(pypi_name)
    return missing


def _default_install_lines(missing: List[str]) -> List[str]:
    """General, always-valid hints."""
    lines = []
    # Single “works everywhere” extra:
    lines.append("pip install -U 'neurosurfer[torch]'")
    # Explicit fallback:
    pkg_line = " ".join(
        sorted(set(missing + ["transformers", "sentence-transformers", "accelerate"]))
    )
    lines.append(f"pip install -U {pkg_line}")
    # Torch CPU fallback (no CUDA wheels required):
    lines.append(
        "pip install -U torch --index-url https://download.pytorch.org/whl/cpu"
    )
    return lines


def _env_specific_torch_hint() -> Optional[str]:
    sys_name, _, arch = _platform_triplet()
    # CUDA-capable Linux x86_64 default hint
    if sys_name == "Linux" and arch in ("x86_64", "amd64"):
        return (
            "pip install -U torch --index-url https://download.pytorch.org/whl/cu124"
        )
    # Apple Silicon: prefer MPS
    if sys_name == "Darwin" and arch in ("arm64", "aarch64"):
        return "pip install -U torch  # macOS arm64 builds include MPS"
    return None


def _bitsandbytes_hint() -> Optional[str]:
    sys_name, _, arch = _platform_triplet()
    if sys_name == "Linux" and arch in ("x86_64", "amd64"):
        return "pip install -U bitsandbytes  # Linux x86_64 is supported"
    # On macOS/Windows, wheels are limited; steer user accordingly
    return "bitsandbytes wheels are limited outside Linux x86_64; consider CPU or other quantization backends."


def _compose_install_hint(missing: List[str]) -> str:
    lines = _default_install_lines(missing)
    torch_env = _env_specific_torch_hint()
    if torch_env:
        lines.append(torch_env)
    if "bitsandbytes" in missing:
        bnb = _bitsandbytes_hint()
        if bnb:
            lines.append(bnb)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return "\n  - " + "\n  - ".join(unique)


def warn_optional_llm_stack():
    """
    Soft-check: warn if the LLM stack is missing. Called on import.
    """
    if _silenced() or _is_ci_like():
        return
    if getattr(warn_optional_llm_stack, "_warned", False):
        return

    missing = _missing_llm_modules()
    if not missing:
        warn_optional_llm_stack._warned = True  # type: ignore
        return

    # Prefer a single consolidated warning (not spammy)
    msg = textwrap.dedent(
        f"""
        Some optional LLM dependencies are not installed: {', '.join(missing)}.
        Neurosurfer core will load, but LLM features (model loading, HF/Unsloth, BnB quant, etc.)
        will be unavailable until you install them.

        To install everything with one extra:
          - pip install 'neurosurfer[torch]'

        Or use one of these commands:
          {_compose_install_hint(missing)}
        """
    ).strip()

    warnings.warn(msg, category=UserWarning, stacklevel=2)
    warn_optional_llm_stack._warned = True  # type: ignore


# ------------- Hard requirement (for code paths that need LLM stack) ----------
def require(module: str, feature: str, install_hint: Optional[str] = None):
    """
    Import a module or raise a RuntimeError with a helpful install hint.
    Use in your code paths that need optional frameworks.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        hint = install_hint or f"pip install {IMPORT_TO_PYPI.get(module, module)}"
        raise RuntimeError(
            f"Optional feature '{feature}' requires '{module}'. Install via: {hint}"
        ) from e


def assert_minimum_runtime():
    """
    Ensure core LLM runtime deps are present; call early in LLM/model startup paths.
    Keep this OUT of global import to avoid hard-failing lightweight installs.
    """
    # Torch: give environment-aware hint
    torch_hint = _env_specific_torch_hint() or "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    require("torch", "Neurosurfer LLM runtime", torch_hint)
    require("transformers", "Model runners", "pip install transformers")
    require("sentence_transformers", "Embedding models", "pip install sentence-transformers")
    # Nice-to-haves (don’t raise if missing; only warn)
    for opt_mod, feat in [
        ("accelerate", "faster training/inference"),
        ("bitsandbytes", "8-bit/4-bit quantization"),
        ("unsloth", "parameter-efficient finetuning")
    ]:
        if _import_optional(opt_mod) is None and not _silenced():
            warnings.warn(
                f"Optional package '{opt_mod}' not found; {feat} will be unavailable. "
                f"Install via: pip install {opt_mod}",
                category=UserWarning,
                stacklevel=2,
            )


# ------------- Reports ---------------------------------------------------------
def diagnostic_report() -> str:
    info = detect_devices()
    details = textwrap.dedent(f"""
    Neurosurfer diagnostic report
    ---------------------------
    Version:           {info.neurosurfer}
    Python:            {info.python}
    OS:                {info.os}

    Torch:             {info.torch}
    Transformers:      {info.transformers}
    Sent-Transformers: {info.sentence_transformers}
    Accelerate:        {info.accelerate}
    bitsandbytes:      {info.bitsandbytes}
    Unsloth:           {info.unsloth}

    CUDA available:    {info.cuda_available}
    CUDA devices:      {info.cuda_device_count} -> {', '.join(info.cuda_device_names) or '-'}
    CUDA version:      {info.cuda_version}
    MPS available:     {info.mps_available}
    MPS built:         {info.mps_built}
    """).strip()
    return details


# ------------- Public API ------------------------------------------------------

__all__ = [
    "DeviceInfo",
    "detect_devices",
    "banner",
    "print_banner_once",
    "warn_optional_llm_stack",
    "require",
    "assert_minimum_runtime",
    "diagnostic_report",
]
