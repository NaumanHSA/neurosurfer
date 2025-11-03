"""
Neurosurfer: A Production-Grade AI Framework
===========================================

Neurosurfer is a comprehensive framework for building intelligent AI applications with support for:
- Multiple LLM backends (Transformers, Unsloth, vLLM, LlamaCpp, OpenAI)
- RAG (Retrieval-Augmented Generation)
- Agentic workflows (ReAct, SQL, RAG agents)
- Vector stores (Chroma, in-memory)
- Tool-based extensibility
- Production-ready FastAPI server

Example:
    >>> from neurosurfer.models.chat_models import TransformersModel
    >>> from neurosurfer.agents import ReActAgent
    >>> from neurosurfer.tools import Toolkit
    >>> llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
    >>> agent = ReActAgent(toolkit=Toolkit(), llm=llm)
    >>> response = agent.run("What is the capital of France?")
"""

import os

from .runtime.checks import (
    print_banner_once,
    warn_optional_llm_stack,
    assert_minimum_runtime,
)
from .runtime.paths import get_cache_dir
from .logger import configure_logging
from .version import __version__

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOGGER = configure_logging()

# ------------------------------------------------------------------------------
# Banner & soft checks
# ------------------------------------------------------------------------------
# Respect NEUROSURF_SILENCE=1 (or NEUROSURF_NO_BANNER=1) to suppress banner/warnings
if os.getenv("NEUROSURF_SILENCE", "").lower() not in ("1", "true", "yes"):
    print_banner_once()

# Single consolidated warning if LLM deps are missing (no import failure)
warn_optional_llm_stack()

# ------------------------------------------------------------------------------
# Cache directory bootstrap (cheap & safe)
# ------------------------------------------------------------------------------
CACHE_DIR = get_cache_dir(create=True)

# ------------------------------------------------------------------------------
# Optional eager assertion (OFF by default)
# If you *really* want to fail fast on import (e.g., in a specific app),
# opt in with: NEUROSURF_EAGER_RUNTIME_ASSERT=1
# ------------------------------------------------------------------------------
if os.getenv("NEUROSURF_EAGER_RUNTIME_ASSERT", "").lower() in ("1", "true", "yes"):
    try:
        assert_minimum_runtime()
    except Exception as e:
        # Don’t raise during import unless explicitly opted-in above
        raise
