"""neurosurfer: build intelligent apps that blend LLM reasoning, tools, and retrieval."""

__version__ = "0.2.0"

from neurosurfer.app.banner import print_startup_banner as _print_startup_banner
_print_startup_banner(__version__)
del _print_startup_banner
