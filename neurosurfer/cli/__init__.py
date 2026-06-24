"""Shim — the CLI was promoted to neurosurfer.app.cli (F6b).

The coding-assistant CLI is part of the product layer. This alias keeps
``neurosurfer.cli`` (and its submodules) importable for back-compat by
redirecting to the real package via sys.modules.
"""
import sys
import neurosurfer.app.cli as _m

sys.modules[__name__] = _m
