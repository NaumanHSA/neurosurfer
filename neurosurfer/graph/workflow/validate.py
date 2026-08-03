"""Compatibility re-export. The implementation moved to :mod:`.validation`.

Validation outgrew one file: twenty-five module-private checks called from a
hand-written sequence, with severity chosen at each of forty append sites and
nothing declaring which rules could fire for which node kind. It is now a
package — see ``validation/__init__.py`` for the shape and the reasoning.

This module stays because ten source files, a script and eight test modules
import from it, and a rename is not a reason to touch all nineteen. New code
should import from ``neurosurfer.graph.workflow.validation``.
"""

from __future__ import annotations

from .validation import (
    DEFER_MARKER,
    INFEASIBLE_MARKER,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_package,
)

__all__ = [
    "DEFER_MARKER",
    "INFEASIBLE_MARKER",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "validate_package",
]
