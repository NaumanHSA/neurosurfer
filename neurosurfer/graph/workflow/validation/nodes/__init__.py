"""Per-node rules, one module per concern.

Importing this package is what *registers* the rules — each module's
``@node_rule`` decorators run on import — so every module has to be imported
here even though nothing references its names. That is the one cost of a
registry, and it is paid in a single place with a test
(``test_every_rule_module_is_imported``) that fails if a module is added and
forgotten.
"""

from __future__ import annotations

from . import _common, agent, bindings, io, settings, tools  # noqa: F401  (imported to register)

__all__ = ["_common", "agent", "bindings", "io", "settings", "tools"]
