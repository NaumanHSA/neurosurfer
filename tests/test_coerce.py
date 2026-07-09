"""Unit tests for the shared loose-output coercion helpers (Pillar 0b)."""

from __future__ import annotations

from neurosurfer.tools.coerce import SHELL_POLICY_ALIASES, coerce_enum


def test_coerce_enum_maps_aliases():
    assert coerce_enum("Deny", SHELL_POLICY_ALIASES, "gated") == "denied"
    assert coerce_enum("READ-ONLY", SHELL_POLICY_ALIASES, "gated") == "readonly"
    assert coerce_enum("ask", SHELL_POLICY_ALIASES, "gated") == "gated"


def test_coerce_enum_unknown_and_nonstring_fall_back():
    assert coerce_enum("wat", SHELL_POLICY_ALIASES, "gated") == "gated"
    assert coerce_enum(123, SHELL_POLICY_ALIASES, "gated") == "gated"
    assert coerce_enum(None, {}, "text") == "text"
