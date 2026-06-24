"""Unit tests for the shared loose-output coercion helpers (Pillar 0b)."""

from __future__ import annotations

from neurosurfer.tools.coerce import (
    SHELL_POLICY_ALIASES,
    coerce_enum,
    coerce_str_list,
    fill_key,
)


def test_coerce_enum_maps_aliases():
    assert coerce_enum("Deny", SHELL_POLICY_ALIASES, "gated") == "denied"
    assert coerce_enum("READ-ONLY", SHELL_POLICY_ALIASES, "gated") == "readonly"
    assert coerce_enum("ask", SHELL_POLICY_ALIASES, "gated") == "gated"


def test_coerce_enum_unknown_and_nonstring_fall_back():
    assert coerce_enum("wat", SHELL_POLICY_ALIASES, "gated") == "gated"
    assert coerce_enum(123, SHELL_POLICY_ALIASES, "gated") == "gated"
    assert coerce_enum(None, {}, "text") == "text"


def test_coerce_str_list():
    assert coerce_str_list(None) == []
    assert coerce_str_list("docs/") == ["docs/"]
    assert coerce_str_list(["a", "b"]) == ["a", "b"]


def test_fill_key_fills_from_first_present_alt():
    assert fill_key({"key": "n"}, "name", ("key", "id")) == {"key": "n", "name": "n"}


def test_fill_key_leaves_present_primary_and_non_dicts_untouched():
    assert fill_key({"name": "x", "id": "y"}, "name", ("id",)) == {"name": "x", "id": "y"}
    assert fill_key("nope", "name", ("id",)) == "nope"
    assert fill_key({"foo": 1}, "name", ("id",)) == {"foo": 1}
