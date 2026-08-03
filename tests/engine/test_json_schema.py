"""JSON Schema → pydantic, the form a node writes its output shape in.

The point of these is the *error* cases as much as the happy ones: a schema that
cannot become a model has to say which part of it could not be read, because
validation surfaces that message to somebody looking at the node, and "invalid
schema" would send them back to re-reading the whole thing.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from neurosurfer.graph.engine.json_schema import JsonSchemaError, model_from_json_schema


def test_scalars_and_required():
    M = model_from_json_schema({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "number"},
            "rank": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
        "required": ["title", "score"],
    })
    assert issubclass(M, BaseModel)
    m = M(title="t", score=1.5)
    assert m.title == "t" and m.score == 1.5
    # Unrequired properties are absent rather than invalid.
    assert m.rank is None and m.ok is None
    with pytest.raises(ValidationError):
        M(score=1.0)


def test_integer_and_number_stay_distinct():
    """Mapping both to float would silently accept 3.5 where an int was declared."""
    M = model_from_json_schema({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    })
    with pytest.raises(ValidationError):
        M(n=3.5)


def test_arrays_of_scalars_and_of_objects():
    M = model_from_json_schema({
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"k": {"type": "string"}},
                    "required": ["k"],
                },
            },
        },
        "required": ["tags", "rows"],
    })
    m = M(tags=["a"], rows=[{"k": "v"}])
    assert m.tags == ["a"]
    # Nested objects become nested models, so validation reaches all the way down.
    assert m.rows[0].k == "v"
    with pytest.raises(ValidationError):
        M(tags=["a"], rows=[{"wrong": "v"}])


def test_nested_object_validates_its_own_required():
    M = model_from_json_schema({
        "type": "object",
        "properties": {
            "meta": {
                "type": "object",
                "properties": {"author": {"type": "string"}},
                "required": ["author"],
            },
        },
        "required": ["meta"],
    })
    assert M(meta={"author": "a"}).meta.author == "a"
    with pytest.raises(ValidationError):
        M(meta={})


def test_enum_restricts_the_value():
    M = model_from_json_schema({
        "type": "object",
        "properties": {"verdict": {"enum": ["yes", "no"]}},
        "required": ["verdict"],
    })
    assert M(verdict="yes").verdict == "yes"
    with pytest.raises(ValidationError):
        M(verdict="maybe")


def test_nullable_via_type_list():
    M = model_from_json_schema({
        "type": "object",
        "properties": {"note": {"type": ["string", "null"]}},
        "required": ["note"],
    })
    assert M(note=None).note is None
    assert M(note="hi").note == "hi"


def test_default_is_honoured():
    M = model_from_json_schema({
        "type": "object",
        "properties": {"n": {"type": "integer", "default": 7}},
    })
    assert M().n == 7


def test_typeless_property_accepts_anything():
    """No `type` is legal JSON Schema and means "anything" — refusing it would
    be inventing a rule the format does not have."""
    M = model_from_json_schema({
        "type": "object",
        "properties": {"blob": {}},
        "required": ["blob"],
    })
    assert M(blob={"a": 1}).blob == {"a": 1}


def test_description_survives_onto_the_field():
    """It is what the model is shown about each property, so losing it would
    quietly make the structured call worse."""
    M = model_from_json_schema({
        "type": "object",
        "properties": {"title": {"type": "string", "description": "A short title."}},
        "required": ["title"],
    })
    assert M.model_fields["title"].description == "A short title."


# ── the errors, which are the part an author actually reads ────────────────


def test_non_object_top_level_is_refused():
    with pytest.raises(JsonSchemaError, match="top level must be an object"):
        model_from_json_schema({"type": "string"})


def test_missing_properties_is_refused():
    with pytest.raises(JsonSchemaError, match="non-empty `properties`"):
        model_from_json_schema({"type": "object"})


def test_unknown_type_names_the_property():
    with pytest.raises(JsonSchemaError, match=r"score: unknown type 'strng'"):
        model_from_json_schema({
            "type": "object",
            "properties": {"score": {"type": "strng"}},
        })


def test_required_naming_a_missing_property_is_refused():
    """A typo here is otherwise invisible: the model builds, and the field the
    author meant to make mandatory simply is not."""
    with pytest.raises(JsonSchemaError, match=r"do not exist: \['titel'\]"):
        model_from_json_schema({
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["titel"],
        })


def test_error_path_reaches_into_a_nested_object():
    with pytest.raises(JsonSchemaError, match=r"meta\.author: unknown type"):
        model_from_json_schema({
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"author": {"type": "nope"}},
                },
            },
        })


def test_model_name_is_made_safe():
    """Node ids carry hyphens; a model name has to be an identifier."""
    M = model_from_json_schema(
        {"type": "object", "properties": {"a": {"type": "string"}}},
        name="my-node_output",
    )
    assert M.__name__.isidentifier()
