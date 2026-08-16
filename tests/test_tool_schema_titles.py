"""A field named `title` must survive schema generation.

`_strip_titles` removes pydantic's `"title": "Genre"` annotation noise so weaker
local models see a flat schema. The one-line version filtered `k != "title"` at
every level — which also deleted a *field named* `title` from `properties`, whose
keys are names the author chose rather than schema keywords.

The result was deterministic and silent: the model was shown a schema with no
`title` field while `required` still listed it, so structured output failed on
every attempt with "title Field required", three retries deep. Tutorial 01's
`MovieReview` starts with exactly that field and had been failing on every run.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from neurosurfer.tools.schema import model_to_schema


class Review(BaseModel):
    title: str
    genre: str = Field(description="what kind of film")
    rating: float


def test_a_field_named_title_survives():
    schema = model_to_schema(Review)
    assert "title" in schema["properties"], "the field was eaten by the annotation strip"
    assert schema["properties"]["title"]["type"] == "string"


def test_the_schema_stays_self_consistent():
    """`required` named `title` while `properties` had lost it — a schema no model
    could satisfy, and the reason the failure looked like the model's fault."""
    schema = model_to_schema(Review)
    for name in schema.get("required", []):
        assert name in schema["properties"], f"required '{name}' is not a property"


def test_annotation_noise_is_still_stripped():
    """The actual purpose, unchanged: pydantic's per-field and top-level `title`
    annotations are still gone."""
    schema = model_to_schema(Review)
    assert "title" not in schema, "top-level annotation survived"
    # Every property is a schema; none of them should carry a title annotation.
    for name, prop in schema["properties"].items():
        assert "title" not in prop, f"annotation survived on {name}"
    # The description is preserved — only titles go.
    assert schema["properties"]["genre"]["description"] == "what kind of film"


def test_nested_models_keep_their_title_fields():
    """`$defs` is name-keyed too, so a nested model's `title` field is the same bug
    one level down."""
    class Inner(BaseModel):
        title: str
        body: str

    class Outer(BaseModel):
        title: str
        inner: Inner

    schema = model_to_schema(Outer)
    assert "title" in schema["properties"]
    defs = schema.get("$defs") or {}
    assert "Inner" in defs, f"expected the nested model in $defs, got {list(defs)}"
    assert "title" in defs["Inner"]["properties"]
    assert "title" not in defs["Inner"], "the nested annotation should still go"
