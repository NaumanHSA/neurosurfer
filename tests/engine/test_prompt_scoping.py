"""What a node is *told*, as distinct from what it can resolve.

Every LLM node prints every graph input. Inside a `map` that meant each body
node was handed the item it was on, the index it was at, **and the whole
collection both were drawn from** — the same list, once per item, in every
prompt. Two reviews made it merely odd; fifty make it quadratic, and a model
reading "reviews: [all fifty]" next to "item: <one>" has no statement of which
one it is supposed to be working on.

These tests hold the two halves apart, because the fix is only safe if the
second half is untouched:

  * **display** — what appears in the user prompt (narrowed here);
  * **resolution** — what `{placeholders}`, expressions and function kwargs can
    reach (unchanged, and asserted to be unchanged).
"""

from __future__ import annotations

from neurosurfer.graph import Base, Graph, GraphExecutor, GraphNode, Map

from ..fakes import ScriptedProvider


def _reviews_graph(body: list[GraphNode], **map_kwargs) -> Graph:
    fan = Map(
        id="per_review",
        over="inputs.reviews",
        item_var="item",
        body=body,
        body_outputs=[n.id for n in body],
        **map_kwargs,
    )
    return Graph(
        name="review_digest",
        nodes=[fan],
        inputs=[{"name": "reviews", "type": "array"}],
        outputs=["per_review"],
    )


REVIEWS = [
    "Battery lasts two days but the screen scratches easily.",
    "Support replied in an hour and fixed it. Delighted.",
]


# ── what a map body is told ───────────────────────────────────────────────────

def test_a_map_body_is_not_handed_the_collection_it_is_iterating_over():
    """The quadratic one. A body handles one element; the list it came from is
    the parent's business, and reciting it per item is O(n²) in the prompt."""
    graph = _reviews_graph([
        Base(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert len(provider.prompts) == 2
    for prompt in provider.prompts:
        assert "reviews:" not in prompt
        # …and not merely renamed: neither review's text appears in the block.
        assert REVIEWS[1] not in prompt.split("Summarise")[-1] or REVIEWS[0] not in prompt


def test_a_map_body_is_not_told_its_own_plumbing():
    """`index` and the item variable are how the container talks to itself. A
    node that used `{item}` already has it interpolated; one that did not has no
    use for it."""
    graph = _reviews_graph([
        Base(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    for prompt in provider.prompts:
        assert "index:" not in prompt
        assert "item:" not in prompt


def test_a_shared_input_a_body_did_not_iterate_over_is_still_shown():
    """The line the narrowing must not cross. Hiding "everything the body did
    not reference" would starve every workflow that leans on the inputs block
    instead of placeholders, so only the container's own names are hidden."""
    graph = Map(
        id="per_review",
        over="inputs.reviews",
        item_var="item",
        body=[Base(id="summarise", instructions="Summarise: {item}")],
        body_outputs=["summarise"],
    )
    g = Graph(
        name="t",
        nodes=[graph],
        inputs=[
            {"name": "reviews", "type": "array"},
            {"name": "house_style", "type": "string"},
        ],
        outputs=["per_review"],
    )
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(g, provider=provider, validate=False).run(
        {"reviews": REVIEWS, "house_style": "terse"}
    )

    for prompt in provider.prompts:
        assert "house_style: terse" in prompt


def test_a_top_level_node_still_sees_every_graph_input():
    """Nothing is hidden outside a container body — a plain graph is unchanged."""
    graph = Graph(
        name="t",
        nodes=[Base(id="a", instructions="do it")],
        inputs=[{"name": "topic", "type": "string"}],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"topic": "otters"})

    assert "topic: otters" in provider.prompts[0]


# ── what a map body can still reach ───────────────────────────────────────────

def test_the_item_still_interpolates_into_the_instruction():
    """Hidden from the prompt block, not from the template. This is the whole
    reason the value is still in the body's inputs at all."""
    graph = _reviews_graph([
        Base(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    systems = " ".join(provider.systems)
    for review in REVIEWS:
        assert review in systems


def test_a_body_can_still_interpolate_the_collection_if_it_asks_for_it():
    """Hiding is about what is *recited*, so a node that names `{reviews}` still
    gets it — the author asked, and the value resolves as it always did."""
    graph = _reviews_graph([
        Base(id="summarise", instructions="Of {reviews} this one is: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert all(REVIEWS[0] in s and REVIEWS[1] in s for s in provider.systems)


def test_a_function_node_in_a_body_still_receives_the_item_as_a_kwarg():
    """A function node is called with the inputs dict as kwargs, so `item` had
    to keep arriving — by an explicit door now rather than by being merged into
    the graph's inputs."""
    graph = _reviews_graph([
        GraphNode(
            id="count",
            kind="function",
            callable=f"{__name__}:_word_count",
        ),
    ])

    result = GraphExecutor(graph, validate=False).run({"reviews": REVIEWS})

    assert result.succeeded, result.errors
    assert result.final["per_review"] == [9, 9]


def _word_count(item: str, **_: object) -> int:
    return len(item.split())


# ── saying it once ────────────────────────────────────────────────────────────

def test_an_input_the_instruction_already_states_is_not_stated_again():
    """The third instance of the shape Phase 6 found: one value, printed twice,
    under two headings that do not agree about what it is."""
    graph = Graph(
        name="t",
        nodes=[Base(id="a", instructions="Summarise this: {article}")],
        inputs=[{"name": "article", "type": "string"}],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"article": "Otters hold hands."})

    assert "Otters hold hands." in provider.systems[0]
    assert "article:" not in provider.prompts[0]


def test_an_input_the_instruction_only_reaches_into_is_still_shown():
    """`{reviews[0]}` states one element, not the list, so the list has not been
    recited and hiding it would take away something the model does not have."""
    graph = Graph(
        name="t",
        nodes=[Base(id="a", instructions="The first is: {reviews[0]}")],
        inputs=[{"name": "reviews", "type": "array"}],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert "reviews:" in provider.prompts[0]


def test_a_name_only_the_unused_field_mentions_is_still_shown():
    """`instructions` wins outright over `purpose`, so a name that appears only
    in `purpose` was never rendered into anything and must still be shown."""
    graph = Graph(
        name="t",
        nodes=[GraphNode(
            id="a", kind="base",
            instructions="Write the summary.",
            purpose="something about {article}",
        )],
        inputs=[{"name": "article", "type": "string"}],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"article": "Otters."})

    assert "article: Otters." in provider.prompts[0]


def test_a_node_with_nothing_left_to_say_still_gets_a_user_turn():
    """Narrowing can empty the block completely — a `map` body whose only input
    is the item it interpolated. An empty user turn is a 400 on Anthropic, so
    the floor is a directive rather than nothing."""
    graph = _reviews_graph([
        Base(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    for prompt in provider.prompts:
        assert prompt.strip()
        # …and it is a directive, not the instruction said a second time.
        assert "Summarise this review" not in prompt
