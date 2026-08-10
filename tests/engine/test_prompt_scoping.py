"""What a node is *told*, as distinct from what it can resolve.

Every LLM node used to be printed every graph input under an `Inputs:` heading.
Inside a `map` that meant each body node was handed the item it was on, the
index it was at, **and the whole collection both were drawn from** — the same
list, once per item, in every prompt. Two reviews made it merely odd; fifty make
it quadratic, and a model reading "reviews: [all fifty]" next to "item: <one>"
has no statement of which one it is supposed to be working on.

The block is gone. A node's turn carries **what its task text names, plus the
outputs of the steps it declared as dependencies** — the same contract a
LangGraph node gets, where nothing formats state into a prompt on the author's
behalf.

These tests hold apart the two things that are easy to conflate, because the
change is only safe if the second is untouched:

  * **display** — what appears in the turn (narrowed to the above);
  * **resolution** — what `{placeholders}`, expressions and function kwargs can
    reach (unchanged, and asserted to be unchanged).

The system prompt is a third thing again, and `test_the_system_prompt_is_the_same`
is the one that would fail first if the task ever crept back into it.
"""

from __future__ import annotations

from neurosurfer.graph import BaseNode, Graph, GraphExecutor, GraphNode, MapNode

from ..fakes import ScriptedProvider


def _reviews_graph(body: list[GraphNode], **map_kwargs) -> Graph:
    fan = MapNode(
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


# ── what a node is told ───────────────────────────────────────────────────────

def test_a_map_body_is_not_handed_the_collection_it_is_iterating_over():
    """The quadratic one. A body handles one element; the list it came from is
    the parent's business, and reciting it per item is O(n²) in the prompt."""
    graph = _reviews_graph([
        BaseNode(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert len(provider.prompts) == 2
    for i, prompt in enumerate(provider.prompts):
        assert REVIEWS[i] in prompt          # the one it is working on
        assert REVIEWS[1 - i] not in prompt  # and not the other


def test_a_map_body_is_not_told_its_own_plumbing():
    """`index` and the item variable are how the container talks to itself. A
    node that used `{item}` has it in the task text already; one that did not
    has no use for it."""
    graph = _reviews_graph([
        BaseNode(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    for prompt in provider.prompts:
        assert "index:" not in prompt
        assert "item:" not in prompt


def test_an_input_the_task_does_not_name_is_not_recited():
    """The rule, at the top level rather than inside a container: a graph input
    is available to interpolate, not something every node is read out."""
    graph = Graph(
        name="t",
        nodes=[BaseNode(id="a", instructions="Write the report.")],
        inputs=[
            {"name": "topic", "type": "string"},
            {"name": "house_style", "type": "string"},
        ],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run(
        {"topic": "otters", "house_style": "terse"}
    )

    assert "otters" not in provider.prompts[0]
    assert "terse" not in provider.prompts[0]


def test_an_input_the_task_does_name_arrives_in_the_task():
    graph = Graph(
        name="t",
        nodes=[BaseNode(id="a", instructions="Write about {topic} in a {house_style} voice.")],
        inputs=[
            {"name": "topic", "type": "string"},
            {"name": "house_style", "type": "string"},
        ],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run(
        {"topic": "otters", "house_style": "terse"}
    )

    assert "Write about otters in a terse voice." in provider.prompts[0]


def test_a_declared_dependency_still_arrives_whole():
    """Narrowing the inputs must not reach the dependency block — that is what
    `depends_on` is *for*, and it is the one thing a node is entitled to."""
    graph = Graph(
        name="t",
        nodes=[
            BaseNode(id="research", instructions="Research it."),
            BaseNode(id="write", depends_on=["research"], instructions="Write it up."),
        ],
        outputs=["write"],
    )
    provider = ScriptedProvider([("FINDINGS", []), ("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({})

    assert "Context from previous nodes:" in provider.prompts[1]
    assert "FINDINGS" in provider.prompts[1]


# ── the system prompt is now the same one every time ──────────────────────────

def test_the_system_prompt_is_the_same_for_every_node_and_every_item():
    """The whole point of moving the task into the turn.

    A system prompt that differs per node — and per *item* inside a map — is a
    prompt-cache prefix that can never be reused. Two body nodes over two
    reviews is four calls and, before this, four distinct system prompts.
    """
    graph = _reviews_graph([
        BaseNode(id="summarise", instructions="Summarise this review: {item}"),
        BaseNode(id="verdict", depends_on=["summarise"], instructions="One word."),
    ])
    provider = ScriptedProvider([("s", []), ("v", []), ("s", []), ("v", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert len(provider.systems) == 4
    assert len(set(provider.systems)) == 1


def test_the_system_prompt_carries_no_task():
    graph = Graph(
        name="t",
        nodes=[BaseNode(id="a", instructions="Summarise the quarterly report.")],
        outputs=["a"],
    )
    provider = ScriptedProvider([("done", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({})

    assert "quarterly report" not in provider.systems[0]
    assert "quarterly report" in provider.prompts[0]


# ── what a map body can still reach ───────────────────────────────────────────

def test_the_item_still_interpolates_into_the_task():
    graph = _reviews_graph([
        BaseNode(id="summarise", instructions="Summarise this review: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    for i, prompt in enumerate(provider.prompts):
        assert f"Summarise this review: {REVIEWS[i]}" in prompt


def test_a_body_can_still_interpolate_the_collection_if_it_asks_for_it():
    """Hidden is not the mechanism — *unnamed* is. A node that names `{reviews}`
    gets it, because the author asked and the value resolves as it always did."""
    graph = _reviews_graph([
        BaseNode(id="summarise", instructions="Of {reviews} this one is: {item}"),
    ])
    provider = ScriptedProvider([("short", []), ("short", [])])

    GraphExecutor(graph, provider=provider, validate=False).run({"reviews": REVIEWS})

    assert all(REVIEWS[0] in p and REVIEWS[1] in p for p in provider.prompts)


def test_a_function_node_in_a_body_still_receives_the_item_as_a_kwarg():
    """A function node is called with the inputs dict as kwargs. None of this
    narrowing touches that — it is about what a *model* is told."""
    graph = _reviews_graph([
        GraphNode(id="count", kind="function", callable=f"{__name__}:_word_count"),
    ])

    result = GraphExecutor(graph, validate=False).run({"reviews": REVIEWS})

    assert result.succeeded, result.errors
    assert result.final["per_review"] == [9, 9]


def _word_count(item: str, **_: object) -> int:
    return len(item.split())
