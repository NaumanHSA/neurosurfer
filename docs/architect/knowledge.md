# Self-knowledge

The Architect does not guess what this installation can do. It reads it.

Two sources, both derived rather than hand-maintained, because a hand-maintained list of
capabilities is a list that is wrong the first time somebody adds a tool.

```python
from neurosurfer.architect.knowledge import KnowledgeBase
```

## The capability manifest

Built from the [tool registry](../guides/tool-registry.md): what tools exist, what each declares it
can do, what credentials each needs, and which needs nothing here can satisfy.

**Versioned by content hash.** Adding or retagging a tool invalidates the manifest automatically —
there is no cache to remember to clear, and no way for the Architect's picture of the installation
to drift from the installation.

This is what `find_capability` and `describe_capability` read.

## The docs index

**These pages are an input to the Architect**, not only documentation for you.

Every markdown file under `docs/` is split at headings and ranked for a query — BM25 when
`rank_bm25` is installed (the `search` extra), otherwise a plain term-frequency overlap score. The
agent reaches it through its `neurosurfer_docs` tool.

Deliberately dependency-light: no embeddings, no vector store, works offline and instantly, and it
is good enough for *"pull the authoritative paragraph about X"*.

```python
from neurosurfer.architect.knowledge.docs_index import DocsIndex, DocSection
```

### What this means for writing docs

Sections are retrieved **without their parents**, which sets two rules:

- **A heading has to name its subject.** `## Loop` under `# Node Kinds` reads as "Loop" to whatever
  pulled it; `## Iterating` does not.
- **A paragraph has to stand alone**, without the three above it.

A subsystem with no page is a subsystem the Architect cannot look up when deciding whether it can
build something. That is why a documentation gap here costs more than a documentation gap usually
does.

### Two pages are excluded

`about/roadmap` and `about/changelog` are dropped from the index.

They name every feature in the project without explaining any of them. A roadmap entry is mostly a
list of capability nouns — which is exactly the vocabulary a build's queries use — so BM25 scores
it highly for almost anything and it **displaces the guide that actually answers**.

Measured on six build-shaped queries it took a top-3 slot in four of them, once beating the graph
guide for *"tool node tool_args required arguments"* with a section about trace spans.

## Why derived, never stored

The same reasoning in both halves: a copy inside the Architect would be a second source of truth
that drifts the moment a tool is added or a server is reconfigured. Both facts are already
authoritative somewhere, so the Architect asks rather than remembers.

## Next

- [Tool Registry](../guides/tool-registry.md) — the manifest's source.
- [Grounding & Refusal](grounding.md) — what the Architect does with what it knows.
- [The Agent](agent.md) — the tools that read this.
