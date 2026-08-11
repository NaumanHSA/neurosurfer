# Verification

The Architect proves its own work **by running it**, before registering anything. A workflow that
looks plausible and a workflow that works are different things, and only one of them is checkable.

Two stages.

## 1. Derive acceptance criteria

One LLM call turns the user's intent plus the staged graph's declared inputs into an
**`AcceptancePlan`**:

- **2–6 explicit success criteria** — what "this worked" means for *this* intent.
- **Concrete test inputs** — real values, derived from the declared inputs.
- **A `Fixture`**, when the workflow reads a file or a directory.

## Fixtures

A workflow that reads a file cannot be tested against a *sentence*.

The run happens in a throwaway sandbox directory that the fixture script **populates first**. A
source path with nothing behind it is a **hard failure that names the fixture it needs** — not a
placeholder string handed to `read_file` so it can fail as "no such file".

That distinction is the whole point: the second produces a diagnosis about a missing file, which is
true and useless. The first produces a diagnosis about a missing fixture, which is the actual
problem.

## 2. Run it and score it

`verify_workflow` runs the staged package on those inputs — in a worker thread, because the runner
is synchronous — and then splits on what happened:

| Outcome | How it is scored |
|---|---|
| **The run failed** | A deterministic diagnosis from the node errors. **No judge call** — criteria cannot pass on a crashed run, so paying a model to say so is waste. |
| **The run was clean** | An LLM judge scores it **per criterion**, and produces a diagnosis plus design suggestions for anything failing. |

### The judge fails closed

**A criterion the judge does not rule on counts as failed.** An unscored criterion is not a pass;
treating it as one would make a judge that returned nothing look like a perfect result.

## What the agent does with the report

The report is rendered for the agent's `test_workflow` tool, and the agent applies fixes with its
**normal graph-editing tools** — `update_node`, `add_node`, rewiring `depends_on`.

This is deliberate: it is **design revision, not field patching.** A verification failure is
evidence the design is wrong, and the repair path is the same one that built it.

See [The Agent](agent.md#fixing-a-failed-verification) for the "fix the smallest thing first" rule
that governs how it responds.

## Verification is fingerprinted

An unchanged design is **not re-run**. The verification result is fingerprinted against the design
it describes, so a build that loops back through validation without changing the graph does not pay
for another full run and another judge call.

Change the graph and the fingerprint no longer matches, so the next check is real.

## Gating

Whether verification blocks registration is the `verify` setting on
[`ArchitectAgent`](agent.md#verify) — `"required"` by default.

```python
ArchitectAgent(provider, verify="required")   # must pass before it registers
ArchitectAgent(provider, verify="encouraged") # prompted, not gated
ArchitectAgent(provider, verify="off")        # skipped
```

`"required"` is the default because `"encouraged"` is prompt-only, and a weaker model simply
skipped the step: two bug-report transcripts went validate → ok → register with no `test_workflow`
call at all.

## The API

```python
from neurosurfer.architect.agent.verify import (
    AcceptanceCriterion, AcceptancePlan, Fixture,
    VerificationReport, derive_acceptance, verify_workflow,
)
```

## Next

- [The Agent](agent.md) — the loop this runs inside.
- [Grounding & Refusal](grounding.md) — the check that happens *before* a node is designed.
- [Validation](../graph/validation.md) — the structural gate, which is a different question.
