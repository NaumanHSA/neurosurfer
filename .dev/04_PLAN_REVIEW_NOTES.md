# 04 — Plan review, as a feature rather than a hook

**Status: notes only. Nothing here is implemented.** Written 2026-08-16 from a
debugging session over `ArchitectAgent.build`; the owner's call, recorded before
it is designed so the reasoning survives.

---

## The complaint

Reviewing a plan is the cheapest possible place to correct a build — a wrong step
costs one edit here and a whole rebuild once it is nodes. `agent.py` says so in a
comment and then makes the feature almost unusable:

1. **It is a callback, not a feature.** `approve_plan=` takes a callable the
   caller must write. There is no built-in way to say "just ask me."
2. **It defaults to off.** `approve_plan=None` skips review entirely, so the
   default behaviour for a human-driven build is *no human in the loop*.
3. **What you are shown is not enough to judge.** `plan.render()` is a flat text
   list of steps. Deciding whether a design is right means seeing its *shape*.
4. **You cannot say what you want changed.** The three accepted answers are yes,
   no, and *here is a complete replacement plan you wrote yourself*. There is no
   path from "make the summary step read the ticket too" to a revised plan.

---

## What it should be

**One parameter.** Something like `approve: bool = True` — needs approval or not,
defaulting to **yes, ask**. The interaction is built in; a caller who wants a
custom surface can still supply one, but nobody should have to write a callback
to get the obvious behaviour.

**Show the plan properly.** The current render is the floor, not the ceiling. At
minimum it needs the plan's *shape* alongside the list — a small graph of the
steps and their `depends_on` edges, so a bag of parallel steps and a pipeline do
not read identically. It is a DAG; draw it.

**Four answers, not three:**

| Answer | Meaning |
|---|---|
| **yes** | build it as planned |
| **no** | reject; nothing is built |
| **a JSON plan** | build exactly this instead — the current edit path |
| **plain English** | *"drop the validation step and have the summary read the ticket"* → **the model replans against that feedback**, and the revised plan comes back for review |

The fourth is the new capability, and the reason for the note. Today rejecting is
a dead end: `blocked_reason` is set and the run ends telling the user to start
over with a better intent. Replanning from feedback keeps the session, the
resolved capabilities and the intent, and changes only what was asked.

That implies a **loop**: plan → review → (feedback → replan → review)\* → build.
Worth deciding up front how many turns it may take before it gives up, the same
way the repair loop has a turn budget.

---

## Things the implementation must not lose

Found while reading the current code; each is load-bearing and easy to break.

- **An edited plan is re-resolved.** `resolve_plan` runs again on anything the
  reviewer hands back, because an edit may name a capability nothing has looked
  up yet. Cheap, and skipping it means building against a stale answer.

- **The re-resolve is guarded by an identity check.** `if decided is not plan:`.
  A reviewer that mutates the plan **in place** and returns the same object
  silently skips re-resolution — its new capability is never looked up. Any
  built-in review surface must not fall into this; going through `to_dict()` /
  `model_validate` is what makes it a different object. A better design would
  re-resolve unconditionally and drop the identity test, since resolving an
  unchanged plan twice costs a catalog lookup and nothing else.

- **`session.plan` is set before `_notify`.** Observers snapshot the session on
  every notify, so narrating first leaves the record holding the plan the user
  just replaced.

- **A malformed edit is not a build failure.** It warns and builds the original.
  Plain-English feedback needs the same tolerance: a replan that comes back
  unusable should fall back, not crash.

- **Supplying a reviewer currently disables the automatic infeasibility gate** —
  the `return` at the end of that branch skips the `acquirable` / unresolved-step
  checks below. Deliberate today ("accepting a plan with unresolved steps is an
  explicit override"), but if review becomes the *default*, that gate is off by
  default too. **This is the sharpest consequence of flipping the default and
  needs an explicit decision**, not an inherited one.

---

## Open questions

- Does `approve=True` make sense for a non-interactive caller (a test, a cron,
  the gateway)? Probably the parameter needs a third state, or the default is
  conditional on there being a TTY / a registered interaction channel.
- Where does the replan prompt live — the planner's own system prompt with the
  previous plan plus feedback, or a separate revise prompt? The planner is one
  structured call today, which is the form a weak model is most reliable at;
  keeping that property matters more than reusing the prompt.
- The gateway already parks a build and waits for `/respond` when
  `review_plan: true`. Whatever this becomes should collapse into that, not sit
  beside it.

---

## Where this sits

Not queued yet. It follows plan 01, and it is a **UX/behaviour change to a
mechanism plan 01 owns**, so it is either a phase of a later Architect plan or a
small plan of its own — decide when it is picked up, not now.
