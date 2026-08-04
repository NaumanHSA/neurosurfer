# Development notes

Design documents for work in progress. Each effort is a **plan** (what is wrong,
what to do about it, in phases) and — once work lands — a **build log** (what
shipped, what it cost, and what the work turned out to be once it was underway).

They ship with the repository on purpose. The plans state *why* a mechanism has
the shape it does, and most of them end up recording a decision that looked wrong
later — which is the part that does not survive in code comments or a changelog.

Not documentation. For that, see [`docs/`](../docs) and the
[CHANGELOG](../CHANGELOG.md).

## The queue

**[ROADMAP.md](ROADMAP.md) is the order.** It names the plans and what each one
owns. Read a plan to find out why a mechanism has its shape; read the roadmap to
decide what to do next.

| Plan | Owns | State |
|---|---|---|
| [01 — The Architect and the validator](01_ARCHITECT_AND_VALIDATOR_PLAN.md) · [build log](01_ARCHITECT_AND_VALIDATOR_BUILD_LOG.md) | Bringing the matured Architect and validator onto the stable line, with the engine floor they stand on — and without the studio. | Current |

## Where the work happens

All of it is on `architect-validator/enhancement`. **`main` is left alone** until
plan 01 is finished, then merged with a version bump — the plan's §5 holds the
checklist. A half-ported Architect on the stable branch is precisely what the
plan's bottom-up ordering exists to prevent, and merging phase by phase would
produce one.

## Conventions

**A ticked box means it shipped.** Items deliberately left undone stay `- [ ]`
with a note saying so, rather than being dropped. That convention exists because
a blanket tick-through once put five unimplemented items in a plan as complete,
and the discrepancy was found by someone reading the code and disbelieving the
document.

A plan's `§0` is its diagnosis and is usually the most useful section: it says
what was actually broken and how that was measured. Every claim in a `§0` should
carry the `file:line` that proves it, so the next person can check rather than
trust.

## Where this work comes from

The Architect and validator were developed for months on a branch called
`architect-v2`, alongside a visual studio. That branch now lives in its own
repository:

```
git remote add studio https://github.com/NaumanHSA/neurosurfer-studio.git
git fetch studio && git log --oneline studio/architect-v2
```

It is **not** being merged. The studio is roughly a third of it and is not wanted
here; what is wanted is the Architect and the validator, which matured a great
deal in that time, and the engine work they stand on. Plan 01 is that port, and
its `.dev/` folder on the studio branch — eight earlier plans and their build
logs — is worth reading for the `§0` diagnoses and the "…and what it taught"
sections, which is where an assumption turned out to be wrong with the transcript
that proved it.
