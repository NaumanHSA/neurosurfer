# Roadmap — one plan, in order

**Status:** current · branch `architect-validator/enhancement`, cut from `main`
(`4065c2f`) and pushed · build logs live beside each plan.

> **`main` is not touched until this plan is finished.** Every phase lands on the
> branch; `main` stays at `4065c2f` as the stable line while a fifty-commit port
> is in flight. When the phases are done and the full suite — **including the live
> tests** — has been run once deliberately, it merges to `main` **with a version
> bump**. See the plan's §5 for the checklist and the `1.1.0`-vs-`2.0.0` argument.

Latest: [01's build log](01_ARCHITECT_AND_VALIDATOR_BUILD_LOG.md) — **all seven
phases are in**; §8 records the release checklist and §9 the 23 commits after it.
The Architect grounds and refuses, verifies by running what it built, and the
three defects the port carried across are fixed.

**§9 is written, and the framing call is made: it is a §9, not a plan 02.** None
of those commits starts from a new diagnosis — each finishes a mechanism this
plan already owns. Node kinds as classes, validation as the first step of every
run, the executor as a package, and a rewritten prompt contract.

**The live tests have now been run against that contract**, and they found two
defects that offline testing could not: a bound argument could not reach an agent
node (`c01894a`), and the contract silently broke the tutorials (`0579b32`).
Both are fixed, with the first carrying the regression tests it should have had.
1190 collected from a 360 baseline — 1171 pass, 15 fail only on Windows (they
fail identically on `main`) — ruff clean.

One box is still open — merge to `main` and bump — and §9 sets out the
`1.1.0`-vs-`2.0.0` argument with the evidence rather than leaving it to the day.

[HANDOFF.md](HANDOFF.md) is superseded by §9 for everything except its §2, which
is still the thing to read before touching the executor package.

**Running the suite:** 24s offline with
`NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9`. Without it, the live tests
default to LM Studio on `:1234` and take twelve minutes if it happens to be up.

---

## Where the Architect stands — 2026-08-06

Parked deliberately, with the graph the current focus. What is known today, so
the next person does not re-derive it:

**It works end to end on a real model.** Three of the four live tests pass on
`gpt-4o-mini`: it builds a workflow, derives acceptance criteria, runs what it
built, judges the output, and refuses an impossible request. Its own
`package/graph.yaml` already follows the new prompt contract — all eleven node
prompts interpolate `{user_intent}` — which is why the contract rewrite did not
disturb it.

**Open — it writes inputs no step reads.** On `gpt-5-mini` the first build of the
branching intent produced:

```
ticket_urgency_routing_and_reply: The workflow asks for 'ticket_text'
but no step uses it, so the value a caller passes is ignored.
```

Five steps, a router among them, and the graph input carrying the ticket named
by none of them. `declared_inputs_are_read_by_something` catches it but only
**warns**, so the workflow stays registerable, and today's backstop is Phase 5
verification noticing the answer ignores the parameter. That is the transcript
`_BUILD_RULES` requires — the rule can now be written from evidence rather than
anticipation. `assemble.py:299` still describes interpolation as an
*authored-tool* concern, which is an understatement under the current contract.

**Open — the branching test measures two things at once.**
`test_agent_designs_branching_workflow_with_real_llm` asserts a router or two
when-guards, but the build must succeed first, and the Architect refuses to
register a workflow that fails its own verification. §8 attributed the failure to
the model being unable to design a branch. That is not what the transcripts show:
`gpt-5-mini` designs the router in its *first* plan and then fails to converge in
the repair loop — 13+ graph runs without a pass, where `gpt-4o-mini` gave up at
12. The design step is not the bottleneck; the repair budget is. Worth splitting
the assertion before treating a red live suite as a model problem.

---

## 01 — The Architect and the validator ⬅ **current**

**[01_ARCHITECT_AND_VALIDATOR_PLAN.md](01_ARCHITECT_AND_VALIDATOR_PLAN.md)**

Bring the matured Architect and validator onto the stable line, with the engine
floor they stand on, and without the studio.

Done when the Architect refuses a request it cannot ground — naming the missing
capability — instead of building something that fails at run time, and the
validator is the gate that proves it in language a person can act on.

**Built bottom-up, on purpose.** The engine can run it → the Architect can see it
→ the validator can prove it → the Architect designs against the proof. The
previous round ran that order backwards and every defect it found was a defect in
the primitive rather than in the authoring.

---

## What is deliberately not a plan of its own

- **The studio.** It lives at `NaumanHSA/neurosurfer-studio` and is not
  maintained here.
- **Validation** belongs to whichever plan owns the mechanism being checked. The
  last round had rules arriving as prompt text ahead of the structure that would
  make them enforceable, which is how a weak model came to be told three
  different times not to write a step it kept writing.
- **Evals** wait for something stable to measure. Nine runs of two prompts is a
  bug-finder, not a benchmark, and nothing above is stable until Phase 5.
- **A release.** The version bump is the *last* step of plan 01, not a phase of
  its own — there is nothing to release until the plan is done, and cutting one
  midway would put a half-ported Architect on `main`, which is the outcome §1
  exists to prevent.
