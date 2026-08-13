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

**§10 closes the four open graph items** — the tool-round budget is a spec field,
the Architect has the input-naming rule its transcript earned, the branching live
test is split into structure and behaviour, and a code node's parameters now
count as reading an input. It also records three defects found while doing them,
and one thing worth knowing before trusting any number below: **the suite was
nine-red on Linux**, eight of those a `caplog`-versus-`propagate=False` gap that
meant the tests could never have passed. Fixed; it is **1235 pass / 7 skip,
ruff clean** now.

**The live tests have now been run against that contract**, and they found two
defects that offline testing could not: a bound argument could not reach an agent
node (`c01894a`), and the contract silently broke the tutorials (`0579b32`).
Both are fixed, with the first carrying the regression tests it should have had.
1227 collected from a 360 baseline — **1207 pass, 16 fail only on Windows** —
ruff clean.

Those sixteen now have a file rather than a number:
**[WINDOWS_TEST_FAILURES.md](WINDOWS_TEST_FAILURES.md)**. Reading them changed
what they mean — seven are a single library bug (subprocess output decoded with
the ANSI codepage, so the tool author cannot verify a tool it wrote on Windows,
and the import-boundary guard is not guarding anything there), and three more
are `os.killpg` called unconditionally, which leaves `python_exec` and
`run_command` unable to time out or kill a child. Neither is this branch's, and
neither blocks the merge; both should be their own work.

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

**Addressed in §10 — it writes inputs no step reads.** The rule is now in
`_BUILD_RULES`, written from the transcript below, and the false positive that
would have made promoting the check to blocking unsafe is fixed. The check itself
is still a **warning**; promoting it is the remaining decision. On `gpt-5-mini`
the first build of the branching intent produced:

```
ticket_urgency_routing_and_reply: The workflow asks for 'ticket_text'
but no step uses it, so the value a caller passes is ignored.
```

Five steps, a router among them, and the graph input carrying the ticket named
by none of them. `declared_inputs_are_read_by_something` catches it but only
**warns**, so the workflow stays registerable, and today's backstop is Phase 5
verification noticing the answer ignores the parameter. That was the transcript
`_BUILD_RULES` required, and §10 wrote the rule from it. `assemble.py`'s
docstring — which described interpolation as an *authored-tool* concern, an
understatement under the current contract — is corrected with it.

**Closed in §10 — the branching test measured two things at once.** It is now two
tests, and the structure half passes on `qwen/qwen3.5-9b` in 124s. The analysis
below is what motivated the split and is kept for the evidence.
The single test it replaced asserted a router or two
when-guards, but the build had to succeed first, and the Architect refuses to
register a workflow that fails its own verification. §8 attributed the failure to
the model being unable to design a branch. That is not what the transcripts show:
`gpt-5-mini` designs the router in its *first* plan, then grinds in the repair
loop — **17 graph runs across 6 verification rounds without converging**, where
`gpt-4o-mini` gave up at 12. The run was stopped there rather than carried to
`max_turns`, so this is "did not converge within 17", not a recorded failure;
the point stands either way, because the design step produced the router
immediately and everything after it was repair.

The bottleneck is the repair loop, not the design — which is what the split
records: structure (is there a router?) and behaviour (does the built graph
satisfy its own judge?) are now separate tests, so a red live suite says which.
**The repair loop's convergence is the thing still open here.** Re-run the
behaviour half with `NEUROSURFER_TEST_MODEL=gpt-5-mini` if the final number is
ever wanted; budget ~20 minutes and the API spend that goes with it.

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
