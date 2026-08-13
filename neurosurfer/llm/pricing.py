"""What a run cost, in money rather than tokens.

`Usage` threads token counts through every layer of this framework — the agent
loop, the graph engine, the trace exporters — and nothing ever converted them
into a number a person budgets in. `pyproject.toml` describes the Langfuse extra
as *"traces, cost, evals"*; the cost half was Langfuse computing it from its own
table, not ours.

**The table is data, not logic.** Rates change, models are added, and a caller
with negotiated pricing needs to override them — so prices live in one dict that
can be replaced wholesale, and every function here is a pure lookup over it.

**An unknown model costs `None`, not zero.** A model missing from the table is
unpriced, and reporting `$0.00` for it would quietly under-report a bill. `None`
is the honest answer and forces the caller to decide what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage

__all__ = [
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER",
    "PRICES",
    "ModelPrice",
    "estimate_cost",
    "format_cost",
    "price_for",
]

#: Cache reads bill at roughly a tenth of the base input rate, and the 5-minute
#: cache write at 1.25×. Both are ratios rather than absolute rates because they
#: are defined that way — a per-model cache rate would drift from its own input
#: rate the moment one of the two changed.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per **million** tokens — the unit every provider publishes."""

    input: float
    output: float
    #: Overrides for providers whose cache pricing is not a multiple of `input`.
    cache_read: float | None = None
    cache_write: float | None = None

    def read_rate(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * CACHE_READ_MULTIPLIER

    def write_rate(self) -> float:
        return (
            self.cache_write
            if self.cache_write is not None
            else self.input * CACHE_WRITE_MULTIPLIER
        )


#: Published list rates. Keys are matched by longest prefix, so a provider that
#: decorates an id — Bedrock's `anthropic.claude-opus-5`, a date suffix, a
#: `-fast` variant — still resolves.
#:
#: Anthropic rates are current as of 2026-06-24. OpenAI rates are the widely
#: published ones and are marked below; verify against your own invoice before
#: trusting either for billing rather than for a rough sense of spend.
PRICES: dict[str, ModelPrice] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-fable-5": ModelPrice(input=10.00, output=50.00),
    "claude-mythos-5": ModelPrice(input=10.00, output=50.00),
    "claude-opus-5": ModelPrice(input=5.00, output=25.00),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00),
    "claude-opus-4-7": ModelPrice(input=5.00, output=25.00),
    "claude-opus-4-6": ModelPrice(input=5.00, output=25.00),
    "claude-opus-4-5": ModelPrice(input=5.00, output=25.00),
    "claude-sonnet-5": ModelPrice(input=3.00, output=15.00),
    "claude-sonnet-4-6": ModelPrice(input=3.00, output=15.00),
    "claude-sonnet-4-5": ModelPrice(input=3.00, output=15.00),
    "claude-haiku-4-5": ModelPrice(input=1.00, output=5.00),
    # ── OpenAI (verify before billing on it) ─────────────────────────────────
    "gpt-4o-mini": ModelPrice(input=0.15, output=0.60),
    "gpt-4o": ModelPrice(input=2.50, output=10.00),
    "gpt-4.1-mini": ModelPrice(input=0.40, output=1.60),
    "gpt-4.1": ModelPrice(input=2.00, output=8.00),
    # ── Anything self-hosted ─────────────────────────────────────────────────
    #
    # Not a claim that inference is free — the electricity and the GPU are real.
    # It is a claim that *this* framework cannot know the rate, and that a local
    # run has no per-token invoice to reconcile against.
    "local": ModelPrice(input=0.0, output=0.0),
}


def price_for(model: str | None) -> ModelPrice | None:
    """The rate for *model*, matched by longest prefix, or ``None`` if unpriced.

    Prefix rather than exact match because a model id arrives decorated in more
    ways than a table can enumerate: `anthropic.claude-opus-5` on Bedrock,
    `claude-opus-4-5-20251101` pinned to a snapshot, `openai/gpt-4o` from a
    proxy. Longest-first so `claude-opus-4-8` never loses to a shorter key that
    happens to also be a prefix.
    """
    if not model:
        return None
    name = model.strip().lower()
    best: tuple[int, ModelPrice] | None = None
    for key, price in PRICES.items():
        if key in name and (best is None or len(key) > best[0]):
            best = (len(key), price)
    return best[1] if best else None


def estimate_cost(model: str | None, usage: Usage | None) -> float | None:
    """US dollars for *usage* on *model*, or ``None`` when the model is unpriced.

    "Estimate" is the honest word: it is list price against counted tokens, so
    it ignores negotiated discounts, batch pricing, and the difference between a
    5-minute and a 1-hour cache write.
    """
    if usage is None:
        return None
    price = price_for(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.read_rate()
        + usage.cache_creation_input_tokens * price.write_rate()
    ) / 1_000_000


def format_cost(cost: float | None) -> str:
    """A cost as a string a person can read at a glance.

    Sub-cent amounts get more decimals rather than rounding to `$0.00`, because
    a run that reports zero looks like a run that was not measured.
    """
    if cost is None:
        return "n/a"
    if cost == 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"
