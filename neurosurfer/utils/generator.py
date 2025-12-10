from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generator, Iterable, List, Tuple, Union, Optional


@dataclass
class GeneratorResult:
    """
    Holds the full results of a generator:

    - yields: list of values produced via `yield`
    - return_value: the final value produced via `return`
    """
    yields: List[Any]
    return_value: Any


@dataclass
class StreamEvent:
    """
    Represents one event emitted by `iterate_with_return`.

    kind: "yield" or "return"
    value: yield value or final return value
    """
    kind: str
    value: Any


class InvalidGeneratorError(Exception):
    """Raised when a non-generator object is passed where a generator is expected."""
    pass


def _ensure_generator(obj) -> Generator:
    """
    Internal: ensures the object is a generator.
    """
    if not hasattr(obj, "__iter__") or not hasattr(obj, "__next__"):
        raise InvalidGeneratorError(f"Expected a generator, got {type(obj).__name__}")
    return obj


def iterate_with_return(gen: Generator) -> Iterable[StreamEvent]:
    """
    Iterate through all yields of a generator AND capture the final return.

    Yields:
        StreamEvent("yield", value)
        StreamEvent("return", return_value)

    Usage:
        for event in iterate_with_return(mygen()):
            if event.kind == "yield":
                ...
            else:
                ...
    """
    gen = _ensure_generator(gen)

    while True:
        try:
            value = next(gen)
        except StopIteration as e:
            # Final return value of the generator
            yield StreamEvent("return", e.value)
            break
        else:
            yield StreamEvent("yield", value)



def consume(gen: Generator) -> GeneratorResult:
    """
    Fully consume a generator and return all yields + final return.

    Returns:
        GeneratorResult(yields=[...], return_value=...)
    """
    gen = _ensure_generator(gen)

    yields = []
    return_value = None

    try:
        while True:
            yields.append(next(gen))
    except StopIteration as e:
        return_value = e.value

    return GeneratorResult(yields=yields, return_value=return_value)
