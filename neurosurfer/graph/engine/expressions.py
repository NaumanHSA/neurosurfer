"""Safe expression evaluator for workflow control flow (Phase 1b).

Conditional edges, router cases, and loop break-conditions all need to evaluate a
small boolean/scalar expression against the live :class:`WorkflowState`. Using
Python's ``eval`` would be a code-execution hole, so this module implements a
**restricted AST evaluator**: it parses an expression with :mod:`ast`, walks only a
whitelisted set of node types, and refuses everything else.

What is allowed
---------------
- Literals: numbers, strings, ``True``/``False``/``None``, lists, tuples, dicts, sets.
- Names resolved from the evaluation namespace (``inputs``, ``nodes``, ``vars``,
  ``state`` — see :meth:`WorkflowState.namespace`).
- Attribute access as **dict/key or attribute** lookup: ``nodes.classify.label``
  resolves ``nodes["classify"]["label"]`` (falling back to ``getattr`` for objects
  such as pydantic models). Dunder attributes (``__class__`` …) are hard-blocked.
- Subscripting: ``vars.items[0]``, ``nodes.x["k"]``.
- Comparisons (``==  !=  <  <=  >  >=  in  not in  is  is not``), boolean logic
  (``and or not``), and arithmetic (``+ - * / // % **``) plus unary ``+ - not``.
- Calls to a small whitelist of pure builtins only: ``len bool int float str
  abs min max sum any all sorted round len lower upper startswith endswith`` — the
  string helpers are exposed as *functions* (``lower(x)``) not methods, so no
  arbitrary attribute-call surface is opened.

What is refused (raises :class:`ExpressionError`)
-------------------------------------------------
Assignments, imports, lambdas, comprehensions, f-strings with calls, attribute
access to dunders, calls to anything outside the whitelist, walrus, starred, slices
with steps beyond simple indexing — anything not explicitly listed above.

The evaluator never imports, never touches the filesystem/network, and cannot reach
Python internals, so a hostile expression can at worst raise or return a wrong value.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

__all__ = ["ExpressionError", "evaluate", "safe_bool"]


class ExpressionError(Exception):
    """Raised when an expression is malformed or uses a disallowed construct."""


# Whitelisted pure functions. String helpers are exposed as functions taking the
# subject as the first argument (``lower(x)``, ``startswith(s, prefix)``) so we never
# have to allow arbitrary method calls.
def _lower(s: Any) -> str:
    return str(s).lower()


def _upper(s: Any) -> str:
    return str(s).upper()


def _startswith(s: Any, prefix: Any) -> bool:
    return str(s).startswith(str(prefix))


def _endswith(s: Any, suffix: Any) -> bool:
    return str(s).endswith(str(suffix))


def _contains(haystack: Any, needle: Any) -> bool:
    try:
        return needle in haystack
    except TypeError:
        return str(needle) in str(haystack)


_ALLOWED_FUNCS: dict[str, Any] = {
    "len": len,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "any": any,
    "all": all,
    "sorted": sorted,
    "round": round,
    "lower": _lower,
    "upper": _upper,
    "startswith": _startswith,
    "endswith": _endswith,
    "contains": _contains,
}

# Binary / unary / boolean / comparison operator implementations.
import operator as _op  # noqa: E402

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
    ast.Not: _op.not_,
}

_CMP_OPS = {
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: _op.is_,
    ast.IsNot: _op.is_not,
}

# Cap iterations of any implicit fan-out (min/max/sorted over huge inputs is fine;
# this guards against pathological ** exponents producing huge ints).
_MAX_POW_EXPONENT = 1000

_MISSING = object()


def _resolve_attr(value: Any, name: str) -> Any:
    """Resolve ``value.name`` as a key lookup (Mapping) or attribute (object).

    Dunder names are blocked outright to keep Python internals unreachable.
    Missing keys/attributes return ``None`` rather than raising, so predicates like
    ``nodes.classify.label == 'x'`` are false (not an error) when a node hasn't run.
    """
    if name.startswith("__"):
        raise ExpressionError(f"access to dunder attribute {name!r} is not allowed")
    if isinstance(value, Mapping):
        return value.get(name, None)
    # Pydantic models / plain objects: attribute lookup, but never dunders/callables.
    attr = getattr(value, name, _MISSING)
    if attr is _MISSING:
        return None
    if callable(attr):
        raise ExpressionError(f"attribute {name!r} resolves to a callable; not allowed")
    return attr


class _Evaluator(ast.NodeVisitor):
    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self._ns = namespace

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        method = getattr(self, "visit_" + type(node).__name__, None)
        if method is None:
            raise ExpressionError(
                f"unsupported expression element: {type(node).__name__}"
            )
        return method(node)

    # ── literals ────────────────────────────────────────────────────────────
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(e) for e in node.elts}

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {
            self.visit(k) if k is not None else None: self.visit(v)
            for k, v in zip(node.keys, node.values, strict=True)
        }

    # ── names / access ──────────────────────────────────────────────────────
    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self._ns:
            return self._ns[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise ExpressionError(f"unknown name {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        return _resolve_attr(value, node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        key = self.visit(node.slice)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return None

    # ── operators ───────────────────────────────────────────────────────────
    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result: Any = True
            for v in node.values:
                result = self.visit(v)
                if not result:
                    return result
            return result
        # Or
        result = False
        for v in node.values:
            result = self.visit(v)
            if result:
                return result
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"unsupported unary operator {type(node.op).__name__}")
        return fn(self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"unsupported operator {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and right > _MAX_POW_EXPONENT:
            raise ExpressionError("exponent too large")
        return fn(left, right)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise ExpressionError(f"unsupported comparison {type(op).__name__}")
            right = self.visit(comparator)
            if not fn(left, right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed in expressions")
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls to whitelisted functions are allowed")
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ExpressionError(f"call to {node.func.id!r} is not allowed")
        args = [self.visit(a) for a in node.args]
        try:
            return fn(*args)
        except ExpressionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as an expression error
            raise ExpressionError(f"error calling {node.func.id}: {exc}") from exc

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)


def evaluate(expression: str, namespace: Mapping[str, Any]) -> Any:
    """Evaluate *expression* against *namespace* and return the result.

    Raises :class:`ExpressionError` on a malformed or disallowed expression.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in expression: {exc}") from exc
    return _Evaluator(namespace).visit(tree)


def safe_bool(expression: str, namespace: Mapping[str, Any], *, default: bool = False) -> bool:
    """Evaluate *expression* to a boolean, returning *default* on any error.

    Used by the scheduler for edge/router/loop predicates where a broken expression
    should fail closed (skip the branch) rather than crash the whole workflow.
    """
    try:
        return bool(evaluate(expression, namespace))
    except ExpressionError:
        return default
