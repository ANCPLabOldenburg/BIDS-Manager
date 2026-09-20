"""Evaluate a parsed BIDS schema expression against a context.

Parsing is reused from ``bidsschematools.expressions.parse``, which returns an
abstract syntax tree built from a handful of node types (:class:`BinOp`,
:class:`RightOp`, :class:`Function`, :class:`Element`, :class:`Property`,
:class:`Array`, :class:`Object`) plus *leaf atoms*: Python ``int``/``float`` for
numbers, and ``str`` for identifiers and quoted-string literals.

This module walks that tree. Two deliberate choices:

* **No ``eval``/``exec``.** Expressions come from the trusted schema, but a real
  tree-walk gives well-defined semantics, good error messages, and keeps code
  execution off the table.
* **Reference parity.** Operator and null-propagation semantics mirror the
  canonical JavaScript implementation in the reference validator. Correctness is
  pinned by the schema's own ``meta.expression_tests``.

A *context* is a mapping from names an expression may reference (``sidecar``,
``suffix``, ``nifti_header`` ...) to their values.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from bidsschematools.expressions import (
    Array,
    BinOp,
    Element,
    Function,
    Object,
    Property,
    RightOp,
    parse,
)

from . import functions as fn

Context = Mapping[str, Any]

# Logical operators short-circuit and therefore must not eagerly evaluate their
# right-hand side; everything else uses the eager binary path.
_LOGICAL_OPS = frozenset({"&&", "||"})


class EvaluationError(Exception):
    """Raised when an expression cannot be evaluated."""


class UnknownFunction(EvaluationError):
    """Raised when an expression calls a function the evaluator does not know.

    Surfacing this distinctly lets callers (e.g. a rule engine) treat an
    unfamiliar function in a newer-than-engine schema as a skipped construct
    rather than a hard failure.
    """


def evaluate_string(expression: str, context: Context) -> Any:
    """Parse ``expression`` and evaluate it against ``context``.

    Parsing is memoized, so repeatedly evaluating the same expression text (the
    common case across many files) parses only once. A string that cannot be
    parsed raises :class:`EvaluationError` rather than leaking the underlying
    parser exception.
    """
    try:
        node = _parse_cached(expression)
    except Exception as exc:  # pyparsing raises its own exception types
        raise EvaluationError(f"could not parse expression {expression!r}: {exc}") from exc
    return evaluate(node, context)


@lru_cache(maxsize=4096)
def _parse_cached(expression: str):
    return parse(expression)


def evaluate(node: Any, context: Context) -> Any:
    """Evaluate a parsed ``node`` (or leaf atom) against ``context``."""
    handler = _NODE_HANDLERS.get(type(node))
    if handler is not None:
        return handler(node, context)
    # Leaf atoms. Booleans are excluded from the numeric branch because
    # ``bool`` is a subclass of ``int`` in Python; the parser never emits bare
    # Python booleans anyway (it uses the identifiers ``true``/``false``).
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return node
    if isinstance(node, str):
        return _atom(node, context)
    # An already-evaluated Python value passed straight through.
    return node


# ---------------------------------------------------------------------------
# Leaf atoms
# ---------------------------------------------------------------------------


def _atom(token: str, context: Context) -> Any:
    """Resolve a leaf string token.

    The parser hands back identifiers and quoted strings as plain ``str``. A
    quoted token is a string literal (the quotes are preserved by the parser);
    ``null``/``true``/``false`` are keywords; anything else is a reference to a
    context variable.
    """
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    return _lookup(context, token)


def _lookup(container: Any, name: str) -> Any:
    """Read ``name`` from a mapping (or attribute), ``None`` when absent."""
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


# ---------------------------------------------------------------------------
# Compound nodes
# ---------------------------------------------------------------------------


def _eval_array(node: Array, context: Context) -> list[Any]:
    return [evaluate(element, context) for element in node.elements]


def _eval_object(node: Object, context: Context) -> dict[str, Any]:
    # Object literals only ever appear as the empty ``{}`` in the schema.
    return {}


def _eval_property(node: Property, context: Context) -> Any:
    target = evaluate(node.name, context)
    if target is None:  # null propagation: null.anything == null
        return None
    return _lookup(target, node.field)


def _eval_element(node: Element, context: Context) -> Any:
    target = evaluate(node.name, context)
    if target is None:  # null propagation: null[i] == null
        return None
    idx = evaluate(node.index, context)
    if isinstance(target, (list, str)) and isinstance(idx, int) and not isinstance(idx, bool):
        if 0 <= idx < len(target):
            return target[idx]
    return None


def _eval_function(node: Function, context: Context) -> Any:
    # Check known-ness before evaluating arguments: an unfamiliar function in a
    # newer-than-engine schema is reported as such without doing argument work.
    if not fn.is_known(node.name):
        raise UnknownFunction(node.name)
    args = [evaluate(arg, context) for arg in node.args]
    try:
        return fn.call(node.name, args, context)
    except (IndexError, TypeError, ValueError):
        # Wrong arity or an unexpected argument type (e.g. a newer schema changed
        # a signature) degrades to null rather than crashing the whole run.
        return None


def _eval_rightop(node: RightOp, context: Context) -> Any:
    if node.op == "!":
        return not fn.truthy(evaluate(node.rh, context))
    raise EvaluationError(f"unsupported unary operator {node.op!r}")


def _eval_binop(node: BinOp, context: Context) -> Any:
    op = node.op
    # Short-circuit logical operators, returning the operand value (not a
    # coerced boolean), matching JavaScript's ``&&`` / ``||``.
    if op in _LOGICAL_OPS:
        left = evaluate(node.lh, context)
        if op == "&&":
            return evaluate(node.rh, context) if fn.truthy(left) else left
        return left if fn.truthy(left) else evaluate(node.rh, context)

    left = evaluate(node.lh, context)
    right = evaluate(node.rh, context)
    return _binary(op, left, right)


# Arithmetic and ordering operators. Across the entire BIDS schema these only
# ever apply to numbers: tabular (string) columns are funneled through
# min/max/count/length first, and a missing field null-propagates. So the engine
# does not need JavaScript's full coercion rules; it needs to be robust. The rule
# engine evaluates hundreds of expressions against every file, so a malformed
# dataset (a number stored as a string, a zero denominator) must degrade to null
# rather than raise. These two groups implement that: coerce when sensible, and
# return null instead of crashing.
_ARITHMETIC_OPS = frozenset({"-", "*", "/", "%", "**"})
_ORDERING_OPS = frozenset({"<", "<=", ">", ">="})


def _binary(op: str, left: Any, right: Any) -> Any:
    """Evaluate a non-short-circuiting binary operator."""
    if op == "==":
        return _equal(left, right)
    if op == "!=":
        return not _equal(left, right)
    if op == "in":
        return _contains(left, right)

    # Arithmetic and ordering propagate null: any null operand yields null.
    if left is None or right is None:
        return None
    if op == "+":
        return _add(left, right)
    if op in _ARITHMETIC_OPS:
        return _arithmetic(op, left, right)
    if op in _ORDERING_OPS:
        return _ordering(op, left, right)
    raise EvaluationError(f"unsupported operator {op!r}")


def _contains(left: Any, right: Any) -> Any:
    """The ``in`` operator: membership in an object's keys / a sequence.

    In the schema ``in`` is used exclusively as key membership on an object
    (``"FlipAngle" in sidecar``), which Python's ``in`` on a mapping matches
    exactly. A null right operand yields null; anything else degrades to a safe
    result rather than raising.
    """
    if right is None:
        return None
    if isinstance(right, (Mapping, list, str)):
        try:
            return left in right
        except TypeError:  # e.g. an unhashable left operand against a mapping
            return False
    return None


def _equal(left: Any, right: Any) -> bool:
    """Equality with null treated specially: null equals only null.

    The schema only ever compares like with like (string to string, number to
    number), so plain equality is correct here; JavaScript's looser cross-type
    coercion is not exercised by any schema rule.
    """
    if left is None or right is None:
        return left is None and right is None
    return left == right


def _add(left: Any, right: Any) -> Any:
    """``+`` is numeric addition, or string concatenation if either side is text."""
    if isinstance(left, str) or isinstance(right, str):
        return fn.js_string(left) + fn.js_string(right)
    try:
        return left + right
    except TypeError:
        return None


def _arithmetic(op: str, left: Any, right: Any) -> Any:
    """Numeric ``-`` ``*`` ``/`` ``%`` ``**``; null on non-numbers or zero division."""
    a, b = _as_number(left), _as_number(right)
    if a is None or b is None:
        return None
    try:
        if op == "-":
            return _simplify(a - b)
        if op == "*":
            return _simplify(a * b)
        if op == "/":
            return _simplify(a / b)
        if op == "%":
            return _simplify(a % b)
        if op == "**":
            return _simplify(a ** b)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    return None


def _ordering(op: str, left: Any, right: Any) -> Any:
    """``<`` ``<=`` ``>`` ``>=``: lexical for two strings, numeric otherwise.

    Returns null (rather than raising) when an operand is not a number and the
    pair is not two strings.
    """
    if isinstance(left, str) and isinstance(right, str):
        a: Any = left
        b: Any = right
    else:
        a = _as_number(left)
        b = _as_number(right)
        if a is None or b is None:
            return None
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _as_number(value: Any) -> float | None:
    """Coerce to a float, or ``None`` if the value is not a number/numeric string."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _simplify(number: float) -> float | int:
    """Present integral floats as ``int`` (so ``4 / 2`` reads as ``2``)."""
    if isinstance(number, float) and number.is_integer():
        return int(number)
    return number


# Dispatch by node type. Defined after the handlers so the functions exist.
_NODE_HANDLERS = {
    Array: _eval_array,
    Object: _eval_object,
    Property: _eval_property,
    Element: _eval_element,
    Function: _eval_function,
    RightOp: _eval_rightop,
    BinOp: _eval_binop,
}
