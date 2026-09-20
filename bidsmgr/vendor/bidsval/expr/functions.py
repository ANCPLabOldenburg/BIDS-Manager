"""Helper functions and value coercions for the BIDS schema expression language.

The BIDS schema's expressions use a small set of built-in functions (``exists``,
``match``, ``length``, ``sorted`` ...) and rely on a specific value model:
JavaScript-style truthiness and null propagation. This module implements both,
mirroring the canonical JavaScript implementation in the reference validator
(``bids-validator``'s ``schema/expressionLanguage.ts``) so that selectors and
checks evaluate identically.

Two semantics are easy to get wrong coming from Python and are worth calling out:

* **Truthiness.** In JavaScript an empty array ``[]`` and an empty object ``{}``
  are *truthy*; in Python they are falsy. :func:`truthy` follows JavaScript.
* **Null propagation.** ``null`` (Python ``None``) flowing into a function
  generally yields ``null`` again, with per-function exceptions (e.g.
  ``intersects`` returns ``False``). The exact behaviour is fixed by the
  schema's own ``meta.expression_tests`` and tested there.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from functools import cmp_to_key
from typing import Any

# A resolver the host can install on the context under this key to give ``exists``
# access to the file tree. Until the file-tree layer lands, ``exists`` simply
# reports "not found" (0), which matches the pure-expression test cases.
EXISTS_RESOLVER_KEY = "__exists_resolver__"


# ---------------------------------------------------------------------------
# Value model: truthiness and JavaScript-style coercions
# ---------------------------------------------------------------------------


def truthy(value: Any) -> bool:
    """Return JavaScript truthiness for ``value``.

    Note the differences from Python: empty lists and dicts are truthy; the
    empty string, ``0``, ``NaN`` and ``None`` are falsy.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return len(value) > 0
    # Arrays, objects and any other reference type are truthy in JavaScript.
    return True


def js_string(value: Any) -> str:
    """Coerce a value to a string the way JavaScript's ``String()`` would.

    Used where the schema relies on stringification, e.g. building a regular
    expression out of a non-string argument.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer():
            return str(int(value))  # JavaScript prints 1.0 as "1"
    return str(value)


def to_number(value: Any) -> float:
    """Coerce a value to a number the way JavaScript's ``Number()`` would.

    Non-numeric strings (such as the BIDS ``n/a``) become ``NaN`` so callers can
    filter them out, matching the reference ``min``/``max``/``sorted`` behaviour.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan


def _simplify_number(number: float) -> float | int:
    """Return an ``int`` for integral floats, else the float unchanged.

    Keeps results like ``min([-1, 'n/a', 1])`` reading as ``-1`` rather than
    ``-1.0`` without changing equality.
    """
    if isinstance(number, float) and number.is_integer():
        return int(number)
    return number


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------


def type_of(value: Any) -> str:
    """The schema ``type`` function: a JavaScript-style type tag."""
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object"


def intersects(left: Any, right: Any) -> Any:
    """Return the intersection of two lists, or ``False`` when there is none.

    Single values are tolerated by treating them as one-element lists. The order
    of the result follows the longer input, matching the reference behaviour.
    """
    a = left if isinstance(left, list) else [left]
    b = right if isinstance(right, list) else [right]
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return False
    # Membership via ``==`` (not hashing) so unhashable or mixed values are safe.
    intersection = [item for item in a if item in b]
    return intersection if intersection else False


def match(target: Any, pattern: Any) -> Any:
    """Regular-expression search. ``null`` target yields ``null``.

    A non-string ``pattern`` is stringified first (so ``null`` becomes the
    literal pattern ``"null"``), matching the reference implementation.
    """
    if target is None:
        return None
    try:
        return re.search(js_string(pattern), js_string(target)) is not None
    except re.error:  # an invalid pattern degrades to null rather than raising
        return None


def substr(value: Any, start: Any, end: Any) -> Any:
    """Substring from ``start`` (inclusive) to ``end`` (exclusive).

    Any ``null`` argument yields ``null``.
    """
    if value is None or start is None or end is None:
        return None
    try:
        return js_string(value)[int(start):int(end)]
    except (TypeError, ValueError):  # non-integer bounds degrade to null
        return None


def min_(values: Any) -> Any:
    """Minimum of a list, ignoring entries that are not numbers (e.g. ``'n/a'``).

    ``null`` -> ``null``.
    """
    numbers = _numeric_entries(values)
    if numbers is None or not numbers:
        return None
    return _simplify_number(min(numbers))


def max_(values: Any) -> Any:
    """Maximum of a list, ignoring entries that are not numbers (e.g. ``'n/a'``).

    ``null`` -> ``null``.
    """
    numbers = _numeric_entries(values)
    if numbers is None or not numbers:
        return None
    return _simplify_number(max(numbers))


def _numeric_entries(values: Any) -> list[float] | None:
    """Coerce ``values`` to numbers, dropping ``NaN``; ``None`` if input is null."""
    if values is None:
        return None
    items = values if isinstance(values, list) else [values]
    return [n for n in (to_number(v) for v in items) if not math.isnan(n)]


def length(value: Any) -> Any:
    """Length of a list or string; ``null`` for anything else (incl. ``null``)."""
    if isinstance(value, (list, str)):
        return len(value)
    return None


def unique(values: Any) -> Any:
    """First-seen-order de-duplication of a list. ``null`` -> ``null``.

    Uses ``==`` for equality, so ``1`` and ``1.0`` collapse and the first
    occurrence (with its type) is kept.
    """
    if values is None:
        return None
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def count(values: Any, target: Any) -> int:
    """Number of elements in ``values`` equal to ``target``."""
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if value == target)


def index(values: Any, item: Any) -> Any:
    """Index of the first ``item`` in ``values``, or ``null`` if absent."""
    if not isinstance(values, list):
        return None
    try:
        return values.index(item)
    except ValueError:
        return None


def allequal(left: Any, right: Any) -> bool:
    """True if both are lists of equal length with element-wise equality."""
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    return len(left) == len(right) and all(a == b for a, b in zip(left, right, strict=False))


def sorted_(values: Any, method: str = "auto") -> Any:
    """Return a new sorted list. ``null`` -> ``null``.

    ``method`` selects the comparison: ``"numeric"`` (numeric order, with
    non-numbers left in place), ``"lexical"`` (string order), or ``"auto"``
    (natural ordering of the values). A stable sort is used so that entries the
    comparison treats as equal keep their original order, matching the reference.
    """
    if not isinstance(values, list):
        return None  # null, or any non-list input, has nothing to sort
    if method == "numeric":
        return _numeric_sorted(values)
    comparator = _lexical_comparator if method == "lexical" else _auto_comparator
    return sorted(values, key=cmp_to_key(comparator))


def _numeric_sorted(values: list[Any]) -> list[Any]:
    """Sort the numeric entries of ``values`` while non-numbers hold their slots.

    Non-numeric entries (for example the BIDS ``"n/a"``) keep their original
    absolute position; the numeric entries are sorted (stably, by numeric value)
    and placed back into the remaining slots in order. This matches the reference
    behaviour for ``sorted(..., "numeric")``.

    Why not a ``cmp_to_key`` comparator: a comparator that maps every ``NaN``
    comparison to "equal" is not a valid total order (it claims ``n/a == 1`` and
    ``n/a == 2`` yet ``1 < 2``), so the sort result depends on the interpreter's
    list-sort merge policy. It happens to match the reference under CPython
    3.10's policy but not under the 3.11+ powersort policy, which silently
    changed the placement of ``n/a`` entries. This position-preserving
    implementation is deterministic across every supported Python version.
    """
    positions = [i for i, value in enumerate(values) if not math.isnan(to_number(value))]
    ordered = sorted((values[i] for i in positions), key=to_number)
    result = list(values)
    for position, value in zip(positions, ordered, strict=True):
        result[position] = value
    return result


def _lexical_comparator(a: Any, b: Any) -> int:
    sa, sb = js_string(a), js_string(b)
    return (sa > sb) - (sa < sb)


def _auto_comparator(a: Any, b: Any) -> int:
    try:
        return (a > b) - (a < b)
    except TypeError:  # mixed types: fall back to string comparison
        sa, sb = js_string(a), js_string(b)
        return (sa > sb) - (sa < sb)


def exists(values: Any, rule: str = "dataset", context: Mapping[str, Any] | None = None) -> int:
    """Count how many of ``values`` exist as paths, per the given ``rule`` mode.

    ``rule`` is one of ``"dataset"``, ``"subject"``, ``"stimuli"``,
    ``"bids-uri"`` or ``"file"``. Resolution needs a file tree, which the host
    supplies by installing a callable on the context under
    :data:`EXISTS_RESOLVER_KEY`. Without it (or for ``null``/empty input) the
    function reports ``0``, which is the correct answer for the pure-expression
    cases and a safe default until the file-tree layer is wired in.
    """
    if values is None:
        return 0
    items = values if isinstance(values, list) else [values]
    if not items:
        return 0
    resolver = None
    if isinstance(context, Mapping):
        resolver = context.get(EXISTS_RESOLVER_KEY)
    if resolver is None:
        return 0
    return sum(1 for item in items if resolver(item, rule))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Each entry receives the already-evaluated argument list and the context.
# Keeping a single table makes the supported function set explicit and easy to
# audit against the schema.
_FUNCTIONS: dict[str, Callable[[list[Any], Mapping[str, Any] | None], Any]] = {
    "type": lambda args, ctx: type_of(args[0]),
    "intersects": lambda args, ctx: intersects(args[0], args[1]),
    "match": lambda args, ctx: match(args[0], args[1]),
    "substr": lambda args, ctx: substr(args[0], args[1], args[2]),
    "min": lambda args, ctx: min_(args[0]),
    "max": lambda args, ctx: max_(args[0]),
    "length": lambda args, ctx: length(args[0]),
    "unique": lambda args, ctx: unique(args[0]),
    "count": lambda args, ctx: count(args[0], args[1]),
    "index": lambda args, ctx: index(args[0], args[1]),
    "allequal": lambda args, ctx: allequal(args[0], args[1]),
    "sorted": lambda args, ctx: sorted_(args[0], args[1] if len(args) > 1 else "auto"),
    "exists": lambda args, ctx: exists(args[0], args[1] if len(args) > 1 else "dataset", ctx),
}


def is_known(name: str) -> bool:
    """Whether ``name`` is a built-in the evaluator can call."""
    return name in _FUNCTIONS


def call(name: str, args: list[Any], context: Mapping[str, Any] | None) -> Any:
    """Invoke built-in ``name`` with already-evaluated ``args``.

    Raises :class:`KeyError` for an unknown function; the evaluator turns that
    into its own, more descriptive error so an unfamiliar function in a newer
    schema degrades gracefully rather than crashing the run.
    """
    return _FUNCTIONS[name](args, context)
