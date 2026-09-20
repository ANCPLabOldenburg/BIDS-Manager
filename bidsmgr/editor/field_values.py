"""What a field holds when the user clears it, and what shape a value takes.

Emptying a box is an ordinary thing to do, and until now it wrote JSON
``null``. That is not "empty", it is a value, and it is the one value BIDS
never accepts: measured against the validator, ``{"Authors": null}`` is an
ERROR while ``{"Authors": []}`` and an absent ``Authors`` are both clean. So
clearing a field produced a violation in a file the user was tidying.

The empty form of a field comes from its declared type, which means it comes
from the schema and changes with the schema version in force:

* ``array`` becomes ``[]``
* ``string`` becomes ``""``
* ``object`` becomes ``{}``
* a number or a boolean has no empty form. There is no numeral meaning
  "unanswered", so the key is REMOVED instead. An absent field is always
  valid; a zero would be a value nobody stated.

The same table answers a second question the Editor needs: given text a user
typed and a field the schema describes, what Python value should be written.
Both live here so a form, a bulk edit and a template cannot disagree about it.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

# Returned when a field has no empty form and clearing it means removing the
# key. Distinct from ``None``, which is a value the caller might legitimately
# want to write, and distinct from any empty container.
REMOVE = object()

# The empty form of each JSON type. A number and a boolean are deliberately
# absent: they have none.
_EMPTY_BY_TYPE: dict[str, Any] = {
    "array": [],
    "string": "",
    "object": {},
}


def _types_of(field) -> tuple[str, ...]:
    """Every JSON type the field accepts, widest information first.

    Prefers ``accepts``, which resolves ``anyOf``; falls back to the plain
    declared type for a spec that predates it.
    """
    accepts = tuple(getattr(field, "accepts", ()) or ())
    if accepts:
        return accepts
    declared = getattr(field, "type", "") or ""
    return (declared,) if declared else ()


def empty_value_for(field) -> Any:
    """What to write when the user clears this field.

    Returns :data:`REMOVE` when the field has no empty form, meaning the key
    should be deleted rather than given a value.
    """
    if field is None:
        # Nothing describes this field, so nothing justifies inventing a
        # value for it. Removing the key is the only safe reading of "empty".
        return REMOVE
    free_text = bool(getattr(field, "accepts_free_text", False))
    # A field that accepts several shapes empties into the first one whose
    # empty value is actually VALID for it, in the order the schema lists
    # them: IntendedFor (a string or an array of them) empties to "" rather
    # than to []. A string variant constrained by an enum does NOT count,
    # because "" is not one of its values. PowerLineFrequency is a number or
    # the string "n/a", and emptying it to "" would be a type error, which is
    # the whole thing this module exists to avoid.
    for kind in _types_of(field):
        if kind == "string" and not free_text:
            continue
        if kind in _EMPTY_BY_TYPE:
            value = _EMPTY_BY_TYPE[kind]
            return list(value) if isinstance(value, list) else (
                dict(value) if isinstance(value, dict) else value
            )
    if getattr(field, "accepts_na", False) or "n/a" in tuple(
        getattr(field, "enum", ()) or ()
    ):
        # No empty form, but the standard does accept "no answer" here, which
        # is a better reading of "cleared" than deleting the key.
        return "n/a"
    return REMOVE


def apply_empty(data: dict, name: str, field) -> bool:
    """Clear ``name`` in ``data`` the way its type says to.

    Returns ``True`` when the mapping changed. Centralised so every caller
    that clears a field does it identically.
    """
    value = empty_value_for(field)
    if value is REMOVE:
        return data.pop(name, REMOVE) is not REMOVE
    if name in data and data[name] == value:
        return False
    data[name] = value
    return True


def is_empty_text(text: Optional[str]) -> bool:
    """Did the user actually clear the box?

    Whitespace counts as cleared. A literal ``"null"`` does not: somebody who
    types the word means the word, and turning it into a JSON null is how the
    thing this module exists to prevent used to happen.
    """
    return text is None or not str(text).strip()


def coerce_text(text: str, field) -> Any:
    """The Python value for text the user typed into ``field``.

    Schema-first: the field's declared type decides, and only when the schema
    says nothing does this fall back to guessing from the text. Guessing is
    what turned ``"60"`` in a free-text field into the number 60.
    """
    if is_empty_text(text):
        return empty_value_for(field)

    text = text.strip()
    types = _types_of(field)

    if "number" in types or "integer" in types:
        number = _as_number(text)
        if number is not None:
            return number
        # Not a number. If a string is also allowed the text stands; if not,
        # it is returned anyway so the validator can say so, rather than this
        # silently dropping what the user typed.
        return text

    if "string" in types and "array" not in types and "object" not in types:
        return text

    if "array" in types:
        item_type = getattr(field, "item_type", "")
        parsed = _try_json(text)
        if isinstance(parsed, list):
            return parsed
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if item_type in ("number", "integer"):
            numbers = [_as_number(p) for p in parts]
            if all(n is not None for n in numbers):
                return numbers
        return parts or [text]

    if "object" in types:
        parsed = _try_json(text)
        return parsed if isinstance(parsed, dict) else text

    # The schema declares nothing usable. Fall back to reading the literal,
    # which is what lets a user type ``true`` or ``3`` into an untyped field.
    parsed = _try_json(text)
    return text if parsed is None else parsed


def _as_number(text: str) -> Optional[Any]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value.is_integer() and "." not in text and "e" not in text.lower():
        return int(value)
    return value


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


__all__ = [
    "REMOVE",
    "apply_empty",
    "coerce_text",
    "empty_value_for",
    "is_empty_text",
]
