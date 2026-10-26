"""Entities the operator wrote into a sequence's name, winning over ours.

A scan proposes each name's entities itself: ``acq`` from the sequence
(dcm2niix's guess, ``tfl3p2``), ``run`` by order of acquisition, ``dir`` from
the phase encoding, ``task`` from a list of known words. An operator who
names a series ``func_task-nback_acq-mb4_run-02`` has already decided some of
them, sometimes on purpose against what the scanner would suggest. For the
entities the user chooses here, what the name says wins:

- the series description (DICOM SeriesDescription), else the protocol name
  (ProtocolName) is read for ``key-value`` tokens, separated by ``_``, spaces,
  ``/`` or ``.``;
- a value is used only when it is valid for the entity in the selected BIDS
  version (a label is letters and digits, an index digits) and the entity is
  allowed for the recording's datatype and suffix;
- an entity the name does not mention keeps the scan's value.

Nothing here changes which recordings exist or what they are (datatype,
suffix): only the values of the chosen entities.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

#: Entities a sequence name may decide, in the order offered. ``sub`` and
#: ``ses`` are not: they come from who and when, and a name typo would move a
#: recording to another person.
OFFERED: tuple[str, ...] = ("task", "acq", "ce", "rec", "dir", "run", "echo", "part",
                            "inv", "mt", "flip", "mod", "trc")

_SPLIT = re.compile(r"[_\s/.]+")


def offered() -> list[str]:
    """The entities offered, as the selected BIDS version knows them."""
    from .. import schema

    known = set(schema.entity_keys())
    return [e for e in OFFERED if e in known]


def label_of(short: str) -> str:
    """``acq`` -> ``acquisition``: the long name the classifiers key on."""
    from .. import schema

    for long_name in schema.entity_order():
        try:
            if schema.entity_info(long_name).name == short:
                return long_name
        except KeyError:
            continue
    return short


def _valid(short: str, value: str) -> bool:
    """``value`` is valid for the entity in the selected BIDS version: its
    format (a label is letters and digits, an index digits) and, where the
    standard lists them, one of its values (``part``: mag, phase, real, imag)."""
    from .. import schema

    try:
        pattern = schema.entity_key_info(short).format.pattern
    except (KeyError, AttributeError):
        return False
    if not value or re.fullmatch(pattern, value) is None:
        return False
    try:
        enum = schema.get_schema().objects.entities[label_of(short)].get("enum")
    except (KeyError, AttributeError, TypeError):
        enum = None
    return not enum or value in enum


def from_name(*texts: str) -> dict[str, str]:
    """``{short key: value}`` written in ``texts`` (the first text that
    states an entity wins), for the entities a name may decide, valid values
    only."""
    found: dict[str, str] = {}
    keys = set(OFFERED)
    for text in texts:
        for token in _SPLIT.split(text or ""):
            key, dash, value = token.partition("-")
            key = key.lower()
            if dash and key in keys and key not in found and _valid(key, value):
                found[key] = value
    return found


def apply(entities: Mapping[str, str], datatype: str, suffix: str,
          wanted: Iterable[str], *texts: str) -> tuple[dict[str, str], dict[str, str]]:
    """``(entities, changed)`` with the wanted entities the name states laid
    over ``entities`` (LONG keys, as the classifiers use them). ``changed``
    maps each long key that moved to its new value."""
    from .. import schema

    wanted = set(wanted)
    stated = {k: v for k, v in from_name(*texts).items() if k in wanted}
    out = dict(entities)
    if not stated:
        return out, {}
    try:
        allowed = set(schema.allowed_entities(datatype, suffix))
    except (KeyError, ValueError):
        allowed = set()
    changed: dict[str, str] = {}
    for short, value in stated.items():
        long = label_of(short)
        if allowed and long not in allowed:
            continue
        if out.get(long) != value:
            out[long] = value
            changed[long] = value
    return out, changed


def describe(wanted: Iterable[str]) -> str:
    keys = [k for k in OFFERED if k in set(wanted)]
    return ", ".join(keys) if keys else "none"


def parse_list(value: Optional[str]) -> list[str]:
    """A comma list (``"acq,task"``) as offered entities, unknown ones refused."""
    keys = [k.strip().lower() for k in (value or "").split(",") if k.strip()]
    unknown = [k for k in keys if k not in OFFERED]
    if unknown:
        raise ValueError(f"not an entity a name can decide: {', '.join(unknown)} "
                         f"(choose from {', '.join(OFFERED)})")
    return [k for k in OFFERED if k in keys]


__all__ = ["OFFERED", "apply", "describe", "from_name", "label_of", "offered", "parse_list"]
