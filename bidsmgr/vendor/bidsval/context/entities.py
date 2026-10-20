"""Parse a BIDS filename into entities, suffix, and extension.

Entities are keyed by their short name as they appear in the filename
(``sub-01_acq-hi_T1w.nii.gz`` -> ``{"sub": "01", "acq": "hi"}``). This matches the
reference validator, which keys the context's ``entities`` by short name only.
(The schema does reference a few entities by long name, e.g. ``entities.density``;
the reference leaves those undefined so the corresponding rules are skipped, and
``bidsval`` follows the same behaviour for parity.)
"""

from __future__ import annotations

from bidsschematools.types.namespace import Namespace

from ..schema import introspect


# VENDORED CHANGE (speed): a name parsed once per schema. Finding a file's
# associations and sidecars re-parsed every name in its folder for every file,
# 4.5 million parses on a 1,500-file folder, and validation grew with the
# square of the files in a folder. Bounded, so a long session cannot grow it
# without limit.
_PARSED: dict[tuple[int, str], tuple[dict[str, str], str, str]] = {}
_PARSED_LIMIT = 500_000


def parse_filename(schema: Namespace, name: str) -> tuple[dict[str, str], str, str]:
    """Return ``(entities, suffix, extension)`` for a filename.

    ``entities`` is keyed by short name. A token without a hyphen in the final
    position is the suffix; malformed tokens (no hyphen, not final) are ignored
    here and flagged by basename validation. The entities are a fresh dict
    each call, so a caller may change it.
    """
    key = (id(schema), name)
    hit = _PARSED.get(key)
    if hit is None:
        if len(_PARSED) > _PARSED_LIMIT:
            _PARSED.clear()
        hit = _PARSED[key] = _parse(schema, name)
    return dict(hit[0]), hit[1], hit[2]


def _parse(schema: Namespace, name: str) -> tuple[dict[str, str], str, str]:
    stem, extension = introspect.split_extension(schema, name)
    parts = stem.split("_")

    suffix = ""
    if parts and "-" not in parts[-1]:
        suffix = parts[-1]
        parts = parts[:-1]

    entities: dict[str, str] = {}
    for part in parts:
        short, sep, value = part.partition("-")
        if sep:  # a well-formed "key-value" token
            entities[short] = value
    return entities, suffix, extension
