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


def parse_filename(schema: Namespace, name: str) -> tuple[dict[str, str], str, str]:
    """Return ``(entities, suffix, extension)`` for a filename.

    ``entities`` is keyed by short name. A token without a hyphen in the final
    position is the suffix; malformed tokens (no hyphen, not final) are ignored
    here and flagged by basename validation.
    """
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
