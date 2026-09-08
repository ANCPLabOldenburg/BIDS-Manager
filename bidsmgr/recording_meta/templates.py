"""Per-sequence metadata templates: the scope between dataset and recording.

BIDS Manager has long had two useful scopes for metadata, the whole dataset and
one row, and most of what a study actually knows sits between them. Every bold
run in a study shares a task description and a set of instructions; the T1w runs
share none of that with them. Stating such a fact at dataset level puts it on
files it does not describe, and stating it per row means saying it once per
recording.

A template is keyed ``"<datatype>/<suffix>"``, optionally narrowed to a single
task with ``"<datatype>/<suffix>@<task>"``. The general key applies first and
the task-specific one refines it.

Nothing here reads or writes a file: resolving which fields apply is a pure
function of the spec and the recording's identity, so the enrichment fixup can
ask for an answer and the GUI can preview one.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import RecordingMetaSpec

# Separates the task qualifier from the datatype/suffix it narrows.
_TASK_SEP = "@"


def template_key(datatype: str, suffix: str, task: Optional[str] = None) -> str:
    """The canonical key for a scope, e.g. ``func/bold`` or ``func/bold@rest``."""
    base = f"{datatype}/{suffix}"
    return f"{base}{_TASK_SEP}{task}" if task else base


def parse_template_key(key: str) -> tuple[str, str, Optional[str]]:
    """``"func/bold@rest"`` -> ``("func", "bold", "rest")``.

    A malformed key yields empty parts rather than raising: the scaffold is a
    file a user may hand-edit, and one bad line must not stop the rest loading.
    """
    head, _, task = key.partition(_TASK_SEP)
    datatype, _, suffix = head.partition("/")
    return datatype.strip(), suffix.strip(), (task.strip() or None)


def resolve_sequence_template(
    spec: Optional[RecordingMetaSpec],
    datatype: str,
    suffix: str,
    task: Optional[str] = None,
) -> dict[str, Any]:
    """The fields a recording of this kind inherits from the templates.

    Least specific first, so a task-specific template refines the general one
    for that datatype/suffix rather than replacing it wholesale.
    """
    if spec is None or not spec.sequence_templates:
        return {}

    out: dict[str, Any] = {}
    for key in (
        template_key(datatype, suffix),
        template_key(datatype, suffix, task) if task else None,
    ):
        if key is None:
            continue
        values = spec.sequence_templates.get(key)
        if isinstance(values, dict):
            out.update(values)
    return out


def validate_sequence_templates(
    spec: Optional[RecordingMetaSpec], field_applies=None,
) -> list[str]:
    """Problems with the templates, as human-readable lines. Empty when clean.

    Two kinds of problem are worth telling a user about. A key that names no
    real datatype or suffix will silently never match anything, which looks
    exactly like the template being ignored. And a field the datatype does not
    accept would be written into a sidecar the standard says must not carry it,
    which is the mistake this whole layer exists to avoid; it is refused at
    apply time regardless, but a user who typed it deserves to hear why it had
    no effect.

    ``field_applies`` is injected so this module stays free of the schema layer
    and remains a pure-data unit; the default resolves it lazily.
    """
    if spec is None or not spec.sequence_templates:
        return []

    if field_applies is None:
        from .. import schema as schema_mod

        field_applies = schema_mod.field_applies

    problems: list[str] = []
    for key, values in spec.sequence_templates.items():
        datatype, suffix, _task = parse_template_key(key)
        if not datatype or not suffix:
            problems.append(
                f"template {key!r}: expected '<datatype>/<suffix>', "
                f"optionally '<datatype>/<suffix>@<task>'"
            )
            continue
        if not isinstance(values, dict):
            problems.append(f"template {key!r}: expected a mapping of field to value")
            continue
        for field in values:
            if not field_applies(field, datatype, suffix):
                problems.append(
                    f"template {key!r}: BIDS does not define {field!r} for "
                    f"{datatype}/{suffix}"
                )
    return problems


__all__ = [
    "parse_template_key",
    "resolve_sequence_template",
    "template_key",
    "validate_sequence_templates",
]
