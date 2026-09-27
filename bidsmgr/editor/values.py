"""Entity values across a whole dataset: what they are, and repadding them.

Rename answers "change this entity on the files I picked". The question this
answers is the other one: "change it everywhere it has this value", which is
what you want when a label is wrong across a dataset rather than in one
folder, and what you want when the width of an index is inconsistent.

Both are the same operation underneath, so neither gets its own engine.
:func:`counts` is the query that makes the first usable in a dialog, and
:func:`plan_padding` turns the second into the list of renames that performs
it. The renames themselves are :func:`bidsmgr.editor.rename.plan_rename`,
already able to move every file carrying a value and to follow every
reference to them.

**Padding is a house style, not a correctness fix.** The standard's ``index``
format accepts ``run-1`` and ``run-01`` equally, and a dataset using one is
not wrong. That is precisely why it belongs in a tool the user reaches for
rather than in the scanner: the scanner writes what the index IS, and how
wide anyone likes to see it written is a preference.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import schema as schema_mod
from .rename import RenameError, RenamePlan, entity_value, plan_rename, walk_dataset

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# What is in use


def counts(root: Path, entity: str) -> list[tuple[str, int]]:
    """Every value ``entity`` takes, with how many files carry it.

    Sorted by the value, numerically when it reads as a number, so ``run-2``
    sits between ``run-1`` and ``run-10`` rather than after them.

    The count is what makes this usable: picking a value to replace from a
    bare list means guessing whether it is the one on four hundred files or
    the one on the single file somebody mistyped.
    """
    tally: Counter[str] = Counter()
    for path in walk_dataset(Path(root)):
        value = entity_value(path.name, entity)
        if value:
            tally[value] += 1
    return sorted(tally.items(), key=lambda kv: (_sort_key(kv[0]), kv[0]))


def _sort_key(value: str) -> tuple[int, float]:
    """Numbers before words, and numbers in numeric order."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, 0.0)


def index_entities() -> list[str]:
    """Every entity the ACTIVE schema declares as an index, in BIDS order.

    Read from the schema rather than listed, so a BIDS version that adds an
    index entity is offered it without an edit here, and one that has never
    heard of ``chunk`` does not offer it. Both the Settings tab and the
    Editor dialog call this, so the two cannot disagree about what exists.
    """
    return [key for key in schema_mod.entity_keys() if is_index(key)]


def is_index(entity: str) -> bool:
    """Whether the schema declares this entity as an index rather than a label.

    Only an index can be repadded: ``acq-01`` and ``acq-1`` are two different
    labels and turning one into the other is a rename, not a reformat.
    """
    try:
        info = schema_mod.entity_key_info(entity)
    except KeyError:
        return False
    return info.format.name == "index"


# --------------------------------------------------------------------------
# Scope: which part of the dataset an edit applies to


@dataclass(frozen=True)
class Scope:
    """A part of the dataset an edit is confined to.

    ``prefix`` is empty for the whole dataset, ``sub-001`` for a subject,
    ``sub-001/ses-pre`` for a session. Matching is on the dataset-relative
    POSIX path, which is exactly the key ``RenamePlan.file_key`` reports, so
    a scope becomes an ``only`` set with no second notion of identity.
    """

    label: str
    prefix: str

    def holds(self, key: str) -> bool:
        return not self.prefix or key.startswith(self.prefix + "/")


def scopes(root: Path) -> list[Scope]:
    """Everywhere an edit could be confined to, widest first.

    The whole dataset, then each subject, then each session inside it. A
    dataset with no sessions gets two levels rather than three; nothing is
    offered that would match no files.
    """
    root = Path(root)
    out = [Scope("The whole dataset", "")]
    for subject in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        out.append(Scope(subject.name, subject.name))
        for session in sorted(p for p in subject.glob("ses-*") if p.is_dir()):
            out.append(Scope(
                f"{subject.name} / {session.name}",
                f"{subject.name}/{session.name}",
            ))
    return out


def counts_in(root: Path, entity: str, scope: Scope) -> list[tuple[str, int]]:
    """:func:`counts`, confined to ``scope``."""
    root = Path(root)
    tally: Counter[str] = Counter()
    for path in walk_dataset(root):
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not scope.holds(key):
            continue
        value = entity_value(path.name, entity)
        if value:
            tally[value] += 1
    return sorted(tally.items(), key=lambda kv: (_sort_key(kv[0]), kv[0]))


def plan_replace(
    root: Path, entity: str, old: str, new: str, scope: Scope,
) -> tuple[RenamePlan, set[str]]:
    """Replace one entity VALUE with another, inside ``scope``.

    Returns the plan and the set of file keys to apply it to. The plan is
    the ordinary dataset-wide one, because everything that makes a rename
    correct lives in it: the folder moves, the scans-row relocation, the
    references, the participants row. Confining the edit is then a matter of
    handing ``apply_rename`` the subset of keys the scope holds, which is
    the same mechanism a partial selection in the rename preview uses.

    Doing it the other way round, planning only within the scope, would
    have meant a second implementation of every one of those consequences.
    """
    plan = plan_rename(Path(root), entity, old, new)
    keys = {
        key for key in plan.file_keys(Path(root)) if scope.holds(key)
    }
    return plan, keys


# --------------------------------------------------------------------------
# Padding


@dataclass(frozen=True)
class Repad:
    """One value becoming another, purely by width."""

    old: str
    new: str
    files: int


def pad(value: str, width: int) -> str:
    """Write ``value`` at ``width`` digits: ``("2", 3)`` to ``"002"``.

    It works in BOTH directions, which is the whole point of a tool called
    "index widths": ``("001", 2)`` is ``"01"``. It used to be ``zfill``
    alone, so it could only ever add zeros, and asking a dataset written as
    ``run-001`` for two digits changed nothing and said nothing. The tool
    looked broken because half of what it claims to do was missing.

    Narrowing is not a loss. An index is a nonnegative integer, so the
    leading zeros are how wide somebody chose to write it and not part of
    the value: ``run-001`` and ``run-01`` are the same run, and the standard
    accepts either.

    **Significant digits are never dropped.** The width is a minimum, so
    ``("100", 2)`` is ``"100"`` and not ``"00"``. Somebody asking for two
    digits on a dataset that reaches a hundred runs means "two digits where
    two will do".

    Two values that would land on the same name (a dataset holding both
    ``run-1`` and ``run-01``) are refused by :func:`plan_padding`, not
    silently fused.
    """
    if not value.isdigit():
        return value
    # Strip the chosen width off first, then write the width that was asked
    # for. ``or "0"`` because ``"000".lstrip("0")`` is empty and zero is a
    # legitimate index.
    return (value.lstrip("0") or "0").zfill(width)


def plan_padding(root: Path, entity: str, width: int) -> list[Repad]:
    """Every value of ``entity`` that would change at ``width``.

    Raises :class:`RenameError` for an entity the schema does not call an
    index, and for a width outside 1 to 6. Six is well past any real run
    count and the limit exists so a typo cannot rewrite a dataset into
    ``run-0000000001``.

    Values that already have the width, and values that are not numbers at
    all, are simply absent from the result rather than listed as no-ops.
    """
    entity = entity.strip()
    if not is_index(entity):
        raise RenameError(
            f"{entity!r} is a label, not an index, so its values have no "
            "width to change. Use a rename to change a label."
        )
    if not 1 <= int(width) <= 6:
        raise RenameError("Width must be between 1 and 6.")

    out: list[Repad] = []
    seen_new: dict[str, str] = {}
    for value, n in counts(root, entity):
        new = pad(value, int(width))
        if new in seen_new:
            # Two values landing on one name. A dataset holding both ``1``
            # and ``01`` is exactly the mess this tool is for, and silently
            # fusing two runs would be the worst possible answer.
            raise RenameError(
                f"{entity}-{value} and {entity}-{seen_new[new]} would both "
                f"become {entity}-{new}. Settle them by hand first: merging "
                "two runs is not something this should decide."
            )
        # Registered BEFORE the no-op check, or a value that already has the
        # width is not there to collide with: ``01`` was skipped and then
        # ``1`` padded onto it unnoticed.
        seen_new[new] = value
        if new != value:
            out.append(Repad(old=value, new=new, files=n))
    return out


@dataclass(frozen=True)
class WidthSplit:
    """One index entity written at more than one width in one dataset."""

    entity: str
    widths: tuple[int, ...]          # every width in use, narrowest first
    examples: tuple[str, ...]        # one value per width
    files: int

    @property
    def suggested(self) -> int:
        """The widest in use. Padding up never loses a digit."""
        return max(self.widths)


def inconsistent_widths(root: Path) -> list[WidthSplit]:
    """Index entities this dataset writes at more than one width.

    ``run-1`` beside ``run-01`` is legal BIDS twice over, which is why no
    validator mentions it, and it is still the thing that makes a dataset
    sort wrongly and a glob miss half its files. This finds it; the padding
    tool settles it.

    Only NUMERIC values count toward a width. A ``run`` whose value is not a
    number is a different problem and saying so here would bury this one.
    """
    out: list[WidthSplit] = []
    for entity in schema_mod.entity_keys():
        if not is_index(entity):
            continue
        by_width: dict[int, list[tuple[str, int]]] = {}
        for value, n in counts(Path(root), entity):
            if not value.isdigit():
                continue
            by_width.setdefault(len(value), []).append((value, n))
        if len(by_width) < 2:
            continue
        widths = tuple(sorted(by_width))
        out.append(WidthSplit(
            entity=entity,
            widths=widths,
            examples=tuple(
                f"{entity}-{by_width[w][0][0]}" for w in widths
            ),
            files=sum(n for group in by_width.values() for _, n in group),
        ))
    return out


def describe_padding(entity: str, plan: list[Repad]) -> str:
    """One line for a dialog: what the padding would do."""
    if not plan:
        return f"Every {entity} value already has that width."
    files = sum(r.files for r in plan)
    shown = ", ".join(f"{entity}-{r.old} to {entity}-{r.new}" for r in plan[:3])
    more = ", ..." if len(plan) > 3 else ""
    return f"{len(plan)} value(s) across {files} file(s): {shown}{more}"


__all__ = [
    "Repad",
    "counts",
    "describe_padding",
    "is_index",
    "pad",
    "plan_padding",
]
