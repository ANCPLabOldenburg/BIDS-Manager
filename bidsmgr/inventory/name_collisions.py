"""Two recordings must never claim the same BIDS name.

When they do, the second one overwrites the first and the dataset quietly loses
a scan. Nothing in the pipeline used to check, and the first time it surfaced
loudly was three ECAT phantoms that all resolved to ``sub-014_pet``: they
collided in staging, every one of them failed, and the only sign was three
ordinary per-task warnings.

**Nothing is written into the inventory.** A clash is a property of the whole
table as it stands, so a note stamped into a cell is stale the moment an entity
is edited: the row stays red after the user has fixed it, which is worse than
saying nothing. The scan reports the count, the GUI derives the state live from
the current names, and the conversion checks again over the rows it is about to
write. Three readings of the same live question, no stored answer to go stale.

**Number what BIDS has an answer for.** ``run`` is the entity for otherwise
identical acquisitions, so when a group clashes and the schema allows ``run``
for that datatype, they are numbered. Deterministically, by acquisition time
where it is known and by source path otherwise, so re-scanning the same tree
gives the same numbers rather than shuffling filenames under the user.

**Say so for the rest, and let the user decide.** A run is not always the right
answer, and it is never the right answer over the top of one somebody already
set. Where the schema forbids ``run``, or a run is already stated and the names
still clash, the tool stops: only the person who ran the study knows whether two
recordings are two runs, two tasks or two sessions. Those rows show red and
conversion refuses.

**Refuse to overwrite.** The check runs again at conversion, over the rows about
to be written, because a TSV can be edited by hand after the scan.

Qt-free, and used by both the CLI and the GUI through the scan.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Shown on a clashing row, by the GUI, from the live state. Not stored.
DUPLICATE_ISSUE = (
    "Another included recording wants this same BIDS name, so one would "
    "overwrite the other. Give them different entities, or exclude one. Which "
    "entity depends on what differs: a task label for different tasks, a "
    "session for different days, a run for genuine repeats."
)


class NameCollisionError(ValueError):
    """Two recordings claim one name, so writing would lose one of them.

    Its own type so the CLI can report it as the ordinary, actionable refusal
    it is rather than as a crash: the user has to change an entity, and a
    traceback tells them nothing about which.
    """


def destination_key(row) -> tuple:
    """What two rows must not share.

    The full destination rather than the basename alone: the same basename
    under a different dataset is a different file. Session and datatype are
    already inside the basename, and are included because a hand-edited row can
    disagree with itself.
    """
    return (
        str(row.get("dataset", "") or "").strip(),
        str(row.get("proposed_datatype", "") or "").strip(),
        str(row.get("proposed_basename", "") or "").strip(),
    )


def _is_included(row) -> bool:
    return str(row.get("include", "1")).strip() not in ("0", "False", "false", "")


def find_collisions(df) -> dict[tuple, list]:
    """``{destination: [row index, ...]}`` for every name claimed more than once.

    Only included rows, and only rows that actually have a name: a row with no
    basename has not been given one yet, which is a different problem.
    """
    seen: dict[tuple, list] = {}
    if df is None or not len(df):
        return {}

    for idx, row in df.iterrows():
        if not _is_included(row):
            continue
        key = destination_key(row)
        if not key[2]:
            continue
        seen.setdefault(key, []).append(idx)

    return {key: rows for key, rows in seen.items() if len(rows) > 1}


def _run_is_allowed(datatype: str, suffix: str) -> bool:
    """Does BIDS let this kind of file carry a ``run``?

    Asked of the schema rather than assumed: the answer differs by datatype and
    changes between versions.
    """
    try:
        from .. import schema as schema_mod

        allowed = schema_mod.allowed_entities(datatype, suffix)
    except Exception:  # noqa: BLE001
        return False
    return any(str(name) == "run" for name in (allowed or ()))


def _entities(row) -> dict:
    import json

    raw = row.get("entities", "")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _order_for_numbering(df, indices: list) -> list:
    """A stable order for the runs about to be numbered.

    Acquisition time first, so run-1 is the one acquired first. Falling back to
    the source path keeps it deterministic for formats that record no time, so
    a re-scan gives the same numbers rather than renaming files under the user.
    """
    def sort_key(idx):
        row = df.loc[idx]
        for column in ("acq_time", "AcquisitionTime", "StudyDate", "scan_start"):
            value = str(row.get(column, "") or "").strip()
            if value:
                return (0, value, str(row.get("source_file", "") or ""))
        return (1, "", str(row.get("source_file", "") or str(idx)))

    return sorted(indices, key=sort_key)


def assign_runs(df) -> int:
    """Number clashing rows as runs where BIDS allows it. Returns how many.

    Modifies ``df`` in place, and only where it is safe:

    * the schema must allow ``run`` for that datatype and suffix, and
    * NO row in the group may already carry one. A run somebody set by hand is
      a stated answer, and renaming over the top of a stated answer is the very
      thing this module exists to prevent.

    Whatever is left clashing is not touched, and shows red instead.
    """
    import json

    n_numbered = 0
    for key, indices in find_collisions(df).items():
        datatype = key[1]
        suffix = str(df.loc[indices[0]].get("bids_guess_suffix", "") or "").strip()

        if any(_entities(df.loc[i]).get("run") for i in indices):
            continue
        if not _run_is_allowed(datatype, suffix):
            continue

        for number, idx in enumerate(_order_for_numbering(df, indices), start=1):
            entities = _entities(df.loc[idx])
            entities["run"] = str(number)
            df.at[idx, "entities"] = json.dumps(entities, sort_keys=True)
            if "run" in df.columns:
                df.at[idx, "run"] = str(number)
            n_numbered += 1
        log.info(
            "named %d recordings run-1..run-%d; they all resolved to %s",
            len(indices), len(indices), key[2],
        )

    if n_numbered:
        # The basenames still say what they said before the run was added.
        from .rebuild import rebuild_from_entities

        rebuild_from_entities(df, in_place=True)
    return n_numbered


def count_collisions(df) -> int:
    """How many rows still want a name another row also wants.

    Reports, and changes nothing. Called after :func:`assign_runs`, so what it
    counts is what could not be resolved automatically. Nothing is written into
    the inventory: a note stored in a cell cannot tell when the user has fixed
    the thing it describes.
    """
    n = 0
    for key, indices in find_collisions(df).items():
        n += len(indices)
        log.warning(
            "%d recordings claim the name %s; one would overwrite the other",
            len(indices), key[2],
        )
    return n


def describe_collisions(df) -> Optional[str]:
    """A message naming what collides, for a caller that must refuse to write.

    ``None`` when every name is unique.
    """
    collisions = find_collisions(df)
    if not collisions:
        return None

    lines = []
    for key, indices in sorted(collisions.items()):
        sources = [
            str(df.loc[i].get("source_file", "") or df.loc[i].get("source_folder", ""))
            for i in indices
        ]
        lines.append(f"  {key[2]} is claimed by {len(indices)} recordings:")
        lines.extend(f"    {source}" for source in sources)
    return (
        "refusing to convert: these recordings would overwrite one another.\n"
        + "\n".join(lines)
        + "\nGive them different entities, or exclude all but one. Which "
        "entity depends on what actually differs: a task label if they are "
        "different tasks, a session if they were acquired on different days, "
        "a run if they are genuine repeats of the same thing."
    )


__all__ = [
    "DUPLICATE_ISSUE",
    "NameCollisionError",
    "assign_runs",
    "count_collisions",
    "describe_collisions",
    "destination_key",
    "find_collisions",
]
