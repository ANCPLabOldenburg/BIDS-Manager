"""Finding the undefaced copy of an image, so the result can be inspected.

Every defacing tool tells the user to check the output, and then gives them no
way to. The advice matters: a defacer that took part of the cerebellum with the
face has ruined the data, and a defacer that missed the nose has not protected
anybody. Both are obvious in a second if you can see the two images together,
and invisible otherwise.

We can offer that because we already keep the original. Two places hold one,
for two different reasons:

``sourcedata/``
    Written only when the user asked for it, and never overwritten by a second
    run, so it is the pristine image however many times the dataset has been
    defaced. It is also the copy that still contains the face, which is why it
    is opt-in.

``.bidsmgr/editor/originals/<op_id>/``
    Written by every Editor operation, because that is what makes undo work.
    The EARLIEST operation holding a given path is the pristine one: a second
    defacing of the same image saves the already-defaced version.

There is a third case with no original at all, and it is not a failure: an
image defaced during conversion never had an undefaced version inside the
dataset, which is the entire point of doing it there. Saying that plainly is
more useful than an empty pane, so :func:`explain_missing` exists.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..project.operations import ORIGINALS_DIR, read_log
from . import status

log = logging.getLogger(__name__)

SOURCEDATA = "sourcedata"


@dataclass(frozen=True)
class Original:
    """An undefaced copy of one image, and where it came from."""

    path: Path
    # "sourcedata" or "history".
    source: str
    # Only for "history": which operation saved it, and what it was called.
    op_id: str = ""
    label: str = ""

    @property
    def description(self) -> str:
        """One line naming the copy, for a header the user reads."""
        if self.source == SOURCEDATA:
            return "the copy kept in sourcedata/"
        if self.label:
            return f"the copy kept when you ran: {self.label}"
        return "the copy kept in the edit history"


def original_for(root: Path, rel: str) -> Optional[Original]:
    """The undefaced copy of ``rel``, or ``None`` if the dataset has none.

    ``sourcedata/`` wins when both exist. It is never overwritten, so it is
    pristine by construction, whereas the history only goes back as far as the
    operations still in the log.
    """
    root = Path(root)
    rel = str(rel).replace("\\", "/")

    mirror = root / SOURCEDATA / rel
    if mirror.is_file():
        return Original(path=mirror, source=SOURCEDATA)

    # Oldest first: the first operation that saved this path holds the version
    # from before anything was done to it.
    for record in read_log(root):
        op_id = record.get("op_id", "")
        if not op_id:
            continue
        if not any(
            str(child.get("path", "")).replace("\\", "/") == rel
            for child in record.get("children", [])
        ):
            continue
        kept = root / ORIGINALS_DIR / op_id / rel
        if kept.is_file():
            return Original(
                path=kept, source="history", op_id=op_id,
                label=str(record.get("label", "")),
            )
    return None


def explain_missing(root: Path, rel: str) -> str:
    """Why there is no original to compare against. Never blank.

    Three different situations, and the user needs to tell them apart: an image
    that was defaced at conversion time is working exactly as intended, an
    image that was never defaced has nothing to compare, and an image whose
    history has been cleared is a real loss.
    """
    root = Path(root)
    target = root / rel
    if not target.is_file():
        return f"{rel} is not in this dataset."

    doc = status.read_sidecar(status.sidecar_for(target))
    if status.defaced_by_others(doc) and not status.defaced_by_us_at_all(doc):
        return (
            f"{rel} records that it was defaced by another tool, so BIDS "
            "Manager never held an undefaced copy of it."
        )
    if not status.defaced_by_us_at_all(doc):
        return (
            f"{rel} has not been defaced, so there is nothing to compare it "
            "against. Deface it first, from Tools."
        )
    return (
        f"{rel} was defaced during conversion, so an undefaced version never "
        "existed inside this dataset. That is deliberate: it is what makes "
        "converting with defacing safer than defacing afterwards. To keep a "
        "copy you can compare against, deface from Tools instead and tick "
        "\"keep the original in sourcedata/\"."
    )


def comparable(root: Path) -> list[str]:
    """Every image in the dataset that has an undefaced copy to compare with.

    Driven from the copies rather than from the images, because that is the
    half that is usually missing.
    """
    root = Path(root)
    out: set[str] = set()

    mirror_root = root / SOURCEDATA
    if mirror_root.is_dir():
        for path in mirror_root.rglob("*.nii*"):
            rel = path.relative_to(mirror_root).as_posix()
            if (root / rel).is_file():
                out.add(rel)

    for record in read_log(root):
        op_id = record.get("op_id", "")
        if not op_id:
            continue
        kept_root = root / ORIGINALS_DIR / op_id
        for child in record.get("children", []):
            rel = str(child.get("path", "")).replace("\\", "/")
            if not rel.endswith((".nii", ".nii.gz")):
                continue
            if (kept_root / rel).is_file() and (root / rel).is_file():
                out.add(rel)

    return sorted(out)


__all__ = ["Original", "SOURCEDATA", "comparable", "explain_missing", "original_for"]
