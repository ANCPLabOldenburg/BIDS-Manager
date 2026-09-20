"""Put the face back, from the copy that still has one.

Undo in the Editor reverses the LAST operation. That is the right tool five
minutes after defacing and the wrong one a week later, once a dozen edits sit
on top: the user would have to undo all of them to reach it, and the history
does not survive forever anyway. What does survive is the copy in
``sourcedata/``, which is never overwritten and is pristine by construction.

So this is the long-lived counterpart to undo: restore the image from its
original wherever that original still is, and take the deidentification
entries back out of the sidecar. It is one reversible operation like any other,
so reverting a revert is an Undo away.

Reverting only makes sense while an undefaced copy exists, which is exactly the
case the comparison view already reasons about, so both read
:mod:`bidsmgr.deface.compare`.

**It does not delete the mirror.** A user who reverts may want to deface again
with a different engine, and removing their only pristine copy as a side effect
of a restore would be the one mistake with no way back.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..project.operations import OperationError, begin_operation
from ..util.cancel import OperationCancelled
from . import compare, status

log = logging.getLogger(__name__)


@dataclass
class RevertOutcome:
    """What a revert did, per file and in total."""

    reverted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    # Images with nothing to revert from, and why. Shown, not hidden: "this one
    # was defaced at conversion time" is the answer a user needs.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.reverted) and not self.failed and not self.cancelled


@dataclass(frozen=True)
class Revertable:
    """One image that can be put back, and where from."""

    relative: str
    path: Path
    original: compare.Original

    @property
    def source(self) -> str:
        return self.original.source


def revertable(root: Path, targets: Optional[Iterable[Path]] = None) -> list[Revertable]:
    """Every image under ``root`` (or under ``targets``) that can be restored.

    Driven from the images the sidecar says WE defaced, not from whatever
    happens to have a copy in ``sourcedata/``: restoring a file this tool never
    touched would be acting on somebody else's data on a guess.
    """
    root = Path(root)
    wanted: Optional[list[Path]] = None
    if targets is not None:
        wanted = [Path(t).resolve() for t in targets]

    out: list[Revertable] = []
    for rel in compare.comparable(root):
        path = root / rel
        if wanted is not None and not _covered(path, wanted):
            continue
        doc = status.read_sidecar(status.sidecar_for(path))
        if not status.defaced_by_us_at_all(doc):
            continue
        original = compare.original_for(root, rel)
        if original is None:
            continue
        out.append(Revertable(relative=rel, path=path, original=original))
    return out


def _covered(path: Path, targets: list[Path]) -> bool:
    resolved = path.resolve()
    for target in targets:
        if resolved == target:
            return True
        if target.is_dir() and target in resolved.parents:
            return True
    return False


def revert_dataset(
    root: Path,
    *,
    targets: Optional[Iterable[Path]] = None,
    items: Optional[Iterable[Revertable]] = None,
    only: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    label: Optional[str] = None,
) -> RevertOutcome:
    """Restore defaced images from their originals, as ONE operation.

    Same shape as :func:`bidsmgr.deface.apply.deface_dataset`, deliberately:
    same ``only`` narrowing for a part-ticked preview, same progress and cancel
    callbacks, same all-or-nothing rollback. A half-reverted dataset is as bad
    as a half-defaced one, and for the same reason.
    """
    root = Path(root)
    chosen = list(items) if items is not None else revertable(root, targets)
    if only is not None:
        wanted = set(only)
        chosen = [c for c in chosen if c.relative in wanted]

    outcome = RevertOutcome()
    if not chosen:
        return outcome

    if label is None:
        n = len(chosen)
        label = f"Restore {n} image{'' if n == 1 else 's'} from the original"

    try:
        with begin_operation(root, label) as op:
            for i, item in enumerate(chosen):
                if cancel_check is not None:
                    cancel_check()
                if progress is not None:
                    progress(i, len(chosen), item.relative)

                if not item.original.path.is_file():
                    # It was there when the list was built and is not now.
                    # Refusing the whole run over it would be worse than
                    # saying so and doing the rest.
                    outcome.skipped.append(
                        (item.relative, "the original is no longer there")
                    )
                    continue

                op.replace_from(item.path, item.original.path)

                sidecar = status.sidecar_for(item.path)
                doc = status.read_sidecar(sidecar)
                if doc or sidecar.exists():
                    op.write_json(sidecar, status.clear(doc))

                outcome.reverted.append(item.relative)

            if progress is not None:
                progress(len(chosen), len(chosen), "")
    except OperationCancelled:
        outcome.reverted.clear()
        outcome.skipped.clear()
        outcome.cancelled = True
        log.info("revert stopped by the user; the dataset was rolled back")
    except (OperationError, OSError) as exc:
        done = len(outcome.reverted)
        where = chosen[done].relative if done < len(chosen) else "the dataset"
        outcome.reverted.clear()
        outcome.skipped.clear()
        outcome.failed.append((where, str(exc)))
        log.warning("revert rolled back at %s: %s", where, exc)

    return outcome


__all__ = ["RevertOutcome", "Revertable", "revert_dataset", "revertable"]
