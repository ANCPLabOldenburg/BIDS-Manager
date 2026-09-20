"""Deface files in a dataset, as one reversible operation.

Everything else in this package avoids datasets. This is the module that does
not: it takes a selection, runs the engine on each image, replaces the bytes,
records the method in the sidecar, and wraps the lot in a single entry in the
Editor's history so one Undo puts the dataset back exactly as it was.

Three decisions worth knowing about.

**The original goes to ``.bidsmgr/``, not to ``sourcedata/``.** BIDSvue mirrors
the pristine image into the dataset so a revert is always possible. That is a
reasonable trade for them and the wrong default here, because *the pristine
copy still contains the face*: a dataset defaced that way and then shared ships
the face inside itself, in a folder every BIDS tool treats as ordinary content.
Our operations log already keeps originals under ``.bidsmgr/``, a dot folder
that tools ignore and that our own delete refuses to touch. ``sourcedata/`` is
available as an explicit choice for people who want the visible mirror and know
what is in it.

**Re-defacing starts from what is on disk.** Not from a pristine copy. Once the
face is gone it is gone, and running a second engine over an already-defaced
image removes a little more, which is harmless. The alternative, keeping the
original around so a second engine can start clean, is the thing the paragraph
above rejects.

**A failure anywhere rolls everything back.** Defacing half a dataset and
stopping is the state hardest to recover from, because the dataset then looks
done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..project.operations import OperationError, begin_operation
from ..util.cancel import OperationCancelled
from .derivatives import dataset_description, derivative_path, description_path
from .engines import DEFAULT_ENGINE_ID, Engine, engine
from .run import DefaceFailed, deface_to_temp
from .select import Candidate, Selection, walk
from .status import read_sidecar, record, sidecar_for

log = logging.getLogger(__name__)

# Where the engine writes its temporary output. Inside the dataset so the final
# move is a rename on one filesystem; hidden and cleaned up after.
STAGING = Path(".bidsmgr") / "deface-staging"

# Where an opt-in visible mirror of the original goes.
SOURCEDATA = "sourcedata"


@dataclass
class DefaceOutcome:
    """What a run did, per file and in total."""

    defaced: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # Files this run CREATED rather than replaced: the derivatives a skull
    # strip writes. Separate from `defaced`, which names the source images,
    # because the user needs to be told where the output went.
    produced: list[str] = field(default_factory=list)
    seconds: float = 0.0
    # Stopped by the user rather than finished or failed. The dataset was
    # rolled back either way; this says which happened, so the GUI does not
    # report a deliberate Stop as an error.
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.defaced) and not self.failed and not self.cancelled


def sourcedata_mirror(root: Path, image: Path) -> Optional[Path]:
    """Where the visible pristine mirror of ``image`` would go.

    ``<root>/sub-01/anat/x.nii.gz`` -> ``<root>/sourcedata/sub-01/anat/x.nii.gz``

    ``None`` when the image is not raw subject data, which the selector should
    already have refused. Defence in depth: a mirror under
    ``sourcedata/derivatives/`` would be a phantom the dataset never asked for.
    """
    root = Path(root)
    try:
        rel = Path(image).relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts or not parts[0].startswith("sub-"):
        return None
    return root / SOURCEDATA / rel


def _write_derivative(op, root: Path, image: Path, result, eng: Engine):
    """Put a stripped image under ``derivatives/``, with what BIDS requires.

    Three files, not one. The image, its sidecar (the source's fields plus the
    record of what was done, so the derivative is self-describing), and the
    pipeline's ``dataset_description.json`` without which the folder is not a
    derivative dataset at all. All through the operation, so one Undo removes
    the lot.
    """
    import bidsmgr

    target = derivative_path(root, image)
    if target is None:
        return None

    description = description_path(root)
    if not description.exists():
        op.write_json(description, dataset_description(
            root, engine_label=eng.label, version=bidsmgr.__version__,
        ))

    op.replace_from(target, result.output)

    # The derivative's own sidecar: what the source said, plus what we did.
    # `Sources` is how BIDS says "this came from that".
    doc = dict(read_sidecar(sidecar_for(image)))
    try:
        doc["Sources"] = [
            f"bids::{Path(image).relative_to(root).as_posix()}"
        ]
    except ValueError:
        pass
    op.write_json(sidecar_for(target), record(doc, eng))
    return target


def deface_dataset(
    root: Path,
    *,
    targets: Optional[Iterable[Path]] = None,
    selection: Optional[Selection] = None,
    engine_id: str = DEFAULT_ENGINE_ID,
    only: Optional[Iterable[str]] = None,
    keep_original_in_sourcedata: bool = False,
    in_place: Optional[bool] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    label: Optional[str] = None,
) -> DefaceOutcome:
    """Deface images under ``root``, reversibly.

    ``selection`` may be supplied by a caller that already computed one, so a
    dialog shows exactly what it then applies. ``only`` narrows it to a set of
    dataset-relative POSIX paths, which is how a part-ticked preview is
    honoured.

    ``progress(done, total, relative_path)`` is called before each file.

    ``cancel_check()`` is called at the same points and may raise
    :class:`bidsmgr.util.cancel.OperationCancelled` to stop the run. Stopping
    rolls the whole operation back, exactly as a failure does, because a
    half-defaced dataset is the state that is hardest to recover from: it looks
    finished. Same contract the scan and convert verbs use.
    """
    root = Path(root)
    eng: Engine = engine(engine_id)

    # Where the result goes. A defaced image is the same scan with voxels
    # blanked, so it replaces the original. A skull-stripped one is a
    # DERIVATIVE: everything outside the brain has been discarded by an
    # algorithm, and writing that over the raw image leaves a dataset whose
    # raw data has been processed. In place stays available because BIDSvue
    # does it and some people want it, but it is not the default.
    if in_place is None:
        in_place = not eng.is_strip

    if selection is None:
        selection = walk(root, targets)
    chosen: list[Candidate] = list(selection.candidates)
    if only is not None:
        wanted = set(only)
        chosen = [c for c in chosen if c.relative in wanted]

    outcome = DefaceOutcome()
    outcome.skipped = [(s.relative, s.reason.value) for s in selection.skipped]
    if not chosen:
        return outcome

    if label is None:
        n = len(chosen)
        verb = "Skull strip" if eng.is_strip else "Deface"
        label = f"{verb} {n} image{'' if n == 1 else 's'} ({eng.label})"

    staging = root / STAGING
    staging.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    try:
        with begin_operation(root, label) as op:
            for i, cand in enumerate(chosen):
                if cancel_check is not None:
                    cancel_check()
                if progress is not None:
                    progress(i, len(chosen), cand.relative)

                result = deface_to_temp(
                    cand.path, engine_id=eng.id, directory=staging,
                )
                produced.append(result.output)
                outcome.seconds += result.seconds

                if not in_place:
                    written = _write_derivative(op, root, cand.path, result, eng)
                    if written is None:
                        outcome.skipped.append((
                            cand.relative,
                            "not raw subject data, so there is nowhere to put "
                            "a derivative of it",
                        ))
                        continue
                    outcome.produced.append(
                        written.relative_to(root).as_posix()
                    )
                    outcome.defaced.append(cand.relative)
                    continue

                # The visible mirror, before the image is replaced, and only
                # when it is not already there: a second deface must not
                # overwrite the pristine copy with a defaced one.
                if keep_original_in_sourcedata:
                    mirror = sourcedata_mirror(root, cand.path)
                    if mirror is not None and not mirror.exists():
                        mirror.parent.mkdir(parents=True, exist_ok=True)
                        op.replace_from(mirror, cand.path)

                op.replace_from(cand.path, result.output)

                sidecar = sidecar_for(cand.path)
                op.write_json(sidecar, record(read_sidecar(sidecar), eng))

                outcome.defaced.append(cand.relative)

            if progress is not None:
                progress(len(chosen), len(chosen), "")
    except OperationCancelled:
        # Stopped on purpose. The rollback has already happened, for the same
        # reason a failure rolls back, but this is not an error: nothing is
        # reported as failed, and the caller decides what to say.
        outcome.defaced.clear()
        outcome.produced.clear()
        outcome.cancelled = True
        log.info("defacing stopped by the user; the dataset was rolled back")
    except (DefaceFailed, OperationError, OSError) as exc:
        # begin_operation already rolled back. Report the file it died on so
        # the user is not left guessing which of forty it was.
        done = len(outcome.defaced)
        where = chosen[done].relative if done < len(chosen) else "the dataset"
        outcome.defaced.clear()
        outcome.produced.clear()
        outcome.failed.append((where, str(exc)))
        log.warning("defacing rolled back at %s: %s", where, exc)
    finally:
        for tmp in produced:
            Path(tmp).unlink(missing_ok=True)
        try:
            staging.rmdir()
        except OSError:
            pass

    return outcome


__all__ = ["DefaceOutcome", "SOURCEDATA", "STAGING", "deface_dataset",
           "sourcedata_mirror"]
