"""Remove identifiable faces from anatomical images.

A head MRI contains a face, and a face can be rendered from one, so a dataset
shared with its faces intact is a dataset shared with its participants
identifiable. Defacing blanks the voxels that carry the face while leaving the
brain alone.

The engines are `niimath <https://github.com/rordenlab/niimath>`_, which
arrives as a pip wheel with its binary inside the same way ``dcm2niix`` does,
and `brainchop <https://github.com/neuroneural/brainchop-cli>`_, which runs the
mindgrab brain-extraction network on tinygrad. No FSL, no FreeSurfer, no ANTs,
no torch, nothing to install by hand.

Both are ORDINARY dependencies, not extras. These features are reached from
the GUI, and somebody who installed by double-clicking cannot add an extra.

The pieces, in the order a caller meets them:

``engines``
    Which defacing engines exist, and the argv each one implies.
``probe``
    Enough of a NIfTI header to tell a volume from a time series.
``select``
    Which images in a dataset can be defaced, and why the rest cannot.
``run``
    Run an engine on one image, into a file the caller owns. No dataset.
``apply``
    Do it to a dataset, reversibly, as one entry in the Editor's history.
``status``
    Read and write the BIDS deidentification fields in the sidecar.
``compare``
    Find the undefaced copy of an image, so the result can be inspected.

Everything here is Qt-free.
"""

from __future__ import annotations

from .compare import Original, comparable, explain_missing, original_for
from .engines import DEFAULT_ENGINE_ID, ENGINES, Engine, engine, engine_ids
from .run import (
    DefaceFailed,
    DefaceResult,
    DefaceUnavailable,
    available,
    deface_to,
    deface_to_temp,
    find_niimath,
    unavailable_reason,
)
from .select import Candidate, Selection, Skip, Skipped, walk

__all__ = [
    "DEFAULT_ENGINE_ID",
    "ENGINES",
    "Candidate",
    "DefaceFailed",
    "DefaceResult",
    "DefaceUnavailable",
    "Engine",
    "Original",
    "Selection",
    "Skip",
    "Skipped",
    "available",
    "comparable",
    "explain_missing",
    "original_for",
    "deface_to",
    "deface_to_temp",
    "engine",
    "engine_ids",
    "find_niimath",
    "unavailable_reason",
    "walk",
]
