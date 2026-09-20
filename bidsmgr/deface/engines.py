"""The defacing engines, as a table rather than a set of branches.

An engine is a name, an argv template, and the deidentification entry it writes
into the sidecar afterwards. Adding one is adding a row here; nothing in
:mod:`bidsmgr.deface.run` learns a new name.

Every engine is one flag on `niimath <https://github.com/rordenlab/niimath>`_,
which registers a template and its mask onto the subject with a 12-DOF affine
and then blanks the voxels the mask marks for removal. niimath is BSD-2-Clause
and ships its binary in a PyPI wheel, which is how `dcm2niix` already reaches
users here: no FSL, no FreeSurfer, no ANTs, no separate download.

The three engines differ in two ways only:

``robustfov``
    Crop the neck and the inferior slices before registering. Steadier on a
    scan with a long neck or a large field of view, which is most clinical
    acquisitions.

``cost``
    The similarity measure. niimath's default is a fast SPM/FLIRT-inspired
    engine; ``hel`` is the AFNI-style Hellinger cost, slower and occasionally
    better on unusual contrast.

Qt-free, and free of any knowledge of datasets. It is a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Where the atlas lives, relative to this file. The template and its mask are
# one asset; see atlas/PROVENANCE.md.
ATLAS_DIR = Path(__file__).resolve().parent / "atlas"
TEMPLATE = ATLAS_DIR / "avg152T1.nii.gz"
MASK = ATLAS_DIR / "avg152T1mask.nii.gz"
# The brain mask for the SAME template, used by the atlas skull-strip engine.
# niimath's own help puts it plainly: "skull-stripping: use -deface with a
# brain mask". The mask decides what survives (>=0.5 keep), so a face mask
# removes the face and a brain mask keeps only the brain. Same flag, same
# registration, opposite sense.
BRAIN_MASK = ATLAS_DIR / "avg152T1brainmask.nii.gz"

# Bumped when the algorithm or the atlas changes in a way that should make
# somebody re-deface. It lands in the sidecar, so a dataset stays honest about
# which version of this produced it.
ENGINE_REVISION = "v1"

# Ours, so a reader can tell our entries from a genuine DICOM one. DICOM
# scheme designators are short and registered; this deliberately is not.
CODING_SCHEME = "BIDSManager"


# What an engine is FOR. Both remove identifiable anatomy; they differ in what
# they keep, and therefore in what the result may be used for. A defaced image
# is still raw data. A skull-stripped one is a derivative.
KIND_DEFACE = "deface"
KIND_STRIP = "strip"

# How an engine computes. "niimath" shells out to the bundled binary;
# "brainchop" runs the mindgrab network in process.
BACKEND_NIIMATH = "niimath"
BACKEND_BRAINCHOP = "brainchop"


@dataclass(frozen=True)
class Engine:
    """One way of removing identifiable anatomy."""

    id: str
    label: str
    description: str
    # Chained before -deface. Crops the neck so registration has less to fit.
    robustfov: bool = False
    # niimath's -cost. None means its default fast engine.
    cost: Optional[str] = None
    # Deface (keep everything but the face) or strip (keep only the brain).
    kind: str = KIND_DEFACE
    backend: str = BACKEND_NIIMATH
    # For a niimath engine: which mask decides what survives.
    mask: Path = MASK
    # For a brainchop engine: which model to run.
    model: str = ""

    @property
    def is_strip(self) -> bool:
        return self.kind == KIND_STRIP

    def argv(self, source: Path, output: Path) -> list[str]:
        """The arguments to hand niimath, in order.

        Built here rather than at the call site so there is one place where the
        shape of the command is decided, and one place to look when niimath
        changes it.
        """
        args = [str(source)]
        if self.robustfov:
            args.append("-robustfov")
        args += ["-deface", str(TEMPLATE), str(self.mask)]
        if self.cost:
            args += ["-cost", self.cost]
        args.append(str(output))
        return args

    @property
    def code_value(self) -> str:
        """What goes in the sidecar. Identifies the engine AND its version."""
        verb = "STRIP" if self.is_strip else "DEFACE"
        stem = self.id.upper().replace("-", "_")
        return f"BIDSMANAGER-{verb}-{stem}-{ENGINE_REVISION}"

    def deid_entry(self) -> dict[str, str]:
        """The ``DeidentificationMethodCodeSequence`` entry for this engine."""
        return {
            "CodingSchemeDesignator": CODING_SCHEME,
            "CodeValue": self.code_value,
            "CodeMeaning": (
                f"BIDS Manager "
                f"{'skull strip' if self.is_strip else 'deface'} "
                f"({self.label})"
            ),
        }

    def deid_method(self) -> str:
        """The ``DeidentificationMethod`` string. For a human to read."""
        if self.backend == BACKEND_BRAINCHOP:
            return (
                f"BIDS Manager {self.label} (brainchop {self.model}, "
                "in-process neural brain extraction)"
            )
        bits = ["niimath -deface"]
        if self.robustfov:
            bits.append("-robustfov")
        if self.cost:
            bits.append(f"-cost {self.cost}")
        return (
            f"BIDS Manager {self.label} "
            f"({' '.join(bits)}, avg152T1 {self.mask.stem})"
        )


ALLINEATE = Engine(
    id="allineate",
    label="allineate",
    description=(
        "Affine registration of a face mask, then blank the face. The fast "
        "default; right for most images."
    ),
)

ALLINEATE_ROBUST = Engine(
    id="allineate-robust",
    label="allineate, neck cropped",
    description=(
        "The same, after cropping the neck and lower head. Steadier when the "
        "scan reaches well below the chin. Note that this one CHANGES THE "
        "IMAGE DIMENSIONS: the cropped slices are gone, so the result no "
        "longer lines up voxel-for-voxel with anything derived from the "
        "original."
    ),
    robustfov=True,
)

ALLINEATE_AFNI = Engine(
    id="allineate-afni",
    label="allineate, Hellinger cost",
    description=(
        "The same, matched with the AFNI-style Hellinger cost instead of the "
        "fast one. Slower; worth trying when the default fits badly."
    ),
    cost="hel",
)

# --------------------------------------------------------------------------
# Skull stripping. Removes everything except the brain, so it removes the face
# too, and much else besides. The output is a DERIVATIVE, which is why
# `apply` writes it under derivatives/ unless told otherwise.

MINDGRAB = Engine(
    id="mindgrab",
    label="mindgrab",
    description=(
        "A small neural network that finds the brain directly in the image. "
        "Much better than fitting an atlas to it. About ten seconds an image "
        "on a laptop CPU, plus a one-off download of the model the first time "
        "it runs."
    ),
    kind=KIND_STRIP,
    backend=BACKEND_BRAINCHOP,
    model="mindgrab",
)

STRIP_ATLAS = Engine(
    id="strip-atlas",
    label="atlas brain mask",
    description=(
        "Affine registration of a brain mask from the same atlas the defacer "
        "uses, keeping what the mask covers. No extra to install and it works "
        "offline, but it is an affine fit of an average brain: expect it to "
        "clip cortex in places and leave dura in others. Use mindgrab when "
        "the result has to be analysed."
    ),
    kind=KIND_STRIP,
    mask=BRAIN_MASK,
)

DEFACE_ENGINES: tuple[Engine, ...] = (
    ALLINEATE, ALLINEATE_ROBUST, ALLINEATE_AFNI,
)
STRIP_ENGINES: tuple[Engine, ...] = (MINDGRAB, STRIP_ATLAS)
ENGINES: tuple[Engine, ...] = DEFACE_ENGINES + STRIP_ENGINES

DEFAULT_ENGINE_ID = ALLINEATE.id
DEFAULT_STRIP_ENGINE_ID = MINDGRAB.id


def engine(engine_id: str) -> Engine:
    """Look one up by id. Raises :class:`KeyError` with the valid ids in it."""
    for eng in ENGINES:
        if eng.id == engine_id:
            return eng
    raise KeyError(
        f"unknown deface engine {engine_id!r}; "
        f"available: {', '.join(e.id for e in ENGINES)}"
    )


def engines_of(kind: str) -> tuple[Engine, ...]:
    """Every engine of one kind, in the order they should be offered."""
    return tuple(e for e in ENGINES if e.kind == kind)


def engine_ids(kind: Optional[str] = None) -> list[str]:
    """Every engine id, or only those of one kind.

    The kind matters at every call site that builds a list for a user: a
    defacing dropdown offering a skull stripper is offering to delete the
    skull when the user asked to blank a face.
    """
    pool = ENGINES if kind is None else engines_of(kind)
    return [e.id for e in pool]


__all__ = [
    "ALLINEATE",
    "ALLINEATE_AFNI",
    "BACKEND_BRAINCHOP",
    "BACKEND_NIIMATH",
    "BRAIN_MASK",
    "DEFACE_ENGINES",
    "DEFAULT_STRIP_ENGINE_ID",
    "KIND_DEFACE",
    "KIND_STRIP",
    "MINDGRAB",
    "STRIP_ATLAS",
    "STRIP_ENGINES",
    "engines_of",
    "ALLINEATE_ROBUST",
    "ATLAS_DIR",
    "CODING_SCHEME",
    "DEFAULT_ENGINE_ID",
    "ENGINES",
    "ENGINE_REVISION",
    "Engine",
    "MASK",
    "TEMPLATE",
    "engine",
    "engine_ids",
]
