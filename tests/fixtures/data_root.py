"""Where real datasets live, on whatever machine is running.

There are two kinds of real data in this suite, and they are reached in two
different ways on purpose.

**Published samples, which any machine can fetch.** The documentation ships
six datasets; ``tests/fixtures/sample_data.py`` downloads and caches them.
Everything a runner needs comes from there, so a fresh Windows or Linux
machine gets full coverage with no manual copying and no local path. That is
the default and it is what CI uses.

**The lab's own datasets, which are not published.** The MRI, MEG, EEG and
OpenNeuroPET collections are large and some are not redistributable, so they
cannot be downloaded. Tests that need them read ``BIDSMGR_TEST_DATA``.

    export BIDSMGR_TEST_DATA=/data/bids_manager/raw_data

**There is deliberately no fallback path.** Eleven files used to name
``/Users/<somebody>/Development/datasets/...`` literally, which meant those
tests could only ever run on one laptop: everywhere else they skipped, and no
amount of data on a runner would change that. Guessing at a location is what
produced that, so this guesses at nothing. Unset means "I do not have it", the
gated tests skip saying exactly that, and nothing silently half-works.

Per-modality gates (``BIDS_MANAGER_REAL_PET_DATA`` and friends) are unchanged.
They answer "do I want this tier"; this answers "and where is it". Two
questions, kept apart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_VAR = "BIDSMGR_TEST_DATA"


def data_root() -> Optional[Path]:
    """The lab's raw-data root, or ``None`` when this machine has none."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def dataset(*parts: str) -> Optional[Path]:
    """A path under the raw-data root, joined portably, or ``None``.

    ``dataset("PET_DICOMS", "PN000001")`` rather than a literal carrying
    slashes, so the separator is the platform's and the call reads the same on
    all three (CROSS_PLATFORM_RULES section 1).

    Returning ``None`` rather than raising lets a module-level constant be
    defined on a machine without the data; the gate below is what stops the
    test running.
    """
    root = data_root()
    return root.joinpath(*parts) if root is not None else None


def have(*parts: str) -> bool:
    """Is that dataset actually here? For a skipif that means what it says."""
    path = dataset(*parts)
    return path is not None and path.exists()


def why_missing(*parts: str) -> str:
    """A skip reason that tells the reader what to do about it."""
    if data_root() is None:
        return (
            f"set {ENV_VAR} to the raw-data root to run this tier. "
            "Published sample datasets are fetched automatically instead; "
            "see tests/fixtures/sample_data.py."
        )
    return f"{'/'.join(parts)} is not under {data_root()}"


__all__ = ["ENV_VAR", "data_root", "dataset", "have", "why_missing"]
