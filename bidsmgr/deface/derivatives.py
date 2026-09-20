"""Where a skull-stripped image goes, and what has to go with it.

A defaced image is still raw data: it is the same scan with some voxels
blanked, and it belongs exactly where the original was. A skull-stripped image
is not. Everything outside the brain has been thrown away by an algorithm that
can be wrong, and BIDS has a word for a file produced by an algorithm from raw
data: a derivative.

So the default is ``derivatives/bidsmgr-skullstrip/``, the raw image is left
alone, and the result carries ``desc-brain`` so it cannot be mistaken for the
scan it came from. Writing in place is offered, because BIDSvue does it and
some people want it, but it is a choice with a consequence and the dialog says
so.

Qt-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# The pipeline folder. Named for the tool and the operation, because a
# dataset can hold several derivative pipelines and "which tool made this"
# is the first question anybody asks of one.
PIPELINE = "bidsmgr-skullstrip"

# The entity BIDS uses to say "a processed version of". Immediately before the
# suffix, which is where the standard puts it.
DESC = "brain"

NIFTI_EXTS = (".nii.gz", ".nii")


def _split_ext(name: str) -> tuple[str, str]:
    for ext in NIFTI_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)], name[-len(ext):]
    return Path(name).stem, Path(name).suffix


def derivative_name(name: str, *, desc: str = DESC) -> str:
    """``sub-01_T1w.nii.gz`` -> ``sub-01_desc-brain_T1w.nii.gz``.

    An existing ``desc-`` is REPLACED rather than added to: two desc entities
    in one filename is not a BIDS name, and stripping something already
    described as, say, ``desc-preproc`` produces a brain-only version of it,
    which is what the new value should say.
    """
    stem, ext = _split_ext(name)
    parts = stem.split("_")
    if len(parts) < 2:
        return f"{stem}_desc-{desc}{ext}"
    suffix = parts[-1]
    entities = [p for p in parts[:-1] if not p.startswith("desc-")]
    return "_".join([*entities, f"desc-{desc}", suffix]) + ext


def derivative_path(
    root: Path, image: Path, *, desc: str = DESC,
    pipeline: str = PIPELINE,
) -> Optional[Path]:
    """Where the stripped version of ``image`` belongs.

    ``None`` when the image is not raw subject data, which is the one case
    where there is no sensible answer: a derivative of a derivative would need
    the user to say which pipeline it belongs to.
    """
    root = Path(root)
    try:
        rel = Path(image).relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts or not parts[0].startswith("sub-"):
        return None
    if "derivatives" in parts:
        return None
    return (
        root / "derivatives" / pipeline / rel.parent
        / derivative_name(rel.name, desc=desc)
    )


def dataset_description(
    root: Path, *, pipeline: str = PIPELINE, engine_label: str = "",
    version: str = "",
) -> dict[str, Any]:
    """The ``dataset_description.json`` a derivative pipeline must have.

    Without it the folder is not a derivative dataset, it is a folder of files
    the validator cannot place. ``SourceDatasets`` names the dataset it came
    from, so the pair stays interpretable if somebody moves one of them.
    """
    source_name = ""
    try:
        source = json.loads(
            (Path(root) / "dataset_description.json").read_text(
                encoding="utf-8"
            )
        )
        source_name = str(source.get("Name", "") or "")
    except (OSError, ValueError):
        pass

    generated_by: dict[str, Any] = {"Name": "BIDS Manager"}
    if version:
        generated_by["Version"] = version
    if engine_label:
        generated_by["Description"] = f"Skull stripping ({engine_label})"

    doc: dict[str, Any] = {
        "Name": (
            f"{source_name} (skull stripped)" if source_name
            else "Skull-stripped images"
        ),
        "BIDSVersion": _bids_version(root),
        "DatasetType": "derivative",
        "GeneratedBy": [generated_by],
    }
    if source_name:
        doc["SourceDatasets"] = [{"Name": source_name}]
    return doc


def _bids_version(root: Path) -> str:
    """Match the raw dataset's BIDSVersion, so the pair does not disagree."""
    try:
        source = json.loads(
            (Path(root) / "dataset_description.json").read_text(
                encoding="utf-8"
            )
        )
        version = str(source.get("BIDSVersion", "") or "")
        if version:
            return version
    except (OSError, ValueError):
        pass
    from ..schema import bids_version

    return bids_version()


def description_path(root: Path, *, pipeline: str = PIPELINE) -> Path:
    return Path(root) / "derivatives" / pipeline / "dataset_description.json"


__all__ = [
    "DESC",
    "PIPELINE",
    "dataset_description",
    "derivative_name",
    "derivative_path",
    "description_path",
]
