"""dcm2niix ``BidsGuess`` classifier — improvement_plan.md M1.

dcm2niix already classifies DICOM series via its built-in ``BidsGuess``
heuristics, which are kept up to date with the BIDS spec. We harvest that
classification by running dcm2niix in **sidecar-only** mode (``-b o``) once
per DICOM source folder and parsing the JSON sidecars. The result is the
highest-fidelity DICOM classifier available without writing our own rules.

This is the first feature to land in ``bidsmgr``: it validates the keystone
``schema/`` (because every BidsGuess output is schema-checked) and seeds
the per-row pipeline that the GUI inspector will surface later.

Reference: architecture.md §4.2 layer 1, ``../improvement_plan.md`` M1.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .. import schema
from ..inventory.types import InventoryRow
from .types import Classification

log = logging.getLogger(__name__)

_SHORT_TO_LONG_CACHE: Optional[dict[str, str]] = None


def _short_to_long_entity_map() -> dict[str, str]:
    """Map BIDS short entity names (``"sub"``) to canonical keys (``"subject"``)."""
    global _SHORT_TO_LONG_CACHE
    if _SHORT_TO_LONG_CACHE is None:
        s = schema.get_schema()
        _SHORT_TO_LONG_CACHE = {
            str(s.objects.entities[k].get("name", k)): k
            for k in s.objects.entities.keys()
        }
    return _SHORT_TO_LONG_CACHE


def parse_bids_guess(guess: Sequence[str]) -> tuple[str, dict[str, str], str]:
    """Parse the ``BidsGuess`` field into ``(datatype, entities, suffix)``.

    dcm2niix encodes BidsGuess as ``[<datatype>, "_<key>-<value>_..._<suffix>"]``.
    For the ``"discard"`` datatype the second element is still parseable but
    callers should treat the row as skip-by-default.
    """

    if not guess or len(guess) < 2:
        raise ValueError(f"Malformed BidsGuess: {guess!r}")

    datatype = str(guess[0])
    tail = str(guess[1]).lstrip("_")
    parts = tail.split("_")
    if not parts:
        raise ValueError(f"BidsGuess has no suffix component: {guess!r}")

    suffix = parts[-1]
    short_to_long = _short_to_long_entity_map()
    entities: dict[str, str] = {}
    for chunk in parts[:-1]:
        if "-" not in chunk:
            continue
        short, _, value = chunk.partition("-")
        long_name = short_to_long.get(short, short)
        # dcm2niix encodes ``run-N`` from DICOM SeriesNumber, which is NOT the
        # BIDS-semantic run (BIDS run-N is for repeated acquisitions of the
        # same parameters within a session). Drop it here; the planner /
        # cli/scan re-derives runs cross-row after grouping.
        if long_name == "run":
            continue
        entities[long_name] = value

    return datatype, entities, suffix


def find_dcm2niix() -> Path:
    """Locate the dcm2niix executable.

    Prefers the binary shipped with the ``dcm2niix`` Python package (a pinned
    pip dependency), falling back to ``$PATH``.
    """

    try:
        import dcm2niix as _pkg  # type: ignore

        bin_path = Path(getattr(_pkg, "bin_path", "") or "")
        if bin_path.exists():
            return bin_path
    except ImportError:
        pass

    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "dcm2niix executable not found. Install the ``dcm2niix`` pip package "
        "or place the binary on PATH."
    )


def _run_dcm2niix_sidecars(
    dicom_dir: Path,
    output_dir: Path,
    *,
    dcm2niix_bin: Optional[Path] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix to produce JSON sidecars only.

    Flags:

    * ``-b o``  : sidecar only (no NIfTI written)
    * ``-ba n`` : do not anonymize the sidecar (keeps SeriesInstanceUID)
    * ``-z n``  : no compression (irrelevant for sidecar-only)
    * ``-f %j`` : filename = SeriesInstanceUID (with ``_e<N>``, ``_ph`` suffixes
                  for multi-echo / phase splits)
    """

    binary = str(dcm2niix_bin or find_dcm2niix())
    cmd = [
        binary,
        "-b", "o",
        "-ba", "n",
        "-z", "n",
        "-o", str(output_dir),
        "-f", "%j",
        str(dicom_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _collect_sidecars(directory: Path) -> list[dict]:
    """Read every ``*.json`` in ``directory`` into memory."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read sidecar %s: %s", path, exc)
            continue
        data["_sidecar_path"] = str(path)
        out.append(data)
    return out


def _validate_classification(datatype: str, suffix: str, entities: dict[str, str]) -> bool:
    """Return ``True`` if the schema accepts this (datatype, suffix, entities) tuple."""
    if datatype == "discard":
        # Not a real datatype — but the row should still produce a Classification
        # marked ``skip=True`` so the GUI can surface the recommendation.
        return True
    if datatype not in schema.list_datatypes():
        return False
    if suffix not in schema.list_suffixes(datatype):
        return False
    allowed = set(schema.allowed_entities(datatype, suffix))
    return all(ent in allowed or ent == "subject" for ent in entities)


def classify_dicom_folder(
    dicom_dir: Path,
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> list[Classification]:
    """Run dcm2niix BidsGuess on ``dicom_dir`` and return classifications.

    Each :class:`InventoryRow` whose ``series_uid`` is present in the sidecar
    output produces one :class:`Classification`. Rows with no matching sidecar
    yield no classification (the caller decides whether to fall back to the
    next classifier in the chain).
    """

    rows = list(rows)
    rows_by_uid: dict[str, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        if r.series_uid:
            rows_by_uid[r.series_uid].append(r)

    use_temp = workdir is None
    if use_temp:
        workdir_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(workdir_ctx.name)
    else:
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = _run_dcm2niix_sidecars(dicom_dir, out_dir, dcm2niix_bin=dcm2niix_bin)
        if proc.returncode != 0:
            log.warning(
                "dcm2niix returncode=%s for %s; stderr=%s",
                proc.returncode, dicom_dir, proc.stderr[-500:],
            )
        sidecars = _collect_sidecars(out_dir)
    finally:
        if use_temp:
            workdir_ctx.cleanup()  # type: ignore[name-defined]

    out: list[Classification] = []
    for sidecar in sidecars:
        guess = sidecar.get("BidsGuess")
        if not guess:
            continue
        try:
            datatype, entities, suffix = parse_bids_guess(guess)
        except ValueError as exc:
            log.debug("could not parse BidsGuess %r: %s", guess, exc)
            continue

        uid = sidecar.get("SeriesInstanceUID")
        matching_rows = rows_by_uid.get(uid, [])
        if not matching_rows:
            log.debug("BidsGuess sidecar with no matching inventory row: uid=%s", uid)
            continue

        skip = (datatype == "discard")
        valid = _validate_classification(datatype, suffix, entities)
        if not valid:
            log.info(
                "BidsGuess output rejected by schema: datatype=%s suffix=%s entities=%s",
                datatype, suffix, entities,
            )
            continue

        rationale = f"dcm2niix BidsGuess: {list(guess)}"
        confidence = 0.0 if skip else 0.85

        for row in matching_rows:
            out.append(
                Classification(
                    row_id=row.row_id,
                    classifier="dcm2niix_bidsguess",
                    datatype=datatype,
                    suffix=suffix,
                    candidate_entities=dict(entities),
                    confidence=confidence,
                    rationale=rationale,
                    skip=skip,
                )
            )
    return out


def classify(
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
) -> list[Classification]:
    """Top-level classifier entry point.

    Groups MRI rows by their containing folder (``row.source.parent`` if
    ``source`` is a file, ``row.source`` if it is a directory) and dispatches
    one dcm2niix invocation per folder.
    """

    rows = [r for r in rows if r.modality == "mri" and r.series_uid]
    if not rows:
        return []

    groups: dict[Path, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        folder = r.source if r.source.is_dir() else r.source.parent
        groups[folder].append(r)

    out: list[Classification] = []
    for folder, group_rows in groups.items():
        out.extend(classify_dicom_folder(folder, group_rows, dcm2niix_bin=dcm2niix_bin))
    return out


__all__ = [
    "classify",
    "classify_dicom_folder",
    "parse_bids_guess",
    "find_dcm2niix",
]
