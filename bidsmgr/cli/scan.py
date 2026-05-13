"""``bidsmgr-scan`` — scan DICOMs, classify, write TSV.

Orchestration is explicit code (architecture.md §12 rule 3 — no Pipeline).

Pipeline:

1. ``inventory.mri_dicom.scan_dicoms_long`` produces the v0.2.5 22-column
   DataFrame, with ``proposed_*`` columns blank and ``modality`` filled
   by the legacy regex dictionary.
2. Build :class:`InventoryRow` objects per DataFrame row, group by source
   folder, and run :func:`classifier.dcm2niix_bidsguess.classify` to
   produce :class:`Classification` records.
3. Run :func:`classifier.sequence_dict.classify` as a fallback for rows
   that BidsGuess didn't fire on (or that BidsGuess produced a
   schema-invalid result for).
4. Strip any ``run`` entity from every classification (BIDS-semantic
   ``run`` is cross-row; the value dcm2niix puts there is DICOM
   SeriesNumber, which is meaningless for BIDS).
5. Group by ``(subject, session, datatype, suffix, other-entities)``
   within each subject+session; assign ``run-1, run-2, …`` only to groups
   with more than one row, ordered by acquisition time. Singletons get no
   run entity.
6. Emit ``proposed_datatype`` / ``proposed_basename`` / ``Proposed BIDS name``
   for every classified row (best effort — even when the schema would
   reject it). ``proposed_issues`` records any required-entity / format
   violations so the GUI / planner can prompt the user.
7. Write the TSV.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re as _re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .. import schema
from ..classifier import dcm2niix_bidsguess, sequence_dict
from ..classifier.types import Classification
from ..inventory import probe_convert as probe_convert_module
from ..inventory._time import parse_dicom_time_seconds as _parse_dicom_time_seconds
from ..inventory.eeg_meg import EEG_MEG_COLUMNS, scan_eeg_meg
from ..inventory.mri_dicom import (
    BIDS_ENTITIES_COLUMNS,
    DATASET_COLUMNS,
    EXTENDED_COLUMNS,
    TSV_COLUMNS,
    scan_dicoms_long,
)
from ..inventory.probe_convert import ProbeFileStats
from ..inventory.types import InventoryRow

log = logging.getLogger(__name__)


# Columns appended after the v0.2.5 22-col contract.
BIDS_GUESS_COLUMNS: tuple[str, ...] = (
    "bids_guess_classifier",
    "bids_guess_datatype",
    "bids_guess_suffix",
    "bids_guess_entities",
    "bids_guess_confidence",
    "bids_guess_skip",
    "proposed_issues",
    "repetition_type",
)

# Columns added when ``--probe-convert`` is enabled. dcm2niix actually
# converts each DICOM series and we record what it produced.
PROBE_COLUMNS: tuple[str, ...] = (
    "probe_n_files",
    "probe_n_nifti",
    "probe_n_volumes",
    "probe_extensions",
)


# Per (datatype, suffix) — how many NIfTI files dcm2niix should produce
# from one input DICOM series. ``None`` means "don't flag, expectations
# vary widely". Anything not listed defaults to 1. Used by the anomaly
# detector to flag e.g. a bold series that produced 2 NIfTIs (the
# operator-cancellation case the user asked about).
EXPECTED_NIFTI_PER_UID: dict[tuple[str, Optional[str]], Optional[int]] = {
    ("anat", "T1w"): 1,
    ("anat", "T2w"): 1,
    ("anat", "FLAIR"): 1,
    ("anat", "T2starw"): 1,
    ("anat", "PDw"): 1,
    ("anat", "MP2RAGE"): None,  # multi-inversion → multiple NIfTIs expected
    ("anat", "T1map"): 1,
    ("anat", "UNIT1"): 1,
    ("anat", "MEGRE"): None,    # multi-echo
    ("func", "bold"): 1,
    ("func", "sbref"): 1,
    ("func", "phase"): 1,
    ("dwi", "dwi"): 1,
    ("dwi", "FA"): 1,
    ("dwi", "ADC"): 1,
    ("dwi", "trace"): 1,
    ("dwi", "colFA"): 1,
    ("dwi", "expADC"): 1,
    ("dwi", "S0map"): 1,
    ("dwi", "sbref"): 1,
    ("fmap", "phasediff"): None,  # 1 DICOM series → mag1+mag2+phasediff (typically 3)
    ("fmap", "magnitude1"): None,
    ("fmap", "magnitude2"): None,
    ("fmap", "fieldmap"): None,
    ("fmap", "epi"): 1,
    ("perf", "asl"): 1,
    ("perf", "m0scan"): 1,
}


# Heuristic thresholds for distinguishing planned repeats from operator
# redos. Tuned against real Siemens Prisma fMRI data.
#
# ``REDO_WINDOW_S`` — two acquisitions with the SAME ``SeriesDescription``
# AND same ``image_type``, with no design marker in the name, falling
# within this many seconds of each other are treated as
# operator-restart-after-quick-check (the technician saw blur/motion and
# recorded again). The earlier one is flagged ``suspected_abort``.
#
# ``ABORT_MIN_FILES`` — both sides of an abort pair must have at least
# this many DICOM files. This prevents the heuristic from flagging a
# tiny derivative output (e.g. 1-file Phoenix mosaic) as a "redo" of its
# 200-file actual-acquisition companion.
#
# ``TRIVIAL_MAX_FILES`` and ``TRIVIAL_RATIO`` — a row is flagged
# ``trivial`` (likely a derivative of another series) when its file
# count is at most ``TRIVIAL_MAX_FILES`` AND there exists a same-name
# companion in the same group with at least ``TRIVIAL_RATIO`` times more
# files. SBRef and other genuinely-small but standalone series stay
# ``planned`` because they have no much-larger same-name companion.
REDO_WINDOW_S: int = 300
ABORT_MIN_FILES: int = 10
TRIVIAL_MAX_FILES: int = 2
TRIVIAL_RATIO: int = 10


_DESIGN_MARKER_RE = _re.compile(
    r"(?:^|[_-])(?:run|split|part)-?\d+", _re.IGNORECASE
)


def _reroute_b0_references_to_fmap_epi(
    rows: list[InventoryRow],
    chosen: dict[str, Classification],
) -> dict[str, Classification]:
    """Reroute likely B0-reference DWI rows to ``fmap/epi``.

    Heuristic: a row currently classified as ``dwi/dwi`` whose
    SeriesDescription contains a recognisable B0 marker (``b0``,
    ``acq-Nb0``, ``b0map``, ``b0_ref``, ``b0rf``) AND whose ``n_files`` is
    less than 50 % of the largest ``dwi/dwi`` peer in the same
    ``(subject, session)`` group is treated as a PEpolar fmap reference
    rather than a real DWI run. Single-volume reference scans for
    distortion correction are exactly what BIDS ``fmap/_epi`` is for.

    The user can re-route back to ``dwi/_dwi`` via the GUI if the b0
    series is meant as a real b=0-only DWI acquisition. We surface the
    decision in ``proposed_issues`` (added downstream by
    ``_augment_dataframe``).
    """

    rows_by_id = {r.row_id.hex: r for r in rows}

    # Per (subject, session), collect the n_files of every dwi/dwi row.
    # Used to compare each candidate against its TRUE peers (excluding
    # the candidate itself).
    peers_files: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for k, c in chosen.items():
        if c.datatype != "dwi" or c.suffix != "dwi":
            continue
        row = rows_by_id.get(k)
        if row is None:
            continue
        gk = (row.subject_hint or "", row.session_hint or "")
        peers_files[gk].append((k, row.n_files or 0))

    for k, c in list(chosen.items()):
        if c.datatype != "dwi" or c.suffix != "dwi":
            continue
        row = rows_by_id.get(k)
        if row is None:
            continue
        if not sequence_dict.looks_like_b0_reference(row.series_description):
            continue
        gk = (row.subject_hint or "", row.session_hint or "")
        # Largest n_files among OTHER dwi rows in the same session.
        other_max = max(
            (n for kk, n in peers_files.get(gk, ()) if kk != k),
            default=0,
        )
        # No real peer → keep as ``dwi`` (could be a b=0-only DWI run; the
        # user reviews). Reroute only when there's a substantially larger
        # DWI peer that this row could be a reference for.
        if other_max == 0:
            continue
        if (row.n_files or 0) >= other_max * 0.5:
            continue
        new_entities = dict(c.candidate_entities)
        # ``epi`` allows direction / acquisition / run / part etc.; nothing
        # to strip.
        chosen[k] = Classification(
            row_id=c.row_id,
            classifier=c.classifier + "+b0_reroute",
            datatype="fmap",
            suffix="epi",
            candidate_entities=new_entities,
            confidence=c.confidence,
            rationale=(
                f"{c.rationale} | rerouted to fmap/epi: SeriesDescription "
                f"{row.series_description!r} matches B0 marker and "
                f"n_files={row.n_files} ≤ 50% of peer max ({other_max})"
            ),
            skip=False,
        )

    return chosen


def _has_design_marker(seq_desc: Optional[str]) -> bool:
    """True when the SeriesDescription encodes a design-level repetition.

    Operators who type ``run-01`` / ``run-02`` / ``split-01`` etc. into the
    sequence name are explicitly signalling planned repeats. Trust them —
    the abort detector skips groups that have any design-marked member.
    """
    if not seq_desc:
        return False
    return bool(_DESIGN_MARKER_RE.search(seq_desc))


# ---------------------------------------------------------------------------
# Step 2 + 3 — building rows and running the classifier chain
# ---------------------------------------------------------------------------


def _rows_from_dataframe(df: pd.DataFrame) -> list[InventoryRow]:
    """Build :class:`InventoryRow` objects from the inventory DataFrame.

    The fmap-collapse step joins multiple SeriesInstanceUIDs with ``|``;
    we expand those so each underlying UID becomes its own InventoryRow.
    The classifier still maps results back via ``series_uid``.
    """

    rows: list[InventoryRow] = []
    for _idx, r in df.iterrows():
        source_dir = r.get("_source_dir") or ""
        if not source_dir:
            continue
        uids_field = str(r.get("series_uid") or "")
        bids_name = str(r.get("BIDS_name") or "").replace("sub-", "") or None
        session = str(r.get("session") or "").replace("ses-", "") or None
        for uid in (u for u in uids_field.split("|") if u):
            rows.append(
                InventoryRow(
                    modality="mri",
                    source=Path(source_dir),
                    series_uid=uid,
                    series_description=str(r.get("sequence") or ""),
                    subject_hint=bids_name,
                    session_hint=session,
                    n_files=int(r.get("n_files") or 0),
                    acq_time=str(r.get("acq_time") or "") or None,
                    fine_modality=str(r.get("modality") or "") or None,
                    image_type=str(r.get("image_type") or "") or None,
                    raw_metadata={
                        "source_folder": str(r.get("source_folder") or ""),
                        "df_index": _idx,
                    },
                )
            )
    return rows


def _is_classification_schema_valid(c: Classification) -> bool:
    """Return ``True`` if (datatype, suffix, candidate_entities) round-trips through the schema."""
    if c.skip or c.datatype == "discard":
        return True  # treated as a "no-emit" decision; doesn't need to be valid
    if c.datatype == "derivatives":
        # Derivatives live outside the raw BIDS schema validation surface;
        # the scan step just needs a non-empty suffix + subject. The actual
        # path is built by ``_propose_basename`` against the derivatives
        # convention, not via ``schema.build_basename``.
        return bool(c.suffix)
    if c.datatype not in schema.list_datatypes():
        return False
    if c.suffix not in schema.list_suffixes(c.datatype):
        return False
    # Validate the candidate entities (subject is filled later by cli/scan).
    test_entities = {"subject": "001", **c.candidate_entities}
    verdicts = schema.validate_entity_set(test_entities, c.datatype, c.suffix)
    return not [v for v in verdicts if v.severity is schema.Severity.ERROR]


def _run_classifier_chain(rows: list[InventoryRow]) -> dict[str, Classification]:
    """Run the BidsGuess classifier first; fall back to sequence_dict.

    Returns a mapping ``row_id (hex) -> Classification``. Each row gets at
    most one classification — the highest-confidence schema-valid result,
    or the sequence_dict fallback if BidsGuess produced nothing usable.
    """

    # Layer 1 — dcm2niix BidsGuess.
    try:
        bg_results = dcm2niix_bidsguess.classify(rows)
    except FileNotFoundError as exc:
        log.warning("BidsGuess skipped: %s", exc)
        bg_results = []

    rows_by_id = {r.row_id.hex: r for r in rows}

    chosen: dict[str, Classification] = {}
    for c in bg_results:
        if not _is_classification_schema_valid(c):
            log.debug("BidsGuess result rejected by schema: %s", c)
            continue
        # BidsGuess often emits a generic ``dwi`` suffix for series whose
        # SeriesDescription clearly marks them as scanner-derivatives
        # (``..._FA``, ``..._ADC``, ``..._TENSOR``). When that happens,
        # let the sequence_dict layer override with the correct
        # specialised suffix.
        row = rows_by_id.get(c.row_id.hex)
        if row and c.datatype == "dwi" and c.suffix == "dwi":
            if sequence_dict.detect_dwi_derivative(row.series_description):
                continue
        key = c.row_id.hex
        existing = chosen.get(key)
        if existing is None or c.confidence > existing.confidence:
            chosen[key] = c

    # Layer 3 — legacy regex/sequence-dictionary classifier (also runs the
    # DWI scanner-derivative detector for the rows we just bounced from
    # BidsGuess).
    needs_fallback = [r for r in rows if r.row_id.hex not in chosen]
    fb_results = sequence_dict.classify(needs_fallback)
    for c in fb_results:
        chosen.setdefault(c.row_id.hex, c)

    return chosen


# ---------------------------------------------------------------------------
# Step 4 + 5 — run normalization
# ---------------------------------------------------------------------------


def _entity_signature(entities: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Stable, hashable representation of an entity dict (excluding ``run``)."""
    return tuple(sorted((k, str(v)) for k, v in entities.items() if k != "run"))


_RUN_HINT_RE = _re.compile(r"(?:^|[_-])run-?0*(\d+)", _re.IGNORECASE)


def _run_hint_from_sequence(text: Optional[str]) -> Optional[int]:
    """Extract the operator-supplied run number from a sequence description, if any."""
    if not text:
        return None
    m = _RUN_HINT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _detect_aborts(
    rows: list[InventoryRow],
    chosen: dict[str, Classification],
    *,
    redo_window_s: int = REDO_WINDOW_S,
    abort_min_files: int = ABORT_MIN_FILES,
    trivial_max_files: int = TRIVIAL_MAX_FILES,
    trivial_ratio: int = TRIVIAL_RATIO,
) -> dict[str, str]:
    """Heuristically separate planned repeats from operator-redo aborts.

    Returns ``row_id_hex -> verdict`` where verdict is one of:

    * ``"isolated"`` — singleton group; no repetition.
    * ``"trivial"`` — ``n_files <= trivial_threshold``. Almost always a
      derivative output (Phoenix mosaic, MoCo summary, scout reformat).
      Excluded from abort pairing.
    * ``"planned"`` — repeat is intentional; either the operator encoded a
      design marker (``run-N`` / ``split-N`` / ``part-N``) in the sequence
      name, or the same-name companion is far enough away in time to be a
      separate planned attempt.
    * ``"suspected_abort"`` — same-name + same-image_type companion sits
      within ``redo_window_s`` later, with neither side trivial and no
      design marker in the name. Operator likely saw blur/motion on a quick
      check and re-recorded; the EARLIER attempt is flagged.

    Three signals that prevent the most common false positives:

    1. Design markers in the SeriesDescription → trust the operator.
    2. ``image_type`` mismatch → magnitude/phase/derived siblings of one
       acquisition (NOT aborts).
    3. ``trivial_threshold`` — 1-file Phoenix mosaics co-exist alongside
       full acquisitions of the same name; they're not redos.
    """

    rows_by_id = {r.row_id.hex: r for r in rows}

    groups: dict[tuple, list[str]] = defaultdict(list)
    for row_key, c in chosen.items():
        if c.skip:
            continue
        row = rows_by_id.get(row_key)
        if row is None:
            continue
        sig = _entity_signature(c.candidate_entities)
        gk = (
            row.subject_hint or "",
            row.session_hint or "",
            c.datatype,
            c.suffix,
            sig,
        )
        groups[gk].append(row_key)

    verdicts: dict[str, str] = {}

    # Pass 1 — base verdict (isolated for singletons, planned for the rest).
    for gk, members in groups.items():
        for k in members:
            verdicts[k] = "isolated" if len(members) == 1 else "planned"

    # Pass 2 — trivial detection. A row is trivial only when it's tiny
    # (``trivial_max_files``) AND another row in the same group with the
    # SAME SeriesDescription has at least ``trivial_ratio``× more files.
    # This isolates "1-file Phoenix mosaic next to 132-file actual run"
    # without false-positively catching standalone short series like SBRef.
    for gk, members in groups.items():
        if len(members) < 2:
            continue
        for k in members:
            row = rows_by_id[k]
            n = row.n_files or 0
            if not (0 < n <= trivial_max_files):
                continue
            same_name = (row.series_description or "").strip()
            for k2 in members:
                if k2 == k:
                    continue
                r2 = rows_by_id[k2]
                if (r2.series_description or "").strip() != same_name:
                    continue
                if (r2.n_files or 0) >= n * trivial_ratio:
                    verdicts[k] = "trivial"
                    break

    # Pass 3 — within each same-params group, look for redo clusters.
    for gk, members in groups.items():
        if len(members) < 2:
            continue
        # Cluster by (SeriesDescription, image_type). Siblings of one
        # acquisition typically differ in image_type (M / P / ND), so this
        # composite key isolates genuine redos of the same physical
        # acquisition from magnitude/phase pairs.
        by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for k in members:
            row = rows_by_id[k]
            name = (row.series_description or "").strip()
            img = (row.image_type or "").strip()
            by_key[(name, img)].append(k)

        for (name, _img), name_members in by_key.items():
            # Trivial outputs sit alongside real acquisitions of the same
            # name; filter them out before checking for redo pairs so we
            # only compare real attempts.
            name_members = [k for k in name_members if verdicts.get(k) != "trivial"]
            if len(name_members) < 2:
                continue
            # Trust operator-encoded design markers.
            if any(_has_design_marker(rows_by_id[k].series_description) for k in name_members):
                continue

            sorted_members = sorted(
                name_members,
                key=lambda k: (
                    _parse_dicom_time_seconds(rows_by_id[k].acq_time)
                    if _parse_dicom_time_seconds(rows_by_id[k].acq_time) is not None
                    else float("inf"),
                    rows_by_id[k].series_uid or "",
                ),
            )

            for i in range(len(sorted_members) - 1):
                k_e = sorted_members[i]
                k_l = sorted_members[i + 1]
                row_e = rows_by_id[k_e]
                row_l = rows_by_id[k_l]
                # Both sides must look like real, complete acquisitions.
                if (row_e.n_files or 0) < abort_min_files:
                    continue
                if (row_l.n_files or 0) < abort_min_files:
                    continue
                t_e = _parse_dicom_time_seconds(row_e.acq_time)
                t_l = _parse_dicom_time_seconds(row_l.acq_time)
                if t_e is None or t_l is None:
                    # No timing information: be conservative — only flag if
                    # there are 3+ same-name same-image_type acquisitions.
                    if len(sorted_members) >= 3:
                        verdicts[k_e] = "suspected_abort"
                    continue
                gap = t_l - t_e
                if 0 <= gap <= redo_window_s:
                    verdicts[k_e] = "suspected_abort"

    return verdicts


def _normalize_runs(
    rows: list[InventoryRow],
    chosen: dict[str, Classification],
    abort_verdicts: Optional[dict[str, str]] = None,
) -> dict[str, Classification]:
    """Reassign ``run-N`` per BIDS semantic.

    BIDS ``run-<index>`` MUST be used when the same set of acquisition
    parameters is repeated within a session. We approximate "same parameters"
    by ``(datatype, suffix, candidate_entities-without-run)`` and group rows
    within ``(subject, session)``.

    Within a group of size > 1, rows are ordered by:

    1. The operator-supplied run hint embedded in ``SeriesDescription``
       (e.g. ``task-foo_run-01_bold``) — this is the user's stated intent.
    2. ``acq_time`` (DICOM AcquisitionTime).
    3. ``series_uid`` lexicographic (Siemens UIDs are time-encoded).
    4. ``row_id`` (deterministic fallback).

    Groups of size 1 lose any pre-existing ``run`` entity.
    """

    rows_by_id = {r.row_id.hex: r for r in rows}
    abort_verdicts = abort_verdicts or {}

    groups: dict[tuple, list[str]] = defaultdict(list)
    for row_key, c in chosen.items():
        if c.skip:
            c.candidate_entities.pop("run", None)
            continue
        # Aborts and trivial derivative outputs are not "planned repeats";
        # exclude both from run-counting so the surviving planned
        # acquisitions get clean ``run-1, run-2, …``. The excluded row
        # keeps no run entity.
        if abort_verdicts.get(row_key) in {"suspected_abort", "trivial"}:
            c.candidate_entities.pop("run", None)
            continue
        row = rows_by_id.get(row_key)
        if row is None:
            continue
        sig = _entity_signature(c.candidate_entities)
        group_key = (
            row.subject_hint or "",
            row.session_hint or "",
            c.datatype,
            c.suffix,
            sig,
        )
        groups[group_key].append(row_key)

    for group_key, member_keys in groups.items():
        if len(member_keys) <= 1:
            for k in member_keys:
                chosen[k].candidate_entities.pop("run", None)
            continue

        def _sort_key(k: str) -> tuple:
            row = rows_by_id[k]
            run_hint = _run_hint_from_sequence(row.series_description)
            # ``None`` first hint sorts last so explicitly-numbered rows
            # take precedence over un-numbered ones.
            return (
                0 if run_hint is not None else 1,
                run_hint if run_hint is not None else 0,
                row.acq_time or "",
                row.series_uid or "",
                row.row_id.hex,
            )

        member_keys.sort(key=_sort_key)
        for idx, k in enumerate(member_keys, start=1):
            chosen[k].candidate_entities["run"] = str(idx)

    return chosen


# ---------------------------------------------------------------------------
# Step 6 — proposed names with issues
# ---------------------------------------------------------------------------


_DERIVATIVES_PIPELINE = "dcm2niix"


def _propose_basename(
    bids_name: str,
    session: str,
    classification: Classification,
) -> tuple[str, str, list[str], dict[str, str]]:
    """Return ``(datatype_or_path, basename, issues, entities)`` for a classification.

    For raw BIDS classifications, the first tuple element is the datatype
    name (``anat`` / ``func`` / ``dwi`` / ``fmap`` / ``perf`` / …).

    For ``classification.datatype == "derivatives"``, the first element is
    the *full* relative directory path (``derivatives/<pipeline>/sub-XXX``
    or ``derivatives/<pipeline>/sub-XXX/ses-Y/dwi``).

    The ``entities`` dict is the canonical entity set used to build the
    basename. The CLI stores it in the row's ``entities`` column as JSON
    so ``bidsmgr-rebuild`` can regenerate the basename later if the
    user edits any entity.
    """

    if classification.skip or classification.datatype == "discard":
        return ("", "", [], {})

    if not bids_name:
        return ("", "", ["BIDS_name missing"], {})

    entities = dict(classification.candidate_entities)
    entities.pop("subject", None)
    entities = {"subject": bids_name, **entities}
    if session:
        entities["session"] = session

    datatype = classification.datatype
    suffix = classification.suffix

    if datatype == "derivatives":
        d, b, i = _propose_derivatives_basename(entities, suffix)
        return d, b, i, entities

    # Strip entities the schema doesn't allow for this (datatype, suffix).
    allowed = set(schema.allowed_entities(datatype, suffix))
    for ent in list(entities.keys()):
        if ent != "subject" and ent not in allowed:
            entities.pop(ent, None)

    verdicts = schema.validate_entity_set(entities, datatype, suffix)
    errors = [v for v in verdicts if v.severity is schema.Severity.ERROR]
    issues = [f"{v.rule_id}: {v.message}" for v in errors]

    if errors:
        for ent in schema.required_entities(datatype, suffix):
            if not entities.get(ent):
                entities[ent] = _placeholder_for_entity(ent)

    try:
        basename = schema.build_basename(entities, datatype, suffix)
    except (ValueError, KeyError) as exc:
        log.debug("could not build basename from %s: %s", classification, exc)
        return (datatype, "", issues + [f"build_basename: {exc}"], entities)

    return datatype, basename, issues, entities


def _propose_derivatives_basename(
    entities: dict[str, str],
    suffix: str,
) -> tuple[str, str, list[str]]:
    """Build a ``derivatives/<pipeline>/sub-XXX/[ses-Y/]dwi`` path with a
    ``_desc-<SUFFIX>_dwi`` basename.

    Used for scanner-derivatives that have no canonical raw BIDS suffix
    (the most common one is ``TENSOR``). Raw BIDS suffixes for DWI
    derivatives — ``ADC``, ``FA``, ``S0map``, ``colFA``, ``expADC``,
    ``trace`` — are handled by the regular path above.
    """

    bids_name = entities.get("subject", "")
    session = entities.get("session", "")
    if not bids_name:
        return ("", "", ["BIDS_name missing"])

    parts = [f"derivatives/{_DERIVATIVES_PIPELINE}", f"sub-{bids_name}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append("dwi")
    target_dir = "/".join(parts)

    # Build a canonical basename. Reuse ``schema.build_basename`` against
    # ``dwi`` + ``_dwi`` so entity ordering matches the rest of the BIDS
    # tree, then append ``_desc-<SUFFIX>``.
    dwi_entities = dict(entities)
    allowed = set(schema.allowed_entities("dwi", "dwi"))
    for ent in list(dwi_entities.keys()):
        if ent != "subject" and ent not in allowed:
            dwi_entities.pop(ent, None)
    try:
        dwi_base = schema.build_basename(dwi_entities, "dwi", "dwi")
    except (ValueError, KeyError) as exc:
        return (target_dir, "", [f"build_basename: {exc}"])

    # ``sub-001_acq-X_run-1_dwi`` → ``sub-001_acq-X_run-1_desc-TENSOR_dwi``.
    if dwi_base.endswith("_dwi"):
        stem = dwi_base[: -len("_dwi")]
        basename = f"{stem}_desc-{suffix}_dwi"
    else:
        basename = f"{dwi_base}_desc-{suffix}"

    issues = [
        f"derivatives: scanner-computed map ({suffix!r}) — no canonical "
        f"raw BIDS suffix; routed under {target_dir}"
    ]
    return target_dir, basename, issues


def _placeholder_for_entity(entity: str) -> str:
    """Schema-format-conforming placeholder used when a required entity is missing."""
    placeholders = {"task": "TASK", "subject": "TBD"}
    return placeholders.get(entity, "TBD")


# ---------------------------------------------------------------------------
# DataFrame integration
# ---------------------------------------------------------------------------


def _augment_dataframe(
    df: pd.DataFrame,
    rows: list[InventoryRow],
    chosen: dict[str, Classification],
    abort_verdicts: Optional[dict[str, str]] = None,
    probe_stats: Optional[dict[str, ProbeFileStats]] = None,
) -> pd.DataFrame:
    """Merge classifier output back into the inventory DataFrame.

    The relationship is N rows per DataFrame row (fmap collapse joins UIDs).
    We pick the *highest-confidence* classification across the underlying
    UIDs as the proposal for the DataFrame row.
    """

    abort_verdicts = abort_verdicts or {}
    probe_stats = probe_stats or {}

    # Index rows by DataFrame index.
    rows_by_df_idx: dict[int, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        df_idx = r.raw_metadata.get("df_index")
        if df_idx is not None:
            rows_by_df_idx[df_idx].append(r)

    for col in BIDS_GUESS_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    if probe_stats:
        for col in PROBE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

    for df_idx, group_rows in rows_by_df_idx.items():
        # Pick best classification for this DataFrame row.
        candidates = [chosen.get(r.row_id.hex) for r in group_rows]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.confidence)

        bids_name = str(df.at[df_idx, "BIDS_name"] or "").replace("sub-", "")
        session = str(df.at[df_idx, "session"] or "").replace("ses-", "")
        datatype, basename, issues, entities_used = _propose_basename(
            bids_name, session, best,
        )

        # Repetition verdict: "isolated" (singleton group), "trivial"
        # (sub-derivative output), "planned" (intentional repeat), or
        # "suspected_abort" (operator-redo). Pick the worst verdict across
        # underlying rows so a partial-abort surfaces rather than being
        # hidden by a collapsed pair.
        verdicts = [abort_verdicts.get(r.row_id.hex, "") for r in group_rows]
        if "suspected_abort" in verdicts:
            rep_type = "suspected_abort"
        elif "planned" in verdicts:
            rep_type = "planned"
        elif "trivial" in verdicts:
            rep_type = "trivial"
        elif "isolated" in verdicts:
            rep_type = "isolated"
        else:
            rep_type = ""

        df.at[df_idx, "bids_guess_classifier"] = best.classifier
        df.at[df_idx, "bids_guess_datatype"] = best.datatype
        df.at[df_idx, "bids_guess_suffix"] = best.suffix
        df.at[df_idx, "bids_guess_entities"] = json.dumps(best.candidate_entities, sort_keys=True)
        df.at[df_idx, "bids_guess_confidence"] = best.confidence
        df.at[df_idx, "bids_guess_skip"] = bool(best.skip)
        df.at[df_idx, "repetition_type"] = rep_type

        annotated_issues = list(issues)
        if rep_type == "suspected_abort":
            annotated_issues.append(
                "suspected_abort: same SeriesDescription as a later companion "
                "within the redo window (likely operator restart after a "
                "noisy / blurry initial attempt)"
            )
            df.at[df_idx, "include"] = 0
        elif rep_type == "trivial":
            annotated_issues.append(
                f"trivial: {best.classifier} produced a candidate but the "
                f"series has very few files — likely a derivative output "
                f"(Phoenix mosaic / MoCo summary / scout reformat)"
            )
            df.at[df_idx, "include"] = 0

        # B0 reference reroute: surfaced via the classifier name suffix.
        if "+b0_reroute" in best.classifier:
            annotated_issues.append(
                "rerouted to fmap/epi: SeriesDescription contains a B0 marker "
                "and file count is much smaller than the longest DWI peer in "
                "this session (likely a PEpolar reference for distortion "
                "correction; user may re-route to dwi/_dwi if it is a real "
                "b=0 DWI run)"
            )

        # fmap multi-output annotation: a single ``gre_field_mapping`` (and
        # similar) DICOM series produces several NIfTI files after dcm2niix
        # conversion (magnitude1, magnitude2, phasediff). The TSV row only
        # shows one representative basename — flag it so the user knows.
        if best.datatype == "fmap" and best.suffix in {
            "phasediff", "magnitude1", "magnitude2", "magnitude", "fieldmap",
        }:
            # Only annotate when image_type marks both magnitude+phase
            # outputs (collapsed fmap row carries combined image_type "MP"
            # or contains "P").
            img_type = str(df.at[df_idx, "image_type"] or "")
            if "P" in img_type and "M" in img_type:
                annotated_issues.append(
                    "fmap multi-output: this single DICOM series produces "
                    "magnitude1 + magnitude2 + phasediff after dcm2niix "
                    "conversion; the proposed basename above is one of three "
                    "files that will be written into fmap/"
                )

        df.at[df_idx, "proposed_issues"] = " | ".join(annotated_issues)

        if best.skip:
            df.at[df_idx, "include"] = 0

        if basename:
            ext = ".tsv" if basename.endswith("_physio") else ".nii.gz"
            df.at[df_idx, "proposed_datatype"] = datatype
            df.at[df_idx, "proposed_basename"] = basename
            df.at[df_idx, "Proposed BIDS name"] = f"{datatype}/{basename}{ext}"

        # Record the canonical entities dict used to build the basename
        # in JSON so ``bidsmgr-rebuild`` can regenerate the basename
        # after the user edits an entity here. ``sort_keys`` keeps the
        # cell stable across reruns for clean diffs.
        if entities_used:
            df.at[df_idx, "entities"] = json.dumps(entities_used, sort_keys=True)

        # Probe-convert columns + anomaly detection. Aggregate every
        # probe stat across the underlying UIDs of this DataFrame row
        # (fmap collapse joins multiple UIDs into one row).
        if probe_stats:
            uids = []
            uids_field = str(df.at[df_idx, "series_uid"] or "")
            for u in uids_field.split("|"):
                u = u.strip()
                if u:
                    uids.append(u)
            n_files_total = 0
            n_nifti_total = 0
            n_volumes_max = 0
            ext_set: set[str] = set()
            seen = False
            for u in uids:
                ps = probe_stats.get(u)
                if ps is None:
                    continue
                seen = True
                n_files_total += ps.n_files
                n_nifti_total += ps.n_nifti
                if ps.n_volumes_max > n_volumes_max:
                    n_volumes_max = ps.n_volumes_max
                ext_set.update(ps.extensions)
            if seen:
                df.at[df_idx, "probe_n_files"] = n_files_total
                df.at[df_idx, "probe_n_nifti"] = n_nifti_total
                df.at[df_idx, "probe_n_volumes"] = n_volumes_max
                df.at[df_idx, "probe_extensions"] = ",".join(sorted(ext_set))

                anomaly = _probe_anomaly(
                    best.datatype, best.suffix,
                    n_nifti=n_nifti_total,
                    n_uids=len(uids),
                )
                if anomaly:
                    annotated_issues.append(anomaly)
                    df.at[df_idx, "proposed_issues"] = " | ".join(annotated_issues)

    return df


def _probe_anomaly(
    datatype: str, suffix: str, *, n_nifti: int, n_uids: int,
) -> Optional[str]:
    """Return a human-readable anomaly note when the converted output
    doesn't match what's expected for this (datatype, suffix), or
    ``None`` if everything looks normal.

    The user's headline case: a bold task that the technician manually
    aborted produces an extra volume that dcm2niix splits into a
    separate NIfTI. Detection: ``func/bold`` should produce 1 NIfTI per
    input UID; getting 2+ flags the row.
    """

    expected = EXPECTED_NIFTI_PER_UID.get((datatype, suffix))
    if expected is None:
        return None
    expected_total = expected * n_uids
    if n_nifti == expected_total:
        return None
    if n_nifti > expected_total:
        return (
            f"probe: dcm2niix produced {n_nifti} NIfTI(s) from {n_uids} input "
            f"DICOM series — expected {expected_total} for {datatype}/{suffix} "
            f"(possible split conversion: extra volume from a manually-cancelled "
            f"acquisition or unexpected multi-echo / multi-part output)"
        )
    return (
        f"probe: dcm2niix produced {n_nifti} NIfTI(s) from {n_uids} input "
        f"DICOM series — expected {expected_total} for {datatype}/{suffix} "
        f"(missing output? check dcm2niix stderr / DICOM integrity)"
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


_SLUG_BAD_RE = _re.compile(r"[^a-z0-9_-]+")
_SLUG_DASH_RE = _re.compile(r"-{2,}")


def _default_dataset_slug(dicom_root: Path) -> str:
    """Derive a safe default dataset slug from the input directory name.

    Lowercases, replaces non-``[a-z0-9_-]`` runs with a single ``-``, and
    collapses repeated dashes. Falls back to ``"dataset"`` for empty input.
    """
    raw = Path(dicom_root).name.lower()
    slug = _SLUG_BAD_RE.sub("-", raw)
    slug = _SLUG_DASH_RE.sub("-", slug).strip("-_")
    return slug or "dataset"


def _unified_column_order(df: pd.DataFrame) -> list[str]:
    """Final unified TSV column order. Locked contract:

    ``TSV(22) + BIDS_GUESS(8) + ENTITIES(1) + DATASET(1) + PROBE(4) +
    EXTENDED(3) + EEG_MEG(12) = 51``.

    The ``entities`` column carries the canonical JSON-encoded BIDS
    entity dict; the converter and ``bidsmgr-rebuild`` use it as the
    source of truth. Display columns (``proposed_basename``, ``task``,
    ``run`` …) are derived from it.

    Columns absent from ``df`` are skipped (so an MRI-only or EEG/MEG-only
    inventory still validates).
    """
    return (
        [c for c in TSV_COLUMNS if c in df.columns]
        + [c for c in BIDS_GUESS_COLUMNS if c in df.columns]
        + [c for c in BIDS_ENTITIES_COLUMNS if c in df.columns]
        + [c for c in DATASET_COLUMNS if c in df.columns]
        + [c for c in PROBE_COLUMNS if c in df.columns]
        + [c for c in EXTENDED_COLUMNS if c in df.columns]
        + [c for c in EEG_MEG_COLUMNS if c in df.columns]
    )


def _finalize_unified_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill missing columns with empty strings so the TSV is complete.

    pandas' ``concat(sort=False)`` introduces NaN in cells where one
    side didn't have the column. We replace those with ``""`` so the
    TSV is clean and consumers don't have to special-case NaN. We also
    ensure all unified-schema columns are present (even if empty).
    """
    all_cols = (
        list(TSV_COLUMNS) + list(BIDS_GUESS_COLUMNS) + list(BIDS_ENTITIES_COLUMNS)
        + list(DATASET_COLUMNS) + list(PROBE_COLUMNS) + list(EXTENDED_COLUMNS)
        + list(EEG_MEG_COLUMNS)
    )
    for col in all_cols:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")


def _empty_unified_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the unified column schema."""
    return _finalize_unified_dataframe(pd.DataFrame())


def _write_files_by_uid_sidecar(output_tsv: Path, files_by_uid: dict[str, list[str]]) -> Path:
    """Write ``<output_tsv>.files_by_uid.json.gz`` next to the inventory.

    The converter reads this sidecar to map each ``series_uid`` back to
    its source DICOM file paths without re-walking the tree.
    """
    sidecar = output_tsv.with_suffix(output_tsv.suffix + ".files_by_uid.json.gz")
    payload = json.dumps({k: list(v) for k, v in files_by_uid.items()}).encode("utf-8")
    with gzip.open(sidecar, "wb") as fh:
        fh.write(payload)
    return sidecar


def run_scan(
    dicom_root: Path,
    output_tsv: Path,
    *,
    n_jobs: int = 1,
    skip_bids_guess: bool = False,
    probe_convert: bool = False,
    dataset: Optional[str] = None,
    line_freq: Optional[float] = None,
    montage: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full scan pipeline and return the DataFrame written to TSV.

    Walks ``dicom_root`` once for each enabled modality scanner:

    * MRI scanner (``scan_dicoms_long``) — finds DICOM series.
    * EEG/MEG scanner (``scan_eeg_meg``) — finds raw EEG/MEG/iEEG/NIRS.

    Both branches run on the same input root; they look at non-
    overlapping file types so both can find their own content even in
    a multimodal study tree. Rows from each branch concatenate into one
    unified TSV (modality-specific columns are blank for rows that
    don't carry them).

    Parameters
    ----------
    n_jobs
        Number of parallel workers. Drives both DICOM-header reading
        during the scan AND the per-series dcm2niix invocations during
        the probe pass.
    probe_convert
        When ``True``, run dcm2niix on each *detected sequence* (one
        invocation per ``SeriesInstanceUID``) by symlinking just that
        series's source DICOMs into a staging folder under
        ``<output_tsv_parent>/.tmp/``. The probe pass produces
        per-series NIfTI / sidecar / bvec / bval; anomalies (e.g. a
        bold series that produced 2 NIfTI files because of an
        operator-aborted volume) surface in ``proposed_issues``. The
        ``.tmp/`` scratch tree is **always wiped** when ``run_scan``
        returns — including on error — so the user is left with the
        inventory TSV and nothing else. MRI rows only.
    """

    dataset_slug = dataset if dataset else _default_dataset_slug(dicom_root)

    df = scan_dicoms_long(
        dicom_root, output_tsv=None, n_jobs=n_jobs, dataset=dataset_slug,
    )

    # EEG/MEG branch: independent walk, merged at the end.
    df_eeg = scan_eeg_meg(
        Path(dicom_root),
        dataset=dataset_slug,
        line_freq=line_freq,
        montage=montage,
    )

    if df.empty and df_eeg.empty:
        log.warning("no DICOMs or EEG/MEG recordings found under %s", dicom_root)
        empty = _empty_unified_dataframe()
        empty.to_csv(output_tsv, sep="\t", index=False)
        return empty

    if df.empty:
        # No MRI rows; just write the EEG/MEG rows in the unified shape.
        for col in BIDS_GUESS_COLUMNS:
            if col not in df_eeg.columns:
                df_eeg[col] = ""
        merged = _finalize_unified_dataframe(df_eeg)
        merged.to_csv(output_tsv, sep="\t", index=False, columns=_unified_column_order(merged))
        print(f"Inventory written to: {output_tsv}")
        log.warning(
            "no DICOMs found; the inventory has only EEG/MEG rows. "
            "files_by_uid sidecar will not be written."
        )
        return merged

    rows = _rows_from_dataframe(df)

    chosen: dict[str, Classification] = {}
    if skip_bids_guess:
        # Only run sequence_dict (legacy regex layer).
        for c in sequence_dict.classify(rows):
            chosen.setdefault(c.row_id.hex, c)
    else:
        chosen = _run_classifier_chain(rows)

    chosen = _reroute_b0_references_to_fmap_epi(rows, chosen)
    abort_verdicts = _detect_aborts(rows, chosen)
    chosen = _normalize_runs(rows, chosen, abort_verdicts)

    probe_stats: dict[str, ProbeFileStats] = {}
    if probe_convert:
        probe_dir = Path(output_tsv).parent / ".tmp"
        files_by_uid = df.attrs.get("files_by_uid", {})
        if not files_by_uid:
            log.warning(
                "probe-convert requested but the inventory is missing "
                "df.attrs['files_by_uid']; skipping."
            )
        else:
            try:
                print(
                    f"probe-convert: per-series dcm2niix into {probe_dir} "
                    f"(n_jobs={n_jobs})"
                )
                probe_stats = probe_convert_module.probe_rows(
                    rows, probe_dir, files_by_uid, n_jobs=n_jobs,
                )
            except FileNotFoundError as exc:
                log.warning("probe-convert skipped: %s", exc)
                probe_stats = {}
            finally:
                # The probe ``.tmp/`` directory is *always* removed once
                # we've harvested the stats — including when the probe
                # itself failed. The user is left with the inventory TSV
                # and no scratch tree. The probe is non-load-bearing for
                # the inventory so cleanup must not raise.
                if probe_dir.exists():
                    try:
                        shutil.rmtree(probe_dir)
                        log.info("probe-convert: removed scratch %s", probe_dir)
                    except OSError as exc:
                        log.warning(
                            "probe-convert: could not remove %s: %s",
                            probe_dir, exc,
                        )

    df = _augment_dataframe(df, rows, chosen, abort_verdicts, probe_stats)
    df.drop(columns=["_source_dir"], errors="ignore", inplace=True)

    # Concatenate MRI + EEG/MEG rows into the unified shape. Columns that
    # only one branch populates get filled with blanks for the other.
    if not df_eeg.empty:
        merged = pd.concat([df, df_eeg], ignore_index=True, sort=False)
    else:
        merged = df
    merged = _finalize_unified_dataframe(merged)

    merged.to_csv(
        output_tsv, sep="\t", index=False,
        columns=_unified_column_order(merged),
    )
    print(f"Inventory written to: {output_tsv}")

    # Always write the per-UID DICOM file map next to the TSV. ``bidsmgr-convert``
    # reads this sidecar to find the source files for each MRI row's dcm2niix call.
    # EEG/MEG rows store ``source_file`` directly in the TSV so they don't
    # need this map.
    files_by_uid = df.attrs.get("files_by_uid", {})
    if files_by_uid:
        sidecar_path = _write_files_by_uid_sidecar(output_tsv, files_by_uid)
        print(f"files_by_uid sidecar written to: {sidecar_path}")
    elif (df["series_uid"] != "").any() if "series_uid" in df.columns else False:
        log.warning(
            "df.attrs['files_by_uid'] is empty but MRI rows are present; "
            "bidsmgr-convert will not be able to convert MRI rows from "
            "this inventory."
        )

    return merged


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="bidsmgr-scan", description=__doc__.split("\n")[0])
    parser.add_argument("dicom_root", help="Directory containing DICOM files (any depth)")
    parser.add_argument("output_tsv", help="Destination TSV file")
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=max(1, round((os.cpu_count() or 1) * 0.8)),
        help="Number of parallel workers used to read DICOM headers",
    )
    parser.add_argument(
        "--no-bids-guess",
        action="store_true",
        help="Skip the dcm2niix BidsGuess classifier and use only the legacy regex layer",
    )
    parser.add_argument(
        "--probe-convert",
        action="store_true",
        help=(
            "After scanning, run dcm2niix on each detected sequence (one "
            "invocation per SeriesInstanceUID) into a hidden "
            "<output_tsv_parent>/.tmp/ staging tree, harvest the actual "
            "files produced, and remove the staging tree. Adds "
            "probe_n_files / probe_n_nifti / probe_n_volumes / "
            "probe_extensions columns to the TSV and surfaces conversion "
            "anomalies (e.g. a bold series that split into two NIfTIs "
            "because of an operator-aborted volume) in proposed_issues. "
            "The .tmp/ directory is always removed when this command "
            "returns — including on error."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "BIDS dataset slug stamped into every row's `dataset` column. "
            "The converter writes each distinct value to "
            "<bids_parent>/<dataset>/. Defaults to a slugified form of the "
            "DICOM root directory name."
        ),
    )
    parser.add_argument(
        "--line-freq", default=None, type=float,
        help=(
            "EEG/MEG only — power-line frequency in Hz, stamped into every "
            "EEG/MEG row's `line_freq` column. Goes into "
            "PowerLineFrequency in the JSON sidecar (BIDS-required). "
            "Typical values: 50 (Europe/most of the world), 60 "
            "(Americas / parts of Asia). Per-row TSV value wins over this."
        ),
    )
    parser.add_argument(
        "--montage", default=None,
        help=(
            "EEG/MEG only — name of an mne built-in montage (e.g. "
            "standard_1005, biosemi64, easycap-M1) stamped into every "
            "EEG/MEG row's `montage` column. The converter applies this "
            "before write_raw_bids → fills electrodes.tsv + "
            "coordsystem.json. Per-row TSV value wins."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    run_scan(
        Path(args.dicom_root),
        Path(args.output_tsv),
        n_jobs=args.jobs,
        skip_bids_guess=args.no_bids_guess,
        probe_convert=args.probe_convert,
        dataset=args.dataset,
        line_freq=args.line_freq,
        montage=args.montage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
