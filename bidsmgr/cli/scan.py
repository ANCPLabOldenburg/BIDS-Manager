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
from ..classifier.user_rules import (
    ExclusionRule,
    UserHint,
    exclusion_matches,
)
from ..classifier import user_rules as user_rules_module
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
from ..inventory.pet import PET_COLUMNS, derive_suggestions as derive_pet_suggestions
from ..inventory.pet_ecat import scan_ecat
from ..inventory.probe_convert import ProbeFileStats
from ..inventory.types import InventoryRow
from ..recording_meta import (
    RecordingMetaSpec,
    dump_spec,
    load_spec,
    scaffold_sidecar_path,
)

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


def _init_object_column(df: pd.DataFrame, col: str, default: str = "") -> None:
    """Create ``df[col]`` with ``object`` dtype, defaulting to ``default``.

    Newer pandas (>= 2.3, the default on Python 3.14) infers a strict
    ``StringDtype`` from a bare ``df[col] = ""`` assignment, which then
    rejects subsequent ``df.at[idx, col] = <float or bool>`` calls with
    a ``TypeError``. Forcing ``object`` dtype on initialisation keeps
    the column permissive across mixed types, matching the legacy
    pandas 2.x behaviour the rest of this module assumes.
    """
    df[col] = pd.Series([default] * len(df), index=df.index, dtype=object)


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
        # A user hint that resolved to dwi/dwi is the user's explicit choice;
        # don't second-guess it with the B0 reroute.
        if c.classifier == "user_hint":
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


# A user-hint classification at or above this confidence is a "force" hint
# (sequence_dict emits 0.95 for force, 0.60 otherwise; dcm2niix BidsGuess is
# 0.85). Force hints override BidsGuess; non-force hints only beat the regex
# fallback.
_FORCE_HINT_CONFIDENCE = 0.90


def _user_hint_classifications(
    rows: list[InventoryRow],
    user_hints: Optional[list[UserHint]],
) -> dict[str, Classification]:
    """Return ``{row_id hex -> Classification}`` for rows matching a user hint,
    schema-validated. ``derivatives`` is rejected (user hints route to raw
    BIDS datatypes only). A hand-edited / invalid hint is dropped with a log
    rather than producing a broken basename."""
    out: dict[str, Classification] = {}
    for c in sequence_dict.classify_user_hints(rows, user_hints):
        if c.datatype == "derivatives":
            log.debug("user hint rejected (derivatives not allowed): %s", c)
            continue
        if not _is_classification_schema_valid(c):
            log.debug("user hint rejected by schema: %s", c)
            continue
        out[c.row_id.hex] = c
    return out


def _run_classifier_chain(
    rows: list[InventoryRow],
    *,
    user_hints: Optional[list[UserHint]] = None,
) -> dict[str, Classification]:
    """Run the BidsGuess classifier first; fall back to sequence_dict.

    Returns a mapping ``row_id (hex) -> Classification``. Each row gets at
    most one classification — the highest-confidence schema-valid result.

    User hints (when supplied) sit by confidence: a ``force`` hint (0.95)
    overrides even BidsGuess (0.85); a default hint (0.60) beats the regex
    fallback (0.40/0.45) but loses to BidsGuess.
    """

    user_cls = _user_hint_classifications(rows, user_hints)
    force_hints = {h: c for h, c in user_cls.items() if c.confidence >= _FORCE_HINT_CONFIDENCE}
    nonforce_hints = {h: c for h, c in user_cls.items() if c.confidence < _FORCE_HINT_CONFIDENCE}

    # Layer 1 — dcm2niix BidsGuess.
    try:
        bg_results = dcm2niix_bidsguess.classify(rows)
    except FileNotFoundError as exc:
        log.warning("BidsGuess skipped: %s", exc)
        bg_results = []

    rows_by_id = {r.row_id.hex: r for r in rows}

    # Seed force user hints (0.95) before BidsGuess so BidsGuess (0.85) can't
    # displace them.
    chosen: dict[str, Classification] = dict(force_hints)
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

    # Non-force user hints (0.60) win on rows BidsGuess did not claim, before
    # the regex fallback gets a turn.
    for key, c in nonforce_hints.items():
        if key not in chosen:
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


def _is_real_answer(value) -> bool:
    """Is this a value, or a placeholder standing in for one?

    A converter writes some keys whatever the answer, so "n/a" means it did not
    know. Counting that as answered is what once hid the fields a user most has
    to supply.
    """
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("n/a", "na", ""):
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def _finish_unified_frame(merged, exclusions) -> None:
    """The last passes over the finished inventory, whatever it holds.

    Extracted because there are two write paths, one for a tree with MRI in it
    and one for a tree without, and they had drifted. A check added to the MRI
    path silently did nothing for an EEG-only dataset: a workshop tree where
    every subject exports a rest and a video recording produced two rows with
    one name per subject, and nothing said so.

    Order matters. Exclusions first, because an excluded row is never written
    and so cannot collide. Names next. The mixed-study heads-up last, so it only
    ever appends to ``proposed_issues`` and never downgrades a more severe row
    state.
    """
    _apply_user_exclusions(merged, exclusions)

    # Two recordings resolving to one name means one silently overwrites the
    # other. Where BIDS has an answer (a run) it is applied; where it does not,
    # or where a run is already stated, the rows are left alone and reported.
    #
    # Nothing is written into the issue column: a note stored there would still
    # be showing after the user had fixed the thing it describes. The GUI
    # derives the red live from the current names, and conversion checks again.
    from ..inventory.name_collisions import assign_runs, count_collisions

    numbered = assign_runs(merged)
    if numbered:
        # Printed, not logged: this CHANGED the filenames the user will get.
        print(
            f"Named {numbered} recordings as runs so their BIDS names stay "
            "unique (they had resolved to the same name)."
        )

    flagged = count_collisions(merged)
    if flagged:
        print(
            f"WARNING: {flagged} recordings share a BIDS name with another "
            "and would overwrite each other. A run could not be assigned "
            "automatically, so give them different entities (a task or "
            "session label), or exclude one. They are shown in red in the "
            "table, and conversion will refuse until they are unique."
        )

    _flag_mixed_study_descriptions(merged)


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
            _init_object_column(df, col)

    if probe_stats:
        for col in PROBE_COLUMNS:
            if col not in df.columns:
                _init_object_column(df, col)
        # Internal (leading underscore -> dropped from the TSV): what the probe
        # conversion actually produced for each series.
        if "_derived_fields" not in df.columns:
            _init_object_column(df, "_derived_fields")

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
            derived: dict = {}
            seen = False
            for u in uids:
                ps = probe_stats.get(u)
                if ps is None:
                    continue
                seen = True
                # What dcm2niix actually wrote for this series. Only the EEG/MEG
                # scanner used to record this, so an MRI row could say nothing
                # about what the conversion answers, and its per-file form asked
                # for the repetition time the converter reads from the header.
                for key, value in (getattr(ps, "sidecar_fields", None) or {}).items():
                    if _is_real_answer(value):
                        derived.setdefault(key, value)
                n_files_total += ps.n_files
                n_nifti_total += ps.n_nifti
                if ps.n_volumes_max > n_volumes_max:
                    n_volumes_max = ps.n_volumes_max
                ext_set.update(ps.extensions)
            if seen and derived:
                df.at[df_idx, "_derived_fields"] = json.dumps(
                    derived, sort_keys=True, default=str,
                )
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

    # PET, from the DICOM Modality tag: classify anything the probe missed,
    # derive the suggestion columns, then exclude the CT companion of a
    # PET/CT study (which raw BIDS cannot hold).
    _classify_pet_rows(df)
    _fill_pet_suggestions(df)
    _flag_ct_companion_rows(df)

    _flag_nonimage_rows(df)
    return df


# Marker prepended to ``proposed_issues`` for a DERIVED non-image series.
# The GUI inventory model keys on the leading ``NONIMAGE_ISSUE_TOKEN``
# substring to paint the row's "not an image" highlight, so keep the two
# in sync (mirrors how the abort highlight keys on ``suspected_abort``).
NONIMAGE_ISSUE_TOKEN = "non-image series"
NONIMAGE_ISSUE = (
    "non-image series: the DICOM headers carry no pixel data "
    "(Rows/Columns absent), so dcm2niix cannot convert it to NIfTI. "
    "This is typically a scanner-derived object such as a Siemens "
    "diffusion TENSOR map. Excluded from conversion."
)


def _is_nonimage_flag(val: object) -> bool:
    """True only when the internal ``_has_pixel_data`` flag is explicitly
    False. NaN / blank / True (fmap-collapsed rows, EEG/MEG rows, real
    images) are treated as images."""
    if val is True or val is False:
        return val is False
    return str(val).strip().lower() in ("false", "0")


def _is_physio_row(df: pd.DataFrame, idx: object) -> bool:
    """True for Siemens CMRR ``_PhysioLog.dcm`` rows.

    Physio DICOMs carry no Image Pixel module (no Rows/Columns) just like a
    DERIVED non-image object, but unlike one they ARE convertible -- the
    ``PhysioDcmBackend`` (bidsphysio) turns them into ``_physio.tsv.gz`` +
    ``_physio.json``. So they must be exempt from the non-image exclusion.
    Identified by the ``physio`` suffix the converter dispatches on, with
    ``modality`` / basename fallbacks for robustness."""
    def _cell(col: str) -> str:
        return str(df.at[idx, col]).strip().lower() if col in df.columns else ""

    return (
        _cell("bids_guess_suffix") == "physio"
        or _cell("modality") == "physio"
        or _cell("proposed_basename").endswith("_physio")
    )


def _flag_nonimage_rows(df: pd.DataFrame) -> None:
    """Exclude + annotate DICOM series that carry no pixel data.

    A series whose DICOMs lack the Image Pixel module (Rows/Columns) is a
    DERIVED non-image object (commonly a Siemens diffusion TENSOR map, also
    structured reports / spectroscopy frames). dcm2niix cannot turn it into
    a NIfTI, so instead of letting the converter attempt it and fail with a
    cryptic ``rc=2`` we flag the row here: force ``bids_guess_skip`` /
    ``include=0`` so it is never converted, and prepend a clear reason to
    ``proposed_issues`` so it surfaces in the inventory table highlighted as
    "not an actual image". Operates on the internal ``_has_pixel_data``
    column (stamped by ``inventory.mri_dicom``, dropped from the final TSV).
    """
    if "_has_pixel_data" not in df.columns:
        return
    for df_idx in df.index:
        if not _is_nonimage_flag(df.at[df_idx, "_has_pixel_data"]):
            continue
        # Physio rows have no pixel data but ARE convertible (bidsphysio).
        if _is_physio_row(df, df_idx):
            continue
        df.at[df_idx, "bids_guess_skip"] = True
        df.at[df_idx, "include"] = 0
        existing = str(df.at[df_idx, "proposed_issues"] or "").strip()
        df.at[df_idx, "proposed_issues"] = (
            f"{NONIMAGE_ISSUE} | {existing}" if existing else NONIMAGE_ISSUE
        )


# A PET/CT or PET/MR study ships its attenuation-correction companion in the
# same export. The MR half is ordinary anat/func and converts normally, but
# BIDS 1.11 has no ``ct`` datatype, so a CT series has nowhere to go in a raw
# dataset. Flag it the way non-image series are flagged: excluded but visible
# and reversible, with the reason on the row.
CT_COMPANION_ISSUE_TOKEN = "ct series"
CT_COMPANION_ISSUE = (
    "ct series: this is the CT half of a PET/CT study. BIDS 1.11 has no "
    "'ct' datatype, so it cannot be placed in a raw BIDS dataset and is "
    "excluded from conversion. Re-tick include only if you intend to route "
    "it somewhere yourself."
)


def _flag_ct_companion_rows(df: pd.DataFrame) -> None:
    """Exclude CT series, which raw BIDS has nowhere to put.

    Keyed on the DICOM ``Modality`` tag rather than the series description,
    because vendors name the CT half anything from ``AC_CT_Brain`` to
    ``Topogram``. Reversible: the row stays in the inventory with ``include=0``
    so the user can override.
    """
    if "_dicom_modality" not in df.columns:
        return
    for idx in df.index:
        if str(df.at[idx, "_dicom_modality"] or "").strip().upper() != "CT":
            continue
        df.at[idx, "bids_guess_skip"] = True
        df.at[idx, "include"] = 0
        existing = str(df.at[idx, "proposed_issues"] or "").strip()
        df.at[idx, "proposed_issues"] = (
            f"{CT_COMPANION_ISSUE} | {existing}" if existing else CT_COMPANION_ISSUE
        )


_SUB_LABEL_RE = _re.compile(r"^sub-(\d+)$")


def _renumber_subjects_after(df: pd.DataFrame, others) -> None:
    """Shift ``df``'s ``sub-NNN`` labels past those already used in ``others``.

    Every modality scanner numbers its own subjects from ``sub-001``. Merging
    two such blocks without reconciling them makes two different people share
    one label, and at convert time the second write lands on top of the first.
    Rather than reconcile identity across modalities (a much harder problem,
    and a separate feature), simply give the incoming block a disjoint range;
    the user renames rows in the inventory to merge them deliberately.

    Rewrites ``BIDS_name`` plus the ``entities`` / basename columns derived
    from it, so the row stays internally consistent.
    """
    if df.empty or "BIDS_name" not in df.columns:
        return

    used: set[int] = set()
    for other in others:
        if other is None or other.empty or "BIDS_name" not in other.columns:
            continue
        for label in other["BIDS_name"].astype(str):
            m = _SUB_LABEL_RE.match(label.strip())
            if m:
                used.add(int(m.group(1)))
    if not used:
        return

    offset = max(used)
    remap: dict[str, str] = {}
    for label in df["BIDS_name"].astype(str):
        m = _SUB_LABEL_RE.match(label.strip())
        if m and label not in remap:
            remap[label] = f"sub-{int(m.group(1)) + offset:03d}"
    if not remap:
        return

    for idx in df.index:
        old = str(df.at[idx, "BIDS_name"]).strip()
        new = remap.get(old)
        if not new:
            continue
        df.at[idx, "BIDS_name"] = new
        token = new[len("sub-"):]
        if "entities" in df.columns:
            try:
                ents = json.loads(df.at[idx, "entities"] or "{}")
            except (TypeError, ValueError):
                ents = {}
            if ents.get("subject"):
                ents["subject"] = token
                df.at[idx, "entities"] = json.dumps(ents, sort_keys=True)
        for col in ("proposed_basename", "Proposed BIDS name"):
            if col in df.columns:
                df.at[idx, col] = str(df.at[idx, col]).replace(old, new, 1)


def _classify_pet_rows(df: pd.DataFrame) -> None:
    """Classify PET series the dcm2niix probe did not reach.

    dcm2niix's BidsGuess already returns ``pet`` for PET series, but only when
    the probe actually ran (it is opt-in, and it needs a successful trial
    conversion). The DICOM ``Modality`` tag says ``PT`` unambiguously and costs
    nothing, so it fills in for the probe here: any PET series still lacking a
    guess gets ``pet``/``pet`` at a confidence just under BidsGuess's 0.85, so
    a real probe result always wins.
    """
    if "_dicom_modality" not in df.columns:
        return
    for idx in df.index:
        if str(df.at[idx, "_dicom_modality"] or "").strip().upper() != "PT":
            continue
        if str(df.at[idx, "bids_guess_datatype"] or "").strip():
            continue  # the probe already classified it
        df.at[idx, "bids_guess_datatype"] = "pet"
        df.at[idx, "bids_guess_suffix"] = "pet"
        df.at[idx, "bids_guess_classifier"] = "dicom_modality"
        df.at[idx, "bids_guess_confidence"] = 0.80
        if "modality" in df.columns:
            df.at[idx, "modality"] = "pet"


def _fill_pet_suggestions(df: pd.DataFrame) -> None:
    """Populate the read-only PET suggestion columns from the scanned tags.

    Suggestions only: nothing here is written into a sidecar. The user adopts
    or overrides them in the dataset-metadata dialog, exactly as with the
    EEG/MEG montage and manufacturer suggestions.
    """
    if "_pet_tags" not in df.columns:
        return
    # The unified backfill runs later, so create the columns here. Object
    # dtype on purpose: a bare ``df[col] = ""`` gives newer pandas a strict
    # StringDtype that then rejects assignment (see _init_object_column).
    for col in PET_COLUMNS:
        if col not in df.columns:
            _init_object_column(df, col)
    for idx in df.index:
        tags = df.at[idx, "_pet_tags"]
        if not isinstance(tags, dict) or not tags:
            continue
        for col, value in derive_pet_suggestions(tags).items():
            if col in df.columns:
                df.at[idx, col] = value


# Marker prepended to ``proposed_issues`` for a user-excluded series. Mirrors
# the non-image precedent: row stays visible (include=0) so the user can
# re-enable it; the GUI surfaces the reason via the issues column / tooltip.
USER_EXCLUDED_ISSUE_TOKEN = "user-excluded"


def _apply_user_exclusions(
    df: pd.DataFrame,
    exclusions: Optional[list[ExclusionRule]],
) -> None:
    """Flag rows matching a user exclusion rule: ``include=0`` +
    ``bids_guess_skip`` + a ``user-excluded`` note prepended to
    ``proposed_issues``. Reversible (the row stays in the inventory; the user
    can re-tick ``include``). Matches the rule against the series description
    (``sequence``) or the relative path (``source_folder`` / ``source_file``).
    """
    if not exclusions:
        return

    def _cell(idx, col: str) -> str:
        return str(df.at[idx, col]).strip() if col in df.columns else ""

    for idx in df.index:
        sequence = _cell(idx, "sequence")
        path = _cell(idx, "source_folder") or _cell(idx, "source_file")
        matched = next(
            (r for r in exclusions
             if exclusion_matches(r, sequence=sequence, path=path)),
            None,
        )
        if matched is None:
            continue
        if "include" in df.columns:
            df.at[idx, "include"] = 0
        if "bids_guess_skip" in df.columns:
            df.at[idx, "bids_guess_skip"] = True
        note = (
            f"{USER_EXCLUDED_ISSUE_TOKEN}: matched the scan exclusion rule "
            f"'{matched.pattern}' ({matched.target}/{matched.match_mode}); "
            "excluded from conversion. Re-tick 'include' to convert it anyway."
        )
        existing = str(df.at[idx, "proposed_issues"] or "").strip() if "proposed_issues" in df.columns else ""
        if "proposed_issues" in df.columns:
            df.at[idx, "proposed_issues"] = f"{note} | {existing}" if existing else note


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


def _write_recording_meta_scaffold(merged: pd.DataFrame, output_tsv: Path):
    """Seed a recording-metadata scaffold from what the scan detected.

    Two sources feed it: EEG/MEG event codes, and the PET tracer facts every
    PET row agrees on. Both are starting points for the user, never final.

    Reads the internal ``_event_codes`` column (populated by the EEG/MEG
    scanner, dropped from the TSV itself), unions the codes across recordings,
    and writes a starter :class:`RecordingMetaSpec` with an event map of blank
    labels next to the inventory TSV. Event labels are left blank for the user
    to fill; the enrichment step skips blank labels, so an unedited scaffold is
    harmless. Returns the path written, or ``None`` when there is nothing to
    seed (no EEG/MEG rows, or no event codes).

    Device fields are intentionally NOT seeded: the manufacturer is a read-only
    suggestion (``manufacturer_suggestion`` column) and mne-bids fills the MEG
    sidecar Manufacturer itself, so seeding a default here would wrongly
    overwrite it. The scaffold is never overwritten if one already exists, so a
    CLI re-scan does not clobber labels the user has filled in (the GUI resets a
    stale scaffold when it starts a fresh scan).
    """
    codes: list[str] = []
    if "source_file" in merged.columns:
        eeg_rows = merged[merged["source_file"].astype(str).str.len() > 0]
        # Union of detected trigger/annotation codes across all recordings.
        for raw in eeg_rows.get("_event_codes", pd.Series([], dtype=object)):
            try:
                for c in json.loads(raw) if isinstance(raw, str) and raw else []:
                    if c and c not in codes:
                        codes.append(c)
            except (ValueError, TypeError):
                continue

    pet_defaults = _seed_pet_defaults(merged)

    if not codes and pet_defaults is None:
        return None

    scaffold_path = scaffold_sidecar_path(output_tsv)
    if scaffold_path.exists():
        # Preserve any labels/edits the user already made on a prior scan.
        return None

    spec = RecordingMetaSpec(
        event_maps={"*": {c: "" for c in sorted(codes)}} if codes else {},
    )
    if pet_defaults is not None:
        spec.pet_defaults = pet_defaults
    scaffold_path.write_text(dump_spec(spec), encoding="utf-8")
    return scaffold_path


def _write_converter_preview(
    merged: pd.DataFrame,
    probe_stats,
    output_tsv: Path,
) -> None:
    """Record what the conversion will answer by itself, per kind of file.

    Written on every scan, into an existing scaffold if there is one, because
    unlike the event map this is DERIVED: it describes the data as it is now,
    so a rescan should refresh it rather than preserve a stale copy. It is
    never merged into a user's answers and never written to a sidecar.

    Two sources, and the cheap one always runs: the header facts the scan read
    out of each recording anyway, and, when the user asked for a probe
    conversion, the sidecars dcm2niix actually produced.
    """
    from ..metadata.converter_preview import (
        merge_previews, preview_by_row, preview_from_inventory, preview_from_probe,
    )

    preview = merge_previews(
        preview_from_inventory(merged),
        preview_from_probe(merged, probe_stats),
    )
    by_row = preview_by_row(merged)
    if not preview and not by_row:
        return

    scaffold_path = scaffold_sidecar_path(output_tsv)
    spec = load_spec(scaffold_path) if scaffold_path.exists() else RecordingMetaSpec()
    spec.converter_preview = preview
    spec.row_preview = by_row
    scaffold_path.write_text(dump_spec(spec), encoding="utf-8")


def _seed_pet_defaults(merged: pd.DataFrame):
    """Seed the scaffold's PET block from values the scan is sure of.

    Only unambiguous facts are seeded, and only when every PET row in the
    dataset agrees on them. A tracer or radionuclide that differs between rows
    is a per-row property, not a dataset default, so seeding one would be
    wrong for the others.

    The dose is NOT seeded: it is per-injection by definition, and a shared
    default would be actively misleading. It stays a per-row suggestion the
    user adopts deliberately.

    Returns ``None`` when there is no PET in the dataset or nothing agreed on.
    """
    if "bids_guess_datatype" not in merged.columns:
        return None
    pet_rows = merged[merged["bids_guess_datatype"].astype(str) == "pet"]
    if pet_rows.empty:
        return None

    def _unanimous(column: str) -> Optional[str]:
        if column not in pet_rows.columns:
            return None
        values = {
            str(v).strip() for v in pet_rows[column] if str(v).strip()
        }
        return values.pop() if len(values) == 1 else None

    from ..recording_meta import PetAcquisitionSpec

    seeded = PetAcquisitionSpec(
        tracer_name=_unanimous("tracer_suggestion"),
        tracer_radionuclide=_unanimous("radionuclide_suggestion"),
    )
    return seeded if seeded.model_dump(exclude_none=True, exclude_defaults=True) else None


def _unified_column_order(df: pd.DataFrame) -> list[str]:
    """Final unified TSV column order. Locked contract:

    ``TSV(24) + BIDS_GUESS(8) + ENTITIES(1) + DATASET(1) + PROBE(4) +
    EXTENDED(3) + EEG_MEG(16) + PET(5) = 62``.

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
        + [c for c in PET_COLUMNS if c in df.columns]
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
        + list(EEG_MEG_COLUMNS) + list(PET_COLUMNS)
    )
    for col in all_cols:
        if col not in df.columns:
            _init_object_column(df, col)
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


# Marker prepended to ``proposed_issues`` for a row whose scan pooled multiple
# distinct DICOM StudyDescriptions. This routes the heads-up through the
# existing severity system: the GUI inventory model classifies any non-error
# ``proposed_issues`` note as a ``warn`` row, so these rows count toward the
# "warnings" chip and appear in the Issues dialog. Awareness-only: the row is
# NOT excluded or altered, and it still converts. The wording deliberately
# avoids the error-token substrings the model treats as fatal
# (``required`` / ``missing`` / ``build_basename`` / ``suspected_abort``).
MIXED_STUDY_ISSUE_TOKEN = "multiple studies in scan"


def _flag_mixed_study_descriptions(df: pd.DataFrame) -> None:
    """Flag rows (and log) when one scan pooled multiple DICOM StudyDescriptions.

    Several distinct StudyDescriptions in a single scanned source usually mean
    two studies were pooled together, or the protocol changed mid-acquisition.
    StudyDescription is not a BIDS entity and is deliberately not shown as a GUI
    column, so the heads-up is surfaced two ways instead: a one-line summary on
    the ``bidsmgr.cli.scan`` logger (the CLI surface, also the GUI Log dock), and
    a non-fatal note appended to each affected row's ``proposed_issues`` so the
    rows read as ``warn`` in the inventory table, count toward the warnings chip,
    and list in the Issues dialog (the GUI surface the user already knows).
    Nothing is excluded or rewritten. EEG/MEG rows have no StudyDescription and
    are ignored.
    """
    if "StudyDescription" not in df.columns:
        return
    studies = df["StudyDescription"].astype(str).str.strip()
    present = studies != ""
    if not present.any():
        return
    distinct = sorted(studies[present].unique())
    if len(distinct) <= 1:
        return

    # CLI / Log-dock surface: one summary line.
    counts = studies[present].value_counts()
    summary = ", ".join(f"'{name}' ({int(counts[name])} series)" for name in distinct)
    log.warning(
        "Multiple distinct DICOM StudyDescriptions found in this scan: %s. "
        "This usually means two studies were pooled or the protocol changed. "
        "If these are separate studies, scan each into its own dataset.",
        summary,
    )

    # GUI severity-system surface: append a per-row warning note so each affected
    # row shows up in the warnings chip + Issues dialog.
    if "proposed_issues" in df.columns:
        for df_idx in df.index[present]:
            this_study = studies.at[df_idx]
            others = ", ".join(f"'{s}'" for s in distinct if s != this_study)
            note = (
                f"{MIXED_STUDY_ISSUE_TOKEN}: this series is '{this_study}'; "
                f"scan also contains {others}"
            )
            existing = str(df.at[df_idx, "proposed_issues"] or "").strip()
            df.at[df_idx, "proposed_issues"] = (
                f"{existing} | {note}" if existing else note
            )

    # The strongest signal is a single subject spanning more than one study.
    if "BIDS_name" in df.columns:
        sub = df.loc[present, ["BIDS_name", "StudyDescription"]].copy()
        sub["BIDS_name"] = sub["BIDS_name"].astype(str).str.strip()
        sub = sub[sub["BIDS_name"] != ""]
        for name, grp in sub.groupby("BIDS_name"):
            subj_studies = sorted(
                grp["StudyDescription"].astype(str).str.strip().unique()
            )
            if len(subj_studies) > 1:
                log.warning(
                    "  subject %s spans %d StudyDescriptions: %s",
                    name,
                    len(subj_studies),
                    ", ".join(f"'{s}'" for s in subj_studies),
                )


def run_scan(
    dicom_root: Path,
    output_tsv: Path,
    *,
    n_jobs: int = 1,
    skip_bids_guess: bool = False,
    probe_convert: bool = False,
    preview_converter_fields: bool = True,
    dataset: Optional[str] = None,
    line_freq: Optional[float] = None,
    montage: Optional[str] = None,
    cancel_check=None,
    user_hints: Optional[list[UserHint]] = None,
    exclusions: Optional[list[ExclusionRule]] = None,
) -> pd.DataFrame:
    """Run the full scan pipeline and return the DataFrame written to TSV.

    Walks ``dicom_root`` once for each enabled modality scanner:

    * MRI scanner (``scan_dicoms_long``) — finds DICOM series, including
      PET DICOM (which dcm2niix converts like any other).
    * EEG/MEG scanner (``scan_eeg_meg``) — finds raw EEG/MEG/iEEG/NIRS.
    * ECAT scanner (``scan_ecat``) — finds ECAT PET, PET's non-DICOM
      native format.

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
        cancel_check=cancel_check,
    )

    # Honour a Stop requested during the (dominant) DICOM read before
    # starting the EEG/MEG walk.
    from ..util.cancel import raise_if_cancelled
    raise_if_cancelled(cancel_check, "scan cancelled by user")

    # EEG/MEG branch: independent walk, merged at the end.
    df_eeg = scan_eeg_meg(
        Path(dicom_root),
        dataset=dataset_slug,
        line_freq=line_freq,
        montage=montage,
    )

    # ECAT branch: a third independent walk. ECAT is PET's non-DICOM native
    # format (Siemens HRRT and older CTI scanners), detected by signature
    # rather than extension, so it cannot collide with the other two walks.
    df_ecat = scan_ecat(Path(dicom_root), cancel_check=cancel_check)
    if not df_ecat.empty:
        df_ecat["dataset"] = dataset_slug or ""
        # Each scanner numbers its subjects from sub-001 independently, so in a
        # tree holding both ECAT and other sources the labels would collide and
        # the later rows would silently overwrite the earlier ones at convert
        # time. Shift the ECAT block past everything already claimed.
        _renumber_subjects_after(df_ecat, (df, df_eeg))
        df_eeg = (
            df_ecat if df_eeg.empty
            else pd.concat([df_eeg, df_ecat], ignore_index=True, sort=False)
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
                _init_object_column(df_eeg, col)
        merged = _finalize_unified_dataframe(df_eeg)
        _finish_unified_frame(merged, exclusions)
        merged.to_csv(output_tsv, sep="\t", index=False, columns=_unified_column_order(merged))
        print(f"Inventory written to: {output_tsv}")
        scaffold_path = _write_recording_meta_scaffold(merged, Path(output_tsv))
        if scaffold_path is not None:
            print(f"Recording-metadata scaffold written to: {scaffold_path}")
        if preview_converter_fields:
            _write_converter_preview(merged, None, Path(output_tsv))
        log.warning(
            "no DICOMs found; the inventory has only EEG/MEG rows. "
            "files_by_uid sidecar will not be written."
        )
        return merged

    rows = _rows_from_dataframe(df)

    chosen: dict[str, Classification] = {}
    if skip_bids_guess:
        # Only run sequence_dict (legacy regex layer). User hints (force +
        # non-force) still apply, taking priority over the regex fallback.
        chosen = dict(_user_hint_classifications(rows, user_hints))
        for c in sequence_dict.classify(rows):
            chosen.setdefault(c.row_id.hex, c)
    else:
        chosen = _run_classifier_chain(rows, user_hints=user_hints)

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

    # Exclusions, name collisions and the mixed-study heads-up, on the unified
    # frame so they cover MRI and EEG/MEG alike. Shared with the EEG-only path
    # above, which is what stopped them drifting apart again.
    _finish_unified_frame(merged, exclusions)

    merged.to_csv(
        output_tsv, sep="\t", index=False,
        columns=_unified_column_order(merged),
    )
    print(f"Inventory written to: {output_tsv}")

    # Seed a recording-metadata scaffold from facts detected in the EEG/MEG
    # headers (event codes, device). The convert/metadata verbs auto-discover
    # it; the user fills the human-only fields (event labels, reference, ...).
    scaffold_path = _write_recording_meta_scaffold(merged, Path(output_tsv))
    if scaffold_path is not None:
        print(f"Recording-metadata scaffold written to: {scaffold_path}")

    # What the conversion will answer by itself, so the metadata template can
    # show it instead of an empty box for a field nobody has to fill in.
    if preview_converter_fields:
        _write_converter_preview(merged, probe_stats, Path(output_tsv))

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
    parser.add_argument(
        "output_tsv", nargs="?", default=None,
        help="Destination TSV file (omit when using --project)",
    )
    parser.add_argument(
        "--project", default=None, type=Path,
        help=(
            "Scan INTO a project dataset folder (created by bidsmgr-create or "
            "the GUI; created/adopted if absent). The inventory is saved as a "
            "new versioned scan under <project>/.bidsmgr/project/scans/ instead "
            "of a loose TSV, so it is resumable in the GUI and never overwrites "
            "an earlier scan. --dataset defaults to the project folder name."
        ),
    )
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
        "--rules-file", default=None,
        help=(
            "Path to a JSON file with user scan rules: "
            "{\"user_hints\": [{\"patterns\": [...], \"datatype\": \"func\", "
            "\"suffix\": \"bold\", \"task\": \"rest\", \"match_mode\": "
            "\"substring\", \"force\": false}], \"scan_exclusions\": "
            "[{\"pattern\": \"calibration\", \"target\": \"sequence\", "
            "\"match_mode\": \"substring\"}]}. Same schema the GUI Settings "
            "'Scan rules' tab persists. Hints extend the classifier; "
            "exclusions mark matching series include=0."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    parser.add_argument(
        "--no-converter-preview", action="store_true",
        help=(
            "Do not record what the conversion will fill in by itself. By "
            "default the scan notes, per kind of file, the values dcm2niix and "
            "mne-bids produce, so the metadata form can show them instead of "
            "an empty box for a field nobody has to answer. Costs nothing "
            "extra: it reads what the scan and any probe conversion already "
            "produced."
        ),
    )
    parser.add_argument(
        "--schema", default=None, metavar="VERSION",
        help=(
            "Which BIDS version to work to (for example 1.11.1). Decides which "
            "fields are declared and which entities a filename may carry. "
            "Default: the newest version shipped."
        ),
    )

    args = parser.parse_args(argv)
    # One version for the whole run, adopted before any schema question is
    # asked. Without this the flag would reach the validator alone and a
    # dataset could be filled in against one version and checked against
    # another.
    from ..schema import set_active_version
    set_active_version(args.schema)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    user_hints: Optional[list[UserHint]] = None
    exclusions: Optional[list[ExclusionRule]] = None
    if args.rules_file:
        try:
            with open(args.rules_file, "r", encoding="utf-8") as fh:
                rules_obj = json.load(fh)
        except (OSError, ValueError) as exc:
            parser.error(f"--rules-file could not be read as JSON: {exc}")
        user_hints, exclusions = user_rules_module.from_json(rules_obj)
        # Reject any bad regex up front so the scan doesn't run a no-op rule.
        for h in user_hints:
            if h.match_mode == "regex":
                for p in h.patterns:
                    err = user_rules_module.validate_regex(p)
                    if err:
                        parser.error(f"--rules-file: bad hint regex {p!r}: {err}")
        for r in exclusions:
            if r.match_mode == "regex":
                err = user_rules_module.validate_regex(r.pattern)
                if err:
                    parser.error(f"--rules-file: bad exclusion regex {r.pattern!r}: {err}")

    if args.project is None and not args.output_tsv:
        parser.error("provide an output_tsv path, or use --project <dataset>")

    if args.project is not None:
        # Project mode: scan into a new versioned bundle under the project.
        from ..cli.create import open_or_create_workspace
        from ..project import workspace
        from ..project.orchestration import import_scan_as_version, row_id

        bids_root = Path(args.project)
        open_or_create_workspace(bids_root)  # ensure scaffold + project bundle
        staging = workspace.scans_dir(bids_root).parent / ".scan_staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        staged_tsv = staging / "inventory.tsv"
        df = run_scan(
            Path(args.dicom_root), staged_tsv,
            n_jobs=args.jobs, skip_bids_guess=args.no_bids_guess,
            probe_convert=args.probe_convert,
            preview_converter_fields=not args.no_converter_preview,
            dataset=args.dataset or bids_root.name,
            line_freq=args.line_freq, montage=args.montage,
            user_hints=user_hints, exclusions=exclusions,
        )
        version = import_scan_as_version(
            bids_root, staged_tsv, raw_root=Path(args.dicom_root),
            row_ids=tuple(row_id(df, i) for i in range(len(df))),
            n_rows=len(df),
        )
        print(
            f"Scan saved as version '{version.version_id}' in project "
            f"{bids_root} ({len(df)} rows)."
        )
        return 0

    run_scan(
        Path(args.dicom_root),
        Path(args.output_tsv),
        n_jobs=args.jobs,
        skip_bids_guess=args.no_bids_guess,
        probe_convert=args.probe_convert,
        preview_converter_fields=not args.no_converter_preview,
        dataset=args.dataset,
        line_freq=args.line_freq,
        montage=args.montage,
        user_hints=user_hints,
        exclusions=exclusions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
