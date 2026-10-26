"""``bidsmgr-convert`` — drive dcm2niix from the inventory TSV into BIDS.

Closes the CLI loop scan → convert. Orchestration is straight-line code
(architecture.md §12 rule 3 — no Pipeline orchestrator).

Pipeline per (dataset, subject, session)::

    Phase 1   parallel per-series dcm2niix into
              <bids_root>/.tmp_bidsmgr/sub-<id>/<datatype>/

    Phase 2   sequential per-subject post-conv:
              - fixups.fieldmaps.apply_fieldmap_renames
              - fixups.scans_tsv.update_scans_tsv (mne-bids has
                already written a scans table for any EEG/MEG)
              - fixups.intended_for.populate_intended_for

    Phase 3   sequential per-subject merge commit:
              new subject  -> os.rename(<staging>/sub-<id>, <bids_root>/sub-<id>)
              (on Windows, retried while a reader holds the tree
              open; see _retry_while_held)
              existing subject -> merge staged files in (add new ses/datatype;
              keep or, with --overwrite, back-up-and-replace colliding files)

Failure handling: per-series failures don't abort the subject; per-subject
post-conv or atomic-commit failures write a JSON error log to
``<bids_root>/.bidsmgr/errors/sub-<id>_<utcstamp>.json`` capturing the
failed task(s), dcm2niix stderr tails, and traceback. **Staging is always
wiped** at the end of the per-subject block, success or fail
(architecture decision: forensic data goes in the error log, not in
``.tmp_bidsmgr/``).

Reference: super_plan.md §14.5.
"""

from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

import pandas as pd
from joblib import delayed  # pools are built by bidsmgr.util.parallel

import bidsmgr

from ..util.oplog import add_log_arguments, dataset_root_for, log_keywords, logged
from ..converter import (
    ConvertResult,
    ConvertTask,
    default_backends,
    dispatch,
    select_backend,
)
from ..metadata.preserve import is_mergeable, merge_file
from ..fixups.dummy_volumes import drop_leading_volumes
from ..fixups import (
    apply_fieldmap_renames,
    attach_companion_files,
    parse_companions,
    clean_mrs_outputs,
    prune_identifiers,
    convert_blood_files,
    deface_staged,
    enrich_pet_sidecars,
    enrich_physio_sidecars,
    enrich_recording_sidecars,
    place_stimulus_presentation,
    populate_intended_for,
    repair_converter_output,
    update_scans_tsv,
)
from ..deface.engines import DEFAULT_ENGINE_ID as DEFACE_DEFAULT_ENGINE
from ..deface.engines import engine_ids as deface_engine_ids
from ..recording_meta import (
    RecordingMetaSpec,
    default_spec,
    load_spec,
    merge_pet,
    resolve_effective,
    scaffold_sidecar_path,
)
from ..converter.sourcedata import MODALITIES as SOURCEDATA_MODALITIES
from ..converter.sourcedata import SourcedataOptions
from ..util.cancel import OperationCancelled, is_cancelled
from ..util.paths import safe_path_component
from ._scaffold import ensure_bidsignore, ensure_dataset_description

log = logging.getLogger(__name__)

# Datatypes converted by the mne-bids backend. These rows are run
# sequentially in Phase 1 because ``write_raw_bids`` mutates shared
# dataset files (participants.tsv / scans.tsv) on every call.
_MNE_BIDS_DATATYPES: frozenset[str] = frozenset({"eeg", "meg", "ieeg", "nirs"})

# Datatypes whose row points at a single source file rather than a DICOM
# series. ECAT PET joins the EEG/MEG family here because one ECAT file holds a
# whole dynamic study, frames included. Unlike them it does NOT go through
# mne-bids, so it stays in the parallel phase.
_FILE_BASED_DATATYPES: frozenset[str] = _MNE_BIDS_DATATYPES | frozenset({"pet"})


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------



def _apply_pet_spreadsheet(spec, path: Path, df) -> object:
    """Fold a PET metadata spreadsheet into ``spec`` as per-row overrides.

    The spreadsheet keys on whatever identifier the study used, so each row is
    matched against the inventory's participant label, BIDS name and source
    filename in turn. A key that matches nothing is reported rather than
    silently dropped: a typo in the spreadsheet is exactly the kind of mistake
    that otherwise surfaces much later as a missing field.
    """
    from ..metadata.pet_spreadsheet import read_pet_spreadsheet
    from ..project.orchestration import row_id as _row_id

    table = read_pet_spreadsheet(path)
    if not table:
        return spec

    # A sheet with no scan identifier is one row per FIELD, and the reader
    # gives it back under the empty key meaning "every PET run". Same
    # convention as the JSON route, so the two cannot disagree about what a
    # study-wide dose means.
    default = table.pop("", None)
    if default is not None:
        spec = spec.model_copy(update={
            "pet_defaults": merge_pet(default, spec.pet_defaults)
            if spec.pet_defaults is not None else default,
        })
    if not table:
        return spec

    # Every identifier an inventory row answers to, mapped to EVERY row that
    # answers to it. A participant-level key must reach all of that
    # participant's scans, not just the first: a subject with two runs got one
    # of them enriched and the other left blank when this mapped one-to-one.
    aliases: dict[str, list[str]] = {}

    def _add(key: str, rid: str) -> None:
        text = key.strip()
        if text and rid not in aliases.setdefault(text, []):
            aliases[text].append(rid)

    for i in df.index:
        rid = _row_id(df, i)
        for column in ("participant_id", "subject", "source_file"):
            if column not in df.columns:
                continue
            value = str(df.at[i, column] or "").strip()
            if not value:
                continue
            _add(value, rid)
            # A participant column may or may not carry the sub- prefix.
            _add(value[4:] if value.startswith("sub-") else f"sub-{value}", rid)
            _add(Path(value).name, rid)

    overrides = dict(spec.pet_overrides)
    matched = 0
    for key, block in table.items():
        rids = aliases.get(key) or aliases.get(Path(key).name) or []
        if not rids:
            log.warning(
                "PET spreadsheet row %r matches no inventory row; ignoring", key,
            )
            continue
        for rid in rids:
            existing = overrides.get(rid)
            if existing is None:
                overrides[rid] = block
            else:
                # Spreadsheet wins over an earlier override: it is the more
                # deliberate statement of the two.
                overrides[rid] = merge_pet(existing, block)
        matched += 1

    log.info(
        "PET spreadsheet: applied %d of %d row(s) to %d recording(s)",
        matched, len(table),
        sum(len(aliases.get(k, [])) for k in table),
    )
    return spec.model_copy(update={"pet_overrides": overrides})


def _apply_pet_metadata_json(spec, path: Path, df) -> object:
    """Fold a PET metadata JSON into ``spec`` as defaults plus per-row overrides.

    The dataset-wide block (the file's unscoped keys) becomes
    ``pet_defaults``; a ``sub-<label>`` block becomes an override on every
    row of that subject. Both layer UNDER anything the user already stated
    in the form, because a file is a starting point and the form is where
    somebody looked at this particular scan.

    The same alias matching as the spreadsheet: a subject-level key must
    reach all of that subject's runs, not just the first.
    """
    from ..metadata.pet_metadata_json import read_pet_metadata_json
    from ..project.orchestration import row_id as _row_id

    blocks = read_pet_metadata_json(path)
    if not blocks:
        return spec

    default = blocks.pop("", None)
    if default is not None:
        spec = spec.model_copy(update={
            "pet_defaults": merge_pet(default, spec.pet_defaults)
            if spec.pet_defaults is not None else default,
        })

    if not blocks:
        return spec

    aliases: dict[str, list[str]] = {}

    def _add(key: str, rid: str) -> None:
        text = key.strip()
        if text and rid not in aliases.setdefault(text, []):
            aliases[text].append(rid)

    for i in df.index:
        rid = _row_id(df, i)
        for column in ("participant_id", "subject", "bids_name", "source_file"):
            if column not in df.columns:
                continue
            value = str(df.at[i, column] or "").strip()
            if not value:
                continue
            _add(value, rid)
            _add(value[4:] if value.startswith("sub-") else f"sub-{value}", rid)
            _add(Path(value).name, rid)

    overrides = dict(spec.pet_overrides)
    for key, block in blocks.items():
        rids = aliases.get(key) or []
        if not rids:
            log.warning(
                "PET metadata block %r matches no inventory row; ignoring", key,
            )
            continue
        for rid in rids:
            existing = overrides.get(rid)
            overrides[rid] = (
                block if existing is None else merge_pet(existing, block)
            )

    return spec.model_copy(update={"pet_overrides": overrides})


@logged("convert", lambda a: (
    dataset_root_for(Path(a["tsv"]).parent)
    or (Path(a["bids_parent"]) / a["dataset"] if a["dataset"] else None),
    {"inventory": a["tsv"], "output": a["bids_parent"], "dataset": a["dataset"],
     "source data": a["raw_root"], "workers": a["n_jobs"],
     "existing files": a["on_existing"] or ("replace" if a["overwrite"] else "skip"),
     "dry run": a["dry_run"], "deface": a["deface"],
     "source data kept": a["sourcedata"],
     "volumes removed from functional runs": a["drop_volumes"],
     "physiology": f'{a["physio_layout"]}{", aligned to the run" if a["physio_align"] else ""}'}))
def run_convert(
    tsv: Path,
    bids_parent: Path,
    *,
    dataset: Optional[str] = None,
    n_jobs: int = 1,
    overwrite: bool = False,
    on_existing: Optional[str] = None,
    dry_run: bool = False,
    dcm2niix_bin: Optional[Path] = None,
    recording_meta: Optional[Path] = None,
    pet_spreadsheet: Optional[Path] = None,
    pet_metadata: Optional[Path] = None,
    raw_root: Optional[Path] = None,
    skip_residuals: bool = True,
    force_edf: bool = False,
    preserve_curation: bool = True,
    deface: bool = False,
    deface_engine: str = DEFACE_DEFAULT_ENGINE,
    cancel_check=None,
    intended_for: str = "uri",
    b0_field: bool = True,
    sourcedata: str = "none",
    sourcedata_modalities=SOURCEDATA_MODALITIES,
    drop_volumes: int = 0,
    physio_layout: str = "by_rate",
    physio_align: bool = False,
) -> int:
    """Convert every commit-ready row in ``tsv`` to BIDS under ``bids_parent``.

    ``drop_volumes`` is removed from the start of every functional run whose
    ``drop_volumes`` cell is empty (a cell wins, 0 included); the run's
    physiology starts at the first volume kept too. ``physio_layout`` is
    ``"by_rate"`` (one table per sampling rate), ``"one_interpolated"`` (one
    table, slower channels interpolated) or ``"one_as_recorded"`` (one table,
    n/a between a slower channel's samples); ``physio_align`` crops physiology
    to its run.

    Returns 0 on success, non-zero if any subject failed.

    ``recording_meta`` points at an optional recording-metadata JSON. Its
    dataset defaults supply EEG/MEG ``line_freq`` / ``montage`` when the
    inventory cell is blank, and its richer fields (reference, ground,
    filters, device, institution, event maps, ...) are folded into the BIDS
    sidecars after the write. When omitted, an EMPTY default spec is used:
    no power-line frequency is invented, and mne-bids writes
    ``PowerLineFrequency: "n/a"`` unless the recording's own header, an
    inventory cell or a template supplies one.

    ``skip_residuals`` (default True) drops the dcm2niix residual/secondary
    outputs -- derived single-volume duplicates split off one input series
    (e.g. ``..._bolda`` next to ``..._bold``). Pass False to keep them.

    ``preserve_curation`` (default True) only matters when a subject already
    exists and a file is being replaced. It merges JSON sidecars and
    ``*_scans.tsv`` field by field instead of overwriting, so metadata somebody
    curated in the Editor survives a re-conversion of the same subject. Pass
    False to have the fresh conversion win outright.

    ``sourcedata`` (``none``, ``copy``, ``archive``) keeps each subject's
    source files in ``<dataset>/sourcedata/`` for ``sourcedata_modalities``
    (``converter/sourcedata``). Refused before anything is written when the
    disk cannot hold them.
    """
    from ..converter.physio_cmrr import LAYOUTS as _PHYSIO_LAYOUTS

    if physio_layout not in _PHYSIO_LAYOUTS:
        raise ValueError(f"unknown physio layout {physio_layout!r}; choose from "
                         f"{', '.join(_PHYSIO_LAYOUTS)}")
    tsv = Path(tsv)
    src_options = SourcedataOptions.parse(sourcedata, sourcedata_modalities)
    bids_parent = Path(bids_parent)

    # Resolve the existing-subject policy. ``--overwrite`` is the back-compat
    # alias for ``replace``; the default is the safe-by-default ``skip`` (merge
    # new files, never touch existing ones).
    if on_existing is None:
        on_existing = "replace" if overwrite else "skip"
    if on_existing not in ("skip", "update", "replace", "error"):
        raise ValueError(
            f"invalid on_existing {on_existing!r}; "
            "expected skip / update / replace / error"
        )

    # One-line version banner so a user's conversion log unambiguously
    # records which copy of the package executed (matters when the same
    # machine has both a pip-installed and a dev-checkout install, or
    # when a Windows user pulled but is still running a stale process).
    log.info(
        "bidsmgr %s @ %s | platform=%s py=%s",
        bidsmgr.__version__,
        Path(bidsmgr.__file__).resolve().parent,
        sys.platform,
        sys.version.split()[0],
    )

    # Candidates the EEG/MEG row resolver uses when ``source_file`` is
    # stored relative-to-raw-root in the inventory TSV. We try (in order):
    # explicit ``raw_root`` arg, then the TSV's parent (the GUI's default
    # is to write the TSV at ``<raw_root>/.bidsmgr_scan.tsv``), then the
    # current working directory. The MRI path is unaffected — DICOM files
    # come from the ``files_by_uid`` sidecar in absolute form.
    source_search_roots: tuple[Path, ...] = tuple(
        p for p in [
            Path(raw_root) if raw_root is not None else None,
            tsv.parent,
        ] if p is not None
    )

    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    if "dataset" not in df.columns:
        raise ValueError(
            f"{tsv} has no `dataset` column — re-run `bidsmgr-scan` to "
            "regenerate the inventory with the new column."
        )

    # Defensive in-memory rebuild: the user may have hand-edited the
    # entities JSON (or display cells) in a spreadsheet without running
    # `bidsmgr-rebuild`. Reconcile here so the conversion always sees
    # the freshest BIDS basenames. The TSV file on disk is untouched.
    if "entities" in df.columns:
        from ..inventory.rebuild import rebuild_from_entities
        df, rebuild_report = rebuild_from_entities(df)
        if rebuild_report.rows_updated:
            log.info(
                "in-memory rebuild reconciled %d rows from entities JSON",
                rebuild_report.rows_updated,
            )
        for w in rebuild_report.warnings[:5]:
            log.warning("rebuild: %s", w)

    df = _filter_convertible_rows(df)
    if dataset:
        df = df[df["dataset"] == dataset].copy()

    # Nothing is written until every row is known to want a different name.
    # The scan resolves what BIDS has an answer for and flags the rest, but a
    # TSV can be edited by hand between the two steps, and the cost of being
    # wrong here is a recording silently overwritten by the next one.
    #
    # This is about two rows in THIS run colliding with each other. Whether a
    # destination already exists from an earlier conversion is a different
    # question, and ``--on-existing`` answers it.
    from ..inventory.name_collisions import (
        NameCollisionError,
        describe_collisions,
    )

    collision_report = describe_collisions(df)
    if collision_report:
        raise NameCollisionError(collision_report)

    # The files_by_uid sidecar is only required when the inventory
    # contains MRI rows. EEG/MEG-only inventories store the recording
    # path in each row's ``source_file`` column.
    has_mri_rows = (
        "series_uid" in df.columns
        and (df["series_uid"].astype(str).str.len() > 0).any()
    )
    files_by_uid = _load_files_by_uid_sidecar(tsv, required=has_mri_rows)
    if raw_root is not None and files_by_uid:
        # The DICOM paths are absolute, from the scan. When the data has
        # been moved and ``raw_root`` says where to (a relink in the GUI, or
        # --raw-root), they are re-based onto it; otherwise unchanged.
        from ..inventory.sources import ScanSources, scanned_root

        files_by_uid = ScanSources(
            Path(raw_root), scanned_root(df, files_by_uid, recorded=raw_root), files_by_uid,
        ).relocated_files_by_uid()

    if df.empty:
        log.warning("no rows to convert (after filtering)")
        return 0

    # Recording-metadata spec for EEG/MEG enrichment. Precedence: an explicit
    # --recording-meta path, else the scaffold the scan wrote next to the TSV
    # (auto-discovered), else an empty default spec, which supplies nothing.
    if recording_meta is not None:
        spec = load_spec(Path(recording_meta))
        log.info("loaded recording metadata from %s", recording_meta)
    else:
        auto_meta = scaffold_sidecar_path(tsv)
        if auto_meta.exists():
            spec = load_spec(auto_meta)
            log.info("auto-loaded recording-metadata scaffold %s", auto_meta)
        else:
            spec = default_spec()

    # A PET metadata spreadsheet layers on top of the spec. A study's
    # radiochemistry usually already exists as a table exported from the lab's
    # records, and retyping it is both tedious and a fresh chance to mistype a
    # dose. Rows are matched on participant label or source filename, whichever
    # the spreadsheet used.
    # The JSON first, then the spreadsheet: a spreadsheet is one row per
    # SCAN and a JSON is usually one block per study, so the more specific
    # statement layers on top of the more general one.
    if pet_metadata is not None:
        spec = _apply_pet_metadata_json(spec, Path(pet_metadata), df)
    if pet_spreadsheet is not None:
        spec = _apply_pet_spreadsheet(spec, Path(pet_spreadsheet), df)

    task_options = TaskOptions(drop_volumes=max(0, int(drop_volumes or 0)),
                               physio_layout=physio_layout, physio_align=bool(physio_align))

    backends = default_backends(dcm2niix_bin=dcm2niix_bin)
    # Capture dcm2niix version once for provenance — it's still the
    # primary backend for MRI rows.
    primary = select_backend("mri", dcm2niix_bin=dcm2niix_bin)
    dcm2niix_version = _dcm2niix_version_string(primary.binary)

    if src_options.enabled and not dry_run:
        # Nothing is written before the copies are known to fit.
        from ..converter import sourcedata as _sd

        needed = 0
        for _name, dataset_df in df.groupby("dataset"):
            for _subject, subject_df in dataset_df.groupby("participant_id"):
                needed += _sd.estimate(_sd.plan(_build_tasks_for_subject(
                    subject_df, bids_parent, files_by_uid,
                    source_search_roots=source_search_roots,
                    skip_residuals=skip_residuals, force_edf=force_edf, spec=spec,
                    options=task_options,
                ), src_options))[1]
        problem = _sd.space_problem(bids_parent, needed)
        if problem:
            log.error("%s Nothing was converted.", problem)
            return 1
        log.info("source data kept in sourcedata/ (%s, %s): %s", src_options.form,
                 ", ".join(m for m in SOURCEDATA_MODALITIES if m in src_options.modalities),
                 _sd.human_size(needed))

    n_failed = 0
    n_new = 0
    n_merged = 0
    n_collision = 0
    cancelled = False
    for dataset_name, dataset_df in df.groupby("dataset"):
        if cancelled:
            break
        if not dataset_name:
            log.warning("skipping %d rows with empty dataset name", len(dataset_df))
            continue

        bids_root = bids_parent / str(dataset_name)
        if not dry_run:
            bids_root.mkdir(parents=True, exist_ok=True)
            ensure_dataset_description(bids_root, generated_by={
                "Name": "bidsmgr",
                "Version": bidsmgr.__version__,
                "Description": "dcm2niix-direct backend",
                "Container": {"Type": "binary", "Tag": dcm2niix_version},
            })
            # Keep .bidsmgr/ + .tmp_bidsmgr/ out of the official bids-validator.
            ensure_bidsignore(bids_root)
            # Pre-convert collision summary: which incoming subjects already
            # exist on disk (incremental add into an existing dataset).
            _log_existing_subject_summary(bids_root, dataset_df, on_existing)

        for subject, subject_df in dataset_df.groupby("participant_id", sort=True):
            # Stop here if the user requested it. Subjects committed before
            # this point stay (the desired "stop now" behaviour); nothing
            # half-written is left because the commit is per-subject atomic.
            if is_cancelled(cancel_check):
                cancelled = True
                break
            if not subject:
                log.warning("skipping rows with empty participant_id")
                continue

            tasks = _build_tasks_for_subject(
                subject_df, bids_root, files_by_uid,
                source_search_roots=source_search_roots,
                skip_residuals=skip_residuals,
                force_edf=force_edf,
                spec=spec,
                options=task_options,
            )
            if not tasks:
                continue

            if dry_run:
                _print_tasks(tasks)
                continue

            # Target dir derives from the task's subject entity (e.g. "001"),
            # the same way _convert_subject builds subj_segment.
            existed = (
                bids_root / f"sub-{safe_path_component(tasks[0].subject)}"
            ).exists()
            try:
                _convert_subject(
                    tasks, bids_root, backends,
                    n_jobs=n_jobs, on_existing=on_existing,
                    dcm2niix_version=dcm2niix_version,
                    cancel_check=cancel_check,
                    spec=spec,
                    preserve_curation=preserve_curation,
                    deface=deface,
                    deface_engine=deface_engine,
                    intended_for=intended_for,
                    b0_field=b0_field,
                    sourcedata=src_options,
                )
                if existed:
                    n_merged += 1
                else:
                    n_new += 1
            except OperationCancelled:
                log.warning("conversion stopped by user (sub-%s not committed)", subject)
                cancelled = True
                break
            except CollisionAbort as exc:
                # Expected stop under on_existing="error": clean message, no
                # forensic traceback. Counts as a failure so rc is non-zero.
                log.warning("collision: %s", exc)
                n_collision += 1
                n_failed += 1
            except Exception:
                log.exception("subject %s failed during conversion", subject)
                n_failed += 1

    # Results summary.
    parts = [f"{n_new} new", f"{n_merged} merged into existing"]
    if n_collision:
        parts.append(f"{n_collision} collision(s)")
    if n_failed:
        parts.append(f"{n_failed} failed")
    log.info("convert summary: %s", ", ".join(parts))

    if cancelled:
        log.warning("conversion stopped by user before all subjects were processed")
        return 130  # 128 + SIGINT, conventional "interrupted" code
    return 0 if n_failed == 0 else 1


# ---------------------------------------------------------------------------
# Per-subject pipeline (the three phases)
# ---------------------------------------------------------------------------


def _convert_subject(
    tasks: list[ConvertTask],
    bids_root: Path,
    backends: list,
    *,
    n_jobs: int,
    on_existing: str,
    dcm2niix_version: str,
    cancel_check=None,
    spec: Optional[RecordingMetaSpec] = None,
    preserve_curation: bool = True,
    deface: bool = False,
    deface_engine: str = DEFACE_DEFAULT_ENGINE,
    intended_for: str = "uri",
    b0_field: bool = True,
    sourcedata: Optional[SourcedataOptions] = None,
) -> None:
    """Run Phases 1–3 for a single (dataset, subject, session) group, then
    keep its source data when asked."""
    subject = tasks[0].subject
    # Subject label is inventory-derived (user-editable). Sanitise into
    # a portable path component so a stray ``|``, ``:`` or trailing dot
    # cannot wedge ``mkdir`` on Windows.
    subj_segment = f"sub-{safe_path_component(subject)}"
    staging = bids_root / ".tmp_bidsmgr" / subj_segment
    staging.mkdir(parents=True, exist_ok=True)
    source_staging = bids_root / ".tmp_bidsmgr" / "sourcedata" / subj_segment

    results: list[ConvertResult] = []
    failed_phase: Optional[str] = None
    exception_obj: Optional[BaseException] = None

    try:
        # Phase 1: parallel per-series dispatch + conversion.
        results = _phase1_parallel_dcm2niix(
            tasks, staging, backends, n_jobs=n_jobs, cancel_check=cancel_check,
        )

        # Phase 2: per-subject post-conv (sequential, fast).
        rename_map = apply_fieldmap_renames(staging)
        n_scans_tsv = update_scans_tsv(staging, rename_map)
        n_intended_for = populate_intended_for(
            staging, subject=subject, session=tasks[0].session,
            form=intended_for, b0_field=b0_field,
            links=_user_fieldmap_links(results, rename_map),
        )
        # EEG/MEG sidecar/channels/events enrichment from the recording-
        # metadata spec (no-op for MRI-only subjects or when spec is None).
        n_enriched = enrich_recording_sidecars(staging, tasks, spec)
        # PET sidecar enrichment: rename DICOM-derived keys to their BIDS
        # names, fill the radiochemistry the scanner cannot record, fix the
        # fields BIDS types as per-frame arrays, and prune the patient
        # identifiers dcm2niix leaves behind under ``-ba n``. Runs even with
        # no spec, since the rename, type-fix and prune passes need no input.
        n_enriched += enrich_pet_sidecars(staging, tasks, spec)
        # MR spectroscopy carries its metadata INSIDE the .nii.gz, as NIfTI
        # extension 44, which is where the NIfTI-MRS standard puts the
        # acquisition parameters a spectrum cannot be read without. It is
        # also where dcm2niix puts the patient's name, ID, date of birth,
        # sex and weight. No sidecar pass can reach inside an image, so
        # this one opens it: measured on real data the moment MRS
        # conversion was enabled, every _svs file carried all five.
        n_enriched += clean_mrs_outputs(staging, tasks)
        # Physio: converter.physio_cmrr writes the fields a physio sidecar
        # cannot do without. This fills what can be DERIVED, asking the
        # active schema which fields it declares rather than hard-coding one
        # BIDS version.
        n_enriched += enrich_physio_sidecars(staging)
        # Copy any per-row curated companion files (events/beh/stim/...) into
        # the staged tree (place + name only; no conversion).
        n_enriched += attach_companion_files(staging, tasks)
        # Dummy scans the user asked to remove, from the image, its sidecar
        # and its events (which the line above may just have attached). Its
        # physiology was written starting at the first volume kept.
        n_enriched += drop_leading_volumes(results)
        # The screen an eye tracker measured gaze on, beside the run's events
        # (which the line above may just have attached).
        n_enriched += place_stimulus_presentation(staging, tasks)
        # A PET run's blood curves, which ARE converted: PMOD exports become the
        # BIDS blood tables, named for the run they belong to. Source data the
        # user pointed at, so it belongs to the conversion.
        n_enriched += convert_blood_files(staging, tasks)
        # Fix what the CONVERTER wrote: a key whose spelling differs from the
        # standard's only in case, and a scalar where the schema declares an
        # array. Runs last so it sees what the modality fixups produced.
        #
        # What the USER stated is NOT applied here. That is the metadata step's
        # job, so this verb produces a faithful conversion and no opinions.
        n_enriched += repair_converter_output(staging)
        # Identifiers, dataset-wide and LAST, so nothing above can put one
        # back. dcm2niix is run with ``-ba n`` to keep SeriesInstanceUID for
        # provenance, which also keeps the patient's name, ID, date of
        # birth, age, sex, size and weight in every sidecar it writes. Until
        # this existed only the PET fixup pruned them, so a PET dataset was
        # clean and an MRI one was not: 58 of 71 MRI sidecars in this lab's
        # own data carried all seven. BIDS is explicit that a sidecar must
        # not carry identifying information, so there is no switch here.
        n_enriched += prune_identifiers(staging)
        # Remove faces LAST, and before the commit, which is the whole point of
        # doing it here: the identifiable image exists only inside the staging
        # directory, for a few seconds, and what lands in the dataset was never
        # identifiable. Nothing is backed up because there is nothing to
        # restore to. A failure is logged and does not fail the conversion.
        if deface:
            try:
                n_defaced = deface_staged(
                    staging, tasks, engine_id=deface_engine,
                )
            except Exception as exc:  # noqa: BLE001
                # A converted subject with a face on it is a problem the user
                # can fix afterwards. A subject that failed to convert because
                # the defacer threw is a problem they cannot. This happened for
                # real: a stale engine id in QSettings raised a KeyError here
                # and took the whole conversion with it.
                n_defaced = 0
                log.warning(
                    "sub-%s: defacing failed (%s: %s); the subject is being "
                    "converted WITHOUT it",
                    subject, type(exc).__name__, exc,
                )
            # Said out loud rather than folded into the enrichment count. A
            # user who asked for defacing needs to see whether it happened,
            # and how many, because the answer 0 is the one that matters.
            log.info(
                "sub-%s: removed faces from %d image(s)", subject, n_defaced,
            )
        _prune_empty_dirs(staging)

        # Phase 3: atomic commit. Use the same sanitised segment we
        # built for the staging dir so the commit target name is
        # consistent across OSes.
        target = bids_root / subj_segment
        _merge_commit(
            staging, target, on_existing=on_existing,
            preserve_curation=preserve_curation,
        )
        _write_provenance(
            bids_root, subj_segment, results, rename_map,
            n_intended_for, n_scans_tsv,
            dcm2niix_version=dcm2niix_version, n_enriched=n_enriched,
        )
        log.info("committed sub-%s to %s", subject, target)
        if sourcedata is not None and sourcedata.enabled:
            _keep_source_data(tasks, sourcedata, bids_root, subj_segment, source_staging,
                              on_existing=on_existing, cancel_check=cancel_check)

    except OperationCancelled:
        # User Stop, not a failure: no forensic error log. The ``finally``
        # wipes staging so this subject leaves nothing half-written; the
        # caller (run_convert) stops the subject loop.
        raise
    except CollisionAbort:
        # on_existing="error" tripped: collision detected before anything was
        # moved, so the target is intact. Not a crash; re-raise without a
        # forensic error log (the caller logs the message).
        raise
    except Exception as exc:
        # Determine which phase blew up: results being non-empty means
        # Phase 1 finished before something went wrong.
        if not results:
            failed_phase = "phase1"
        elif not (bids_root / subj_segment).exists():
            failed_phase = "phase2_or_phase3"
        else:
            failed_phase = "phase3_provenance"
        exception_obj = exc
        _write_error_log(
            bids_root, subject, tasks, results, exc, failed_phase,
        )
        raise
    finally:
        # Always wipe staging (success or fail; forensic data is in the error log).
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if source_staging.exists():
            shutil.rmtree(source_staging, ignore_errors=True)
        try:
            source_staging.parent.rmdir()
        except OSError:
            pass
        # Try to remove the empty .tmp_bidsmgr container too.
        # mne-bids may have written dataset-level files there (README,
        # dataset_description.json, participants.tsv/json) when it
        # treated `.tmp_bidsmgr/` as its BIDS root for this subject —
        # those are stale (the metadata module writes the canonical
        # versions later at the real BIDS root). Strip them along with
        # any macOS-injected ``.DS_Store`` before rmdir.
        tmp_root = bids_root / ".tmp_bidsmgr"
        if tmp_root.is_dir():
            for stray in tmp_root.iterdir():
                # Preserve any in-progress sub-* dirs (other subjects
                # currently mid-conversion; sequential loop today, but
                # be defensive). Remove plain files only.
                if stray.is_file():
                    try:
                        stray.unlink()
                    except OSError:
                        pass
            try:
                tmp_root.rmdir()
            except OSError:
                pass

        # Surface per-series failures via log even if no exception was raised.
        if exception_obj is None:
            for r in results:
                if not r.success:
                    log.warning(
                        "sub-%s task %s failed: %s",
                        subject, r.task.basename, r.error,
                    )
                    # The dcm2niix stderr tail is captured on the result
                    # but useless if never logged — surface it so the
                    # failure is actionable without digging through the
                    # JSON error log. ``rc=2`` is generic; the tail is
                    # what tells you "MAX_PATH overflow" vs "bad header"
                    # vs "no DICOMs found".
                    tail = (r.dcm2niix_stderr_tail or "").strip()
                    if tail:
                        log.warning(
                            "sub-%s task %s dcm2niix stderr (tail):\n%s",
                            subject, r.task.basename, tail,
                        )


def _phase1_parallel_dcm2niix(
    tasks: list[ConvertTask],
    staging: Path,
    backends: list,
    *,
    n_jobs: int,
    cancel_check=None,
) -> list[ConvertResult]:
    """Run per-series conversion in parallel; return per-task results.

    Backend selection is per-task: each task is dispatched to the
    first registered backend whose ``can_handle()`` returns True.

    A Stop request raises :class:`OperationCancelled` once the in-flight
    task(s) return -- new tasks stop dispatching, the subject's staging is
    wiped by the caller's ``finally``, and nothing is committed.
    """
    from ..util.cancel import OperationCancelled, is_cancelled

    if is_cancelled(cancel_check):
        raise OperationCancelled("conversion cancelled by user")

    # mne-bids' ``write_raw_bids`` rewrites shared dataset files
    # (``participants.tsv`` and the subject's ``*_scans.tsv``) on *every*
    # call. Running several EEG/MEG recordings for one subject in parallel
    # therefore races on those files: one worker truncates+rewrites while
    # another reads, and mne-bids blows up ("loadtxt: input contained no
    # data" -> IndexError). dcm2niix (MRI) and physio tasks write only their
    # own independent outputs and are safe to parallelise. So split the work:
    # mne-bids-backed tasks run sequentially, the rest run in the pool.
    serial = [t for t in tasks if t.datatype in _MNE_BIDS_DATATYPES
              and t.modality != "eyetrack"]
    parallel_tasks = [t for t in tasks if t not in serial]

    out: list[ConvertResult] = []

    for t in serial:
        if is_cancelled(cancel_check):
            raise OperationCancelled("conversion cancelled by user")
        out.append(_phase1_one(backends, t, staging))

    if parallel_tasks:
        if n_jobs == 1 or len(parallel_tasks) == 1:
            for t in parallel_tasks:
                if is_cancelled(cancel_check):
                    raise OperationCancelled("conversion cancelled by user")
                out.append(_phase1_one(backends, t, staging))
        else:
            from ..util.parallel import preferred_backend, run_parallel

            # Few tasks aren't worth a process pool; otherwise take the fast
            # one unless bidsmgr.util.parallel has already seen a pool die in
            # this session (run_parallel falls back and remembers).
            if len(parallel_tasks) < 4:
                parallel_backend = "threading"
            else:
                parallel_backend = preferred_backend() or "loky"

            # Collect into a fresh list rather than straight into ``out``: a
            # retry on the threading backend re-runs every task, and appending
            # to ``out`` would leave the first attempt's partial results behind
            # as duplicates.
            def _collect(gen):
                collected: list[ConvertResult] = []
                for res in gen:
                    collected.append(res)
                    if is_cancelled(cancel_check):
                        raise OperationCancelled("conversion cancelled by user")
                return collected

            out.extend(run_parallel(
                lambda: (
                    delayed(_phase1_one)(backends, t, staging)
                    for t in parallel_tasks
                ),
                n_jobs=n_jobs,
                consume=_collect,
                backend=parallel_backend,
                return_as="generator",
            ))

    return out


def _phase1_one(backends: list, task: ConvertTask, staging: Path) -> ConvertResult:
    """Picklable single-task wrapper.

    Dispatches to the first matching backend per :func:`dispatch`.
    Catches any exception so a single bad task doesn't take down the
    whole subject.

    ``staging`` is the per-subject root; this wrapper picks the session
    subdir (when the task has a session) so the backend writes to
    ``<staging>/ses-<label>/<datatype>/``.
    """
    # Session label is inventory-derived (user-editable). Run it
    # through ``safe_path_component`` so an illegal-on-Windows char
    # (``|``, ``:``, trailing dot, …) cannot wedge ``mkdir`` here.
    if task.session:
        target_root = staging / f"ses-{safe_path_component(task.session)}"
    else:
        target_root = staging
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        backend = dispatch(backends, task)
    except LookupError as exc:
        return ConvertResult(
            task=task, success=False, error=str(exc),
        )
    try:
        return backend.convert(task, target_root)
    except Exception as exc:
        log.exception("backend %s raised for %s", backend.name, task.basename)
        return ConvertResult(
            task=task, success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty subdirectories under ``root`` (bottom-up).

    Avoids committing empty datatype folders (e.g. ``func/`` left behind
    when its only series failed conversion). ``root`` itself is preserved.
    """
    if not root.is_dir():
        return
    # walk bottom-up so deeper dirs are evaluated before their parents.
    for path in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # not empty; leave it.
                pass


class CollisionAbort(Exception):
    """Raised by :func:`_merge_commit` under ``on_existing="error"`` when the
    staged subject has files that already exist on disk. The caller logs the
    message (no forensic error log) and counts the subject as failed.
    """


_ON_EXISTING_VERB = {
    "skip": "merge new files, keep existing",
    "update": "merge, replace changed files",
    "replace": "merge, replace colliding files (backed up)",
    "error": "abort on any file collision",
}


def _log_existing_subject_summary(bids_root: Path, dataset_df, on_existing: str) -> None:
    """Pre-convert heads-up: which incoming subjects already exist on disk.

    Awareness for incremental conversion: a subject already on disk will be
    merged into per ``on_existing``. Informational only (the per-subject merge
    counts + the final summary report the actual outcome). ``participant_id`` values
    (``sub-XXX``) match the on-disk subject dir names for the common case.
    """
    if "participant_id" not in dataset_df.columns:
        return
    incoming = sorted({str(s).strip() for s in dataset_df["participant_id"] if str(s).strip()})
    on_disk = {p.name for p in bids_root.glob("sub-*") if p.is_dir()}
    if not incoming or not on_disk:
        return  # brand-new dataset (or no subjects): nothing to flag
    already = [s for s in incoming if s in on_disk]
    new = [s for s in incoming if s not in on_disk]
    msg = (
        f"converting into existing dataset '{bids_root.name}' "
        f"({len(on_disk)} subject(s) present): {len(new)} new"
    )
    if already:
        shown = ", ".join(already[:8]) + ("" if len(already) <= 8 else " ...")
        msg += (
            f", {len(already)} already present ({shown}) "
            f"-> {_ON_EXISTING_VERB[on_existing]}"
        )
    log.info(msg)


def sourcedata_estimate(df: pd.DataFrame, sources, options: SourcedataOptions
                        ) -> tuple[int, int, list[str]]:
    """``(files, bytes, modalities)`` that keeping the source data of the
    rows a conversion would convert adds, for the confirmation before Run.
    ``sources`` is the scan's :class:`~bidsmgr.inventory.sources.ScanSources`
    (where the data is now)."""
    from ..converter import sourcedata as _sd

    if not options.enabled:
        return 0, 0, []
    rows = _filter_convertible_rows(df.fillna("").astype(str))
    n = size = 0
    kinds: set[str] = set()
    for _, row in rows.iterrows():
        kind = _sd.kind_for(row.get("modality", ""), row.get("datatype", ""),
                            str(row.get("bids_name", "")).rsplit("_", 1)[-1])
        if kind not in options.modalities:
            continue
        uid = str(row.get("series_uid", "")).strip()
        if uid:
            files = sources.series_files(uid)
        else:
            found = sources.file(str(row.get("source_file", "")))
            files = _sd.recording_files(found) if found is not None else []
        if not files:
            continue
        kinds.add(kind)
        items = [_sd.Item(kind=kind, destination="", files=[(f, f.name) for f in files])]
        count, nbytes = _sd.estimate(items)
        n += count
        size += nbytes
    return n, size, [k for k in _sd.MODALITIES if k in kinds]


def _keep_source_data(tasks, options: SourcedataOptions, bids_root: Path, subj_segment: str,
                      staging: Path, *, on_existing: str, cancel_check=None) -> None:
    """Keep the subject's source data in ``sourcedata/``, after the subject
    itself committed. A failure here is said and does not undo the
    conversion; a stop is a stop."""
    from ..converter import sourcedata as _sd

    try:
        report = _sd.keep_subject(
            tasks, options, staging=staging,
            target=bids_root / _sd.SOURCEDATA / subj_segment, on_existing=on_existing,
            backup_root=bids_root / ".bidsmgr" / "backup", stamp=_utc_stamp(),
            cancel_check=cancel_check)
    except OperationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - the conversion itself succeeded
        log.warning("%s: the source data could not be kept in sourcedata/ (%s: %s)",
                    subj_segment, type(exc).__name__, exc)
        return
    if report is not None:
        log.info("%s: source data in sourcedata/: %s", subj_segment, report.summary())
        for line in report.failed[:10]:
            log.warning("%s: not copied to sourcedata/: %s", subj_segment, line)


#: How long the commit keeps trying to move a staged file or tree into place
#: while Windows reports it as held open, and how long it waits between tries.
#: Windows only; see :func:`_retry_while_held`. Ten seconds is two orders of
#: magnitude more than any blocker measured here (worst case 156 ms, under a
#: load harsher than a real session's) and still bounded, so a handle that is
#: genuinely stuck — a staged file open in another program — is reported
#: rather than waited on forever.
_COMMIT_RETRY_DEADLINE_S = 10.0
_COMMIT_RETRY_SLEEP_S = 0.05


def _retry_while_held(move, what: Path) -> None:
    """Run ``move`` and, on Windows only, retry it while the OS says "held".

    Everywhere else ``move`` is called exactly once and its exception
    propagates untouched, which is what the commit has always done: POSIX
    renames a path no matter who is reading it.

    Windows refuses with ``WinError 5`` while ANY process holds a handle
    anywhere inside the tree being moved, and merely ENUMERATING a directory
    holds one. Conversion stages into ``<bids_root>/.tmp_bidsmgr/``, which
    lives inside the dataset the GUI is watching, so a single tree refresh,
    validation pass, antivirus scan or open Explorer window is enough to break
    the commit. Measured on this machine: with one background
    ``schema.warm_dataset`` pass running — which the Editor now does on every
    tree refresh — 4 of 12 commits failed. That is the bug this exists for.

    Such a blocker is transient, so the move is simply retried. Past the
    deadline the original error is raised, so a stuck handle still fails the
    subject the way it did before, with nothing half-committed.
    """
    if os.name != "nt":
        move()
        return
    deadline = time.monotonic() + _COMMIT_RETRY_DEADLINE_S
    while True:
        try:
            move()
            return
        except PermissionError:
            # WinError 5 (access denied) and 32 (sharing violation) both land
            # here. A target that already exists raises FileExistsError and is
            # NOT retried: that is a logic error, not a held handle.
            if time.monotonic() >= deadline:
                log.error(
                    "%s is still held open by another program after %.0fs; on "
                    "Windows a directory cannot be moved while anything is "
                    "reading inside it. Close anything open under the dataset "
                    "(a viewer, an Explorer window, a backup or antivirus "
                    "scan) and convert again.",
                    what, _COMMIT_RETRY_DEADLINE_S,
                )
                raise
            time.sleep(_COMMIT_RETRY_SLEEP_S)


def _rename_into_place(staging_subject: Path, target: Path) -> None:
    """Move a staged subject tree wholesale onto its final name.

    The atomic fast path of :func:`_merge_commit`, kept atomic: one
    ``os.rename``, retried on Windows while a reader holds the tree open.
    """
    _retry_while_held(lambda: os.rename(staging_subject, target), staging_subject)


def _move_staged_file(src: Path, dst: Path) -> None:
    """Move one staged file to its place in the dataset.

    ``shutil.move`` everywhere but Windows, where a reader holding this one
    file gets the same patience the wholesale move gets.
    """
    _retry_while_held(lambda: shutil.move(str(src), str(dst)), src)


def _merge_commit(
    staging_subject: Path, target: Path, *, on_existing: str,
    preserve_curation: bool = True,
) -> tuple[int, int, int]:
    """Commit a staged subject into the dataset, merging into an existing one.

    A brand-new subject is moved in wholesale via the atomic ``os.rename`` fast
    path. When the subject already exists, the staged tree is merged FILE BY FILE
    so adding a new session or datatype lands beside the existing data instead of
    replacing the whole subject (the incremental-conversion case). Returns
    ``(added, replaced, kept)`` file counts.

    ``on_existing`` governs a staged file that collides with one already on disk:

    * ``"skip"``    keep the existing file (default; existing data never lost);
    * ``"update"``  replace it only if the content differs (identical files kept);
    * ``"replace"`` always back it up then replace it;
    * ``"error"``   abort the whole subject (commit nothing) if ANY file collides.

    A replaced file is first moved to
    ``<bids_root>/.bidsmgr/backup/<sub>_<utcstamp>/<relpath>``, so the merge path
    is reconstructable even though atomicity drops from subject- to file-level.

    ``preserve_curation`` changes what "replace" means for the two file kinds a
    person edits: a JSON sidecar and ``*_scans.tsv`` are merged FIELD by field
    rather than overwritten, so an afternoon in the Editor survives a second
    conversion of the same subject while the fresh pass still contributes
    everything it newly knows. See :mod:`bidsmgr.metadata.preserve`. Every
    other file is replaced exactly as before, and the backup is taken either
    way, so the pre-merge state is always recoverable.
    """
    staged = [p for p in sorted(staging_subject.rglob("*")) if p.is_file()]

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _rename_into_place(staging_subject, target)
        return (len(staged), 0, 0)

    colliding = [p for p in staged if (target / p.relative_to(staging_subject)).exists()]
    if colliding and on_existing == "error":
        rels = ", ".join(str(p.relative_to(staging_subject)) for p in colliding[:5])
        more = "" if len(colliding) <= 5 else f" (+{len(colliding) - 5} more)"
        raise CollisionAbort(
            f"{target.name}: {len(colliding)} file(s) already exist ({rels}{more}); "
            f"aborted (use --on-existing skip/update/replace to proceed)"
        )

    backup_dir = target.parent / ".bidsmgr" / "backup" / f"{target.name}_{_utc_stamp()}"
    added = replaced = kept = 0
    preserved: list[str] = []
    for src in staged:
        rel = src.relative_to(staging_subject)
        dst = target / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            _move_staged_file(src, dst)
            added += 1
            continue
        # Colliding file.
        if on_existing == "skip":
            kept += 1
            continue
        if on_existing == "update" and filecmp.cmp(str(src), str(dst), shallow=False):
            kept += 1  # identical content; leave the existing file in place
            continue
        bdst = backup_dir / rel
        bdst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(dst), str(bdst))
        if preserve_curation and is_mergeable(dst):
            outcome = merge_file(dst, src)
            if outcome is not None:
                src.unlink(missing_ok=True)
                if isinstance(outcome, list) and outcome:
                    preserved.extend(f"{rel}:{f}" for f in outcome)
                replaced += 1
                continue
        # Not mergeable, or the merge could not be done correctly: the
        # ordinary replace, with the original already backed up above.
        dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        _move_staged_file(src, dst)
        replaced += 1
    if preserved:
        shown = ", ".join(preserved[:6])
        more = "" if len(preserved) <= 6 else f" (+{len(preserved) - 6} more)"
        log.info(
            "kept %d curated field(s) through the re-conversion: %s%s",
            len(preserved), shown, more,
        )
    if added or replaced or kept:
        tail = (
            " (use --on-existing replace to overwrite the kept files)"
            if kept and on_existing == "skip" else ""
        )
        log.info(
            "merged into existing %s: %d added, %d replaced, %d kept%s",
            target, added, replaced, kept, tail,
        )
    return (added, replaced, kept)


# ---------------------------------------------------------------------------
# Task construction from TSV rows
# ---------------------------------------------------------------------------


def _user_fieldmap_links(results: list[ConvertResult],
                         rename_map: dict) -> dict[str, list[str]]:
    """The user's own fieldmap choices, from row ids to the image names the
    conversion produced: ``{fieldmap image: [run images]}``."""
    images: dict[str, list[str]] = {}
    for res in results:
        names = []
        for path in res.staged_files:
            final = Path(rename_map.get(path, path))
            if final.name.endswith((".nii", ".nii.gz")):
                names.append(final.name)
        images[res.task.row_id] = names
    links: dict[str, list[str]] = {}
    for res in results:
        chosen = res.task.fieldmap_targets
        if not chosen:
            continue
        targets = [name for rid in chosen for name in images.get(rid, [])]
        for name in images.get(res.task.row_id, []):
            links[name] = targets
    return links


def _filter_convertible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that are include=1 with a non-empty bids_name
    and are not flagged ``bids_guess_skip``.
    """
    out = df.copy()
    if "include" in out.columns:
        out = out[out["include"].astype(str) == "1"]
    if "bids_name" in out.columns:
        out = out[out["bids_name"].astype(str).str.len() > 0]
    if "bids_guess_skip" in out.columns:
        out = out[~out["bids_guess_skip"].astype(str).str.lower().isin({"true", "1"})]
    return out


class TaskOptions(NamedTuple):
    """What a conversion asks of every task, beyond its row."""

    #: Volumes removed from a functional run whose cell is empty.
    drop_volumes: int = 0
    physio_layout: str = "by_rate"
    physio_align: bool = False


def _build_tasks_for_subject(
    subject_df: pd.DataFrame,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    source_search_roots: tuple[Path, ...] = (),
    skip_residuals: bool = True,
    force_edf: bool = False,
    spec: Optional[RecordingMetaSpec] = None,
    options: TaskOptions = TaskOptions(),
) -> list[ConvertTask]:
    tasks: list[ConvertTask] = []
    for _, row in subject_df.iterrows():
        task = _row_to_task(
            row, bids_root, files_by_uid,
            source_search_roots=source_search_roots,
            skip_residuals=skip_residuals,
            force_edf=force_edf,
            spec=spec,
            options=options,
        )
        if task is not None:
            tasks.append(task)
    return _physio_follows_its_run(tasks, _physio_links(subject_df))


def _physio_links(df: pd.DataFrame) -> dict[str, str]:
    """``{physio row id: run row id}`` for the logs of functional runs
    (``fixups.dummy_volumes.physio_runs``), in the ids the tasks carry."""
    from ..fixups.dummy_volumes import physio_runs
    from ..project.orchestration import row_id

    positions = {label: k for k, label in enumerate(df.index)}
    frame = df.reset_index(drop=True)
    return {row_id(frame, positions[log_]): row_id(frame, positions[run])
            for log_, run in physio_runs(df).items()}


def _physio_follows_its_run(tasks: list[ConvertTask],
                            links: dict[str, str]) -> list[ConvertTask]:
    """A physiology log takes the volumes removed from the run it was
    recorded with (``links``): its tables then start where the run's image
    starts. A log with no run while volumes are removed from this subject's
    runs is said so in the log, since its tables then keep every volume."""
    from ..fixups.dummy_volumes import applies

    by_id = {t.row_id: t for t in tasks}
    cutting = any(applies(t.datatype, t.suffix) and t.drop_volumes > 0 for t in tasks)
    out = []
    for t in tasks:
        if t.datatype == "func" and t.suffix == "physio":
            run = by_id.get(links.get(t.row_id, ""))
            if run is not None:
                if run.drop_volumes != t.drop_volumes:
                    t = t.model_copy(update={"drop_volumes": run.drop_volumes})
            elif cutting:
                log.warning(
                    "%s: no functional run being converted was recorded with this "
                    "physiological log, so no volumes are removed from it", t.basename)
        out.append(t)
    return out


def _drop_volumes_for(row: pd.Series, datatype: str, suffix: str, default: int) -> int:
    """Volumes to remove from this row (``fixups.dummy_volumes.volumes_to_remove``);
    a cell that is not a number is reported and the default used."""
    from ..fixups.dummy_volumes import volumes_to_remove

    try:
        return volumes_to_remove(row.get("drop_volumes", ""), datatype, suffix, default)
    except ValueError:
        log.warning("drop_volumes %r is not a number; %d used", row.get("drop_volumes"), default)
        return volumes_to_remove("", datatype, suffix, default)


def _row_to_task(
    row: pd.Series,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    source_search_roots: tuple[Path, ...] = (),
    skip_residuals: bool = True,
    force_edf: bool = False,
    spec: Optional[RecordingMetaSpec] = None,
    options: TaskOptions = TaskOptions(),
) -> Optional[ConvertTask]:
    """Detect the row's shape and dispatch to the right builder.

    Two shapes exist. A DICOM series row carries a non-empty ``series_uid``
    and many source files. A file-based row carries a single path in
    ``source_file``: EEG, MEG, iEEG and NIRS recordings, and ECAT PET, whose
    whole dynamic study lives in one file. Anything else returns None and the
    caller skips it.
    """
    series_uid = str(row.get("series_uid", "")).strip()
    if series_uid:
        return _row_to_task_mri(
            row, bids_root, files_by_uid, skip_residuals=skip_residuals,
            options=options,
        )

    source_file = str(row.get("source_file", "")).strip()
    if source_file:
        return _row_to_task_file_based(
            row, bids_root, source_search_roots=source_search_roots,
            force_edf=force_edf, spec=spec,
        )

    return None


def _row_to_task_mri(
    row: pd.Series,
    bids_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    skip_residuals: bool = True,
    options: TaskOptions = TaskOptions(),
) -> Optional[ConvertTask]:
    series_uid = str(row.get("series_uid", "")).strip()
    if not series_uid:
        return None

    # v0.2.5 collapses magnitude+phase fmap rows into one row with the
    # two SeriesInstanceUIDs joined by ``|``. Pull files for every UID.
    uids = [u for u in series_uid.split("|") if u]
    files: list[str] = []
    for uid in uids:
        files.extend(files_by_uid.get(uid, []))
    if not files:
        log.warning(
            "no files_by_uid entry for series %s — skipping (rebuild the "
            "scan inventory and its files_by_uid sidecar)", series_uid,
        )
        return None

    participant = str(row.get("participant_id", "")).strip()
    subject = participant[len("sub-"):] if participant.startswith("sub-") else participant
    if not subject:
        return None

    raw_session = str(row.get("session", "")).strip()
    session = (
        raw_session[len("ses-"):] if raw_session.startswith("ses-") else (raw_session or None)
    )
    if session == "":
        session = None

    datatype = str(row.get("datatype", "")).strip()
    suffix = (
        str(row.get("bids_guess_suffix", "")).strip()
        or _suffix_from_basename(str(row.get("bids_name", "")))
    )
    basename = str(row.get("bids_name", "")).strip()
    if not (datatype and suffix and basename):
        return None

    # Derivatives-bound rows go under <bids_root>/derivatives/<pipeline>/...
    # rather than the live raw tree. v1 doesn't ship the derivatives fixup,
    # so for now skip them with a warning.
    if datatype.startswith("derivatives/"):
        log.info(
            "row series=%s suggests derivatives output (%s); skipping until "
            "the derivatives fixup lands", series_uid, datatype,
        )
        return None

    # Expected outputs differ by suffix. Physio rows produce
    # ``.tsv.gz`` + ``.json`` from the physio backend; everything
    # else dcm2niix-direct handles produces ``.nii.gz`` + ``.json``,
    # plus DWI extras.
    if suffix == "physio":
        expected_outputs = (".tsv.gz", ".json")
    elif suffix == "dwi":
        expected_outputs = (".nii.gz", ".json", ".bval", ".bvec")
    else:
        expected_outputs = (".nii.gz", ".json")

    # Surface the row's canonical entities dict (the source of truth
    # the user edited) — informational for backends that want to inspect
    # individual entity values without re-parsing the basename.
    entities_dict = _parse_entities_json(row)

    return ConvertTask(
        row_id=series_uid,
        series_uid=series_uid,
        source_files=tuple(Path(p) for p in files),
        dataset=str(row.get("dataset", "")).strip(),
        bids_root=bids_root,
        subject=subject,
        session=session,
        datatype=datatype,
        suffix=suffix,
        modality=str(row.get("modality", "")).strip().lower(),
        entities=entities_dict,
        basename=basename,
        expected_outputs=expected_outputs,
        repetition_type=str(row.get("repetition_type", "")).strip(),
        skip_residuals=skip_residuals,
        companion_files=parse_companions(row.get("companion_files", "")),
        fieldmap_targets=_parse_fieldmap_targets(row),
        drop_volumes=_drop_volumes_for(row, datatype, suffix, options.drop_volumes),
        physio_layout=options.physio_layout,
        physio_align=options.physio_align,
    )


def _parse_fieldmap_targets(row: pd.Series) -> tuple[str, ...]:
    """The runs the user chose for a fieldmap row (``fieldmap_targets``, a
    JSON list of row ids); ``()`` when the rule decides."""
    raw = str(row.get("fieldmap_targets", "") or "").strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    return tuple(str(x) for x in data) if isinstance(data, list) else ()


def _parse_entities_json(row: pd.Series) -> dict[str, str]:
    """Parse the row's ``entities`` JSON column. Returns ``{}`` on error."""
    raw = str(row.get("entities", "")).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    # Normalise values to str (Pydantic ConvertTask wants dict[str, str]).
    return {str(k): str(v) for k, v in data.items()}


def _row_to_task_file_based(
    row: pd.Series,
    bids_root: Path,
    *,
    source_search_roots: tuple[Path, ...] = (),
    force_edf: bool = False,
    spec: Optional[RecordingMetaSpec] = None,
) -> Optional[ConvertTask]:
    """Build a :class:`ConvertTask` for a row whose source is one file.

    That covers EEG, MEG, iEEG and NIRS recordings, which the mne-bids
    backend reads, and ECAT PET, which the nibabel-based ``EcatDirect``
    backend reads. Both carry the path directly in ``source_file``, relative
    to the scan input root or absolute.
    """
    source_file = str(row.get("source_file", "")).strip()
    if not source_file:
        return None

    participant = str(row.get("participant_id", "")).strip()
    subject = participant[len("sub-"):] if participant.startswith("sub-") else participant
    if not subject:
        return None

    raw_session = str(row.get("session", "")).strip()
    session = (
        raw_session[len("ses-"):] if raw_session.startswith("ses-")
        else (raw_session or None)
    )
    if session == "":
        session = None

    datatype = str(row.get("datatype", "")).strip().lower()
    eyetrack = str(row.get("modality", "")).strip().lower() == "eyetrack"
    if datatype not in _FILE_BASED_DATATYPES and not eyetrack:
        # Either not actually a file-based row, or the user clobbered the
        # column. Skip.
        return None

    suffix = (
        str(row.get("bids_guess_suffix", "")).strip()
        or _suffix_from_basename(str(row.get("bids_name", "")))
        or datatype
    )
    basename = str(row.get("bids_name", "")).strip()
    if not basename:
        return None

    # Resolve source_file to an absolute path. The scan stores it
    # relative to the scan input root; the convert input is the BIDS
    # parent — different reference. Try in order: each ``source_search_root``
    # the caller provided (the GUI passes the raw_root explicitly; the
    # CLI auto-fills the TSV's parent), then the current working
    # directory, then ``.resolve()`` as a last resort.
    src_path = Path(source_file)
    if not src_path.is_absolute():
        candidates: list[Path] = []
        for root in source_search_roots:
            candidates.append(Path(root) / src_path)
        candidates.append(Path.cwd() / src_path)
        candidates.append(src_path.resolve())
        for c in candidates:
            if c.exists():
                src_path = c
                break
        else:
            # Say so here, where the reason is known. Left unresolved, the row
            # falls past the backend that reads its format (which checks the
            # file's signature and so needs it to exist) into the DICOM one,
            # and fails with "no source DICOMs were accessible": true, and no
            # help at all in working out that a path could not be resolved.
            log.warning(
                "could not find the source for %s: %r is relative and does "
                "not exist under any of %s. Pass --raw-root <the folder that "
                "was scanned> so relative paths can be resolved.",
                basename, source_file,
                [str(r) for r in source_search_roots] or ["the working directory"],
            )

    # The row's ``entities`` column is the source of truth (the in-memory
    # rebuild ran above so it's already reconciled with display cells).
    # Fall back to individual columns / basename parsing for legacy
    # inventories that don't have the ``entities`` column.
    entities = _parse_entities_json(row)
    if not entities.get("task"):
        task_value = str(row.get("task", "")).strip()
        if task_value:
            entities["task"] = task_value
    if not entities.get("run"):
        run_value = str(row.get("run", "")).strip()
        if not run_value:
            m = re.search(r"_run-(\d+)", basename)
            if m:
                run_value = m.group(1)
        if run_value:
            entities["run"] = run_value

    # mne-bids writes the format-native data file (.edf / .vhdr / .fif /
    # …) plus channels.tsv + datatype JSON. We don't validate the exact
    # extension — the backend collects whatever lands in the datatype dir.
    expected_outputs: tuple[str, ...] = (".json",)

    # Resolve EEG/MEG line_freq + montage. Precedence: the inventory cell
    # wins; else the recording-metadata dataset default (resolved per row by
    # source path); else NOTHING.
    #
    # There used to be a final `line_freq = 50.0` here, so PowerLineFrequency
    # was always populated. It was always populated and sometimes wrong: 50 Hz
    # is Europe, and a recording made in the US, Canada, Japan or Brazil is 60.
    # The sidecar said 50 in a BIDS-required field with nothing to distinguish
    # it from a value somebody measured, and the recordings most likely to
    # reach this line are exactly the ones nobody had thought about.
    #
    # Leaving it None is not a gap. mne-bids writes
    # PowerLineFrequency: "n/a" when raw.info["line_freq"] is unset, which the
    # schema allows (the field is anyOf number or the literal "n/a") and which
    # is the true statement. A guess that validates is worse than an honest
    # "not stated" that also validates.
    #
    # The recording's OWN header still wins over all of this: the mne_bids
    # backend only writes our value when raw.info carries none.
    # The datatype selects the per-modality block: what a study says about its
    # EEG amplifier must not reach its MEG recordings.
    eff_acq = (
        resolve_effective(spec, source_file, None, str(row.get("bids_guess_datatype", "") or "") or None).acquisition
        if spec is not None else None
    )

    line_freq_raw = str(row.get("line_freq", "")).strip()
    try:
        line_freq = float(line_freq_raw) if line_freq_raw else None
    except ValueError:
        line_freq = None
    if line_freq is None and eff_acq is not None:
        line_freq = eff_acq.power_line_freq

    montage = str(row.get("montage", "")).strip() or None
    if montage is None and eff_acq is not None:
        montage = eff_acq.montage

    # Per-row reference/ground overrides (the spec default applies in the
    # enrichment fixup when these are blank).
    eeg_reference = str(row.get("eeg_reference", "")).strip() or None
    eeg_ground = str(row.get("eeg_ground", "")).strip() or None

    return ConvertTask(
        row_id=source_file,  # source path is unique per recording
        series_uid="",       # blank for EEG/MEG rows
        source_files=(src_path,),
        dataset=str(row.get("dataset", "")).strip(),
        bids_root=bids_root,
        subject=subject,
        session=session,
        datatype=datatype,
        suffix=suffix,
        modality=str(row.get("modality", "")).strip().lower(),
        entities=entities,
        basename=basename,
        expected_outputs=expected_outputs,
        repetition_type=str(row.get("repetition_type", "")).strip(),
        line_freq=line_freq,
        montage=montage,
        eeg_reference=eeg_reference,
        eeg_ground=eeg_ground,
        force_edf=force_edf,
        companion_files=parse_companions(row.get("companion_files", "")),
        eyetrack=_eyetrack_settings(spec, source_file) if eyetrack else {},
    )


def _eyetrack_settings(spec: Optional[RecordingMetaSpec], row_id: str) -> dict:
    """What the user said about an eye-tracking recording (screen, timing)."""
    from ..recording_meta import resolve_eyetrack

    if spec is None:
        return {}
    return resolve_eyetrack(spec, row_id).model_dump(exclude_none=True)


def _suffix_from_basename(basename: str) -> str:
    """Last underscore-delimited token of a BIDS basename = its suffix."""
    if not basename:
        return ""
    return basename.rsplit("_", 1)[-1]


def _row_data(row: pd.Series, sources) -> tuple[dict[str, list[str]], Optional[Path]]:
    """Where one row's source data is now: ``(files_by_uid, None)`` for a
    DICOM series, ``({}, path)`` for a recording file. Raises with the reason
    when it cannot be found."""
    from ..inventory.probe_convert import series_files

    uid = str(row.get("series_uid", "")).strip()
    source_file = str(row.get("source_file", "")).strip()
    if not uid:
        if not source_file:
            raise ValueError("this row names no source data")
        path = sources.file(source_file)
        if path is None:
            raise FileNotFoundError(f"{source_file} is not in {sources.root}")
        return {}, path
    uids = [u for u in uid.split("|") if u]
    files_by_uid = {u: [str(p) for p in sources.series_files(u)] for u in uids}
    if not any(files_by_uid.values()):
        # A scan without its files_by_uid record: look in the row's folder.
        folder = sources.folder(str(row.get("source_folder", "")))
        if folder is not None:
            files_by_uid = {u: series_files(folder, u) for u in uids}
    if not any(files_by_uid.values()):
        raise FileNotFoundError(
            f"no DICOM of this series was found under {sources.root}")
    return files_by_uid, None


def preview_row(row, work_dir: Path, sources) -> list[Path]:
    """What converting this one row makes, for a look before the real thing.

    ``row`` is an inventory row (a mapping), ``sources`` the scan's
    :class:`~bidsmgr.inventory.sources.ScanSources` (where its data is now).
    The row goes through the conversion's own task builder and backends
    (dcm2niix, bidsphysio for physiology, nibabel for ECAT) into
    ``work_dir``, then the fieldmap renames, so what is shown is split and
    named the way the dataset will have it: a fieldmap's magnitude images
    and phase difference, a physio recording's table, a spectrum.

    An EEG or MEG recording is not converted: the viewer reads the source
    file as it is. A series the plan does not convert (a scout, an
    unclassified series) has no task, and is converted plainly by dcm2niix,
    which is still the quickest way to find out what it is.

    Returns the files a viewer opens; raises with the reason otherwise.
    """
    from ..inventory.probe_convert import preview_series
    from ..viz.data.formats import is_recording_path, kind_of

    row = pd.Series({k: ("" if v is None else v) for k, v in dict(row).items()})
    files_by_uid, path = _row_data(row, sources)
    if path is not None:
        # An EyeLink .edf shares its extension with an EEG recording and is
        # nothing like one: it is converted (to the BEP020 physio tables the
        # signal viewer plots), not read as it is.
        eyetrack = str(row.get("modality", "")).strip().lower() == "eyetrack"
        if is_recording_path(path) and not eyetrack:
            return [path]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    roots = (sources.root,) if sources.root is not None else ()
    task = _row_to_task(row, work_dir, files_by_uid, source_search_roots=roots)
    if task is None:
        if files_by_uid:
            return preview_series([f for files in files_by_uid.values() for f in files],
                                  work_dir)
        raise ValueError("the plan for this row converts nothing")
    staging = work_dir / f"sub-{safe_path_component(task.subject)}"
    result = _phase1_one(default_backends(), task, staging)
    if not result.success:
        raise RuntimeError(result.error or "the conversion made nothing")
    apply_fieldmap_renames(staging)
    shown = sorted(p for p in staging.rglob("*") if p.is_file() and kind_of(p))
    if not shown:
        raise RuntimeError("the conversion made no file that can be shown")
    return shown


def sidecars_for_row(row, work_dir: Path, sources, *,
                     spec: Optional[RecordingMetaSpec] = None,
                     options: TaskOptions = TaskOptions()) -> dict[str, dict]:
    """The sidecars converting this one row writes, by file name.

    The row goes through the conversion's own task builder and backend into
    ``work_dir``, then the per-subject steps that shape a sidecar: fieldmap
    names, PET and MR spectroscopy fields, physio fields, the converter
    repairs and the identifiers. What the user stated is not applied, as in
    the conversion: that is the metadata step's job. ``spec`` is passed for
    what a backend needs from it (an eye tracker's settings, the PET block).

    How the Editor rebuilds a sidecar that was deleted from the data it was
    converted from. Raises with the reason when the row cannot be converted.
    """
    row = pd.Series({k: ("" if v is None else v) for k, v in dict(row).items()})
    files_by_uid, _path = _row_data(row, sources)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    roots = (sources.root,) if sources.root is not None else ()
    task = _row_to_task(row, work_dir, files_by_uid, source_search_roots=roots, spec=spec,
                        options=options)
    if task is None:
        raise ValueError("the plan for this row converts nothing")
    if task.suffix == "physio":
        # Alone, the row has no run to follow: the number recorded for it.
        task = task.model_copy(update={"drop_volumes": options.drop_volumes})
    staging = work_dir / f"sub-{safe_path_component(task.subject)}"
    result = _phase1_one(default_backends(), task, staging)
    if not result.success:
        raise RuntimeError(result.error or "the conversion made nothing")
    apply_fieldmap_renames(staging)
    drop_leading_volumes([result])
    enrich_pet_sidecars(staging, [task], spec)
    clean_mrs_outputs(staging, [task])
    enrich_physio_sidecars(staging)
    repair_converter_output(staging)
    prune_identifiers(staging)
    out: dict[str, dict] = {}
    for path in sorted(staging.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out[path.name] = data
    return out


# ---------------------------------------------------------------------------
# Files_by_uid sidecar
# ---------------------------------------------------------------------------


def _load_files_by_uid_sidecar(
    tsv: Path, *, required: bool = True,
) -> dict[str, list[str]]:
    """Load the per-UID DICOM file map.

    Required for inventories with MRI rows (``series_uid`` populated).
    EEG/MEG-only inventories don't need this sidecar — the recording
    paths live in each row's ``source_file`` column. Pass
    ``required=False`` for those cases; missing sidecar returns ``{}``.
    """
    sidecar = tsv.with_suffix(tsv.suffix + ".files_by_uid.json.gz")
    if not sidecar.exists():
        if required:
            raise FileNotFoundError(
                f"missing files_by_uid sidecar: {sidecar} — "
                "re-run `bidsmgr-scan` to produce it next to the inventory TSV"
            )
        return {}
    with gzip.open(sidecar, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Provenance + error log
# ---------------------------------------------------------------------------


def _write_provenance(
    bids_root: Path,
    subject_segment: str,
    results: list[ConvertResult],
    rename_map: dict[Path, Path],
    n_intended_for: int,
    n_scans_tsv: int,
    *,
    dcm2niix_version: str,
    n_enriched: int = 0,
) -> None:
    """Convert provenance at ``<root>/.bidsmgr/provenance/<subject>.json``.

    One place, not one per subject. It used to live in the subject folder, and
    a hidden directory inside every subject caused problems out of all
    proportion to what it held: deleting a subject left the folder behind
    because the tool's own state is excluded from a delete, renaming a subject
    onto another had to invent somewhere for the loser's copy to go, and every
    walker in the codebase had to know to skip it. The dataset already has
    exactly one place for tool state, at its root, and this is tool state.
    """
    prov_dir = Path(bids_root) / ".bidsmgr" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "generated_at": _utc_stamp(),
        "dcm2niix_version": dcm2niix_version,
        "bidsmgr_version": bidsmgr.__version__,
        "tasks": [
            {
                "row_id": r.task.row_id,
                "series_uid": r.task.series_uid,
                "datatype": r.task.datatype,
                "suffix": r.task.suffix,
                "basename": r.task.basename,
                "outputs": [p.name for p in r.staged_files],
                # What was asked of it, so converting it again (the
                # Editor's sidecars from the source data) does the same.
                "drop_volumes": r.task.drop_volumes,
                "physio_layout": r.task.physio_layout,
                "physio_align": r.task.physio_align,
                "n_input_dicoms": len(r.task.source_files),
                "success": r.success,
                "error": r.error,
                "dcm2niix_returncode": r.dcm2niix_returncode,
                "duration_s": r.duration_s,
            }
            for r in results
        ],
        "fieldmap_renames": [
            {"from": old.name, "to": new.name} for old, new in rename_map.items()
        ],
        "intended_for_updated": n_intended_for,
        "scans_tsv_rewritten": n_scans_tsv,
        "recording_sidecars_enriched": n_enriched,
    }
    record["subject_dir"] = subject_segment
    name = safe_path_component(subject_segment) or "subject"
    (prov_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")


def _write_error_log(
    bids_root: Path,
    subject: str,
    tasks: list[ConvertTask],
    results: list[ConvertResult],
    exc: BaseException,
    failed_phase: str,
) -> None:
    """Forensic log when a subject fails. Never raises."""
    try:
        err_dir = bids_root / ".bidsmgr" / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        path = err_dir / f"sub-{safe_path_component(subject)}_{_utc_stamp()}.json"
        payload = {
            "schema_version": 1,
            "subject": subject,
            "failed_at_phase": failed_phase,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "tasks": [_task_dump(t) for t in tasks],
            "results_so_far": [_result_dump(r) for r in results],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log.error("wrote error log to %s", path)
    except Exception:  # pragma: no cover - logging must not raise
        log.exception("could not write error log")


def _task_dump(t: ConvertTask) -> dict:
    return {
        "row_id": t.row_id, "series_uid": t.series_uid,
        "subject": t.subject, "session": t.session,
        "datatype": t.datatype, "suffix": t.suffix, "basename": t.basename,
        "n_source_files": len(t.source_dicom_files),
        "expected_outputs": list(t.expected_outputs),
        "repetition_type": t.repetition_type,
    }


def _result_dump(r: ConvertResult) -> dict:
    return {
        "basename": r.task.basename,
        "success": r.success, "error": r.error,
        "dcm2niix_returncode": r.dcm2niix_returncode,
        "stderr_tail": r.dcm2niix_stderr_tail,
        "duration_s": r.duration_s,
        "outputs": [p.name for p in r.staged_files],
    }


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def _dcm2niix_version_string(binary: Path) -> str:
    """Capture ``dcm2niix --version`` output for provenance."""
    try:
        proc = subprocess.run(
            [str(binary), "-h"], capture_output=True, text=True, timeout=10,
        )
        first = (proc.stdout or proc.stderr).splitlines()[0:2]
        return " ".join(line.strip() for line in first if line.strip()) or "unknown"
    except Exception:
        return "unknown"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_tasks(tasks: Iterable[ConvertTask]) -> None:
    for t in tasks:
        print(
            f"DRY: sub-{t.subject}"
            f"{' ses-' + t.session if t.session else ''} "
            f"{t.datatype}/{t.basename} "
            f"({len(t.source_dicom_files)} files)"
        )


# ---------------------------------------------------------------------------
# argparse entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point. Reports a name collision as a refusal, not a crash."""
    from ..inventory.name_collisions import NameCollisionError

    try:
        return _main(argv)
    except NameCollisionError as exc:
        log.error("%s", exc)
        return 1


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-convert",
        description=(
            "Convert DICOM series listed in an inventory TSV into BIDS using "
            "dcm2niix. Each distinct `dataset` value in the TSV becomes a "
            "sibling BIDS root under <bids_parent>."
        ),
    )
    parser.add_argument(
        "tsv", nargs="?", default=None,
        help="Inventory TSV produced by `bidsmgr-scan` (omit when using --project)",
    )
    parser.add_argument(
        "bids_parent", nargs="?", default=None,
        help=(
            "Parent directory; each `dataset` value becomes a sibling BIDS "
            "root underneath. Omit when using --project (output is the project)."
        ),
    )
    parser.add_argument(
        "--project", default=None, type=Path,
        help=(
            "Convert the project dataset's active (latest) scan version into "
            "its locked root. Resolves the inventory + output + recording "
            "metadata from <project>/.bidsmgr/project, replaying the curation "
            "edits recorded in the GUI. Supersedes the positional tsv / "
            "bids_parent."
        ),
    )
    parser.add_argument(
        "--version", default=None,
        help=(
            "With --project, convert a specific scan version instead of the "
            "latest. Accepts the version id or its index (see bidsmgr-project)."
        ),
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Limit conversion to rows with this exact `dataset` value.",
    )
    parser.add_argument(
        "--jobs", "-j", type=int,
        default=max(1, round((os.cpu_count() or 1) * 0.8)),
        help="Number of dcm2niix invocations to run in parallel.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Alias for --on-existing replace: when a subject already exists, back "
            "up and replace colliding files (new sessions/datatypes still merge in)."
        ),
    )
    parser.add_argument(
        "--on-existing", choices=("skip", "update", "replace", "error"),
        default=None,
        help=(
            "Policy when an incoming subject already exists on disk. New "
            "sessions/datatypes always merge in; this governs colliding files: "
            "'skip' keep existing (default), 'update' replace only changed files, "
            "'replace' back up + replace colliding files, 'error' abort the "
            "subject if anything would collide. --overwrite is an alias for replace."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the tasks that would be converted; write nothing.",
    )
    parser.add_argument(
        "--dcm2niix", default=None, type=Path,
        help="Path to a specific dcm2niix binary (defaults to bundled).",
    )
    parser.add_argument(
        "--recording-meta", default=None, type=Path,
        help=(
            "Path to a recording-metadata JSON (EEG/MEG enrichment). Its "
            "dataset defaults supply line_freq / montage for blank inventory "
            "cells, and its richer fields fill sidecar reference / ground / "
            "filters / device / institution, retype auxiliary channels, and "
            "map event codes to labels. Optional; omitting it means no "
            "power-line frequency is invented, so PowerLineFrequency is "
            "\"n/a\" unless the recording header, an inventory cell or a "
            "template states one. Set per-row line_freq / montage in the "
            "inventory TSV columns to override per recording."
        ),
    )
    parser.add_argument(
        "--pet-spreadsheet", default=None, type=Path,
        help=(
            "Path to a PET metadata spreadsheet (.tsv / .csv / .xlsx / .ods). "
            "One row per scan, keyed by participant label, BIDS name or source "
            "filename, with columns for tracer, dose, mass, administration and "
            "timing. Column names are matched loosely (Injected Dose, "
            "injected_radioactivity and InjectedRadioactivity all work). Values "
            "layer on top of the recording-metadata spec as per-row overrides."
        ),
    )
    parser.add_argument(
        "--pet-metadata", default=None, type=Path,
        help=(
            "Path to a PET metadata JSON keyed by BIDS sidecar field names "
            "(TracerName, InjectedRadioactivity, ModeOfAdministration, ...). "
            "A flat object applies to every PET run; an object keyed by "
            "sub-<label> scopes each block to that subject. This is the shape "
            "pypet2bids accepts via --set-default-metadata-json, including its "
            "nifti_json wrapper, so a file written for that tool is read here "
            "unmodified. Keys BIDS does not define for a PET sidecar are "
            "reported and dropped, never written."
        ),
    )
    parser.add_argument(
        "--intended-for", choices=("uri", "relative"), default="uri",
        help="How a fieldmap's IntendedFor names its runs: BIDS URIs (default) or "
             "paths relative to the subject (with the session). Older fMRIPrep "
             "(sdcflows 2.10 and earlier) ignores URIs; B0FieldIdentifier is read by "
             "every version.")
    parser.add_argument(
        "--no-b0-field-identifier", action="store_true",
        help="Do not write B0FieldIdentifier / B0FieldSource next to IntendedFor.")
    parser.add_argument(
        "--sourcedata", choices=("none", "copy", "archive"), default="none",
        help="Keep each subject's source files in <dataset>/sourcedata/: as they are "
             "(copy) or one .tar.gz per series (archive). Default: none. The copies are "
             "not anonymised.")
    parser.add_argument(
        "--sourcedata-modalities", default=",".join(SOURCEDATA_MODALITIES),
        metavar="LIST",
        help="Which source data --sourcedata keeps, comma separated, from "
             f"{', '.join(SOURCEDATA_MODALITIES)} (default: all).")
    parser.add_argument(
        "--raw-root", default=None, type=Path,
        help=(
            "Folder the scan was run against. Used as the first "
            "candidate when resolving EEG/MEG rows' relative "
            "``source_file`` paths. If omitted, the TSV's parent "
            "directory is tried (the GUI's default puts the TSV "
            "inside the raw root)."
        ),
    )
    parser.add_argument(
        "--drop-volumes", type=int, default=0, metavar="N",
        help=(
            "Remove the first N volumes (dummy scans) of every functional run "
            "whose drop_volumes cell in the table is empty; a cell wins, 0 "
            "included. The image, its sidecar (NumberOfVolumesDiscardedByUser, "
            "the acquisition time), its time in the scans table, its events and "
            "the physiology recorded with it then start at the first volume kept."
        ),
    )
    parser.add_argument(
        "--physio-layout", choices=("by-rate", "one-interpolated", "one-as-recorded"),
        default="by-rate",
        help=(
            "Physiological recordings. BIDS asks one sampling frequency per table: "
            "one table per sampling rate, the samples as recorded (by-rate, default); "
            "one table for the run at the fastest rate, slower channels interpolated "
            "(one-interpolated); or one table at the fastest rate with each sample in "
            "the row of its own time and n/a between a slower channel's samples "
            "(one-as-recorded). All carry the scanner trigger as the last column."
        ),
    )
    parser.add_argument(
        "--physio-align", action="store_true",
        help=(
            "Crop physiological recordings to their run: from its first volume "
            "(after any removed with --drop-volumes) to the end of its last, so "
            "StartTime is 0."
        ),
    )
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help=(
            "Keep dcm2niix residual/secondary outputs. By default the "
            "converter drops the derived single-volume duplicates dcm2niix "
            "splits off one input series (e.g. ..._bolda alongside ..._bold, "
            "or _Eq_ / _ROI markers). These are not real acquired images and "
            "have no valid BIDS suffix. Pass this flag to keep them."
        ),
    )
    parser.add_argument(
        "--deface", action="store_true",
        help=(
            "Remove the face from anatomical and PET images before the "
            "subject is committed, so the identifiable image never enters the "
            "dataset at all."
        ),
    )
    parser.add_argument(
        "--deface-engine", default=DEFACE_DEFAULT_ENGINE,
        choices=deface_engine_ids(),
        help="which defacing engine to use (default: %(default)s)",
    )
    parser.add_argument(
        "--force-edf", action="store_true",
        help=(
            "Re-encode EEG / iEEG recordings to EDF on write instead of "
            "keeping the source format. Harmonises a study to one BIDS-native "
            "format, and makes a non-BIDS-native but mne-readable source "
            "(GDF, EGI, ...) convertible. MEG / NIRS are unaffected."
        ),
    )
    parser.add_argument(
        "--overwrite-curation", action="store_true",
        help=(
            "When re-converting a subject that already exists, let the fresh "
            "conversion win outright. By default a JSON sidecar and a "
            "*_scans.tsv are merged field by field instead, so metadata "
            "curated in the Editor survives the second pass. Only has an "
            "effect together with --on-existing update / replace."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    parser.add_argument(
        "--schema", default=None, metavar="VERSION",
        help=(
            "Which BIDS version to work to (for example 1.11.1). Decides which "
            "fields are declared and which entities a filename may carry. "
            "Default: the newest version shipped."
        ),
    )

    parser.add_argument(
        "--anonymise", choices=("off", "recommended", "strict"), default="off",
        help="Anonymise each converted dataset's metadata when the conversion is done "
             "(see bidsmgr-anonymise). If you run bidsmgr-metadata afterwards, give it "
             "--anonymise instead: it writes participants.tsv, which comes after.")

    add_log_arguments(parser)
    args = parser.parse_args(argv)
    # One version for the whole run, adopted before any schema question is
    # asked. Without this the flag would reach the validator alone and a
    # dataset could be filled in against one version and checked against
    # another.
    from ..schema import set_active_version
    set_active_version(args.schema)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.project is not None:
        # Project mode: convert a scan version into the locked root.
        from ..project.orchestration import (
            find_version, latest_version, version_dataframe,
        )

        bids_root = Path(args.project)
        if args.version:
            version = find_version(bids_root, args.version)
            if version is None:
                parser.error(
                    f"no scan version {args.version!r} in {bids_root} "
                    f"(see `bidsmgr-project {bids_root}`)"
                )
        else:
            version = latest_version(bids_root)
        if version is None:
            parser.error(
                f"no scans in project {bids_root}; run `bidsmgr-scan "
                f"<raw> --project {bids_root}` first"
            )
        # Replay the GUI curation edits onto the inventory, keep the dataset
        # column in lock-step with the (possibly renamed) folder, and bake it
        # back into the version inventory so dcm2niix's files_by_uid sidecar
        # (which sits next to it) stays aligned.
        df = version_dataframe(version, apply_edits=True)
        if "dataset" in df.columns:
            df["dataset"] = bids_root.name
        df.to_csv(version.inventory, sep="\t", index=False)
        # Where the scan's data is: its files are believed over its record,
        # and an explicit --raw-root over both.
        from ..inventory import sources

        found, seen = sources.resolve(
            df, sources.load_files_by_uid(version.inventory),
            recorded=version.raw_root, label=version.source_label,
        )
        source_root = args.raw_root or found.root
        if args.raw_root is None and seen.state in ("missing", "partial"):
            print(f"warning: {sources.describe(seen, found.root)} Pass --raw-root "
                  "<where the folder is now>, or locate it in the GUI.", file=sys.stderr)
        rc = run_convert(
            version.inventory,
            bids_root.parent,
            dataset=bids_root.name,
            n_jobs=args.jobs,
            overwrite=args.overwrite,
            on_existing=args.on_existing,
            dry_run=args.dry_run,
            dcm2niix_bin=args.dcm2niix,
            recording_meta=args.recording_meta,
            pet_spreadsheet=args.pet_spreadsheet,
            pet_metadata=args.pet_metadata,
            raw_root=source_root,
            skip_residuals=not args.keep_residuals,
            preserve_curation=not args.overwrite_curation,
            force_edf=args.force_edf,
            deface=args.deface,
            deface_engine=args.deface_engine,
            intended_for=args.intended_for,
            b0_field=not args.no_b0_field_identifier,
            sourcedata=args.sourcedata,
            sourcedata_modalities=args.sourcedata_modalities,
            drop_volumes=args.drop_volumes,
            physio_layout=args.physio_layout.replace("-", "_"),
            physio_align=args.physio_align,
            **log_keywords(args),
        )
        return _anonymise_after(rc, bids_root.parent, [bids_root.name], args)

    if not args.tsv or not args.bids_parent:
        parser.error("provide tsv and bids_parent, or use --project <dataset>")

    rc = run_convert(
        Path(args.tsv),
        Path(args.bids_parent),
        dataset=args.dataset,
        n_jobs=args.jobs,
        overwrite=args.overwrite,
        on_existing=args.on_existing,
        dry_run=args.dry_run,
        dcm2niix_bin=args.dcm2niix,
        recording_meta=args.recording_meta,
        pet_spreadsheet=args.pet_spreadsheet,
        pet_metadata=args.pet_metadata,
        raw_root=args.raw_root,
        skip_residuals=not args.keep_residuals,
        preserve_curation=not args.overwrite_curation,
        force_edf=args.force_edf,
        deface=args.deface,
        deface_engine=args.deface_engine,
        intended_for=args.intended_for,
        b0_field=not args.no_b0_field_identifier,
        sourcedata=args.sourcedata,
        sourcedata_modalities=args.sourcedata_modalities,
        drop_volumes=args.drop_volumes,
        physio_layout=args.physio_layout.replace("-", "_"),
        physio_align=args.physio_align,
        **log_keywords(args),
    )
    names = [args.dataset] if args.dataset else _dataset_names(Path(args.tsv))
    return _anonymise_after(rc, Path(args.bids_parent), names, args)


def _dataset_names(tsv: Path) -> list[str]:
    try:
        df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False,
                         usecols=lambda c: c == "dataset")
    except (OSError, ValueError):
        return []
    return sorted({n for n in df.get("dataset", []) if n})


def _anonymise_after(rc: int, parent: Path, names: list[str], args) -> int:
    """``--anonymise``: each converted dataset, once the conversion is done."""
    if rc != 0 or args.dry_run or getattr(args, "anonymise", "off") == "off":
        return rc
    from .metadata import _anonymise

    failed = 0
    for name in names:
        root = parent / name
        if (root / "dataset_description.json").is_file():
            failed += _anonymise(root, args.anonymise)
    return 1 if failed else rc


if __name__ == "__main__":
    sys.exit(main())
