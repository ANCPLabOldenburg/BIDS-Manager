"""Inventory raw EEG / MEG / iEEG / NIRS recordings into the unified TSV.

Sibling of ``inventory/mri_dicom.py``: probes every candidate file with
``mne.io.read_raw(preload=False)`` and emits one row per recording. Files
mne can't read are silently skipped, so dropping a folder of mixed
content produces a clean inventory of recordings only.

Output rows match the **unified** TSV schema (see
``cli/scan.py``): universal columns (subject, BIDS_name, session,
dataset, include, modality, modality_bids, proposed_datatype,
proposed_basename, "Proposed BIDS name", source_folder) plus the
EEG/MEG-specific :data:`EEG_MEG_COLUMNS` group below. MRI-specific
columns (``series_uid``, demographics, study-level, BidsGuess, probe)
are left blank for EEG/MEG rows; the scan orchestrator fills them with
empty strings before write.

Reference: ported from
``BIDS-Manager/bids_manager/eeg_meg_inventory.py`` (v0.2.5).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .. import schema as schema_mod

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified-TSV column group for EEG/MEG-specific fields.
# Appended last in the unified-TSV column order:
#   TSV(22) + BIDS_GUESS(8) + DATASET(1) + PROBE(4) + EXTENDED(3) + EEG_MEG(9)
# ---------------------------------------------------------------------------

EEG_MEG_COLUMNS: tuple[str, ...] = (
    "task",            # BIDS task entity (e.g. "rest", "flanker", "driving").
                       # User-editable; the converter writes it verbatim into
                       # the BIDS basename. Blank for MRI rows.
    "run",             # BIDS run entity (integer). User-editable; when non-empty
                       # the converter emits ``_run-<N>`` in the basename.
    "format",          # raw EEG/MEG format short label: EDF | BrainVision | FIF | CNT | ...
    "source_file",     # absolute file path (or folder path for .ds/.mff)
    "n_channels",      # mne probe: int
    "sfreq",           # mne probe: float Hz
    "duration_sec",    # mne probe: float
    "n_times",         # mne probe: int
    "recording_time",  # ISO timestamp from raw.info['meas_date'], or empty
    "has_positions",   # 1 / 0 — whether montage carries non-NaN coordinates
    "line_freq",       # User-editable; power-line frequency in Hz (50 or 60 typical).
                       # Written into PowerLineFrequency in the JSON sidecar
                       # (BIDS-required for EEG/MEG/iEEG). Empty = use CLI default.
    "montage",         # User-editable; mne built-in montage name (e.g. standard_1005,
                       # biosemi64, easycap-M1). When set, the backend applies it
                       # before write_raw_bids → fills electrodes.tsv +
                       # coordsystem.json. Empty = leave positions to whatever the
                       # recording carries.
)


# Extensions mne-bids accepts as raw inputs.
_RECOGNISED_EXTS: tuple[str, ...] = (
    # MEG
    ".fif", ".fif.gz", ".con", ".sqd", ".pdf",
    # EEG
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".eeg", ".egi", ".mff",
    # iEEG
    ".mef", ".nwb",
    # NIRS
    ".snirf",
)
# Folder-shaped recordings (CTF MEG, EGI MFF). Treated as single candidates.
_DIR_FORMATS: tuple[str, ...] = (".ds", ".mff")


# ``mne`` is heavy; lazy-import so the module is cheap to import in
# environments without mne (e.g. fresh venv, unit tests that mock probes).
try:
    import mne  # noqa: F401
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# ---------------------------------------------------------------------------
# Probe result + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """What we extract from a single recording during inventory."""

    source: Path
    sfreq: float
    n_channels: int
    n_times: int
    duration_sec: float
    recording_time: str   # ISO-ish string or empty
    datatype: str         # 'eeg' | 'meg' | 'ieeg' | 'nirs' | ''
    has_positions: bool
    fmt: str              # short label: 'EDF', 'FIF', 'BrainVision', ...


def _detect_datatype(raw) -> str:
    """Best-effort datatype label from the channel kinds present in ``raw``."""
    try:
        ch_types = set(raw.get_channel_types())
    except Exception:
        ch_types = set()
    if "meg" in ch_types or any(t in ch_types for t in ("mag", "grad", "ref_meg")):
        return "meg"
    if any(t in ch_types for t in ("seeg", "ecog", "dbs")):
        return "ieeg"
    if "eeg" in ch_types:
        return "eeg"
    if any(t.startswith("fnirs") for t in ch_types):
        return "nirs"
    return ""


def _has_positions(raw) -> bool:
    """Return True when the recording carries usable channel positions.

    Canonical check: ``raw.get_montage()`` returns a montage with at least
    one position vector that's not all-NaN. ``raw.info['dig']`` alone is
    unreliable because some readers populate fiducials but no electrodes.
    """
    import math

    try:
        montage = raw.get_montage()
    except Exception:
        return False
    if montage is None:
        return False
    try:
        positions = montage.get_positions().get("ch_pos") or {}
    except Exception:
        return False
    for vec in positions.values():
        if vec is None:
            continue
        try:
            if any(v is not None and not math.isnan(float(v)) for v in vec):
                return True
        except Exception:
            continue
    return False


def _format_label(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fif.gz") or name.endswith(".fif"):
        return "FIF"
    if name.endswith(".vhdr"):
        return "BrainVision"
    if name.endswith(".edf"):
        return "EDF"
    if name.endswith(".bdf"):
        return "BDF"
    if name.endswith(".gdf"):
        return "GDF"
    if name.endswith(".set"):
        return "EEGLAB"
    if name.endswith(".cnt"):
        return "CNT"
    if name.endswith(".eeg"):
        return "Nihon Kohden"
    if name.endswith((".sqd", ".con")):
        return "KIT"
    if name.endswith(".pdf"):
        return "4D"
    if name.endswith(".ds"):
        return "CTF"
    if name.endswith(".mef"):
        return "MEF"
    if name.endswith(".nwb"):
        return "NWB"
    if name.endswith((".snirf",)):
        return "SNIRF"
    if name.endswith((".mff",)):
        return "EGI MFF"
    return path.suffix.lstrip(".").upper() or "?"


def _probe(path: Path) -> Optional[ProbeResult]:
    """Attempt to read ``path`` with mne; return a probe result or None.

    Mne is imported lazily inside the function body so callers in
    test-only environments can monkey-patch ``mne.io.read_raw``
    cleanly.
    """
    if not _HAS_MNE:
        return None

    import mne as _mne

    try:
        raw = _mne.io.read_raw(str(path), preload=False, verbose="ERROR")
    except Exception:
        return None

    sfreq = float(raw.info.get("sfreq", 0.0))
    n_channels = int(len(raw.ch_names))
    n_times = int(raw.n_times)
    duration_sec = float(n_times / sfreq) if sfreq > 0 else 0.0

    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        recording_time = ""
    else:
        try:
            recording_time = meas_date.isoformat()
        except Exception:
            recording_time = str(meas_date)

    return ProbeResult(
        source=path,
        sfreq=sfreq,
        n_channels=n_channels,
        n_times=n_times,
        duration_sec=duration_sec,
        recording_time=recording_time,
        datatype=_detect_datatype(raw),
        has_positions=_has_positions(raw),
        fmt=_format_label(path),
    )


# ---------------------------------------------------------------------------
# Walking & subject-identity heuristics
# ---------------------------------------------------------------------------


def candidate_paths(root: Path) -> List[Path]:
    """Walk ``root`` and yield candidate recording paths.

    Folder-shaped formats (``.ds``, ``.mff``) are yielded as the directory
    itself; everything else as a file. We never descend into folder-shaped
    recordings. BrainVision triplets (``.vhdr`` + ``.vmrk`` + ``.eeg``)
    collapse to the ``.vhdr`` only.
    """
    candidates: List[Path] = []
    for cur, dirs, files in os.walk(root):
        cur_path = Path(cur)
        suff = cur_path.suffix.lower()
        if suff in _DIR_FORMATS:
            candidates.append(cur_path)
            dirs[:] = []
            continue
        # Folder-shaped recordings encountered as subdirs: capture them
        # BEFORE we prevent descent (otherwise the filter removes them
        # from the loop iteration too).
        for d in list(dirs):
            if d.lower().endswith(_DIR_FORMATS):
                candidates.append(cur_path / d)
        # Don't descend into folder-shaped recordings — their internals
        # are managed by the reader, not bidsmgr.
        dirs[:] = [d for d in dirs if not d.lower().endswith(_DIR_FORMATS)]
        for fname in files:
            lname = fname.lower()
            for ext in _RECOGNISED_EXTS:
                if lname.endswith(ext):
                    candidates.append(cur_path / fname)
                    break

    # BrainVision: keep only .vhdr; .eeg/.vmrk are paired sidecars.
    pruned: List[Path] = []
    for p in candidates:
        ln = p.name.lower()
        if ln.endswith(".vhdr"):
            pruned.append(p)
        elif ln.endswith(".eeg") or ln.endswith(".vmrk"):
            if p.with_suffix(".vhdr").exists():
                continue
            pruned.append(p)
        else:
            pruned.append(p)
    return sorted(pruned)


def _bids_id_from_filename(name: str) -> str:
    """Return a BIDS-safe subject token derived from ``name``.

    Strips file extensions, then a leading ``sub-`` or ``sub_`` prefix,
    then keeps alphanumerics. Strips extension first so a flat-layout
    fallback like ``Subject20_1.edf`` becomes ``Subject201``, not
    ``Subject201edf``.
    """
    # Strip every extension layer (handles .nii.gz / .tar.gz / .vhdr /
    # .edf / etc.). ``Path("a.fif.gz").stem`` is ``"a.fif"``; we want
    # the bare stem, so iterate.
    s = name
    while "." in s and not s.startswith("."):
        s = Path(s).stem
        if "." not in s:
            break
    low = s.lower()
    if low.startswith("sub-") or low.startswith("sub_"):
        s = s[4:]
    return re.sub(r"[^0-9A-Za-z]+", "", s)


def extract_task_run(stem: str) -> tuple[str, str]:
    r"""Extract ``(task, run)`` from a recording's filename stem.

    Tries patterns in order of specificity:

    1. **BIDS form** ``task-X[_run-Y]`` anywhere in the stem — wins.
    2. **Underscore form** ``task_X(_run_Y)?`` (Klingelbach style):
       ``task_driving_run_05_00.fif`` → task=``driving``, run=``05``.
       The trailing ``_00`` (file-numbering artefact) is dropped.
    3. **PhysioBank form** ``[A-Z]+(\d{2,})`` (e.g. eegmmidb's
       ``S001R03``): the trailing ``R<digits>`` is treated as the run
       number; the part of the stem matching the parent folder name is
       discarded so the task field stays empty for the user to fill.
    4. **Bare run** ``_run[-_]<N>`` anywhere — extracts run; rest of
       stem becomes the task hint.
    5. **Fallback** — sanitised stem as task, no run.

    Returns ``(task, run)`` — either may be empty.
    """
    if not stem:
        return "", ""

    # 1. BIDS literal — strongest signal.
    m = re.search(r"task-([A-Za-z0-9]+)(?:_run-(\d+))?", stem)
    if m:
        return m.group(1), (m.group(2) or "")

    # 2. Underscore form: task_<name>(_run_<N>)?
    m = re.match(
        r"^task[_\-]+([A-Za-z]+)(?:[_\-]+run[_\-]+(\d+))?",
        stem, flags=re.IGNORECASE,
    )
    if m:
        return m.group(1), (m.group(2) or "")

    # 3. Bare ``_run-N`` or ``_run_N`` somewhere in the stem.
    m = re.search(r"(?:^|[_\-])run[_\-](\d+)", stem, flags=re.IGNORECASE)
    if m:
        run = m.group(1)
        # Strip the run token (and everything after) to derive a task hint.
        head = stem[: m.start()]
        task = re.sub(r"[^0-9A-Za-z]+", "", head)
        return task, run

    # 4. PhysioBank ``...R<digits>`` (eegmmidb): single capital R then digits.
    m = re.search(r"R(\d+)", stem)
    if m:
        # Everything before the R is usually the subject id; drop it.
        # Everything after is usually empty for PhysioBank. Leave task
        # blank so the user fills in the proper dataset task name.
        return "", m.group(1)

    # 5. Fallback: sanitised stem.
    return re.sub(r"[^0-9A-Za-z]+", "", stem), ""


def guess_subject_session_task(
    path: Path, root: Path,
) -> tuple[str, str, str, str]:
    """Heuristic: pull subject/session/task/run hints from ``path``.

    BIDS form (``sub-XXX/ses-YYY/...``) wins for subject and session.
    Task and run come from :func:`extract_task_run` applied to the
    filename stem (after stripping ``.nii.gz``/``.fif`` etc).

    Returns ``(subject, session, task, run)``. Empty strings where no
    hint was found.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = list(rel.parts)
    sub = ses = ""
    task = run = ""
    for part in parts:
        if not sub:
            m = re.match(r"sub-([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
            if m:
                sub = m.group(1)
        if not ses:
            m = re.match(r"ses-([A-Za-z0-9]+)", part, flags=re.IGNORECASE)
            if m:
                ses = m.group(1)
    if not sub and parts:
        sub = _bids_id_from_filename(parts[0])

    # Task / run from the filename stem (handles many conventions).
    stem = path.stem
    if stem.lower().endswith(".fif"):
        stem = stem[:-4]
    task, run = extract_task_run(stem)

    # mne-bids requires a non-empty task. When ``extract_task_run``
    # legitimately can't find one (e.g. PhysioBank stems where the
    # filename is just ``<subject><run>`` and we already pulled the
    # run), fall back to the sanitised stem so the row stays
    # convertible. The user sees this in the TSV and can rename to a
    # meaningful value before convert.
    if not task:
        task = re.sub(r"[^0-9A-Za-z]+", "", stem) or "task"
    return sub, ses, task, run


# ---------------------------------------------------------------------------
# Public scan entry point
# ---------------------------------------------------------------------------


def scan_eeg_meg(
    root: Path,
    *,
    dataset: Optional[str] = None,
    candidates: Optional[List[Path]] = None,
    line_freq: Optional[float] = None,
    montage: Optional[str] = None,
) -> pd.DataFrame:
    """Walk ``root`` and return a DataFrame of probed EEG/MEG recordings.

    Parameters
    ----------
    root
        Directory containing raw EEG/MEG/iEEG/NIRS files. May contain
        nested subdirectories.
    dataset
        Optional dataset slug stamped into every row's ``dataset``
        column. The CLI fills in a default before this function runs.
    candidates
        Optional pre-computed list of candidate paths. When provided
        we skip the directory walk; used by ``cli/scan.py`` so it can
        do one combined walk for both modalities.

    Returns
    -------
    pandas.DataFrame
        One row per probed recording. Columns are universal scan
        columns + :data:`EEG_MEG_COLUMNS`. Empty DataFrame (with the
        correct columns) when no recordings are found or mne is
        unavailable.
    """
    if candidates is None:
        root = Path(root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        candidates = candidate_paths(root)
    else:
        root = Path(root).resolve()

    if not _HAS_MNE:
        log.warning(
            "mne is not installed; EEG/MEG recordings will not be probed. "
            "Install mne and mne-bids to enable EEG/MEG support."
        )
        return _empty_dataframe()

    rows: List[dict] = []
    bids_counter = 0
    bids_id_for_subject: dict[str, str] = {}

    for path in candidates:
        probe = _probe(path)
        if probe is None:
            continue

        sub_hint, ses_hint, task_hint, run_hint = guess_subject_session_task(path, root)
        sub_token = sub_hint or _bids_id_from_filename(path.parent.name)
        if not sub_token:
            sub_token = path.stem

        if sub_token not in bids_id_for_subject:
            bids_counter += 1
            bids_id_for_subject[sub_token] = f"sub-{bids_counter:03d}"
        bids_name = bids_id_for_subject[sub_token]

        datatype = probe.datatype or "eeg"
        # Build the BIDS basename via the schema engine so the row's
        # ``proposed_basename`` is canonical and consumable by both the
        # converter and the metadata audit.
        # Use the schema's canonical entity names: ``subject`` and
        # ``session`` (not the short ``sub``/``ses`` BIDS tokens).
        # ``schema.build_basename`` expects these keys.
        entities: dict[str, str] = {}
        sub_token_clean = bids_name[len("sub-"):] if bids_name.startswith("sub-") else bids_name
        entities["subject"] = sub_token_clean
        if ses_hint:
            entities["session"] = ses_hint
        if task_hint:
            entities["task"] = task_hint
        if run_hint:
            # BIDS run is an integer; strip leading zeros.
            entities["run"] = str(int(run_hint)) if run_hint.isdigit() else run_hint
        # For EEG/MEG/iEEG the file's own datatype-specific suffix matches
        # the datatype key (eeg, meg, ieeg, nirs).
        suffix = datatype
        try:
            basename = schema_mod.build_basename(entities, datatype, suffix)
        except Exception as exc:
            log.debug(
                "schema.build_basename failed for %s (%s/%s): %s",
                bids_name, datatype, suffix, exc,
            )
            # Fallback: stitch manually so the row still has a useful name.
            parts_for_name = [bids_name]
            if ses_hint:
                parts_for_name.append(f"ses-{ses_hint}")
            if task_hint:
                parts_for_name.append(f"task-{task_hint}")
            if run_hint:
                parts_for_name.append(f"run-{int(run_hint) if run_hint.isdigit() else run_hint}")
            parts_for_name.append(suffix)
            basename = "_".join(parts_for_name)

        try:
            rel_source = path.relative_to(root).as_posix()
        except ValueError:
            rel_source = str(path)

        # The canonical entities dict — same one used to build the
        # basename, JSON-encoded for the TSV's ``entities`` column.
        # This is the **source of truth**: ``bidsmgr-rebuild`` and the
        # converter both read from here.
        entities_for_tsv = dict(entities)
        # Translate the schema's "subject" key back to the BIDS-id
        # representation expected by the convert orchestrator.
        if "sub" in entities_for_tsv and "subject" not in entities_for_tsv:
            pass  # already in canonical form

        rows.append({
            "subject": sub_token,
            "BIDS_name": bids_name,
            "session": f"ses-{ses_hint}" if ses_hint else "",
            "source_folder": str(path.parent.relative_to(root))
                if path.parent != root else "",
            "include": 1,
            "modality": datatype,
            "modality_bids": datatype,
            "proposed_datatype": datatype,
            "proposed_basename": basename,
            "Proposed BIDS name": f"{basename}",
            "entities": json.dumps(entities_for_tsv, sort_keys=True),
            "task": task_hint,
            "run": run_hint,
            "format": probe.fmt,
            "source_file": rel_source,
            "n_channels": probe.n_channels,
            "sfreq": probe.sfreq,
            "duration_sec": round(probe.duration_sec, 3),
            "n_times": probe.n_times,
            "recording_time": probe.recording_time,
            "has_positions": int(probe.has_positions),
            "line_freq": "" if line_freq is None else line_freq,
            "montage": montage or "",
            "dataset": dataset or "",
            # Surface the classifier-style columns so the inspector's
            # ``suffix`` and ``data`` columns show useful values for
            # EEG/MEG rows. MRI rows get these from the BidsGuess
            # classifier; EEG/MEG rows are scanner-deterministic so
            # we set them directly here.
            "bids_guess_classifier": "eeg_meg_scanner",
            "bids_guess_datatype": datatype,
            "bids_guess_suffix": datatype,  # eeg / meg / ieeg / nirs
            "bids_guess_confidence": "1.00",
            "bids_guess_skip": False,
        })

    if not rows:
        return _empty_dataframe()

    df = pd.DataFrame(rows)
    df = _infer_sessions_from_recording_time(df)
    return df


def _infer_sessions_from_recording_time(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster a subject's recordings by date and assign ses-1 / ses-2 / ….

    Mirrors ``inventory/mri_dicom.py``'s longitudinal session inference,
    which uses ``StudyDate`` from DICOM headers. For EEG/MEG we use the
    date part of mne's ``raw.info['meas_date']`` (already stored in the
    row's ``recording_time`` column as an ISO timestamp).

    Rules:

    * Skip rows whose ``session`` column is already populated (path-
      based ``ses-XXX`` token wins — user intent first).
    * Skip subjects with only 0 or 1 distinct recording dates.
    * For subjects with 2+ distinct dates, assign ``ses-1, ses-2, …``
      ordered chronologically. Update both ``session`` AND
      ``proposed_basename`` so the canonical BIDS name reflects it.
    """
    if df.empty or "recording_time" not in df.columns:
        return df

    out = df.copy()
    # Pull the date prefix (YYYY-MM-DD) from the ISO timestamp.
    date_part = out["recording_time"].astype(str).str.slice(0, 10)
    # Sentinel for missing — pandas will treat empty as a normal group.
    date_part = date_part.where(date_part.str.len() == 10, "")

    for bids_name, idx in out.groupby("BIDS_name").groups.items():
        rows_idx = list(idx)
        # Don't touch rows the path-regex already gave a session.
        with_session = [
            i for i in rows_idx
            if str(out.at[i, "session"]).strip()
        ]
        if with_session:
            continue
        unique_dates = sorted({
            d for d in (date_part.at[i] for i in rows_idx) if d
        })
        if len(unique_dates) < 2:
            continue
        date_to_ses = {
            d: f"ses-{n + 1}" for n, d in enumerate(unique_dates)
        }
        for i in rows_idx:
            d = date_part.at[i]
            ses_label = date_to_ses.get(d)
            if not ses_label:
                continue
            out.at[i, "session"] = ses_label
            # Splice the session into the BIDS basename.
            out.at[i, "proposed_basename"] = _splice_session_into_basename(
                str(out.at[i, "proposed_basename"]),
                ses_label,
            )
            out.at[i, "Proposed BIDS name"] = out.at[i, "proposed_basename"]
            # Keep the canonical ``entities`` JSON in sync — the rebuild
            # engine reads from here.
            ent_raw = str(out.at[i, "entities"]).strip()
            if ent_raw:
                try:
                    ent = json.loads(ent_raw)
                except json.JSONDecodeError:
                    ent = {}
                if isinstance(ent, dict):
                    ses_value = ses_label[len("ses-"):] if ses_label.startswith("ses-") else ses_label
                    ent["session"] = ses_value
                    out.at[i, "entities"] = json.dumps(ent, sort_keys=True)
    return out


def _splice_session_into_basename(basename: str, session_label: str) -> str:
    """Insert ``ses-N`` into a BIDS basename right after ``sub-XXX``.

    No-op if the basename already contains a ``_ses-`` token.
    """
    if not basename or "_ses-" in basename:
        return basename
    parts = basename.split("_")
    if not parts or not parts[0].startswith("sub-"):
        # Unusual shape; prepend instead of failing.
        return f"{session_label}_{basename}"
    parts.insert(1, session_label)
    return "_".join(parts)


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the EEG/MEG row columns present."""
    cols = [
        "subject", "BIDS_name", "session", "source_folder", "include",
        "modality", "modality_bids", "proposed_datatype",
        "proposed_basename", "Proposed BIDS name",
    ] + list(EEG_MEG_COLUMNS) + ["dataset"]
    return pd.DataFrame(columns=cols)


__all__ = [
    "EEG_MEG_COLUMNS",
    "ProbeResult",
    "candidate_paths",
    "guess_subject_session_task",
    "scan_eeg_meg",
]
