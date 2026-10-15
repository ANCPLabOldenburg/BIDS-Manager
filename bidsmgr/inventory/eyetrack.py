"""Finding EyeLink eye-tracking recordings in a raw folder.

One inventory row per recording (``.edf`` by its ``SR_RESEARCH`` signature,
so a European Data Format ``.edf`` EEG file is never taken for one, and
``.asc`` by its header), with ``modality`` = ``eyetrack``. The row names
the BIDS eye-tracking file (BEP020, BIDS 1.11): a ``physio`` recording with
``recording-eye1`` (left), ``eye2`` (right), or both for a binocular file,
in ``beh/`` until the user says which run it belongs to (then ``func/``,
``eeg/``, ...). The ``task`` is what the file name says, else for the user
to fill in.

The header facts (tracker, sampling rate, eyes, length) come from the file's
preamble and, when a reader is installed, from a pass over its events only
(``inventory/eyelink``), which is fast. With no reader the row is still
listed, excluded, with the reason and what to install.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .eyelink import edf_reader, is_eyelink_asc, is_eyelink_edf, read_preamble, read_recording

log = logging.getLogger(__name__)

#: The ``recording`` label of each eye (the standard's recommendation).
RECORDING_LABEL = {"left": "eye1", "right": "eye2"}
NO_READER_TOKEN = "needs an EyeLink reader"


def is_eyetrack_file(path) -> bool:
    return is_eyelink_edf(path) or is_eyelink_asc(path)


def find_eyetrack_files(root) -> list[Path]:
    """Every EyeLink ``.edf`` / ``.asc`` under ``root`` (by content)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            low = name.lower()
            if not low.endswith((".edf", ".asc")):
                continue
            fp = Path(dirpath) / name
            if is_eyetrack_file(fp):
                out.append(fp)
    return sorted(out)


def _subject_from_path(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    for part in parts:
        if part.lower().startswith("sub-"):
            return part[4:]
    return path.parent.name if len(parts) > 1 else path.stem


def _when(date: str) -> tuple[str, str]:
    """``(YYYYMMDD, HHMMSS)`` from the preamble's ``DATE`` (the tracker
    computer's clock), as the DICOM columns hold them."""
    from datetime import datetime

    try:
        when = datetime.strptime(" ".join(date.split()), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return "", ""
    return when.strftime("%Y%m%d"), when.strftime("%H%M%S")


def probe(path: Path) -> dict:
    """What a row shows about a recording: the preamble, and, when a reader
    is here, the eyes, sampling rate and length (events only, no samples)."""
    facts = {"preamble": read_preamble(path), "eyes": (), "sfreq": 0.0, "duration": 0.0,
             "reader": "", "why": ""}
    reader, why = edf_reader() if is_eyelink_edf(path) else ("asc", "")
    facts["reader"], facts["why"] = reader, why
    if not reader:
        return facts
    try:
        rec = read_recording(path, samples=False)
    except Exception as exc:  # noqa: BLE001 - the row says why
        facts["why"] = f"{path.name} could not be read: {exc}"
        facts["reader"] = ""
        return facts
    facts["eyes"] = rec.eyes
    facts["sfreq"] = rec.sfreq
    facts["duration"] = (sum(b - a for a, b in rec.blocks) / 1000.0 if rec.blocks
                         else rec.duration_s)
    facts["n_messages"] = len(rec.messages)
    return facts


def recording_label(eyes) -> str:
    """The ``recording`` entity of the row (the first eye's)."""
    eyes = tuple(eyes or ())
    return RECORDING_LABEL[eyes[0]] if eyes else "eye1"


def scan_eyetrack(root_dir, *, dataset: Optional[str] = None, cancel_check=None
                  ) -> pd.DataFrame:
    """Inventory every EyeLink recording under ``root_dir``."""
    from .. import schema as schema_mod
    from ..util.cancel import OperationCancelled, is_cancelled
    from .eeg_meg import extract_task_run

    root = Path(root_dir)
    rows: list[dict] = []
    ids: dict[str, str] = {}
    for idx, fp in enumerate(find_eyetrack_files(root)):
        if (idx & 3) == 0 and is_cancelled(cancel_check):
            raise OperationCancelled("scan cancelled by user")
        facts = probe(fp)
        subject = _subject_from_path(fp, root)
        if subject not in ids:
            ids[subject] = f"sub-{len(ids) + 1:03d}"
        participant = ids[subject]
        stem = fp.name.rsplit(".", 1)[0]
        task, run = extract_task_run(stem)
        entities = {"subject": participant[4:], "recording": recording_label(facts["eyes"])}
        if task:
            entities["task"] = task
        if run:
            entities["run"] = str(int(run)) if str(run).isdigit() else run
        try:
            basename = schema_mod.build_basename(entities, "beh", "physio")
        except Exception:  # noqa: BLE001
            basename = f"{participant}_recording-{entities['recording']}_physio"
        rel = fp.relative_to(root)
        pre = facts["preamble"]
        eyes = facts["eyes"]
        notes = []
        include = 1
        if not facts["reader"]:
            include = 0
            notes.append(f"{NO_READER_TOKEN}: {facts['why']}")
        if len(eyes) == 2:
            notes.append("binocular: both eyes are written (recording-eye1 and eye2)")
        rows.append({
            "subject": subject,
            "participant_id": participant,
            "datatype": "beh",
            "bids_name": basename,
            "bids_path": basename,
            "entities": json.dumps(entities, sort_keys=True),
            "task": task or "",
            "run": entities.get("run", ""),
            "source_folder": str(rel.parent) if rel.parent != Path(".") else root.name,
            "source_file": str(rel),
            "format": "EyeLink EDF" if is_eyelink_edf(fp) else "EyeLink ASC",
            "sequence": stem,
            "include": include,
            "n_files": 1,
            "modality": "eyetrack",
            "sequence_kind": "eyetrack",
            "bids_guess_datatype": "beh",
            "bids_guess_suffix": "physio",
            "bids_guess_classifier": "eyelink_header",
            "bids_guess_confidence": 0.90,
            "bids_guess_skip": not facts["reader"],
            "manufacturer_suggestion": "SR Research",
            "sfreq": facts["sfreq"] or "",
            "duration_sec": round(facts["duration"], 3) if facts["duration"] else "",
            "study_date": _when(pre.get("date", ""))[0],
            "acq_time": _when(pre.get("date", ""))[1],
            "recording_time": pre.get("date", ""),
            "issues": "; ".join(notes),
        })
        if not facts["reader"]:
            log.warning("eye tracking: %s (%s)", facts["why"], fp)
    out = pd.DataFrame(rows)
    if not out.empty and dataset is not None:
        out["dataset"] = dataset
    return out


__all__ = ["NO_READER_TOKEN", "RECORDING_LABEL", "find_eyetrack_files", "is_eyetrack_file",
           "probe", "recording_label", "scan_eyetrack"]
