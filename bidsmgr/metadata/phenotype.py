"""Write the BIDS ``phenotype/`` directory from user-supplied measure tables.

Phenotype is participant-level, non-imaging measurement data (questionnaires,
clinical scales, cognitive batteries) that does not fit a single
``participants.tsv`` row. Each input table (TSV/CSV/XLSX/ODS), keyed by
``participant_id``, becomes ``phenotype/<measure>.tsv`` plus a
``<measure>.json`` data dictionary. Modality-agnostic: an MRI or EEG/MEG
dataset uses it identically.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix in (".tsv", ".txt"):
            return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if suffix == ".csv":
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        return pd.read_excel(path, dtype=str).fillna("")
    except Exception as exc:
        log.warning("could not read phenotype table %s: %s", path, exc)
        return None


def measure_name(stem: str) -> str:
    """Sanitise a file stem into a BIDS-safe measure label."""
    name = re.sub(r"[^A-Za-z0-9]+", "", stem)
    return name or "measure"


def load_sidecar_dictionary(table_path: Path) -> dict:
    """Load an optional user-authored data dictionary (codebook) for a table.

    For a measure / participants table ``foo.tsv`` (or ``.csv`` / ``.xlsx``), a
    sibling ``foo.json`` lets the user supply real column descriptions, ``Levels``,
    and ``Units`` (plus, for phenotype, a ``MeasurementToolMetadata`` block) - a
    codebook expressed as the BIDS data dictionary itself.
    Returns ``{}`` when there is no sibling JSON or it cannot be parsed.
    """
    path = Path(table_path)
    sidecar = path.with_suffix(".json")
    if sidecar == path or not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read codebook %s: %s", sidecar, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_phenotype(
    bids_root: Path,
    phenotype_files: Optional[Sequence[Path]],
    report,
) -> None:
    """Emit ``phenotype/<measure>.tsv`` + ``.json`` for each input table.

    Each table must have a ``participant_id`` column; tables that lack one are
    skipped with a warning. The JSON is a minimal data dictionary describing
    each non-id column. ``report`` collects written files and warnings, matching
    the rest of the metadata engine.
    """
    if not phenotype_files:
        return

    pheno_dir = bids_root / "phenotype"
    for raw_path in phenotype_files:
        path = Path(raw_path)
        if not path.exists():
            report.warnings.append(f"phenotype file not found: {path}")
            continue
        df = _read_table(path)
        if df is None:
            report.warnings.append(f"could not read phenotype file: {path}")
            continue
        if "participant_id" not in df.columns:
            report.warnings.append(
                f"phenotype file {path} has no 'participant_id' column; skipping"
            )
            continue

        pheno_dir.mkdir(parents=True, exist_ok=True)
        measure = measure_name(path.stem)

        tsv_out = pheno_dir / f"{measure}.tsv"
        df.to_csv(tsv_out, sep="\t", index=False)
        report.files_written.append(tsv_out)

        # Data dictionary: merge any user-authored codebook (sibling JSON) so
        # real Descriptions / Levels / Units (and an optional
        # MeasurementToolMetadata block) flow through; auto-fill the rest with
        # the bare column name. The codebook entry wins where present.
        codebook = load_sidecar_dictionary(path)
        json_out = pheno_dir / f"{measure}.json"
        data_dict: dict = {}
        tool_meta = codebook.get("MeasurementToolMetadata")
        if isinstance(tool_meta, dict) and tool_meta:
            data_dict["MeasurementToolMetadata"] = tool_meta
        for col in df.columns:
            if col == "participant_id":
                continue
            entry = codebook.get(str(col))
            if isinstance(entry, dict) and entry:
                data_dict[str(col)] = entry
            else:
                data_dict[str(col)] = {"Description": str(col)}
        json_out.write_text(json.dumps(data_dict, indent=2) + "\n", encoding="utf-8")
        report.files_written.append(json_out)
        log.info(
            "phenotype: wrote %s (%d rows, %d columns%s)",
            tsv_out.name, len(df), len(df.columns),
            "; codebook merged" if codebook else "",
        )


__all__ = ["write_phenotype", "load_sidecar_dictionary", "measure_name"]
