#!/usr/bin/env python3
"""
run_heudiconv_from_heuristic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Launch HeuDiConv using *auto_heuristic.py*,
handling cleaned-vs-physical folder names automatically.
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
import subprocess
import os
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterable, List, Optional
import pandas as pd
import re

from bidsphysio.dcm2bids import dcm2bidsphysio
from pydicom import dcmread
from pydicom.dataset import Dataset

from bids_manager.schema_renamer import normalize_study_name

# Acceptable DICOM file extensions (lower case)
# Some Siemens datasets omit file extensions; we therefore supplement the
# extension check with a quick sniff of the header for the ``DICM`` tag.
DICOM_EXTS = (".dcm", ".ima")


def is_dicom_file(path: str) -> bool:
    """Return ``True`` if *path* appears to be a DICOM file."""

    name = Path(path).name.lower()
    if name.endswith(DICOM_EXTS):
        return True
    if "." in name:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

# ────────────────── helpers ──────────────────
def load_heuristic_module(heur: Path) -> ModuleType:
    """Return the imported heuristic module located at ``heur``."""

    spec = importlib.util.spec_from_file_location("heuristic", heur)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_sid_map(heur: Path) -> Dict[str, str]:
    """Load the ``SID_MAP`` dictionary from a heuristic file."""

    module = load_heuristic_module(heur)
    return module.SID_MAP  # type: ignore[attr-defined]


def _is_included(value: object) -> bool:
    """Return ``True`` when the *include* column represents a truthy value."""

    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no"}


def _map_physio_templates(module: ModuleType, df: pd.DataFrame) -> Dict[int, str]:
    """Return mapping of DataFrame index → BIDS template for physio rows."""

    info_func = getattr(module, "infotodict", None)
    if info_func is None:
        return {}

    seqinfo = []
    id_to_idx: Dict[str, int] = {}
    counter = 0
    for idx, row in df.iterrows():
        folder = Path(str(row.get("source_folder", "") or ".")).name
        sequence = str(row.get("sequence", ""))
        uid_field = str(row.get("series_uid", ""))
        uids = [u for u in uid_field.split("|") if u] or [""]
        for uid in uids:
            sid = f"physio{counter}"
            counter += 1
            seqinfo.append(
                SimpleNamespace(
                    series_description=sequence,
                    dcm_dir_name=folder,
                    series_uid=uid,
                    series_id=sid,
                )
            )
            id_to_idx[sid] = idx

    if not seqinfo:
        return {}

    mapping: Dict[int, str] = {}
    info = info_func(seqinfo)
    for key, ids in info.items():
        if not isinstance(key, tuple) or not key:
            continue
        template = key[0]
        if not isinstance(template, str):
            continue
        for sid in ids:
            idx = id_to_idx.get(sid)
            if idx is not None and idx not in mapping:
                mapping[idx] = template
    return mapping


def _split_uid_field(uid_field: object) -> List[str]:
    """Return list of SeriesInstanceUID values encoded in ``uid_field``."""

    if isinstance(uid_field, (list, tuple, set)):
        values = [str(v).strip() for v in uid_field]
    else:
        values = [s.strip() for s in str(uid_field or "").split("|")]
    return [v for v in values if v]


def _candidate_dicom_paths(dicom_dir: Path) -> List[Path]:
    """Return sorted list of DICOM files contained in ``dicom_dir``."""

    return sorted(
        f for f in dicom_dir.iterdir()
        if f.is_file() and is_dicom_file(str(f))
    )


def _resolve_direct_physio_path(raw_root: Path, dicom_dir: Path, row: pd.Series) -> Optional[Path]:
    """Return direct physio DICOM path hinted in ``row`` if available."""

    path_keys: Iterable[str] = ("dicom_path", "dicom_file", "physio_path", "file_path")
    for key in path_keys:
        value = row.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            # Try relative to the recorded folder first, then the root.
            rel_candidate = dicom_dir / candidate
            if rel_candidate.exists():
                return rel_candidate
            candidate = raw_root / candidate
        if candidate.exists():
            return candidate
    return None


def _looks_like_physio(dataset: Dataset, expected: str) -> bool:
    """Return ``True`` when ``dataset`` metadata resembles a physio recording."""

    def _to_text(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(v) for v in value)
        return str(value)

    expected = expected.strip().lower()
    keywords = [
        expected,
        "physio",
        "physiolog",
        "physlog",
        "pulse",
        "resp",
    ]

    for attr in ("SeriesDescription", "ProtocolName", "SequenceName"):
        text = _to_text(getattr(dataset, attr, "")).strip().lower()
        if not text:
            continue
        if any(keyword and keyword in text for keyword in keywords):
            return True

    image_type = getattr(dataset, "ImageType", None)
    if image_type:
        text = " ".join(str(v).lower() for v in image_type if v)
        if any(keyword and keyword in text for keyword in keywords):
            return True

    return False


def _valid_physio_candidate(row: pd.Series, path: Path) -> bool:
    """Return ``True`` when ``path`` points to the intended physio DICOM."""

    try:
        dataset = dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:
        print(f"Failed to read physio DICOM {path}: {exc}")
        return False

    uid_values = set(_split_uid_field(row.get("series_uid", "")))
    series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
    if uid_values:
        if series_uid not in uid_values:
            print(
                "Skipping physio candidate",
                path,
                "because SeriesInstanceUID",
                series_uid or "<missing>",
                "does not match expected values",
                ",".join(sorted(uid_values)) or "<missing>",
            )
            return False
        return True

    sequence = str(row.get("sequence", ""))
    if _looks_like_physio(dataset, sequence):
        return True

    print(f"Skipping physio candidate {path}: metadata does not look like a physio recording.")
    return False


def _resolve_physio_dicom(raw_root: Path, dicom_dir: Path, row: pd.Series) -> Optional[Path]:
    """Return the DICOM file corresponding to ``row`` if present."""

    direct = _resolve_direct_physio_path(raw_root, dicom_dir, row)
    if direct is not None:
        if _valid_physio_candidate(row, direct):
            return direct
        return None

    dicom_files = _candidate_dicom_paths(dicom_dir)
    if not dicom_files:
        return None

    for path in dicom_files:
        if _valid_physio_candidate(row, path):
            return path

    return None


def _resolve_physio_source_folder(raw_root: Path, source_folder: str) -> Path:
    """Return the folder containing the physio DICOM files.

    The *source_folder* column in the heuristic table should contain a path
    that is relative to ``raw_root``.  Some datasets, however, repeat the name
    of ``raw_root`` as the first path component (for example ``OL_3844`` inside
    ``.../OL_3844``).  This helper collapses those redundant segments so the
    resulting path correctly resolves to the actual DICOM directory.
    """

    cleaned = str(source_folder or "").strip()
    if not cleaned:
        return raw_root

    candidate_path = Path(cleaned)
    if candidate_path.is_absolute():
        # Absolute paths are returned as-is so existing behaviour is preserved.
        return candidate_path

    combined = raw_root / candidate_path
    if combined.exists():
        return combined

    # Remove any duplicated occurrences of the raw folder name from the front
    # of the candidate path (``OL_3844/OL_3844`` → ``OL_3844`` → ``.``).
    parts = list(candidate_path.parts)
    while parts and parts[0] == raw_root.name:
        parts.pop(0)

    if not parts:
        # The path only repeated the raw folder name; fall back to raw_root.
        return raw_root

    trimmed = raw_root / Path(*parts)
    if trimmed.exists():
        return trimmed

    # If the trimmed path still does not exist, preserve the previous
    # behaviour and return the combined (non-existent) path so callers can
    # emit the usual warning message.
    return combined


def convert_physio_series(raw_root: Path,
                          bids_out: Path,
                          module: ModuleType,
                          df: pd.DataFrame) -> None:
    """Convert physiological DICOM series described in ``df`` using bidsphysio."""

    if df.empty:
        return

    if "modality" not in df.columns:
        return

    physio_df = df[df["modality"].astype(str).str.lower() == "physio"].copy()
    if physio_df.empty:
        return

    if "include" not in physio_df.columns:
        return

    physio_df = physio_df[physio_df["include"].apply(_is_included)]
    if physio_df.empty:
        return

    template_map = _map_physio_templates(module, physio_df)
    if not template_map:
        print("No physio entries found in heuristic; skipping physiology conversion.")
        return

    converted: set[str] = set()
    for idx, row in physio_df.iterrows():
        template = template_map.get(idx)
        if not template:
            print(f"Skipping physio series '{row.get('sequence', '')}': no heuristic template match.")
            continue

        source_folder = str(row.get("source_folder", ""))
        dicom_dir = _resolve_physio_source_folder(raw_root, source_folder)
        if not dicom_dir.exists():
            print(f"Physio source folder missing: {dicom_dir}")
            continue

        dicom_path = _resolve_physio_dicom(raw_root, dicom_dir, row)
        if dicom_path is None:
            print(f"No DICOM physiolog files found in {dicom_dir}; skipping.")
            continue

        prefix_path = bids_out / template
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_str = str(prefix_path)
        if prefix_str in converted:
            continue

        print(f"Converting physio {dicom_path} → {prefix_path}")
        physio_data = dcm2bidsphysio.dcm2bids(dicom_path)
        physio_data.save_to_bids_with_trigger(prefix_str)
        converted.add(prefix_str)


def clean_name(raw: str) -> str:
    """Return alphanumeric-only version of ``raw``."""

    return "".join(ch for ch in raw if ch.isalnum())

def safe_stem(text: str) -> str:
    """Return filename-friendly version of *text* (used for study names).

    Study identifiers are cleaned with :func:`normalize_study_name` first so we
    collapse accidental repeats such as ``"study_study"`` into a single token
    before substituting filesystem-friendly underscores.
    """

    cleaned = normalize_study_name(text)
    return re.sub(r"[^0-9A-Za-z_-]+", "_", cleaned).strip("_")


def physical_by_clean(raw_root: Path) -> Dict[str, str]:
    """Return mapping cleaned_name → relative folder path for all subdirs."""
    mapping: Dict[str, str] = {
        "": "",
        ".": "",
        raw_root.name: "",
        clean_name(raw_root.name): "",
    }
    for p in raw_root.rglob("*"):
        if not p.is_dir():
            continue
        rel = str(p.relative_to(raw_root))
        base = p.name
        mapping.setdefault(rel, rel)
        mapping.setdefault(clean_name(rel), rel)
        mapping.setdefault(base, rel)
        mapping.setdefault(clean_name(base), rel)
    return mapping



def detect_depth(folder: Path) -> int:
    """Minimum depth (#subdirs) from *folder* to any DICOM file."""

    for root, _dirs, files in os.walk(folder):
        if any(is_dicom_file(os.path.join(root, f)) for f in files):
            rel = Path(root).relative_to(folder)
            return len(rel.parts)
    raise RuntimeError(f"No DICOMs under {folder}")


def heudi_cmd(raw_root: Path,
              phys_folders: List[str],
              heuristic: Path,
              bids_out: Path,
              depth: int) -> List[str]:
    """Build the ``heudiconv`` command for the given parameters."""
    wild = "*/" * depth

    if len(phys_folders) == 1 and phys_folders[0] == "":
        subj = clean_name(raw_root.name) or "root"
        return [
            "heudiconv",
            "--files",
            str(raw_root),
            "-s",
            subj,
            "-f",
            str(heuristic),
            "-c",
            "dcm2niix",
            "-o",
            str(bids_out),
            "-b",
            "--minmeta",
            "--overwrite",
        ]

    # Use "*" instead of "*.*" so DICOMs without extensions are also matched
    template = f"{raw_root}/" + "{subject}/" + wild + "*"
    subjects = [p or clean_name(raw_root.name) for p in phys_folders]
    return [
        "heudiconv",
        "-d",
        template,
        "-s",
        *subjects,
        "-f",
        str(heuristic),
        "-c",
        "dcm2niix",
        "-o",
        str(bids_out),
        "-b",
        "--minmeta",
        "--overwrite",
    ]


def _parse_age(value: str) -> str:
    """Return numeric age from DICOM-style age strings (e.g. '032Y')."""
    m = re.match(r"(\d+)", str(value))
    if not m:
        return str(value)
    age = m.group(1).lstrip("0")
    return age or "0"


def write_participants(summary_path: Path, bids_root: Path) -> None:
    """Generate ``participants.tsv`` from ``subject_summary.tsv``.

    The generated table always contains the legacy demographic columns and the
    newly requested ``FamilyName`` and ``patientID`` columns (in that order) so
    downstream tooling can rely on a stable layout regardless of the input
    summary.
    """
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path, sep="\t", keep_default_na=False)
    # Ensure the demographics we expect are present even when reading a legacy
    # ``subject_summary.tsv`` that predates the new columns.
    for column in ("FamilyName", "PatientID"):
        if column not in df.columns:
            df[column] = ""

    part_df = (
        df[
            [
                "BIDS_name",
                "GivenName",
                "PatientSex",
                "PatientAge",
                "FamilyName",
                "PatientID",
            ]
        ]
        .drop_duplicates(subset=["BIDS_name"])
        .copy()
    )
    if part_df.empty:
        return

    part_df["PatientAge"] = part_df["PatientAge"].apply(_parse_age)
    part_df.rename(
        columns={
            "BIDS_name": "participant_id",
            "GivenName": "given_name",
            "PatientSex": "sex",
            "PatientAge": "age",
            "PatientID": "patientID",
        },
        inplace=True,
    )

    # Write columns in a deterministic order so the new fields appear at the
    # end of the table as requested.
    column_order = [
        "participant_id",
        "given_name",
        "sex",
        "age",
        "FamilyName",
        "patientID",
    ]
    part_df = part_df[column_order]

    part_path = bids_root / "participants.tsv"
    part_df.to_csv(part_path, sep="\t", index=False)


# ────────────────── main runner ──────────────────
def run_heudiconv(raw_root: Path,
                  heuristic: Path,
                  bids_out: Path,
                  per_folder: bool = True,
                  mapping_df: Optional[pd.DataFrame] = None) -> None:
    """Run HeuDiConv using ``heuristic`` and write output to ``bids_out``."""

    heur_module      = load_heuristic_module(heuristic)
    sid_map          = heur_module.SID_MAP              # type: ignore[attr-defined]
    clean2phys       = physical_by_clean(raw_root)
    cleaned_ids      = sorted(sid_map.keys())
    phys_folders     = [clean2phys[c] for c in cleaned_ids]

    depth = detect_depth(raw_root / phys_folders[0])

    print("Raw root    :", raw_root)
    print("Heuristic   :", heuristic)
    print("Output BIDS :", bids_out)
    print("Folders     :", phys_folders)
    print("Depth       :", depth, "\n")

    bids_out.mkdir(parents=True, exist_ok=True)

    if per_folder:
        for phys in phys_folders:
            print(f"── {phys} ──")
            cmd = heudi_cmd(raw_root, [phys], heuristic, bids_out, depth)
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            print()
    else:
        cmd = heudi_cmd(raw_root, phys_folders, heuristic, bids_out, depth)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    summary_path = bids_out / ".bids_manager" / "subject_summary.tsv"
    physio_rows = pd.DataFrame()
    if mapping_df is not None:
        dataset = bids_out.name
        mdir = bids_out / ".bids_manager"
        study_mask = mapping_df["StudyDescription"].fillna("").apply(safe_stem) == dataset
        # Work on a copy so we can freely add fallback columns without
        # affecting the cached mapping DataFrame passed by the caller.
        sub_df = mapping_df[study_mask].copy()
        if not sub_df.empty:
            mdir.mkdir(exist_ok=True)
            mapping_path = mdir / "subject_mapping.tsv"

            if summary_path.exists():
                old_summary = pd.read_csv(summary_path, sep="\t", keep_default_na=False)
                combined = pd.concat([old_summary, sub_df], ignore_index=True)
                combined.drop_duplicates(inplace=True)
            else:
                combined = sub_df
            combined.to_csv(summary_path, sep="\t", index=False)

            physio_rows = sub_df

            # Preserve the extended subject demographics so the mapping table
            # exposes both the historical and the newly requested columns.
            map_columns = ["GivenName", "BIDS_name", "FamilyName", "PatientID"]
            for column in map_columns:
                if column not in sub_df.columns:
                    sub_df[column] = ""
            new_map = sub_df[map_columns].drop_duplicates().copy()
            new_map.rename(columns={"PatientID": "patientID"}, inplace=True)
            new_map = new_map[["GivenName", "BIDS_name", "FamilyName", "patientID"]]
            if mapping_path.exists():
                old_map = pd.read_csv(mapping_path, sep="\t", keep_default_na=False)
                # Backfill missing columns from legacy mapping files before
                # concatenating so we maintain a consistent schema.
                if "PatientID" in old_map.columns and "patientID" not in old_map.columns:
                    old_map.rename(columns={"PatientID": "patientID"}, inplace=True)
                for column in ("FamilyName", "patientID"):
                    if column not in old_map.columns:
                        old_map[column] = ""
                old_map = old_map[["GivenName", "BIDS_name", "FamilyName", "patientID"]]
                new_map = pd.concat([old_map, new_map], ignore_index=True)
                new_map.drop_duplicates(
                    subset=["GivenName", "BIDS_name", "FamilyName", "patientID"],
                    inplace=True,
                )
            new_map.to_csv(mapping_path, sep="\t", index=False)

    if physio_rows.empty and summary_path.exists():
        try:
            physio_rows = pd.read_csv(summary_path, sep="\t", keep_default_na=False)
        except Exception as exc:
            print(f"Could not load summary for physio conversion: {exc}")
            physio_rows = pd.DataFrame()

    convert_physio_series(raw_root, bids_out, heur_module, physio_rows)

    # Always refresh participants.tsv from the accumulated summary
    write_participants(summary_path, bids_out)


# ────────────────── CLI interface ──────────────────
def main() -> None:
    """Command line interface for ``run-heudiconv``."""

    import argparse

    parser = argparse.ArgumentParser(description="Run HeuDiConv using one or more heuristics")
    parser.add_argument("dicom_root", help="Root directory containing DICOMs")
    parser.add_argument("heuristic", help="Heuristic file or directory with heuristic_*.py files")
    parser.add_argument("bids_out", help="Output BIDS directory")
    parser.add_argument("--subject-tsv", help="Path to subject_summary.tsv", default=None)
    parser.add_argument("--single-run", action="store_true", help="Use one heudiconv call for all subjects")
    args = parser.parse_args()

    mapping_df = None
    if args.subject_tsv:
        mapping_df = pd.read_csv(args.subject_tsv, sep="\t", keep_default_na=False)

    heur_path = Path(args.heuristic)
    heuristics = [heur_path] if heur_path.is_file() else sorted(heur_path.glob("heuristic_*.py"))
    for heur in heuristics:
        dataset = heur.stem.replace("heuristic_", "")
        out_dir = Path(args.bids_out) / dataset
        run_heudiconv(Path(args.dicom_root), heur, out_dir, per_folder=not args.single_run, mapping_df=mapping_df)


if __name__ == "__main__":
    main()

