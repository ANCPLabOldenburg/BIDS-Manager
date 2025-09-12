#!/usr/bin/env python3
"""
build_heuristic_from_tsv.py — **v10**
====================================
Simple heuristic that:
1. **Keeps every sequence**, including SBRef.
2. **Uses the raw SeriesDescription** (cleaned) as the filename stem – no
   added `rep-*`, task, or echo logic.
3. Skips only modalities listed in `SKIP_BY_DEFAULT` (`report`, `physio`).
"""

from __future__ import annotations
from pathlib import Path
from textwrap import dedent
from typing import Optional
import pandas as pd
import re

# Import the same preview logic the GUI uses so heuristic names match preview
try:
    from .renaming.schema_renamer import (
        load_bids_schema,
        SeriesInfo,
        build_preview_names,
    )
    from .renaming.config import DEFAULT_SCHEMA_DIR
except Exception:
    # Fallback for direct script execution
    from renaming.schema_renamer import (  # type: ignore
        load_bids_schema,
        SeriesInfo,
        build_preview_names,
    )
    from renaming.config import DEFAULT_SCHEMA_DIR  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SKIP_BY_DEFAULT = {"report", "physio"}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def clean(text: str) -> str:
    """Return alphanumerics only (for variable names)."""
    return re.sub(r"[^0-9A-Za-z]+", "", str(text))


def safe_stem(seq: str) -> str:
    """Clean SeriesDescription for use in a filename."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", seq.strip()).strip("_")


def dedup_parts(*parts: str) -> str:
    """Return underscore-joined *parts* with consecutive repeats removed."""
    tokens: list[str] = []
    for part in parts:
        for t in str(part).split("_"):
            if t and (not tokens or t != tokens[-1]):
                tokens.append(t)
    return "_".join(tokens)


def _normalize_subject(bids_name: str) -> str:
    """Return plain subject id without the optional "sub-" prefix.

    HeuDiConv expects subject ids without the leading entity tag when used
    inside folder names or file basenames. Users sometimes provide values like
    "sub-001" in the TSV. To avoid paths such as "sub-sub-001" we strip the
    prefix here and keep only the alphanumeric identifier.
    """
    s = str(bids_name or "").strip()
    if s.lower().startswith("sub-"):
        s = s[4:]
    # Keep alphanumerics only for safety inside filenames
    return clean(s)


def _normalize_session(session: str) -> str:
    """Return plain session id without the optional "ses-" prefix.

    Some inventories already contain values like "ses-pre". BIDS filenames must
    include the entity as "ses-<ID>" exactly once, so we remove any duplicate
    prefix and sanitize the identifier.
    """
    s = str(session or "").strip()
    if s.lower().startswith("ses-"):
        s = s[4:]
    return clean(s)


def _rewrite_fmap_suffix_if_needed(datatype: str, base: str, sequence: str) -> str:
    """Return a possibly adjusted basename for fieldmaps.

    HeuDiConv will use the suffix from ``base``. When the modality is fmap,
    prefer magnitude1/magnitude2/phasediff based on simple cues in the original
    sequence text. This mirrors the user's expectations from the preview and
    avoids ambiguous ``_fieldmap`` files.
    """
    if datatype != "fmap":
        return base
    stem, _, suffix = base.rpartition("_")
    sl = (sequence or "").lower()
    if any(tok in sl for tok in ("fieldmap1", "run-01_fmap", "echo-1")):
        return f"{stem}_magnitude1"
    if any(tok in sl for tok in ("fieldmap2", "run-02_fmap", "echo-2")):
        return f"{stem}_magnitude2"
    if any(tok in sl for tok in ("phase", "phasediff", "fmap")):
        return f"{stem}_phasediff"
    return base


def _sanitize_template_for_dcm2niix(template: str) -> str:
    """Return a filename template safe for the dcm2niix -f argument.

    dcm2niix's format string is executed via the shell on some platforms. The
    preview logic occasionally emits repetition markers like "(2)" to indicate
    duplicates. Parentheses in the template cause a shell parse error. We strip
    any trailing "(N)" group and any stray parentheses to keep names valid,
    without altering the BIDS semantics (suffix/entities remain intact).
    """
    # Drop a single trailing "(number)" at the end of the base
    out = re.sub(r"\(\d+\)$", "", template)
    # Remove any remaining literal parentheses to be safe
    out = out.replace("(", "").replace(")", "")
    return out


def _detect_dwi_derivative(sequence: str) -> Optional[str]:
    """Return standardized map name if sequence looks like a DWI derivative.
    
    Only these specific map types are considered derivatives according to the
    user's strict policy. Anything else is treated as raw DWI.
    """
    s = (sequence or "").lower()
    if "colfa" in s:
        return "ColFA"
    if "fa" in s and "colfa" not in s:
        return "FA"
    if "tensor" in s:
        return "TENSOR"
    if "adc" in s:
        return "ADC"
    if "trace" in s:
        return "TRACE"
    if "tracew" in s:
        return "TRACE"
    return None


def _strip_run_tokens(sequence: str) -> str:
    """Remove any run-N tokens from sequence name before processing.
    
    This ensures we don't have run- artifacts in the original sequence names
    that could cause confusion with our repetition handling.
    """
    # Remove run-N patterns (run-01, run-1, etc.)
    sequence = re.sub(r"_?run-\d+_?", "_", sequence, flags=re.IGNORECASE)
    sequence = re.sub(r"^run-\d+_", "", sequence, flags=re.IGNORECASE)
    sequence = re.sub(r"_run-\d+$", "", sequence, flags=re.IGNORECASE)
    # Clean up multiple underscores
    sequence = re.sub(r"_+", "_", sequence).strip("_")
    return sequence


def guess_task_from_sequence(sequence: str) -> str:
    """Extract task name from sequence description.
    
    This function mirrors the logic in schema_renamer.py to ensure
    consistent task detection between preview and heuristic generation.
    """
    if not sequence:
        return "unknown"
    
    # First strip any run tokens that might interfere
    seq_lower = sequence.lower()
    
    # Check for resting state patterns first
    resting_patterns = ["rs", "_rs", "rs_", "rest", "resting"]
    for pattern in resting_patterns:
        if pattern in seq_lower:
            return "rest"
    
    # Check for common task patterns
    task_hints = ["exec", "paradigma", "paradigm", "task", "activation", "movie", "nback", 
                  "flanker", "stroop", "motor", "checker", "checkerboard"]
    for hint in task_hints:
        if hint in seq_lower:
            return clean(hint)
    
    # If no specific task hint found, use the full sanitized sequence 
    # to ensure uniqueness (prevents different sequences from having same BIDS name)
    return clean(sequence) or "unknown"


def generate_bids_name(row, rep_num: int, rep_count: int, only_last_repeated: bool = False) -> str:
    """Generate a BIDS basename using the SAME logic as the preview/table.
    
    This ensures perfect alignment between what users see in the GUI and
    what gets generated during conversion. All naming logic is now unified
    through the schema_renamer module.
    """
    subj = _normalize_subject(row["BIDS_name"])  # fixes "sub-sub-001"
    session_raw = row.get("session", "")
    if pd.isna(session_raw):
        session_raw = ""
    ses = _normalize_session(session_raw)
    
    # Keep the original sequence so that run information can be extracted later
    # by ``propose_bids_basename``.  This mirrors the behaviour of the GUI.
    sequence = str(row.get("sequence", ""))
    modality = str(row.get("modality", ""))

    # Collect optional entities such as task hints.  We include ``task_hits`` so
    # the same automatic task detection used by the GUI applies when building a
    # heuristic directly from a TSV file.
    extra: dict[str, str] = {}
    for key in ("task", "task_hits", "acq", "run", "dir", "echo"):
        if row.get(key):
            extra[key] = str(row.get(key))

    # Create SeriesInfo exactly like the GUI does
    schema = load_bids_schema(DEFAULT_SCHEMA_DIR)
    series = SeriesInfo(
        subject=subj,
        session=ses or None,
        modality=modality,
        sequence=sequence,
        rep=(None if only_last_repeated else int(rep_num) if rep_count > 1 else None),
        extra=extra,
    )
    
    # Use the exact same preview logic to get the proposed name
    datatype, base = build_preview_names([series], schema)[0][1:]
    
    # The preview logic already handles:
    # - DWI derivative detection and desc-<MAP> naming
    # - Task detection with fallback to full sequence for uniqueness  
    # - Repetition handling with _rep-N
    # - Fieldmap suffix normalization
    # - Preservation of run numbers when present
    
    return base


# -----------------------------------------------------------------------------
# Core writer
# -----------------------------------------------------------------------------

def write_heuristic(df: pd.DataFrame, dst: Path, only_last_repeated: bool = False) -> None:
    """Write a HeuDiConv heuristic from ``df`` to ``dst``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table generated by :mod:`dicom_inventory` describing the DICOM series.
    dst : Path
        Destination ``heuristic_<name>.py`` file.
    only_last_repeated : bool
        If True, only keep the last repetition of repeated sequences.
    """

    print("Building heuristic (v10)…")
    buf: list[str] = []

    # 1 ─ header -----------------------------------------------------------
    buf.append(
        dedent(
            '''\
            """AUTO-GENERATED HeuDiConv heuristic (v10)."""
            from typing import Tuple

            def create_key(template: str,
                           outtype: Tuple[str, ...] = ("nii.gz",),
                           annotation_classes=None):
                if not template:
                    raise ValueError("Template must be non-empty")
                return template, outtype, annotation_classes
            '''
        )
    )

    # 2 ─ SID_MAP ----------------------------------------------------------
    sid_pairs = {(clean(str(r.source_folder)) or clean(Path(r.source_folder or '.').name), r.BIDS_name) for r in df.itertuples()}
    buf.append("\nSID_MAP = {\n")
    for folder, bids in sorted(sid_pairs):
        buf.append(f"    '{folder}': '{bids}',\n")
    buf.append("}\n\n")

    # 3 ─ template keys ----------------------------------------------------
    # Include series UID (or rep) in the key to handle repeated sequences
    seq2key: dict[tuple[str, str, str, str, str], str] = {}
    key_defs: list[tuple[str, str]] = []

    # Handle repetitions
    rep_counts = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False)["sequence"].transform("count")
    )
    rep_index = (
        df.groupby(["BIDS_name", "session", "sequence"], dropna=False).cumcount() + 1
    )
    
    # Filter for only last repeated if requested
    if only_last_repeated:
        # Create a mask to keep only the last occurrence of each repeated sequence
        df_working = df.copy()
        df_working['rep_counts'] = rep_counts
        df_working['rep_index'] = rep_index
        
        # Keep only the last repetition for sequences that repeat
        mask = (df_working['rep_counts'] == 1) | (df_working['rep_index'] == df_working['rep_counts'])
        df_working = df_working[mask].copy()
        
        # Recalculate indices for the filtered dataframe
        rep_counts = pd.Series(1, index=df_working.index)  # All are now unique
        rep_index = pd.Series(1, index=df_working.index)   # All are now first (and only)
        df = df_working
    

    key_def_set = set()
    for idx, row in df.iterrows():
        ses_raw = row.get("session", "")
        ses = _normalize_session("" if pd.isna(ses_raw) else str(ses_raw).strip())
        folder = Path(str(row.get("source_folder", "."))).name
        rep_num = rep_index.loc[idx]
        rep_count = rep_counts.loc[idx]
        uid_field = str(row.get("series_uid", ""))
        bids = row["BIDS_name"]
        
        # Determine container based on modality and derivatives
        container = row.get("modality_bids", "misc") or "misc"
        sequence = str(row.get("sequence", ""))
        
        # Check if this is a derivative using the same logic as preview
        is_derivative = _detect_dwi_derivative(sequence) is not None
        if is_derivative:
            container = "derivatives"
        
        # Generate proper BIDS name
        bids_filename = generate_bids_name(row, rep_num, rep_count, only_last_repeated)
        
        # Resolve a robust container path from coarse modality labels
        modb = (row.get("modality_bids", "") or "").lower()
        mod_coarse = (str(row.get("modality", "")) or "").lower()
        if modb in {"func", "dwi", "anat", "fmap"}:
            container_resolved = modb
        elif mod_coarse == "bold":
            container_resolved = "func"
        elif mod_coarse == "dwi":
            container_resolved = "dwi"
        else:
            container_resolved = "misc"

        # Build path based on whether it's a derivative
        if is_derivative:
            # Derivatives go to derivatives/pipeline/sub-/ses-/dwi/
            subj_norm = _normalize_subject(bids)
            path_parts = ["derivatives", "dcm2niix", f"sub-{subj_norm}"]
            if ses:
                path_parts.append(f"ses-{ses}")
            path_parts.append("dwi")
            path = "/".join(path_parts)
        else:
            # Regular BIDS structure
            subj_norm = _normalize_subject(bids)
            path_parts = [f"sub-{subj_norm}"]
            if ses:
                path_parts.append(f"ses-{ses}")
            path_parts.append(container_resolved)
            path = "/".join(path_parts)
        
        # Ensure template is safe for dcm2niix (no parentheses)
        template = _sanitize_template_for_dcm2niix(f"{path}/{bids_filename}")
        
        # Create unique key variable
        key_parts = [bids, ses, clean(sequence)]
        if rep_count > 1:
            key_parts.append(f"run{rep_num}")
        key_var = "key_" + clean("_".join(p for p in key_parts if p))
        
        # Ensure unique key names
        original_key_var = key_var
        counter = 1
        while key_var in key_def_set:
            key_var = f"{original_key_var}_{counter}"
            counter += 1
        
        key_defs.append((key_var, template))
        key_def_set.add(key_var)

        uid_list = [u for u in uid_field.split("|") if u] or [""]
        for uid in uid_list:
            key_id = (row["sequence"], row["BIDS_name"], ses, folder, uid)
            if key_id in seq2key:
                continue
            seq2key[key_id] = key_var

    for var, tpl in key_defs:
        buf.append(f"{var} = create_key('{tpl}')\n")
    buf.append("\n")

    # 4 ─ infotodict() ----------------------------------------------------
    buf.append("def infotodict(seqinfo):\n    \"\"\"Return mapping SeriesDescription → key list.\"\"\"\n")
    for var in seq2key.values():
        buf.append(f"    {var}_list = []\n")
    buf.append("    info = {\n")
    for var in seq2key.values():
        buf.append(f"        {var}: {var}_list,\n")
    buf.append("    }\n\n")

    buf.append("    for s in seqinfo:\n")
    for (seq, _b, _s, folder, uid), var in seq2key.items():
        seq_esc = seq.replace("'", "\\'")
        fol_esc = folder.replace("'", "\\'")
        uid_esc = str(uid).replace("'", "\\'")
        buf.append(
            f"        if s.series_description == '{seq_esc}' and s.dcm_dir_name == '{fol_esc}' and getattr(s, 'series_uid', '') == '{uid_esc}':\n"
        )
        buf.append(f"            {var}_list.append(s.series_id)\n")
    buf.append("    return info\n")

    dst.write_text("".join(buf), encoding="utf-8")
    print("Heuristic written →", dst.resolve())


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def generate(tsv: Path, out_dir: Path, only_last_repeated: bool = False) -> None:
    """Generate heuristic files for each study described in ``tsv``.

    Parameters
    ----------
    tsv : Path
        Path to ``subject_summary.tsv`` produced by :mod:`dicom_inventory`.
    out_dir : Path
        Directory where the heuristic files will be written.
    only_last_repeated : bool
        If True, only keep the last repetition of repeated sequences.
    """

    df = pd.read_csv(tsv, sep="\t", keep_default_na=False)

    # Drop rows with unwanted modalities
    mask = df.modality.isin(SKIP_BY_DEFAULT)
    if mask.any():
        df.loc[mask, "include"] = 0
        print(f"Auto‑skipped {mask.sum()} rows ({', '.join(SKIP_BY_DEFAULT)})")

    df = df[df.include == 1]

    out_dir.mkdir(parents=True, exist_ok=True)

    for study, sub_df in df.groupby("StudyDescription"):
        fname = safe_stem(study or "unknown")
        heur = out_dir / f"heuristic_{fname}.py"
        write_heuristic(sub_df, heur, only_last_repeated)
        folders = " ".join(sorted({clean(f) or clean(Path(f or '.').name) for f in sub_df.source_folder.unique()}))
        print(dedent(f"""
        heudiconv -d "<RAW_ROOT>/{{subject}}/**/*.*" -s {folders} -f {heur.name} -c dcm2niix -o <BIDS_OUT>/{fname} -b --minmeta --overwrite"""))


def main() -> None:
    """Entry point for the ``build-heuristic`` command line utility."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate HeuDiConv heuristic(s) from TSV")
    parser.add_argument("tsv", help="Path to subject_summary.tsv file")
    parser.add_argument("out_dir", help="Directory to write heuristic files")
    parser.add_argument("--only-last-repeated", action="store_true", 
                        help="Only keep the last repetition of repeated sequences")
    args = parser.parse_args()

    generate(Path(args.tsv), Path(args.out_dir), args.only_last_repeated)


if __name__ == "__main__":
    main()

