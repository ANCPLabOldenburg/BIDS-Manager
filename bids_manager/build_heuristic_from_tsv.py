#!/usr/bin/env python3
"""
build_heuristic_from_tsv.py — **v10**
====================================
Simple heuristic that:
1. **Keeps every sequence**, including SBRef.
2. **Uses the raw SeriesDescription** (cleaned) as the filename stem – no
   added `run-*`, task, or echo logic.
3. Skips only modalities listed in `SKIP_BY_DEFAULT` (`report`,
   `physio`, `refscan`).
"""

from __future__ import annotations
from pathlib import Path
from textwrap import dedent
from typing import Tuple
import pandas as pd
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SKIP_BY_DEFAULT = {"report", "physio", "refscan"}

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


# -----------------------------------------------------------------------------
# Core writer
# -----------------------------------------------------------------------------

def write_heuristic(df: pd.DataFrame, dst: Path) -> None:
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
    sid_pairs = {(clean(str(r.source_folder)), r.BIDS_name) for r in df.itertuples()}
    buf.append("\nSID_MAP = {\n")
    for folder, bids in sorted(sid_pairs):
        buf.append(f"    '{folder}': '{bids}',\n")
    buf.append("}\n\n")

    # 3 ─ template keys ----------------------------------------------------
    seq2key: dict[tuple[str, str, str, str], str] = {}
    key_defs: list[tuple[str, str]] = []

    for row in df.itertuples():
        ses = str(row.session).strip() if pd.notna(row.session) and str(row.session).strip() else ""
        folder = Path(str(row.source_folder)).name
        key_id = (row.sequence, row.BIDS_name, ses, folder)
        if key_id in seq2key:
            continue

        bids = row.BIDS_name
        container = row.modality_bids or "misc"
        stem = safe_stem(row.sequence)

        base = dedup_parts(bids, ses, stem)
        path = "/".join(p for p in [bids, ses, container] if p)
        template = f"{path}/{base}"

        parts = [bids, ses, stem]
        key_var = "key_" + clean("_".join(p for p in parts if p))
        seq2key[key_id] = key_var
        key_defs.append((key_var, template))

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
    for (seq, _b, _s, folder), var in seq2key.items():
        seq_esc = seq.replace("'", "\\'")
        fol_esc = folder.replace("'", "\\'")
        buf.append(f"        if s.series_description == '{seq_esc}' and s.dcm_dir_name == '{fol_esc}':\n")
        buf.append(f"            {var}_list.append(s.series_id)\n")
    buf.append("    return info\n")

    dst.write_text("".join(buf), encoding="utf-8")
    print("Heuristic written →", dst.resolve())


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def generate(tsv: Path, out_dir: Path) -> None:
    df = pd.read_csv(tsv, sep="\t")

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
        write_heuristic(sub_df, heur)
        folders = " ".join(sorted({clean(f) for f in sub_df.source_folder.unique()}))
        print(dedent(f"""
        heudiconv -d "<RAW_ROOT>/{{subject}}/**/*.dcm" -s {folders} -f {heur.name} -c dcm2niix -o <BIDS_OUT>/{fname} -b --minmeta --overwrite"""))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate HeuDiConv heuristic(s) from TSV")
    parser.add_argument("tsv", help="Path to subject_summary.tsv file")
    parser.add_argument("out_dir", help="Directory to write heuristic files")
    args = parser.parse_args()

    generate(Path(args.tsv), Path(args.out_dir))


if __name__ == "__main__":
    main()

