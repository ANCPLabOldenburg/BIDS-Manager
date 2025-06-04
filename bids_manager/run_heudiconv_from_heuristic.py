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
from typing import Dict, List


# ────────────────── helpers ──────────────────
def load_sid_map(heur: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("heuristic", heur)
    module = importlib.util.module_from_spec(spec)         # type: ignore
    assert spec.loader
    spec.loader.exec_module(module)                        # type: ignore
    return module.SID_MAP                                  # type: ignore


def clean_name(raw: str) -> str:
    return "".join(ch for ch in raw if ch.isalnum())


def physical_by_clean(raw_root: Path) -> Dict[str, str]:
    """
    Return mapping folder_name → folder_name  (no cleaning),
    only first-level dirs under *raw_root*.
    """
    return {p.name: p.name for p in raw_root.iterdir() if p.is_dir()}



def detect_depth(folder: Path) -> int:
    """Minimum depth (#subdirs) from *folder* to any .dcm file."""
    for root, _dirs, files in os.walk(folder):
        if any(f.lower().endswith(".dcm") for f in files):
            rel = Path(root).relative_to(folder)
            return len(rel.parts)
    raise RuntimeError(f"No DICOMs under {folder}")


def heudi_cmd(raw_root: Path,
              phys_folders: List[str],
              heuristic: Path,
              bids_out: Path,
              depth: int) -> List[str]:
    wild = "*/" * depth
    template = f"{raw_root}/" + "{subject}/" + wild + "*.dcm"
    return [
        "heudiconv",
        "-d", template,
        "-s", *phys_folders,
        "-f", str(heuristic),
        "-c", "dcm2niix",
        "-o", str(bids_out),
        "-b", "--minmeta", "--overwrite",
    ]


# ────────────────── main runner ──────────────────
def run_heudiconv(raw_root: Path,
                  heuristic: Path,
                  bids_out: Path,
                  per_folder: bool = True) -> None:

    sid_map          = load_sid_map(heuristic)          # cleaned → sub-XXX
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


# ────────────────── CLI interface ──────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run HeuDiConv using a heuristic")
    parser.add_argument("dicom_root", help="Root directory containing DICOMs")
    parser.add_argument("heuristic", help="Heuristic .py file")
    parser.add_argument("bids_out", help="Output BIDS directory")
    parser.add_argument("--single-run", action="store_true", help="Use one heudiconv call for all subjects")
    args = parser.parse_args()

    run_heudiconv(Path(args.dicom_root), Path(args.heuristic), Path(args.bids_out), per_folder=not args.single_run)


if __name__ == "__main__":
    main()

