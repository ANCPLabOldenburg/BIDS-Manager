"""Compare BIDS Manager's PET sidecars against pypet2bids', file by file.

The question a PET reviewer asks first is whether this tool loses anything
against the reference implementation. This answers it with numbers rather than
assertion: both tools are run over the same sources, and every field either one
produces is accounted for.

Run::

    python tools/compare_pet2bids.py <inventory.tsv> <our_bids_root> <raw_root>

Both sides are run through their DOCUMENTED interfaces, which for pypet2bids
means the ``dcm2niix4pet`` and ``ecatpet2bids`` console scripts rather than
importing their classes: their CLI is what they support, and calling it that way
is the only fair comparison.

The output is deliberately blunt about four things:

* fields only we produce
* fields only they produce, which is the list that matters to us
* fields where the two disagree, which is where one of us is wrong
* sources one tool converts and the other does not, at all

Nothing here is a test. It is a measurement, re-runnable when either side
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Keys dcm2niix or a converter writes about itself. Neither tool is judged on
# them, because they are not metadata about the scan.
NOT_METADATA = {
    "ConversionSoftware", "ConversionSoftwareVersion", "BidsGuess", "Filename",
    "ImageSize", "PixelDimensions",
}

# Words a converter writes when it does not know. Counting them as produced
# fields would flatter whichever tool emits its whole template, which is why
# both sides are filtered the same way.
PLACEHOLDERS = {"unknown", "n/a", "na", "none", "null", ""}


def answered(sidecar: dict) -> dict:
    """The fields of a sidecar that actually say something.

    Applied to BOTH tools. We omit a field we cannot answer; they write the key
    with an empty value. Comparing raw key counts would score that as a win for
    them, which is not a difference in what was read out of the scanner.
    """
    out = {}
    for name, value in sidecar.items():
        if name in NOT_METADATA:
            continue
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in PLACEHOLDERS:
            continue
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            continue
        out[name] = value
    return out


def schema_pet_fields() -> set[str]:
    """The names BIDS declares for a PET sidecar."""
    try:
        from bidsmgr import schema as schema_mod

        return {f.name for f in schema_mod.sidecar_fields("pet", "pet")}
    except Exception:  # noqa: BLE001
        return set()


def dcm2niix_path() -> str:
    """The same dcm2niix we pin, so neither side has a different converter."""
    try:
        import dcm2niix

        root = Path(dcm2niix.__file__).parent
        for found in root.rglob("dcm2niix*"):
            if found.is_file() and os.access(found, os.X_OK):
                return str(found)
    except Exception:  # noqa: BLE001
        pass
    return shutil.which("dcm2niix") or ""


def run_theirs_dicom(source_dir: Path, script: Path) -> tuple[dict, str]:
    """Their DICOM converter on one series folder. ``({}, reason)`` on failure."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "sub-001"
        proc = subprocess.run(
            [str(script), str(source_dir), "--destination-path", str(out)],
            capture_output=True, text=True, timeout=1800,
        )
        produced = sorted(Path(scratch).rglob("*.json"))
        if not produced:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1][:120] if tail else f"rc={proc.returncode}, no json"
            return {}, reason
        try:
            return json.loads(produced[0].read_text()), ""
        except (OSError, ValueError) as exc:
            return {}, f"unreadable output: {exc}"


def run_theirs_ecat(source_file: Path, script: Path) -> tuple[dict, str]:
    """Their ECAT converter on one file."""
    with tempfile.TemporaryDirectory() as scratch:
        nifti = Path(scratch) / "sub-001_pet.nii.gz"
        proc = subprocess.run(
            [str(script), str(source_file), "--nifti", str(nifti), "--convert"],
            capture_output=True, text=True, timeout=1800,
        )
        produced = sorted(Path(scratch).rglob("*.json"))
        if not produced:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1][:120] if tail else f"rc={proc.returncode}, no json"
            return {}, reason
        try:
            return json.loads(produced[0].read_text()), ""
        except (OSError, ValueError) as exc:
            return {}, f"unreadable output: {exc}"


def our_sidecar(bids_root: Path, basename: str) -> dict:
    """Our sidecar for one row, by the basename the conversion gave it."""
    for found in bids_root.rglob(f"{basename}.json"):
        try:
            return json.loads(found.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def compare(inventory: Path, bids_root: Path, raw_root: Path) -> int:
    import pandas as pd

    binaries = Path(sys.executable).parent
    dicom_script = binaries / "dcm2niix4pet"
    ecat_script = binaries / "ecatpet2bids"
    for script in (dicom_script, ecat_script):
        if not script.exists():
            print(f"missing {script}; is pypet2bids installed?")
            return 2

    env_note = dcm2niix_path()
    if env_note:
        os.environ["DCM2NIIX_PATH"] = env_note
    os.environ["PET2BIDS_TELEMETRY_ENABLED"] = "0"

    declared = schema_pet_fields()
    df = pd.read_csv(inventory, sep="\t", dtype=str).fillna("")
    rows = df[(df.get("proposed_datatype") == "pet") & (df["include"] != "0")]

    only_ours: Counter = Counter()
    only_theirs: Counter = Counter()
    disagree: Counter = Counter()
    our_wins: list[str] = []
    their_wins: list[str] = []
    both = ours_only_converted = theirs_only_converted = neither = 0

    print(f"comparing {len(rows)} PET series\n")
    print(f"{'source':46} {'ours':>6} {'theirs':>7}  note")
    print("-" * 92)

    for _, row in rows.iterrows():
        fmt = str(row.get("format", "")).upper()
        basename = str(row.get("proposed_basename", ""))
        mine = answered(our_sidecar(bids_root, basename))

        if fmt == "ECAT":
            source = raw_root / str(row.get("source_file", ""))
            theirs_raw, why = run_theirs_ecat(source, ecat_script)
            label = source.parent.name
        else:
            source = raw_root / str(row.get("source_folder", ""))
            theirs_raw, why = run_theirs_dicom(source, dicom_script)
            label = f"{source.parent.name}/{source.name}"
        theirs = answered(theirs_raw)

        if mine and theirs:
            both += 1
        elif mine:
            ours_only_converted += 1
            our_wins.append(f"{label}: they wrote nothing ({why})")
        elif theirs:
            theirs_only_converted += 1
            their_wins.append(f"{label}: we wrote nothing")
        else:
            neither += 1

        note = "" if (mine and theirs) else ("THEY FAILED" if mine else "WE FAILED")
        print(f"{label[:46]:46} {len(mine):>6} {len(theirs):>7}  {note}")

        # Only series BOTH converted are diffed. Counting fields from a series
        # the other tool crashed on would report the crash a second time, as
        # thirty field wins.
        if not (mine and theirs):
            continue
        for name in set(mine) - set(theirs):
            only_ours[name] += 1
        for name in set(theirs) - set(mine):
            only_theirs[name] += 1
        for name in set(mine) & set(theirs):
            if mine[name] != theirs[name]:
                disagree[name] += 1

    def show(title: str, counter: Counter, mark_schema: bool = True) -> None:
        print(f"\n{title} ({len(counter)} distinct)")
        if not counter:
            print("   none")
            return
        for name, n in counter.most_common():
            flag = ""
            if mark_schema:
                flag = "  <- BIDS field" if name in declared else "  (not a BIDS PET field)"
            print(f"   {name:38} in {n:2} series{flag}")

    print("\n" + "=" * 92)
    print(f"converted by both: {both}   only us: {ours_only_converted}   "
          f"only them: {theirs_only_converted}   neither: {neither}")
    for line in our_wins:
        print(f"   we converted, they did not -> {line}")
    for line in their_wins:
        print(f"   they converted, we did not -> {line}")

    print(f"\nfield comparison below covers the {both} series both tools converted")
    show("FIELDS ONLY WE PRODUCE", only_ours)
    show("FIELDS ONLY THEY PRODUCE", only_theirs)
    show("FIELDS WHERE WE DISAGREE", disagree)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inventory", type=Path)
    ap.add_argument("bids_root", type=Path)
    ap.add_argument("raw_root", type=Path)
    args = ap.parse_args(argv)
    return compare(args.inventory, args.bids_root, args.raw_root)


if __name__ == "__main__":
    raise SystemExit(main())
