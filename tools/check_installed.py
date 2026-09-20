#!/usr/bin/env python3
"""Exercise an INSTALLED bids-manager, the way a user gets it.

Run with the interpreter of a clean environment that has the wheel installed,
from a directory that is NOT the source checkout::

    python tools/check_installed.py

Why this exists, and why it is not a pytest file
------------------------------------------------
``pyproject.toml`` sets ``pythonpath = ["."]``, so every pytest run puts the
SOURCE TREE first on ``sys.path``. That is right for developing and it means
the test suite can never see a packaging defect: the tests import
``./bidsmgr`` whatever was installed. Measured, not assumed, on 2026-09-14.

That blind spot is not theoretical. The vendored ``bidsval`` shipped with no
schema files for weeks. The source tree was complete, every test passed, and
the wheel was broken, because package data is opt-in and nothing ever imported
the installed copy.

So this runs outside pytest, refuses to proceed if it can see the source tree,
and goes all the way from "the wheel is installed" to "a dataset converted".

What it checks
--------------
1. the package imports, and imports from site-packages rather than a checkout;
2. the data files the wheel must carry are present AND readable through the
   installed package, not just present in the zip;
3. every console script the project declares exists and runs;
4. a synthetic EEG + MEG dataset converts end to end: scan, convert,
   metadata, validate, with zero errors (MRI needs DICOM, so it is not
   synthesised here; see the note on ``_raw_tree``);
5. the GUI imports and a main window constructs offscreen.

Exit code 0 when all of it holds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Every subprocess started here gets UTF-8 output.
#
# Python picks its stdout encoding from the console, and a Windows console
# through a PIPE reports cp1252. `bidsmgr-scan --help` prints an arrow, which
# cp1252 cannot encode, so argparse raised UnicodeEncodeError and the command
# exited 1 under CI while working perfectly in a terminal, because Windows
# Terminal is UTF-8. The difference was the pipe, not the tool.
#
# A test that runs a CLI owns the environment it runs it in, so it says what
# encoding it wants rather than inheriting whatever the harness had.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

PASS, FAIL = [], []


def check(label: str):
    """Decorator running one check and recording the outcome."""
    def wrap(fn):
        try:
            detail = fn()
        except Exception as exc:                       # noqa: BLE001
            FAIL.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {label}\n        {type(exc).__name__}: {exc}")
        else:
            PASS.append(label)
            print(f"  ok    {label}" + (f"  ({detail})" if detail else ""))
        return fn
    return wrap


# ---------------------------------------------------------------------------
# 1. It is the installed package, not a checkout
# ---------------------------------------------------------------------------

print("\n== the installed package ==")


@check("imports, and from site-packages")
def _imported() -> str:
    import bidsmgr

    where = Path(bidsmgr.__file__).resolve()
    if "site-packages" not in where.parts:
        raise AssertionError(
            f"imported from {where}, which is not an installed location. "
            "Run this from OUTSIDE the source checkout."
        )
    return f"{bidsmgr.__version__} at .../{where.parent.name}"


# ---------------------------------------------------------------------------
# 2. The data the wheel had to carry, reachable through the package
# ---------------------------------------------------------------------------

print("\n== package data ==")


@check("the vendored bidsval schemas resolve")
def _schemas() -> str:
    from bidsmgr.vendor.bidsval import schema as bv

    versions = bv.available_versions()
    if len(versions) < 5:
        raise AssertionError(f"only {len(versions)} bundled schema(s)")
    for version in versions:
        if bv.resolve(version) is None:
            raise AssertionError(f"{version} did not resolve")
    return f"{len(versions)} versions"


@check("the Qt stylesheet is readable")
def _qss() -> str:
    import bidsmgr.gui as gui

    qss = Path(gui.__file__).parent / "theme.qss"
    size = qss.read_text(encoding="utf-8")
    return f"{len(size):,} chars"


@check("the schema adapter answers")
def _adapter() -> str:
    from bidsmgr.schema import bids_version, sidecar_fields

    n = len(sidecar_fields("anat", "T1w"))
    if n < 50:
        raise AssertionError(f"only {n} fields for anat/T1w")
    return f"anat/T1w has {n} fields, BIDS {bids_version()}"


# ---------------------------------------------------------------------------
# 3. The console scripts a user actually types
# ---------------------------------------------------------------------------

print("\n== console scripts ==")

SCRIPTS = [
    "bidsmgr-create", "bidsmgr-scan", "bidsmgr-convert",
    "bidsmgr-rebuild", "bidsmgr-metadata", "bidsmgr-validate",
    "bidsmgr-project", "bidsmgr-adopt",
]

for _name in SCRIPTS:
    @check(f"{_name} --help")
    def _script(name=_name) -> str:
        exe = shutil.which(name)
        if exe is None:
            raise AssertionError("not on PATH")
        proc = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace", env=CHILD_ENV,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"exit {proc.returncode}: {(proc.stderr or '')[:160]}"
            )
        return ""


# ---------------------------------------------------------------------------
# 4. A dataset, converted end to end
# ---------------------------------------------------------------------------

print("\n== a dataset, end to end ==")

WORK = Path(tempfile.mkdtemp(prefix="bidsmgr-installed-"))


def _raw_tree() -> Path:
    """A tiny source tree the scanner can actually read: EEG and MEG.

    NOT MRI, deliberately. The scanner's MRI input is DICOM, so a bare NIfTI
    dropped in here is simply not inventoried. A first version of this check
    wrote one and called the tree multimodal; the run reported two inventory
    rows for four files, which is how the mistake surfaced.

    Synthesising DICOM is possible but it would be a fabricated series rather
    than anything a scanner produced, so MRI conversion stays a real-data
    concern. EEG and MEG are genuine here: mne writes the same EDF and FIF a
    real amplifier does, and they exercise the whole mne-bids path.
    """
    import mne
    import numpy as np

    raw = WORK / "raw"
    for subject in ("sub-01", "sub-02"):
        folder = raw / subject
        folder.mkdir(parents=True, exist_ok=True)

        eeg_info = mne.create_info(["Cz", "Pz", "Oz"], 100.0, "eeg")
        mne.io.RawArray(
            np.zeros((3, 200)), eeg_info, verbose="error",
        ).export(
            str(folder / f"{subject}_task-rest_eeg.edf"),
            fmt="edf", verbose="error", overwrite=True,
        )

        meg_info = mne.create_info(
            ["MEG0111", "MEG0112"], 100.0, "mag",
        )
        mne.io.RawArray(
            np.zeros((2, 200)), meg_info, verbose="error",
        ).save(
            folder / f"{subject}_task-rest_meg_raw.fif",
            verbose="error", overwrite=True,
        )
    return raw


@check("a synthetic source tree builds")
def _built() -> str:
    raw = _raw_tree()
    files = [p for p in raw.rglob("*") if p.is_file()]
    return f"{len(files)} files, {sum(p.stat().st_size for p in files):,} bytes"


@check("bidsmgr-scan produces an inventory")
def _scan() -> str:
    out = WORK / "inv.tsv"
    proc = subprocess.run(
        [shutil.which("bidsmgr-scan"), str(WORK / "raw"), str(out)],
        capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace", env=CHILD_ENV,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
        )
    if not out.is_file():
        raise AssertionError("no inventory written")
    rows = out.read_text(encoding="utf-8").splitlines()
    return f"{len(rows) - 1} row(s)"


@check("bidsmgr-convert writes a BIDS tree")
def _convert() -> str:
    proc = subprocess.run(
        [shutil.which("bidsmgr-convert"), str(WORK / "inv.tsv"),
         str(WORK / "bids"), "--raw-root", str(WORK / "raw")],
        capture_output=True, text=True, timeout=1800,
        encoding="utf-8", errors="replace", env=CHILD_ENV,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
        )
    subjects = sorted((WORK / "bids").rglob("sub-*"))
    if not subjects:
        raise AssertionError("no subject folders produced")
    return f"{len({p.name for p in subjects if p.is_dir()})} subject dir(s)"


def _bids_root() -> Path:
    """The dataset root convert produced, which may sit one level down."""
    root = WORK / "bids"
    if (root / "dataset_description.json").is_file():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "dataset_description.json").is_file():
            return child
    raise AssertionError("no dataset_description.json anywhere under bids/")


@check("bidsmgr-metadata completes")
def _metadata() -> str:
    root = _bids_root()
    proc = subprocess.run(
        [shutil.which("bidsmgr-metadata"), str(root)],
        capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace", env=CHILD_ENV,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
        )
    return "participants.tsv" if (root / "participants.tsv").is_file() else ""


@check("every recording is listed in a scans table")
def _scans() -> str:
    root = _bids_root()
    tables = sorted(root.rglob("*_scans.tsv"))
    if not tables:
        raise AssertionError("no *_scans.tsv written")
    listed = 0
    for table in tables:
        listed += len(table.read_text(encoding="utf-8").splitlines()) - 1
    recordings = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in (".edf", ".fif", ".gz")
        and p.parent.name in ("anat", "eeg", "func", "meg")
    ]
    if listed < len(recordings):
        raise AssertionError(
            f"{listed} row(s) for {len(recordings)} recording(s)"
        )
    return f"{listed} row(s) across {len(tables)} table(s)"


@check("bidsmgr-validate reports no errors")
def _validate() -> str:
    from bidsmgr.editor.validator import validate

    report = validate(_bids_root())
    errors = report.counts.get("err", 0)
    if errors:
        raise AssertionError(f"{errors} error(s): {report.counts}")
    return f"{report.counts}"


# ---------------------------------------------------------------------------
# 5. The GUI constructs
# ---------------------------------------------------------------------------

print("\n== the GUI ==")


@check("a main window constructs offscreen")
def _gui() -> str:
    """Built the way ``bidsmgr.main`` builds it: a ThemeManager, then the
    window. Constructing it any other way would test a path no user takes."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    app = QApplication.instance() or QApplication([])
    theme = ThemeManager(app)
    theme.apply("dark")
    window = MainWindow(theme)
    window.resize(800, 600)
    # The two views are the whole application; a window that builds without
    # them has not really built.
    panels = [type(w).__name__ for w in window.findChildren(object)]
    for needed in ("ConverterPanel", "EditorPanel"):
        if needed not in panels:
            raise AssertionError(f"no {needed} in the window")
    window.close()
    return "converter and editor present"


# ---------------------------------------------------------------------------

shutil.rmtree(WORK, ignore_errors=True)

print(f"\n{'=' * 62}")
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
for label, why in FAIL:
    print(f"    FAILED  {label}: {why}")
print("=" * 62)
sys.exit(1 if FAIL else 0)
