"""Fetch the published sample datasets, and cache them outside the checkout.

The tests that matter most need REAL inputs. Synthetic EEG and MEG are honest
(mne writes the same EDF and FIF an amplifier does) but there is no honest
synthetic DICOM: a fabricated series is a test of the fabrication. And DICOM
is where this tool does its hardest work, so a suite without it is missing the
part most likely to break.

The documentation already publishes exactly what is needed. The multimodal
sample is **56 MB** and holds, for one participant:

* ``mri/``  35 Siemens DICOM: T1w, a BOLD run, two fieldmap series, a
  localizer, and a PhoenixZIPReport (the non-image series the scanner is
  supposed to detect and exclude, which is a regression target in itself);
* ``pet/``  35 DICOM, one FDG scan on a GE Advance, plus ``dose_sheet.csv``
  for the PET spreadsheet reader;
* ``eeg/``  two 64-channel EDF recordings;
* ``meg/``  one 325-channel Elekta FIF.

One folder, four modalities, one session. That is the shape the recently
reported ``*_scans.tsv`` defect needed, and no synthetic fixture produced it.

Caching, and why it lives outside the repository
------------------------------------------------
``actions/checkout`` runs ``git clean -ffdx`` by default, which deletes
anything untracked in the workspace INCLUDING ignored files. A download cached
inside the checkout would therefore be re-fetched on every CI run. The cache
goes in a user directory instead, keyed by dataset and version, so a runner
downloads each dataset once and never again until the version changes.

Offline is not a failure. A machine with no network and no cache skips the
tests that need data rather than failing them, because "this machine cannot
reach the internet" is not a defect in the code under test.
"""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest


@dataclass(frozen=True)
class Sample:
    """One published dataset: where to get it and what it should contain."""

    name: str
    token: str          # the cloud.uol.de share token
    megabytes: int      # measured, so a surprise download is a visible one
    root: str           # the top-level folder inside the archive
    summary: str

    @property
    def url(self) -> str:
        return f"https://cloud.uol.de/s/{self.token}/download"


# Measured 2026-09-14 with a HEAD request. The two large ones are listed so
# nobody has to guess why they are not used by default.
SAMPLES: dict[str, Sample] = {
    "multimodal": Sample(
        "multimodal", "o6XCk6zH9DYpoes", 56,
        "BIDS_Manager_multimodal_sample",
        "one participant, MRI + PET DICOM, EEG EDF, MEG FIF. The CI default.",
    ),
    "pet": Sample(
        "pet", "CGcjfTpxzFWnrdz", 8, "",
        "PET DICOM and ECAT.",
    ),
    "eeg": Sample(
        "eeg", "T66zc5mN4eeZPGK", 34, "",
        "the EEG tutorial dataset.",
    ),
    "mri": Sample(
        "mri", "g9gMPpwL7Xg49y9", 330, "",
        "the MRI tutorial dataset. Too big for a routine run.",
    ),
    "mri_advanced": Sample(
        "mri_advanced", "ZxaZCtHJPLjtDbR", 4483, "",
        "4.5 GB. Manual use only.",
    ),
    "meg": Sample(
        "meg", "btGeke5NNkDcs6G", 8303, "",
        "8.3 GB. Manual use only.",
    ),
}

# Bump when a published archive is replaced, so every cache re-fetches.
VERSION = "v1"

# Anything this size or smaller is fetched without asking. The others need
# BIDSMGR_ALLOW_BIG_DOWNLOADS=1, so a stray test cannot pull 8 GB onto a
# laptop over a conference wifi.
AUTO_FETCH_MB = 100


def cache_root() -> Path:
    """Where downloads live. Outside the checkout, on purpose (see above)."""
    override = os.environ.get("BIDSMGR_TEST_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "bidsmgr-test-data"


def _ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle when it is there, the system default otherwise.

    The same helper the vendored bidsval carries, for the same reason: a
    python.org interpreter on macOS is not wired into the system trust store,
    so a plain urlopen fails with "unable to get local issuer certificate"
    while curl on the same machine succeeds. Measured here on 2026-09-14.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:            # pragma: no cover - certifi normally present
        return ssl.create_default_context()


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    # Downloaded to a .part and renamed, so an interrupted fetch cannot leave
    # a truncated archive that looks cached.
    with urllib.request.urlopen(
        url, timeout=120, context=_ssl_context(),
    ) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(target)


def fetch(name: str, *, allow_big: Optional[bool] = None) -> Path:
    """The extracted dataset directory, downloading it once if needed.

    Raises ``LookupError`` when the data is neither cached nor reachable, so
    the caller decides whether that is a skip or a failure.
    """
    try:
        sample = SAMPLES[name]
    except KeyError:
        raise LookupError(f"no sample called {name!r}") from None

    where = cache_root() / VERSION / sample.name
    marker = where / ".complete"
    if marker.is_file():
        return where

    if allow_big is None:
        allow_big = os.environ.get("BIDSMGR_ALLOW_BIG_DOWNLOADS") == "1"
    if sample.megabytes > AUTO_FETCH_MB and not allow_big:
        raise LookupError(
            f"{sample.name} is {sample.megabytes} MB; set "
            "BIDSMGR_ALLOW_BIG_DOWNLOADS=1 to fetch it"
        )

    archive = cache_root() / VERSION / f"{sample.name}.zip"
    try:
        if not archive.is_file():
            _download(sample.url, archive)
        where.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(where)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        shutil.rmtree(where, ignore_errors=True)
        raise LookupError(f"could not fetch {sample.name}: {exc}") from exc

    marker.write_text(f"{sample.url}\n{VERSION}\n", encoding="utf-8")
    # The archive is the big thing and the extraction is what tests read, so
    # the zip goes once the extraction succeeded.
    archive.unlink(missing_ok=True)
    return where


def raw_root(name: str = "multimodal") -> Path:
    """The ``raw/`` folder a scan should be pointed at."""
    top = fetch(name)
    inner = top / SAMPLES[name].root if SAMPLES[name].root else top
    candidate = inner / "raw"
    if candidate.is_dir():
        return candidate
    # A sample whose layout differs: hand back whatever single directory the
    # archive contains rather than guessing further.
    subdirs = [p for p in inner.iterdir() if p.is_dir()] if inner.is_dir() else []
    return subdirs[0] if len(subdirs) == 1 else inner


# ---------------------------------------------------------------------------
# PET, which ships every one of its source formats in one published sample
# ---------------------------------------------------------------------------
#
# Thirteen PET tests used to gate on ``BIDS_MANAGER_REAL_PET_DATA=1`` plus a
# local copy of the OpenNeuroPET phantom set, which meant they ran on one
# laptop and skipped on every runner: the coverage existed on paper and nowhere
# else. The published sample carries all three sources the tool has to read,
# so the honest gate is "could this machine get the sample", which any machine
# with a network can.
#
# Each accessor returns ``None`` rather than raising when the piece is not
# there, so a module-level constant can be defined on a machine with no data
# and the skip is decided by the gate rather than by an import error.


def _pet_tree() -> Optional[Path]:
    try:
        top = fetch("pet")
    except LookupError:
        return None
    inner = top / SAMPLES["pet"].root if SAMPLES["pet"].root else top
    return inner if inner.is_dir() else top


def pet_dicom_dir() -> Optional[Path]:
    """A folder of real PET DICOM, as a scanner wrote it."""
    tree = _pet_tree()
    if tree is None:
        return None
    hits = sorted(p for p in tree.rglob("*.dcm"))
    return hits[0].parent if hits else None


def pet_ecat_file() -> Optional[Path]:
    """The ECAT ``.v``, PET's non-DICOM native format.

    Found by extension here because the sample's layout is known. The
    SCANNER-side detection deliberately does not: ``inventory/pet_ecat`` keys
    on the ``MATRIX7x`` signature, since ``.v`` is not a reserved extension.
    """
    tree = _pet_tree()
    if tree is None:
        return None
    hits = sorted(tree.rglob("*.v"))
    return hits[0] if hits else None


def pet_blood_dir() -> Optional[Path]:
    """The folder holding the PMOD ``.bld`` blood curves."""
    tree = _pet_tree()
    if tree is None:
        return None
    hits = sorted(tree.rglob("*.bld"))
    return hits[0].parent if hits else None


def pet_dose_sheet(ecat: bool = False) -> Optional[Path]:
    """The lab's own dose table, which the scanner does not record."""
    tree = _pet_tree()
    if tree is None:
        return None
    hits = sorted(tree.rglob("dose_sheet*.csv"))
    for path in hits:
        if ("ecat" in path.name.lower()) == ecat:
            return path
    return hits[0] if hits else None


def require(name: str = "multimodal") -> Path:
    """``raw_root``, but skipping the test when the data cannot be had.

    Offline with an empty cache is a fact about the machine, not a defect in
    the code, so it skips with the reason rather than failing.
    """
    try:
        return raw_root(name)
    except LookupError as exc:
        pytest.skip(str(exc))


__all__ = [
    "AUTO_FETCH_MB",
    "pet_blood_dir",
    "pet_dicom_dir",
    "pet_dose_sheet",
    "pet_ecat_file",
    "SAMPLES",
    "VERSION",
    "Sample",
    "cache_root",
    "fetch",
    "raw_root",
    "require",
]
