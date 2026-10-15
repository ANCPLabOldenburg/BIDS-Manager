"""Which files are recordings, and which viewer a file opens in.

Routing a click must not read the file: a 500 MB BOLD and a 32 KB spectrum
look the same until you do. So the kind of a file is decided from its name,
its folder and (for a table) its sidecar, here, once, for every host. The
one exception is DICOM, which vendors write with any name or none: its
132-byte preamble is read, never more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..bids import full_ext

#: Recording files the signal viewer opens.
#:
#: BrainVision sidecars (``.eeg`` / ``.vmrk``) are deliberately absent: the
#: user opens the ``.vhdr``. ``mne.io.read_raw`` dispatches every entry,
#: with the vendor-specific fallbacks in :func:`signal.read_recording`.
RECORDING_FILE_EXTS: frozenset[str] = frozenset({
    ".fif", ".fif.gz", ".con", ".sqd", ".kdf",
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".egi", ".mef", ".nwb",
})
#: Folder-shaped recordings (CTF, EGI).
RECORDING_DIR_EXTS: frozenset[str] = frozenset({".ds", ".mff"})

#: Volume files the volume viewer opens.
VOLUME_EXTS: frozenset[str] = frozenset({".nii", ".nii.gz", ".mgz", ".mgh"})

Kind = Literal["volume", "signal", "spectrum", "physio", ""]


def is_recording_path(path) -> bool:
    """True when ``path`` is an EEG/MEG/iEEG recording the viewer can open."""
    p = Path(path)
    ext = full_ext(p)
    if p.is_dir():
        return ext in RECORDING_DIR_EXTS
    return ext in RECORDING_FILE_EXTS


def is_mrs_path(path) -> bool:
    """True when ``path`` should open in the spectrum viewer.

    Decided by the DATATYPE folder and the schema's own suffix list, never by
    opening the file.
    """
    p = Path(path)
    if full_ext(p) not in (".nii", ".nii.gz"):
        return False
    if p.parent.name != "mrs":
        return False
    try:
        from ... import schema

        suffixes = set(schema.list_suffixes("mrs"))
    except Exception:  # noqa: BLE001
        suffixes = {"svs", "mrsi", "unloc", "mrsref"}
    stem = p.name.split(".")[0]
    return stem.rsplit("_", 1)[-1] in suffixes


def kind_of(path, *, sidecar_checked: Optional[bool] = None) -> Kind:
    """Which viewer ``path`` opens in ("" for none).

    A ``.tsv``/``.tsv.gz`` is a ``physio`` recording only when its sidecar
    declares a sampling frequency; pass ``sidecar_checked`` when the caller
    already knows, or it is read here (a few hundred bytes).
    """
    p = Path(path)
    if is_mrs_path(p):
        return "spectrum"
    if full_ext(p) in VOLUME_EXTS:
        return "volume"
    if is_recording_path(p):
        return "signal"
    if full_ext(p) in (".tsv", ".tsv.gz"):
        if sidecar_checked is None:
            from .physio import read_timing

            sidecar_checked = read_timing(p) is not None
        return "physio" if sidecar_checked else ""
    return ""


#: What the Editor shows a file that is not an image, a recording or a
#: table in (``document_kind``).
Document = Literal["gradients", "picture", "markdown", "html", "text", "dicom", ""]

#: Pictures a dataset carries: electrode and anatomical photos (``_photo``),
#: figures under ``derivatives`` and ``code``.
PICTURE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".webp"})
#: Text the standard names without an extension (README, CHANGES, LICENSE)
#: and the usual others.
TEXT_NAMES: frozenset[str] = frozenset({
    "readme", "changes", "license", "authors", "copying", "notice", ".gitignore",
    ".gitattributes", ".datalad"})
TEXT_EXTS: frozenset[str] = frozenset({
    ".txt", ".rst", ".csv", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".log", ".py",
    ".sh", ".bash", ".m", ".r", ".jl", ".js", ".bat", ".ps1", ".vmrk", ".xml", ".tex",
    ".bib", ".sql", ".cff"})
MARKDOWN_EXTS: frozenset[str] = frozenset({".md", ".markdown"})


def document_kind(path) -> Document:
    """How a file that is not a volume, a recording or a table is shown
    ("" for none: the Editor shows what it is and offers the system's app).

    ``.bval``/``.bvec`` are the gradient table; README, README.md and other
    Markdown render; CHANGES, LICENSE and the like are text; a DICOM file
    (by its content) opens in the DICOM header pane."""
    p = Path(path)
    name = p.name.lower()
    ext = full_ext(p)
    if ext in (".bval", ".bvec"):
        return "gradients"
    if ext in PICTURE_EXTS:
        return "picture"
    if ext in MARKDOWN_EXTS:
        return "markdown"
    if ext in (".html", ".htm"):
        return "html"
    if name in TEXT_NAMES or ext in TEXT_EXTS or name.startswith(("readme.", "changes.")):
        return "text"
    # A DICOM is recognised by its content: Siemens ``.IMA``, Philips files
    # named by their UID and extensionless files are all DICOM. Reading the
    # 132-byte preamble on a click is cheap; pixels are never read here.
    from ...inventory.mri_dicom import is_dicom_file

    if p.is_file() and is_dicom_file(str(p)):
        return "dicom"
    return ""


__all__ = [
    "Document", "MARKDOWN_EXTS", "PICTURE_EXTS", "RECORDING_DIR_EXTS", "RECORDING_FILE_EXTS",
    "TEXT_EXTS", "TEXT_NAMES", "VOLUME_EXTS", "document_kind", "is_mrs_path",
    "is_recording_path", "kind_of",
]
