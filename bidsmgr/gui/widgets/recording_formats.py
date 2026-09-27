"""Which files are recordings, and how their extensions are spelled.

Three lines of knowledge with four consumers (the Editor's dispatch, the
recording pane, the time-series view's CTF check, the tests), so it lives on
its own rather than being imported out of whichever widget happened to
define it first.
"""

from __future__ import annotations

from pathlib import Path

# Recognised recording extensions (the Editor dispatch set).
#
# Deliberately excludes BrainVision sidecars (``.eeg`` / ``.vmrk``) and the
# raw partners opened via their header file: the user opens ``.vhdr``, not the
# binary ``.eeg``. ``mne.io.read_raw`` auto-dispatches every entry below (with
# the format-specific fallbacks in ``recording_viewer_pane._read_raw``).
VIEWER_FILE_EXTS: frozenset[str] = frozenset({
    ".fif", ".fif.gz", ".con", ".sqd", ".kdf",
    ".vhdr", ".edf", ".bdf", ".gdf", ".set", ".cnt", ".egi", ".mef", ".nwb",
})
VIEWER_DIR_EXTS: frozenset[str] = frozenset({".ds", ".mff"})


def full_ext(path) -> str:
    """Lower-case extension, preserving the ``.fif.gz`` double suffix."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".fif.gz"):
        return ".fif.gz"
    return p.suffix.lower()


def is_recording_path(path) -> bool:
    """True when *path* is an EEG/MEG/iEEG recording the viewer can open."""
    p = Path(path)
    ext = full_ext(p)
    if p.is_dir():
        return ext in VIEWER_DIR_EXTS
    return ext in VIEWER_FILE_EXTS


__all__ = [
    "VIEWER_DIR_EXTS",
    "VIEWER_FILE_EXTS",
    "full_ext",
    "is_recording_path",
]
