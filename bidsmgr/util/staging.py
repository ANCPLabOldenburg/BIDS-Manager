"""Putting one series' DICOMs where dcm2niix can read them.

dcm2niix converts a FOLDER, so every caller that converts one series (the
scan's probe, the conversion, the series preview) first gathers that series'
files into a folder of its own, under short sequential names (``000000.dcm``,
``000001.dcm``, ...). Vendor DICOM names are ~80 characters and push a deep
Windows path past its 260-character ceiling; dcm2niix orders frames by their
DICOM tags, never by filename, so renaming is safe.

A file is LINKED where it can be and COPIED where a link could fail, and a
link that could fail is any link on Windows that involves a network share:

- a link stored ON a share is not followed by default (Windows' "remote to
  remote" and "remote to local" link rules are off), so a project kept on a
  share gave dcm2niix a folder of links it could not open, and dcm2niix
  answered ``rc=2`` "Unable to find any DICOM images";
- a link from a local folder TO a share is followed by default, but a policy
  can switch that off, and an institution that manages its shares often does;
- creating a link at all needs a privilege an ordinary Windows account lacks
  (``WinError 1314``), and a share may refuse it with another error.

So on Windows, a source or a staging folder on a share is copied, and any
error creating a link switches the rest of the series to copies. macOS and
Linux link as before: they mount a share into the one file tree and follow
links into it like any other folder.

:func:`staging_folder` keeps the copies off the share as well: copying a
series from a share to a staging folder on the same share and reading it
back moves every byte over the network three times, where a local
temporary folder moves it once.

Link targets are made absolute with ``os.path.abspath``, never
``Path.resolve``: on Windows ``resolve`` turns a mapped drive into its UNC
path, or on a DFS share into whichever server answered, which may not be
readable from here.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from .paths import is_network_path, long_path

log = logging.getLogger(__name__)


def staging_folder(preferred: Path) -> Path:
    """Where to stage: ``preferred``, or a new local temporary folder when
    ``preferred`` is on a network share (Windows only). The caller removes
    whichever it gets once dcm2niix has run."""
    if is_network_path(preferred):
        return Path(tempfile.mkdtemp(prefix="bidsmgr-dicoms-"))
    return Path(preferred)


def stage_files(files: Iterable[str | Path], staging_dir: Path) -> int:
    """Gather ``files`` into ``staging_dir`` under short sequential names.

    The folder is emptied first, so staging again is idempotent. Files that
    do not exist are skipped. Returns how many were staged.
    """
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(long_path(staging_dir))
    staging_dir.mkdir(parents=True)
    copy = is_network_path(staging_dir)
    n = 0
    for idx, fp in enumerate(files):
        source = os.path.abspath(os.fspath(fp))
        if not os.path.exists(long_path(source)):
            continue
        suffix = Path(source).suffix or ".dcm"
        target = long_path(staging_dir / f"{idx:06d}{suffix}")
        if copy or is_network_path(source):
            shutil.copyfile(long_path(source), target)
        else:
            try:
                os.symlink(long_path(source), target)
            except OSError as exc:
                log.info("staging: cannot link here (%s); copying the rest of the "
                         "series instead", exc)
                copy = True
                shutil.copyfile(long_path(source), target)
        n += 1
    return n


__all__ = ["stage_files", "staging_folder"]
