"""Cross-platform path helpers.

The conversion pipeline composes path components from inventory data:
subject id, session label, dataset slug, dcm2niix BIDS basename, and
(historically) a raw SeriesInstanceUID. macOS and Linux accept almost
any byte sequence as a path component, but Windows rejects a number of
characters at the filesystem-syscall level (raising ``WinError 123``
"The filename, directory name, or volume label syntax is incorrect")
and also caps a full path at 260 characters unless an opt-in long-path
mode is used.

This module is the single place where we encode those rules. The
inventory layer is free to keep its canonical logical ids (which may
contain ``|`` or other characters that are perfectly valid as dict
keys but not as path components); every site that turns one of those
ids into a path must funnel through here.

Why a module-level helper rather than inlining the hash:

* multiple call sites (per-series staging dir, fmap pair UIDs, future
  derivatives stamping) need the same sanitisation;
* tests can pin the behaviour cross-platform on a CI runner that may
  not exhibit the Windows failure;
* keeps ``converter/backends/dcm2niix_direct.py`` free of OS trivia.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


# Characters Windows refuses inside a path component. ``/`` and ``\``
# are path separators on every OS so they're always illegal *inside* a
# segment; the rest are Windows-only but stripping them everywhere
# keeps file layouts identical across OSes (avoiding a "works on Linux
# but breaks if synced to a Windows share" trap).
WINDOWS_RESERVED_CHARS: frozenset[str] = frozenset('<>:"/\\|?*')

# Reserved device names on Windows — illegal as the *stem* of any path
# component regardless of case or extension (e.g. ``CON.txt`` is also
# rejected). We refuse these by appending an underscore.
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9",
})

# Control characters (ASCII 0–31) also crash Windows ``CreateFile``;
# regex applied below.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f]")

# Path-component length budget. Windows MAX_PATH = 260 for a *full*
# absolute path, so we cap each component conservatively to leave room
# for the BIDS root + datatype tree. dcm2niix itself produces basenames
# under ~80 chars, so 96 is plenty of headroom for our own composed
# components.
_MAX_COMPONENT_LEN = 96

# A name-collision-safe length budget for a stable hash suffix appended
# when a sanitised component would otherwise lose information (e.g. the
# raw input contained illegal chars or was too long).
_HASH_SUFFIX_LEN = 10


def safe_path_component(
    raw: str,
    *,
    fallback: str = "unnamed",
    max_len: int = _MAX_COMPONENT_LEN,
) -> str:
    """Return ``raw`` reshaped into a portable path-component string.

    Guarantees on the output:

    * no character from :data:`WINDOWS_RESERVED_CHARS`;
    * no ASCII control character;
    * no trailing dot or whitespace (Windows silently strips them
      from path components, which corrupts cross-OS comparisons);
    * does not match a name in :data:`WINDOWS_RESERVED_NAMES`;
    * length ≤ ``max_len``;
    * never empty (falls back to ``fallback``).

    The transformation is deterministic, idempotent, and information-
    preserving for already-safe inputs (e.g. ``"sub-001"`` is returned
    unchanged). When the input needed lossy edits (illegal char
    substitution or length truncation), a short SHA-1 hash suffix is
    appended so two distinct illegal inputs cannot collide on the
    same output.

    Examples
    --------
    >>> safe_path_component("sub-001")
    'sub-001'
    >>> safe_path_component("foo|bar")           # doctest: +SKIP
    'foo_bar_<10hex>'
    >>> safe_path_component("trailing.")         # doctest: +SKIP
    'trailing__<10hex>'
    """
    if not raw:
        return fallback

    cleaned = _CONTROL_CHARS_RE.sub("_", raw)
    for ch in WINDOWS_RESERVED_CHARS:
        if ch in cleaned:
            cleaned = cleaned.replace(ch, "_")

    # Windows strips trailing dots / spaces silently. Replace them so
    # the visible string matches what gets stored on disk.
    if cleaned and cleaned[-1] in (".", " "):
        cleaned = cleaned.rstrip(". ") or fallback

    # Reserved device names — append an underscore on the stem.
    stem = cleaned.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"

    needs_disambiguation = (cleaned != raw) or len(cleaned) > max_len

    if needs_disambiguation:
        # Truncate to leave room for the hash suffix + underscore.
        budget = max(1, max_len - _HASH_SUFFIX_LEN - 1)
        head = cleaned[:budget].rstrip(". _") or fallback
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
        cleaned = f"{head}_{digest}"

    return cleaned or fallback


def long_path(path: Path | str) -> str:
    """Return ``path`` annotated with the Windows long-path prefix when needed.

    On Windows, the Win32 API caps absolute paths at 260 characters
    unless the path is prefixed with ``\\\\?\\`` (UNC paths use
    ``\\\\?\\UNC\\``). Pass the result of this helper to any low-level
    Win32 syscall (``os.scandir``, ``os.symlink``, ``shutil.rmtree``,
    ``subprocess.run`` arguments) that may otherwise hit the limit.

    On macOS / Linux this is a no-op — the platform check is the only
    code path that runs.

    Notes
    -----
    The returned value is a ``str`` so callers can pass it to APIs
    that accept ``str`` or path-like. Using ``Path()`` on the
    long-prefixed string round-trips correctly on Windows but the
    prefix is preserved.
    """
    if os.name != "nt":
        return str(path)

    p = os.fspath(path)
    # Already prefixed — don't double-wrap.
    if p.startswith("\\\\?\\"):
        return p

    # Only prefix when there's a real risk of exceeding MAX_PATH.
    # Sub-260 paths work unchanged on every Windows version we target.
    if len(p) < 248:
        return p
    return _prefixed(p)


def long_path_for_tree(path: Path | str) -> str:
    """Like :func:`long_path`, but for a directory somebody else will WALK.

    Use this for a directory handed to a tool that opens the files INSIDE it.
    :func:`long_path` is for a path that is itself about to be opened.

    The difference is the whole point, and it is easy to miss: ``long_path``
    measures the string it is given, but the length that decides whether a
    child can be opened belongs to the CHILD. A folder of Siemens DICOMs at
    200 characters sits comfortably under the 248 threshold and is handed over
    unprefixed, while its ~90-character filenames put every file at 290 —
    thirty past the ceiling.

    That is exactly how it presented: dcm2niix walked the folder, could open
    nothing, and reported ``rc=2`` "Unable to find any DICOM images in ... (or
    subfolders 5 deep)", which reads as "this is not DICOM" rather than "this
    path is too long". The BidsGuess classifier contributed nothing, every
    series fell through to the regex layer, and a gradient fieldmap came back
    ``phasediff`` where macOS and Linux said ``magnitude1``.

    Measured on the reporting dataset, same files and same binary: 0 sidecars
    from the 290-character path, 15 from a short one.

    So there is no threshold here. We cannot know the longest child without
    walking the tree ourselves, and a wrong guess costs the entire
    classification, so Windows always gets the prefix and pays a few bytes of
    argument. The path is made absolute and normalised first because
    ``\\\\?\\`` accepts only a fully-qualified, normalised path: it switches OFF
    the normalisation that would otherwise tidy up ``/`` and ``..``.

    ``os.path.abspath``, NOT ``Path.resolve``. On Windows ``resolve`` asks the
    file system for the FINAL path, which turns a mapped drive into its UNC
    path (``Z:\\data`` into ``\\\\server\\share\\data``) and, on a DFS share,
    into whichever file server answered. Neither is the folder the user
    picked, and the second may not be readable at all. ``abspath`` only
    normalises the text.

    A no-op off Windows, exactly like its sibling: macOS and Linux return
    ``str(path)`` and have no ceiling to work around.
    """
    if os.name != "nt":
        return str(path)
    p = os.fspath(path)
    if p.startswith("\\\\?\\"):
        return p
    return _prefixed(os.path.abspath(p))


def _prefixed(p: str) -> str:
    """Add the Win32 long-path prefix, in the form UNC paths need.

    A path that already carries a prefix (``\\\\?\\`` or the device form
    ``\\\\.\\``) is returned as it is: read as a UNC path it would become
    ``\\\\?\\UNC\\?\\...``, which names nothing.
    """
    if p.startswith(("\\\\?\\", "\\\\.\\")):
        return p
    # UNC paths (``\\\\server\\share\\...``) get the special ``UNC\\`` form.
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


#: What ``GetDriveTypeW`` answers for a drive mapped to a network share.
_DRIVE_REMOTE = 4


def _drive_type(root: str) -> int:
    """``GetDriveTypeW`` for a drive root such as ``Z:\\``; 0 when unknown."""
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
    except (AttributeError, OSError, ValueError):
        return 0


def is_network_path(path: Path | str) -> bool:
    """Whether ``path`` is on a network share, as Windows sees it.

    A UNC path (``\\\\server\\share\\...``, also in its ``\\\\?\\UNC\\`` form)
    or a path on a drive letter mapped to a share. Windows treats such a path
    differently from a local one in ways that matter to staging: it does not
    follow a symbolic link stored on a share (its default "remote to remote"
    and "remote to local" link rules are off), and a policy can switch off
    links from a local folder to a share too.

    Always ``False`` off Windows: macOS and Linux mount a share into the one
    file tree and follow links into it like any other folder.
    """
    if os.name != "nt":
        return False
    p = os.fspath(path)
    if p[:8].upper() == "\\\\?\\UNC\\":
        return True
    if p.startswith(("\\\\?\\", "\\\\.\\")):
        p = p[4:]
    elif not p.startswith("\\\\"):
        p = os.path.abspath(p)
    if p.startswith("\\\\"):
        return True
    drive = os.path.splitdrive(p)[0]
    if len(drive) == 2 and drive[1] == ":":
        return _drive_type(drive + "\\") == _DRIVE_REMOTE
    return False


def missing_folder_message(path: Path | str) -> str:
    """Why ``path`` cannot be read, in words a user can act on.

    On Windows a drive letter that does not exist at all is most often a
    drive mapped to a network share in ANOTHER session: mapped drives belong
    to the login session that mapped them, so a program started "as
    administrator" (or by another user, or as a service) does not see them,
    while File Explorer does.
    """
    text = f"no such folder: {os.fspath(path)}"
    if os.name != "nt":
        return text
    drive = os.path.splitdrive(os.path.abspath(os.fspath(path)))[0]
    if len(drive) == 2 and drive[1] == ":" and not os.path.exists(drive + "\\"):
        text += (
            f". Drive {drive} is not visible to BIDS Manager. A drive mapped to a "
            "network share belongs to the Windows session that mapped it, so a "
            "program started as administrator, or by another user, does not see "
            "it. Start BIDS Manager normally, or use the share's \\\\server\\share "
            "path instead of the drive letter."
        )
    return text
