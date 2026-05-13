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

    # UNC paths (``\\server\share\...``) get the special ``UNC\`` form.
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p
