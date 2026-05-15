"""Qt-free self-update helpers.

The Windows constraint
======================

On Windows a running Python process holds file handles on its loaded
modules (especially compiled extensions: ``PyQt6/Qt6/bin/*.dll``,
``dcm2niix.exe``, ``numpy/*.pyd``...). ``pip install --upgrade
bids-manager`` invoked from inside the GUI then fails with::

    PermissionError: [WinError 32] The process cannot access the file
    because it is being used by another process.

The workaround is to spawn a **detached** helper process that:

1. Waits for the GUI process to exit (so all file handles are released);
2. Runs ``python -m pip install --upgrade bids-manager``;
3. Optionally restarts the GUI.

The helper script is a stand-alone ``.py`` that gets **copied to a
temp directory** before being launched, so the running helper survives
pip replacing ``bidsmgr.gui._update_helper`` on disk.

Failure policy
==============

Network helpers (``fetch_latest_pypi``) **never raise**. They return
``None`` on any failure (no internet, DNS error, SSL error, timeout,
malformed JSON, ...). Callers treat ``None`` as "couldn't tell".
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


PYPI_PACKAGE = "bids-manager"
PYPI_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE}/json"
USER_AGENT = "BIDS-Manager-GUI-update-check"


def installed_version() -> str:
    """Return the running bidsmgr version (``bidsmgr.__version__``)."""
    from .. import __version__
    return str(__version__)


def is_editable_install() -> bool:
    """True when bidsmgr is installed in editable mode (``pip install -e .``).

    Editable installs should not be auto-updated. We detect them via the
    PEP 660 ``direct_url.json`` marker that pip writes into the dist-info.
    Any error reading the metadata is treated as "not editable" so we
    err on the side of allowing updates.
    """
    try:
        from importlib.metadata import distribution, PackageNotFoundError
        try:
            dist = distribution(PYPI_PACKAGE)
        except PackageNotFoundError:
            return False
        raw = dist.read_text("direct_url.json")
        if not raw:
            return False
        data = json.loads(raw)
        return bool(data.get("dir_info", {}).get("editable"))
    except Exception:
        return False


def fetch_latest_pypi(timeout: float = 8.0) -> Optional[str]:
    """Return the latest stable version on PyPI, or ``None`` on any failure.

    Never raises. Tries certifi → system CA → unverified SSL, in that
    order. Returns ``None`` if every attempt fails or the response is
    not parseable.
    """
    req = Request(PYPI_URL, headers={"User-Agent": USER_AGENT})
    contexts = []
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    try:
        contexts.append(ssl.create_default_context())
    except Exception:
        pass
    try:
        contexts.append(ssl._create_unverified_context())
    except Exception:
        pass

    for ctx in contexts:
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            version = str(payload.get("info", {}).get("version", "")).strip()
            return version or None
        except Exception as exc:
            log.debug("PyPI fetch attempt failed: %s", exc)
    return None


def is_newer(latest: str, current: str) -> bool:
    """Return True when ``latest`` is strictly newer than ``current``.

    Uses ``packaging.version`` when available; falls back to a string
    comparison that at least catches the equality case correctly.
    """
    if not latest or not current:
        return False
    try:
        from packaging.version import Version
        return Version(latest) > Version(current)
    except Exception:
        return latest != current


# ---------------------------------------------------------------------------
# Detached-helper spawn
# ---------------------------------------------------------------------------


def _python_executable() -> str:
    """Best-effort path to the venv's Python interpreter.

    ``sys.executable`` is the running interpreter, which is exactly the
    one pip needs to upgrade the package it's already imported from.
    """
    return sys.executable or "python"


def _copy_helper_to_temp() -> Path:
    """Copy ``_update_helper.py`` to a temp dir + return the new path.

    The copy isolates the running helper from any pip operation that
    may replace files inside the installed ``bidsmgr.gui`` package.
    """
    src = Path(__file__).with_name("_update_helper.py")
    tmpdir = Path(tempfile.mkdtemp(prefix="bidsmgr-update-"))
    dst = tmpdir / "_update_helper.py"
    shutil.copy2(src, dst)
    return dst


def launch_update_helper(*, restart: bool = True) -> bool:
    """Spawn the detached helper, then return.

    The caller is expected to immediately quit the GUI so the helper
    can replace the package files. Returns ``True`` if the spawn call
    itself succeeded; the actual pip install runs after this returns.
    """
    try:
        helper = _copy_helper_to_temp()
    except Exception as exc:
        log.error("Could not stage update helper: %s", exc)
        return False

    python = _python_executable()
    parent_pid = os.getpid()

    cmd = [
        python, str(helper),
        "--parent-pid", str(parent_pid),
        "--python", python,
        "--package", PYPI_PACKAGE,
    ]
    if restart:
        # Restart by re-launching the same entry point the user just
        # used. ``sys.argv[0]`` is e.g. ``.../env/Scripts/bidsmgr.exe``
        # on Windows or ``.../env/bin/bidsmgr`` on POSIX.
        cmd += ["--restart-cmd", sys.argv[0]]

    try:
        if platform.system() == "Windows":
            # CREATE_NEW_CONSOLE pops a visible console window for pip's
            # output. DETACHED_PROCESS would hide it but the user gets
            # no feedback during a multi-minute pip install.
            CREATE_NEW_CONSOLE = 0x00000010
            CREATE_BREAKAWAY_FROM_JOB = 0x01000000
            flags = CREATE_NEW_CONSOLE | CREATE_BREAKAWAY_FROM_JOB
            subprocess.Popen(
                cmd,
                creationflags=flags,
                close_fds=True,
            )
        else:
            # POSIX: detach into a new session so SIGHUP from the GUI
            # exit doesn't take the helper down with it. Output goes to
            # ~/.bidsmgr/update.log so the user can debug if needed.
            log_path = Path.home() / ".bidsmgr" / "update.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_path, "ab", buffering=0)
            subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        return True
    except Exception as exc:
        log.error("Could not spawn update helper: %s", exc)
        return False


__all__ = [
    "PYPI_PACKAGE",
    "fetch_latest_pypi",
    "installed_version",
    "is_editable_install",
    "is_newer",
    "launch_update_helper",
]
