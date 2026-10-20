"""Qt-free self-update helpers.

The Windows constraint
======================

On Windows a running Python process holds file handles on its loaded
modules (especially compiled extensions: ``PyQt6/Qt6/bin/*.dll``,
``dcm2niix.exe``, ``numpy/*.pyd``...). Upgrading ``bids-manager``
from inside the GUI then fails with::

    PermissionError: [WinError 32] The process cannot access the file
    because it is being used by another process.

The workaround is to spawn a **detached** helper process that:

1. Waits for the GUI process to exit (so all file handles are released);
2. Installs the update;
3. Optionally restarts the GUI.

The helper script is a stand-alone ``.py`` that gets **copied to a
temp directory** before being launched, so the running helper survives
the update replacing ``bidsmgr.gui._update_helper`` on disk.

Installer installations
=======================

An installation made by the BIDS Manager installer has an
``installation.json`` beside its environment, naming the uv that built
it and the settings that keep that uv to itself. The update then goes
through that uv, exactly as the installer did: no pip cache or log
outside the install folder, no pip or uv settings of the user's, and
the record kept current. Every other installation (pip, conda, an
installer 1.x) updates with pip in its own environment, as before.

A release can need a newer Python than the environment has. The update
then offers the newest release this Python can run, or, for an
installer installation, moves the environment to the Python the newest
release needs.

Failure policy
==============

Network helpers (``fetch_latest``) **never raise**. They return
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


PYPI_PACKAGE = "bids-manager"
PYPI_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE}/json"
USER_AGENT = "BIDS-Manager-GUI-update-check"
MANIFEST_NAME = "installation.json"


@dataclass(frozen=True)
class Latest:
    """What PyPI offers this installation.

    ``version`` is the newest release this Python can install, ``newest``
    the newest release of all. They differ only when a release needs a
    newer Python, and then ``needs_python`` names it (``"3.13"``).
    """

    version: str
    newest: str
    needs_python: str = ""


@dataclass(frozen=True)
class Installation:
    """An installation made by the BIDS Manager installer."""

    manifest: Path
    uv: Path


@dataclass(frozen=True)
class UpdatePlan:
    """The update to offer: a version, and the Python to move to first ("" = none)."""

    version: str
    move_to_python: str = ""


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


def find_installation(prefix: Optional[Path] = None) -> Optional[Installation]:
    """The installer's record of the running environment, or ``None``.

    The record sits beside the environment (``<install>/env`` and
    ``<install>/installation.json``). It counts only when it names this
    very environment and its uv is still there, so a stray or copied
    record never steers an update.
    """
    env = Path(prefix or sys.prefix)
    manifest = env.parent / MANIFEST_NAME
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        named_env = Path(record["environment"])
        uv = Path(record["uv"]["executable"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        if named_env.resolve() != env.resolve():
            return None
    except OSError:
        return None
    if not uv.is_file():
        return None
    return Installation(manifest=manifest, uv=uv)


def _fetch_pypi(timeout: float) -> Optional[dict]:
    """The package's PyPI JSON, or ``None``. Tries certifi, then the
    system's certificates, then no verification, in that order."""
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
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            log.debug("PyPI fetch attempt failed: %s", exc)
    return None


def fetch_latest(timeout: float = 8.0) -> Optional[Latest]:
    """What PyPI offers the running Python, or ``None`` on any failure."""
    payload = _fetch_pypi(timeout)
    if payload is None:
        return None
    try:
        return latest_from(payload, platform.python_version())
    except Exception as exc:
        log.debug("PyPI response not understood: %s", exc)
        return None


def _runs_on(requires_python: str, python: str) -> bool:
    """True when a release's ``Requires-Python`` admits ``python``."""
    if not requires_python:
        return True
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ImportError:
        return True
    try:
        return SpecifierSet(requires_python).contains(python, prereleases=True)
    except InvalidSpecifier:
        return True


def minimum_python(requires_python: str, python: str) -> str:
    """The first Python minor version after ``python``'s that a release
    admits (``">=3.13"`` on 3.12 gives ``"3.13"``), or "" if none does."""
    try:
        major, minor = (int(part) for part in python.split(".")[:2])
    except ValueError:
        return ""
    for candidate in range(minor + 1, minor + 20):
        if any(_runs_on(requires_python, f"{major}.{candidate}.{patch}") for patch in range(30)):
            return f"{major}.{candidate}"
    return ""


def latest_from(payload: dict, python: str) -> Optional[Latest]:
    """Read PyPI's JSON for ``python``: the newest stable release it can
    install and, when that is not the newest release, the Python the
    newest one needs. Yanked releases and pre-releases are never offered."""
    info = payload.get("info") or {}
    newest = str(info.get("version") or "").strip()
    if not newest:
        return None
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return Latest(version=newest, newest=newest)

    releases = []
    for name, files in (payload.get("releases") or {}).items():
        try:
            parsed = Version(name)
        except InvalidVersion:
            continue
        live = [f for f in (files or []) if not f.get("yanked")]
        if parsed.is_prerelease or not live:
            continue
        releases.append((parsed, name, str(live[0].get("requires_python") or "")))
    if not releases:
        releases = [(Version(newest), newest, str(info.get("requires_python") or ""))]
    releases.sort(reverse=True)

    installable = next((name for _v, name, needs in releases if _runs_on(needs, python)), "")
    newest_needs = next(
        (needs for _v, name, needs in releases if name == newest),
        str(info.get("requires_python") or ""),
    )
    needs_python = "" if _runs_on(newest_needs, python) else (
        minimum_python(newest_needs, python) or newest_needs)
    return Latest(version=installable, newest=newest, needs_python=needs_python)


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


def plan_update(latest: Latest, current: str, *, can_move_python: bool) -> Optional[UpdatePlan]:
    """The update to offer, or ``None`` when there is none to offer.

    An installer installation can move to the Python the newest release
    needs, so it is offered the newest release. Any other installation
    is offered the newest release its own Python can run: offering one
    it cannot install would report success, change nothing and ask
    again at every start.
    """
    if latest.needs_python and can_move_python and is_newer(latest.newest, current):
        return UpdatePlan(version=latest.newest, move_to_python=latest.needs_python)
    if is_newer(latest.version, current):
        return UpdatePlan(version=latest.version)
    return None


# ---------------------------------------------------------------------------
# Detached-helper spawn
# ---------------------------------------------------------------------------


def _python_executable() -> str:
    """Best-effort path to the venv's Python interpreter.

    ``sys.executable`` is the running interpreter, which is exactly the
    one the update is for.
    """
    return sys.executable or "python"


def _base_python() -> str:
    """The Python the environment was built from, outside it.

    Moving to a newer Python replaces the environment, and Windows cannot
    delete a program that is running, so the helper of an installer
    installation runs on this one rather than the environment's own.
    """
    base = getattr(sys, "_base_executable", "") or ""
    if base and Path(base).is_file():
        return base
    return _python_executable()


def _copy_helper_to_temp() -> Path:
    """Copy ``_update_helper.py`` to a temp dir + return the new path.

    The copy isolates the running helper from any update that may
    replace files inside the installed ``bidsmgr.gui`` package.
    """
    src = Path(__file__).with_name("_update_helper.py")
    tmpdir = Path(tempfile.mkdtemp(prefix="bidsmgr-update-"))
    dst = tmpdir / "_update_helper.py"
    shutil.copy2(src, dst)
    return dst


def update_log_path(installation: Optional[Installation]) -> Path:
    """Where the helper's output goes on macOS and Linux: inside the
    install folder for an installer installation, so the uninstaller
    takes it too; otherwise ``~/.bidsmgr/update.log``."""
    if installation is not None:
        return installation.manifest.parent / "update.log"
    return Path.home() / ".bidsmgr" / "update.log"


def launch_update_helper(*, restart: bool = True, move_to_python: str = "") -> bool:
    """Spawn the detached helper, then return.

    The caller is expected to immediately quit the GUI so the helper
    can replace the package files. Returns ``True`` if the spawn call
    itself succeeded; the actual install runs after this returns.
    ``move_to_python`` ("3.13") rebuilds the environment on that Python
    first, which only an installer installation can do.
    """
    installation = find_installation()
    if move_to_python and installation is None:
        log.error("Moving to Python %s needs the installer's uv", move_to_python)
        return False
    try:
        helper = _copy_helper_to_temp()
    except Exception as exc:
        log.error("Could not stage update helper: %s", exc)
        return False

    python = _python_executable()
    runner = python if installation is None else _base_python()
    cmd = [
        runner, "-I", str(helper),
        "--parent-pid", str(os.getpid()),
        "--python", python,
        "--package", PYPI_PACKAGE,
    ]
    if installation is not None:
        cmd += ["--installation", str(installation.manifest)]
        if move_to_python:
            cmd += ["--move-to-python", move_to_python]
    if restart:
        # Restart by re-launching the same entry point the user just
        # used. ``sys.argv[0]`` is e.g. ``.../env/Scripts/bidsmgr.exe``
        # on Windows or ``.../env/bin/bidsmgr`` on POSIX.
        cmd += ["--restart-cmd", sys.argv[0]]

    try:
        if platform.system() == "Windows":
            # CREATE_NEW_CONSOLE pops a visible console window for the
            # installer's output. DETACHED_PROCESS would hide it but the
            # user gets no feedback during a multi-minute install.
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
            # exit doesn't take the helper down with it.
            log_path = update_log_path(installation)
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
    "Installation",
    "Latest",
    "PYPI_PACKAGE",
    "UpdatePlan",
    "fetch_latest",
    "find_installation",
    "installed_version",
    "is_editable_install",
    "is_newer",
    "latest_from",
    "launch_update_helper",
    "minimum_python",
    "plan_update",
    "update_log_path",
]
