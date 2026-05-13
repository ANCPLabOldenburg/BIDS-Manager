"""Make the Qt xcb platform plugin loadable on minimal Linux hosts.

Qt 6.5+ refuses to load the ``xcb`` platform plugin unless
``libxcb-cursor.so.0`` (the ``libxcb-cursor0`` Debian package) is
present on the system. Distributions like vanilla Ubuntu Server, fresh
WSL images, and container base images often omit it, producing::

    qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb"

This module fixes that without requiring sudo:

1. If a Wayland session is active, prefer the ``wayland`` plugin with
   ``xcb`` as a fallback — Wayland does not need ``libxcb-cursor``.
2. If the system library is already loadable, do nothing.
3. Otherwise download ``libxcb-cursor0`` into a user cache directory
   (via ``apt-get download``, which does not need root) and preload it
   with ``ctypes`` so Qt's ``dlopen`` finds it already mapped.

Call :func:`prepare` before importing ``PyQt6.QtWidgets``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "bidsmgr" / "qtlibs"


def _try_load_xcb_cursor() -> bool:
    """Return True if ``libxcb-cursor.so.0`` can be dlopen'd."""
    try:
        ctypes.CDLL("libxcb-cursor.so.0", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def _fetch_via_apt(target: Path) -> bool:
    """Download libxcb-cursor0 with apt-get (no sudo) and extract the .so."""
    if not (shutil.which("apt-get") and shutil.which("dpkg")):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        try:
            res = subprocess.run(
                ["apt-get", "download", "libxcb-cursor0"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        if res.returncode != 0:
            return False
        debs = list(tmpdir.glob("libxcb-cursor0*.deb"))
        if not debs:
            return False
        extracted = tmpdir / "x"
        try:
            subprocess.run(
                ["dpkg", "-x", str(debs[0]), str(extracted)],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return False
        # The deb ships e.g. usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0.0.0
        # plus a libxcb-cursor.so.0 symlink. We want the real file.
        real = None
        for cand in extracted.rglob("libxcb-cursor.so.0*"):
            if cand.is_file() and not cand.is_symlink():
                real = cand
                break
        if real is None:
            for cand in extracted.rglob("libxcb-cursor.so.0"):
                if cand.is_file():
                    real = cand
                    break
        if real is None:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real, target)
        return True


def prepare() -> None:
    """Best-effort: ensure the Qt xcb (or wayland) plugin can load."""
    if not sys.platform.startswith("linux"):
        return
    # Respect an explicit user choice.
    if os.environ.get("QT_QPA_PLATFORM"):
        return

    # If Wayland is active, prefer it — it doesn't need libxcb-cursor.
    # Keep xcb as a fallback for hybrid sessions.
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"

    # System library already present?
    if _try_load_xcb_cursor():
        return

    # Cached copy from a previous run?
    cached = _cache_dir() / "libxcb-cursor.so.0"
    if cached.exists():
        try:
            ctypes.CDLL(str(cached), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            cached.unlink(missing_ok=True)

    # Fetch with apt-get download (works without sudo on Debian/Ubuntu).
    if _fetch_via_apt(cached):
        try:
            ctypes.CDLL(str(cached), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass

    # Nothing worked. Leave a hint before Qt aborts so the user sees a
    # path forward that doesn't require root.
    print(
        "bidsmgr: libxcb-cursor.so.0 is missing and could not be "
        "auto-installed. The Qt xcb platform plugin will likely fail "
        "to load. Try one of:\n"
        "  - apt-get download libxcb-cursor0   (no sudo, Debian/Ubuntu)\n"
        "  - conda install -c conda-forge xcb-util-cursor\n"
        "  - sudo apt install libxcb-cursor0",
        file=sys.stderr,
    )
