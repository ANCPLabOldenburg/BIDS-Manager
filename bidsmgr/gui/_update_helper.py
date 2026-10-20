"""Detached self-update helper. Runs *after* the GUI has exited.

This script is copied to a temp directory by ``updater.launch_update_helper``
before it is spawned, so it does not depend on the installed
``bidsmgr.gui`` package files staying intact during the update (which is
exactly what would be a problem on Windows). It uses the standard library
only.

Lifecycle::

    1. wait for --parent-pid to terminate
    2. install the update:
       - with --installation, through the uv the installer recorded,
         with the installer's settings, then bring the record up to date;
         --move-to-python first rebuilds the environment on that Python,
         and puts the previous one back if anything fails;
       - otherwise ``python -m pip install --upgrade <package>``
    3. optionally re-launch the GUI via --restart-cmd

On Windows the helper runs in its own console window so the user sees the
progress; on POSIX the launcher redirects it to a log file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


PARENT_WAIT_TIMEOUT_S = 90.0  # how long we'll wait for the GUI to exit
PARENT_POLL_INTERVAL_S = 0.5
RENAME_RETRY_S = 10.0  # Windows: a virus scanner may hold a file for a moment

# The installer's build (uv's target triple) -> uv's name for a Python of it.
_PYTHON_PLATFORMS = {
    "aarch64-apple-darwin": "macos-aarch64-none",
    "x86_64-apple-darwin": "macos-x86_64-none",
    "x86_64-unknown-linux-gnu": "linux-x86_64-gnu",
    "aarch64-unknown-linux-gnu": "linux-aarch64-gnu",
    "x86_64-pc-windows-msvc": "windows-x86_64-none",
}

# Variables that would point the installer's uv or the new environment
# somewhere else.
_FOREIGN = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH")


def _wait_for_parent_posix(pid: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Process exists but we can't signal it. Treat as alive.
            pass
        time.sleep(PARENT_POLL_INTERVAL_S)
    return False


def _wait_for_parent_windows(pid: int, timeout_s: float) -> bool:
    import ctypes

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    # Not used, and kept on purpose: it is the OTHER value
    # WaitForSingleObject returns, so the pair is what makes the
    # `result == WAIT_OBJECT_0` below legible as "exited" rather than
    # "timed out". Deleting it to satisfy the linter would trade a
    # documented Win32 contract for a tidier report.
    WAIT_TIMEOUT = 0x102  # noqa: F841

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        # Couldn't open it. Either it's already gone, or we lack
        # permission. Either way the update won't be blocked by it.
        return True
    try:
        result = kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
        return result == WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


def wait_for_parent(pid: int, timeout_s: float = PARENT_WAIT_TIMEOUT_S) -> bool:
    """Block until pid exits or *timeout_s* elapses. Returns True if gone."""
    if pid <= 0:
        return True
    if platform.system() == "Windows":
        return _wait_for_parent_windows(pid, timeout_s)
    return _wait_for_parent_posix(pid, timeout_s)


def run_pip_upgrade(python: str, package: str) -> int:
    """Run ``python -m pip install --upgrade <package>``. Returns rc."""
    print(f"[update] running: {python} -m pip install --upgrade {package}",
          flush=True)
    try:
        proc = subprocess.run(
            [python, "-m", "pip", "install", "--upgrade", package],
            check=False,
        )
        return proc.returncode
    except Exception as exc:
        print(f"[update] pip invocation failed: {exc}", flush=True)
        return 1


# ---------------------------------------------------------------------------
# Installer installations: the recorded uv
# ---------------------------------------------------------------------------


def uv_environment(record: dict, base: dict) -> dict:
    """``base`` without anything meant for another uv or environment,
    plus the settings the installer ran its uv with."""
    env = {key: value for key, value in base.items()
           if not key.startswith("UV_") and key not in _FOREIGN}
    env.update({key: str(value) for key, value in (record.get("uv_environment") or {}).items()})
    return env


def run_uv(record: dict, args: list) -> int:
    """Run the recorded uv; once more with the system's certificates if
    it fails, which networks that inspect HTTPS need (as the installer does)."""
    uv = record["uv"]["executable"]
    env = uv_environment(record, dict(os.environ))
    print(f"[update] running: uv {' '.join(str(a) for a in args)}", flush=True)
    try:
        rc = subprocess.run([uv, *args], env=env, check=False).returncode
        if rc != 0:
            print("[update] trying again with the system's certificates", flush=True)
            rc = subprocess.run([uv, *args], env={**env, "UV_SYSTEM_CERTS": "1"},
                                check=False).returncode
        return rc
    except Exception as exc:
        print(f"[update] uv could not run: {exc}", flush=True)
        return 1


def python_request(record: dict, version: str) -> str:
    """uv's name for Python ``version`` built for this installation's
    platform (``cpython-3.13-macos-aarch64-none``), so a move keeps the
    build the installer chose (Windows on ARM stays on x64)."""
    target = (record.get("installer") or {}).get("target", "")
    suffix = _PYTHON_PLATFORMS.get(target)
    return f"cpython-{version}-{suffix}" if suffix else version


def installed_pythons(record: dict) -> set:
    """uv's names for the Pythons in the install folder, one per minor
    version and build (``cpython-3.13-macos-aarch64-none``)."""
    root = (record.get("uv_environment") or {}).get("UV_PYTHON_INSTALL_DIR")
    found = set()
    if root:
        for entry in Path(root).glob("cpython-*"):
            match = re.fullmatch(r"cpython-(\d+\.\d+)\.\d+[^-]*-(.+)", entry.name)
            if match:
                found.add(f"cpython-{match.group(1)}-{match.group(2)}")
    return found


def remove_pythons(record: dict, requests) -> None:
    """Remove Pythons no environment uses any more, best effort: each is
    about 70 MB. On Windows the one this helper runs on cannot be removed
    while it runs, so it stays until the next move."""
    uv = record["uv"]["executable"]
    env = uv_environment(record, dict(os.environ))
    for request in sorted(requests):
        try:
            rc = subprocess.run([uv, "python", "uninstall", request], env=env,
                                check=False).returncode
        except Exception:
            rc = 1
        if rc != 0:
            print(f"[update] {request} is kept (in use, or could not be removed).", flush=True)


def upgrade_with_uv(record: dict, python: str, package: str) -> int:
    return run_uv(record, ["pip", "install", "--upgrade", "--compile-bytecode",
                           "--python", python, package])


def _rename(src: Path, dst: Path) -> None:
    """``src.rename(dst)``, retried for a few seconds on Windows, where a
    virus scanner can hold a file the moment after a program closes it."""
    deadline = time.time() + (RENAME_RETRY_S if platform.system() == "Windows" else 0)
    while True:
        try:
            src.rename(dst)
            return
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(0.5)


def _starts(python: str) -> bool:
    try:
        return subprocess.run([python, "-I", "-c", "import bidsmgr, PyQt6.QtWidgets"],
                              check=False).returncode == 0
    except Exception:
        return False


def move_to_python(record: dict, version: str, python: str, package: str) -> int:
    """Rebuild the environment on Python ``version`` and install the
    package into it. The previous environment is set aside first and put
    back if any step fails, so a failed move leaves BIDS Manager as it was."""
    env_dir = Path(record["environment"])
    request = python_request(record, version)
    already_there = request in installed_pythons(record)
    install = ["python", "install", request, "--no-bin"]
    if platform.system() == "Windows":
        install.append("--no-registry")
    if run_uv(record, install) != 0:
        print(f"[update] Python {version} could not be installed.", flush=True)
        return 1

    previous = env_dir.with_name(env_dir.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    _rename(env_dir, previous)
    ok = (run_uv(record, ["venv", "--seed", "--prompt", "BIDS-Manager",
                          "--python", request, str(env_dir)]) == 0
          and upgrade_with_uv(record, python, package) == 0
          and _starts(python))
    if ok:
        shutil.rmtree(previous, ignore_errors=True)
        print(f"[update] moved to Python {version}.", flush=True)
        return 0
    print("[update] the move did not complete; putting the previous environment back.",
          flush=True)
    shutil.rmtree(env_dir, ignore_errors=True)
    _rename(previous, env_dir)
    if not already_there:
        remove_pythons(record, {request})
    return 1


def record_update(manifest: Path, record: dict, python: str, moved_to: str = "") -> None:
    """Bring ``installation.json`` up to date with what is now installed."""
    try:
        out = subprocess.run(
            [python, "-I", "-c",
             "import json, platform; from importlib.metadata import version; "
             "print(json.dumps([version('bids-manager'), platform.python_version()]))"],
            capture_output=True, text=True, check=True,
        ).stdout
        package_version, python_version = json.loads(out.strip().splitlines()[-1])
    except Exception as exc:
        print(f"[update] could not read the installed versions: {exc}", flush=True)
        return
    record.setdefault("package", {})["version"] = package_version
    record.setdefault("python", {})["version"] = python_version
    if moved_to:
        # The mark tells the installer this Python was needed by a release,
        # so running it again keeps the environment rather than rebuilding
        # it on the installer's older default.
        record["python"]["requested"] = moved_to
        record["python"]["moved_by_update"] = True
    record["updated"] = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0).isoformat()
    try:
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[update] could not update {manifest.name}: {exc}", flush=True)


def update_installation(manifest: Path, python: str, package: str, move_to: str) -> int:
    """The update of an installer installation, through its own uv."""
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if move_to:
        rc = move_to_python(record, move_to, python, package)
    else:
        rc = upgrade_with_uv(record, python, package)
    if rc == 0:
        record_update(manifest, record, python, move_to)
        if move_to:
            # Last, because on macOS and Linux this may remove the very
            # Python this helper runs on (harmless once nothing more is
            # loaded from it).
            remove_pythons(record, installed_pythons(record) - {python_request(record, move_to)})
    return rc


def restart_gui(restart_cmd: str) -> None:
    """Re-launch the GUI in a detached process and exit the helper."""
    if not restart_cmd:
        return
    try:
        if platform.system() == "Windows":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(
                [restart_cmd],
                creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                [restart_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        print(f"[update] re-launched: {restart_cmd}", flush=True)
    except Exception as exc:
        print(f"[update] could not relaunch GUI: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="bidsmgr-update-helper")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--python", required=True,
                        help="Path to the venv's Python interpreter.")
    parser.add_argument("--package", default="bids-manager")
    parser.add_argument("--installation", default="",
                        help="The installer's installation.json: update through its uv.")
    parser.add_argument("--move-to-python", default="",
                        help="With --installation: rebuild the environment on this "
                             "Python (for example 3.13) before updating.")
    parser.add_argument("--restart-cmd", default="",
                        help="GUI entry point to relaunch after upgrade. "
                             "Empty disables restart.")
    args = parser.parse_args()

    print(f"[update] waiting for parent pid {args.parent_pid} to exit...",
          flush=True)
    if not wait_for_parent(args.parent_pid):
        print(
            f"[update] timed out after {PARENT_WAIT_TIMEOUT_S:.0f}s; "
            "the GUI is still running, aborting the update.",
            flush=True,
        )
        _pause_console_on_windows()
        return 2
    # A small grace period: even after the process exits, Windows may
    # take a moment to release every file handle the OS held on its
    # behalf.
    time.sleep(0.5)

    if args.installation:
        try:
            rc = update_installation(Path(args.installation), args.python,
                                     args.package, args.move_to_python)
        except Exception as exc:
            print(f"[update] the update could not run: {exc}", flush=True)
            rc = 1
    else:
        rc = run_pip_upgrade(args.python, args.package)
    if rc != 0:
        print(f"[update] the update exited with code {rc}.", flush=True)
        _pause_console_on_windows()
        return rc

    print("[update] upgrade completed successfully.", flush=True)
    if args.restart_cmd:
        restart_gui(args.restart_cmd)
    else:
        _pause_console_on_windows()
    return 0


def _pause_console_on_windows() -> None:
    """On Windows we run in a CREATE_NEW_CONSOLE, so pause so the user
    can read the output before the window auto-closes.
    """
    if platform.system() != "Windows":
        return
    try:
        input("\nPress Enter to close this window...")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
