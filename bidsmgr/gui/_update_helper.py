"""Detached self-update helper. Runs *after* the GUI has exited.

This script is copied to a temp directory by ``updater.launch_update_helper``
before it is spawned, so it does not depend on the installed
``bidsmgr.gui`` package files staying intact during the pip install
(which is exactly what would be a problem on Windows).

Lifecycle::

    1. wait for --parent-pid to terminate
    2. pip install --upgrade <package>
    3. optionally re-launch the GUI via --restart-cmd

Output is whatever ``pip`` prints. On Windows the helper runs in its
own console window so the user sees pip's progress; on POSIX the
launcher redirects it to ``~/.bidsmgr/update.log``.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time


PARENT_WAIT_TIMEOUT_S = 90.0  # how long we'll wait for the GUI to exit
PARENT_POLL_INTERVAL_S = 0.5


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
    WAIT_TIMEOUT = 0x102

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        # Couldn't open it. Either it's already gone, or we lack
        # permission. Either way pip won't be blocked by it.
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
    parser.add_argument("--restart-cmd", default="",
                        help="GUI entry point to relaunch after upgrade. "
                             "Empty disables restart.")
    args = parser.parse_args()

    print(f"[update] waiting for parent pid {args.parent_pid} to exit...",
          flush=True)
    if not wait_for_parent(args.parent_pid):
        print(
            f"[update] timed out after {PARENT_WAIT_TIMEOUT_S:.0f}s; "
            "the GUI is still running, aborting pip install.",
            flush=True,
        )
        _pause_console_on_windows()
        return 2
    # A small grace period: even after the process exits, Windows may
    # take a moment to release every file handle the OS held on its
    # behalf.
    time.sleep(0.5)

    rc = run_pip_upgrade(args.python, args.package)
    if rc != 0:
        print(f"[update] pip exited with code {rc}.", flush=True)
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
