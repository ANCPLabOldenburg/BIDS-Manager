# Installer changelog

The installers have their own version, separate from BIDS Manager's: an
installer puts BIDS Manager on a computer, and BIDS Manager then updates
itself. `INSTALLER_VERSION` in each script, the first heading below and the
installers in `Installers.zip` always agree; `tests/unit/test_installers.py`
checks it.

## 2.0.3 (2026-10-19)

- The installer can uninstall. When BIDS Manager is already installed, the
  menu offers `[x] Uninstall` beside Update, Reinstall and Abort, and asks
  once more (No by default). It removes the install folder and everything
  that starts BIDS Manager: the app on macOS (unregistered from Launch
  Services first), the menu entries, Desktop files and icon on Linux, the
  Desktop and Start-menu shortcuts on Windows. It does the same for an
  installation made by installer 1.x. Datasets and settings are kept.
  `BIDSMGR_UNINSTALL=1` uninstalls without asking; `BIDSMGR_YES=1` alone
  never uninstalls.
- Reinstall and Uninstall wait for BIDS Manager to be closed on macOS and
  Linux too, as Windows already did: removing a program while it runs
  leaves it to fail at random later.
- The macOS uninstaller placed on the Desktop also unregisters the app.

## 2.0.2 (2026-10-16)

- macOS: BIDS-Manager.app starts natively on Apple Silicon. Its launcher
  is a script, and macOS started it under Rosetta; Python still ran
  natively, but every program it started ran as Intel, so tinygrad 0.14's
  `clang -mcpu=native` was refused and skull stripping failed from the app
  while it worked from a Terminal. The app now declares
  `LSRequiresNativeExecution` and `LSArchitecturePriority`. Installer 1.x's
  app had the same flaw; its tinygrad never needed clang.
- Python 3.10 by default again (it was 3.12 in 2.0.0 and 2.0.1), the version
  installer 1.x used and every version pin was tested with: a newer Python
  brings newer libraries, among them tinygrad 0.14 for brainchop.
- The launchers put the environment's own programs first on PATH, as
  installer 1.x's did by activating the environment.

## 2.0.1 (2026-10-16)

- Every call to the new environment's Python uses `-I`. Without it Python
  looks in the current folder first, so an installer started from inside a
  BIDS Manager checkout checked, recorded and took the icon from that copy
  instead of the one it had just installed.
- An update keeps an environment whose Python is newer than the installer's
  default. BIDS Manager's own update can now move an installation to a newer
  Python, and re-running the installer must not move it back to an older
  Python, and so to an older BIDS Manager.
- Works with BIDS Manager's update button, which from the next release
  updates an installer installation through this uv and keeps
  `installation.json` current.

## 2.0.0 (2026-10-16)

The installers are built on uv instead of a portable Python kept in this
repository.

- uv 0.13.1 is downloaded into the install folder and checked against its
  published SHA-256 before it is used. A uv already on the computer is never
  called, and its settings (`uv.toml`, `UV_*` variables) are ignored.
- Python 3.12 comes from that uv, into the install folder. Nothing is added
  to the system's `PATH` or the Windows registry.
- New platforms: Intel Macs, Linux on ARM64, Windows 11 on ARM (through its
  x64 emulation, because dcm2niix has no ARM64 build for Windows).
- No administrator rights anywhere. On Windows the installer no longer
  installs the Visual C++ 2008 runtime that EyeLink `.edf` reading needs
  (that required elevation); it says when the runtime is missing.
- Running the installer again updates BIDS Manager to the newest release.
  Before, it reinstalled the version already there.
- An installation from installer 1.x is converted in place: its portable
  Python 3.10 is removed and the environment is rebuilt on Python 3.12.
- `installation.json` in the install folder records the installer, uv,
  Python and BIDS Manager versions, and how to call that uv.
- The scripts are tracked uncompressed in `Installers/`; `Installers.zip` is
  built from them by `tools/build_installers.py`, without macOS metadata.
- Optional settings for testing and for administrators: `BIDSMGR_INSTALL_DIR`,
  `BIDSMGR_PYTHON`, `BIDSMGR_PACKAGE`, `BIDSMGR_YES`, `BIDSMGR_NO_SHORTCUTS`.

## 1.0.2 (Windows) and 1.0.1 (macOS, Linux)

Download a portable Python 3.10 from `external/python-embed/` in this
repository, create an environment, `pip install bids-manager`, and register a
launcher. Apple Silicon Macs, Linux x86_64 and Windows x64 only. The Windows
installer 1.0.2 added the Visual C++ 2008 runtime for EyeLink, installed with
administrator rights.
