# BIDS Manager installers

One file per system. Each installs BIDS Manager for the current user, in a
folder of its own, without administrator rights. Running it again updates
BIDS Manager to the newest release.

| System | File | How to run it |
|---|---|---|
| macOS 11 or newer, Apple Silicon or Intel | `MacOS/install_BIDS_Manager.command` | Double-click it. If macOS refuses a file from the internet: right-click it, Open, Open. |
| Linux, x86_64 or ARM64 (Ubuntu, Debian, Fedora...) | `Linux/install_BIDS_Manager.sh` | `bash install_BIDS_Manager.sh` |
| Windows 10 or 11 (x64), Windows 11 on ARM | `Windows/install_BIDS_Manager.bat` | Double-click it. If Windows says it protected your PC: More info, Run anyway. |

## What it installs

Everything goes into `~/BIDS-Manager` (`%USERPROFILE%\BIDS-Manager` on
Windows):

```
BIDS-Manager/
  uv/                 uv, checked against its published SHA-256
  python/             Python 3.10, downloaded by that uv
  env/                the environment BIDS Manager runs in
  cache/              uv's download cache, which makes updates quick
  installation.json   what is installed, with which versions
  install.log         what the last run did
  uninstall_BIDS_Manager.*
```

plus a way to start it: `BIDS-Manager.app` in `~/Applications` on macOS, a
menu entry and a Desktop file on Linux, shortcuts on the Desktop and in the
Start menu on Windows.

The installer uses only its own uv. A uv already on the computer is never
called, and the user's uv settings (`uv.toml`, `UV_*` variables) do not
reach it. The Python it downloads is not added to `PATH` or to the Windows
registry, so other programs never see it.

## Updating

BIDS Manager's Check updates button, or running the installer again. From
the release after BIDS Manager 1.5.0, the button updates through the
installer's own uv, so an update leaves nothing outside the install folder;
and when a new release needs a newer Python, it moves the installation to
it, putting the previous one back if anything goes wrong. (1.5.0's button
updates with pip, which works but keeps its download cache outside.)

## Removing it

Run the installer again and choose `[x] Uninstall`, or run
`uninstall_BIDS_Manager` (on the Desktop, in the menu, or in the install
folder). Either removes the install folder and the launchers. Datasets and
BIDS Manager's settings are not touched.

## Options

For testing and for administrators. Environment variables, all optional:

| Variable | Effect |
|---|---|
| `BIDSMGR_INSTALL_DIR` | Install somewhere else (an absolute path to a folder of its own) |
| `BIDSMGR_PYTHON` | Another Python version (default `3.10`) |
| `BIDSMGR_PACKAGE` | What to install: `bids-manager==1.5.0`, a wheel file, ... |
| `BIDSMGR_YES=1` | Take the default answer to every question |
| `BIDSMGR_NO_SHORTCUTS=1` | No app, menu entry or shortcuts |
| `BIDSMGR_UNINSTALL=1` | Remove BIDS Manager without asking (`BIDSMGR_YES=1` alone never does) |
| `BIDSMGR_TARGET` | macOS and Linux: force a build, for example `x86_64-apple-darwin` on an Apple Silicon Mac (it runs through Rosetta) |

A test installation that leaves the real one alone, on macOS or Linux:

```bash
BIDSMGR_INSTALL_DIR=/tmp/bm-test BIDSMGR_YES=1 BIDSMGR_NO_SHORTCUTS=1 \
    bash install_BIDS_Manager.command
```

On Windows, in a Command Prompt:

```bat
set BIDSMGR_INSTALL_DIR=%TEMP%\bm-test
set BIDSMGR_YES=1
set BIDSMGR_NO_SHORTCUTS=1
install_BIDS_Manager.bat
```

## For maintainers

- The scripts in this folder are the source. `Installers.zip` is built from
  them with `python tools/build_installers.py`, and
  `tests/unit/test_installers.py` fails when the zip is out of date or the
  scripts disagree with each other.
- A change: edit the scripts, raise `INSTALLER_VERSION` in all three, add a
  `CHANGELOG.md` entry, rebuild the zip.
- A new uv: change `UV_VERSION` and the SHA-256 of each archive, published
  next to it as `https://github.com/astral-sh/uv/releases/download/<version>/<archive>.sha256`.
- The Windows file is ASCII with CRLF line endings, the others use LF;
  `.gitattributes` keeps them so on every checkout.
