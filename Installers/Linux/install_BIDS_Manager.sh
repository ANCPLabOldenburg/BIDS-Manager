#!/usr/bin/env bash
# =======================================================================
# BIDS Manager installer for Linux (x86_64 and ARM64, glibc)
# Installer version 2.0.3. History: Installers/CHANGELOG.md
# Licence: MIT, ANCP Lab, University of Oldenburg
# https://github.com/ANCPLabOldenburg/BIDS-Manager
#
# Run it with:  bash install_BIDS_Manager.sh
#
# Without administrator rights, and entirely inside ~/BIDS-Manager, it
# installs:
#
#   uv/        a pinned copy of uv, checked against its published SHA-256.
#              A uv you already have is never used, nor its settings.
#   python/    the Python version below, downloaded by that uv
#   env/       an environment with bids-manager from PyPI
#
# and adds BIDS Manager to the application menu. Running it again updates
# BIDS Manager to the newest release, or uninstalls it.
#
# Optional settings, for testing and for administrators:
#   BIDSMGR_INSTALL_DIR     where to install (default ~/BIDS-Manager)
#   BIDSMGR_PYTHON          Python version (default 3.10)
#   BIDSMGR_PACKAGE         what to install: "bids-manager==1.5.0", a wheel file...
#   BIDSMGR_YES=1           take the default answer to every question
#   BIDSMGR_NO_SHORTCUTS=1  no menu entry, no Desktop files, no icon
#   BIDSMGR_TARGET          force a build: x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu
#   BIDSMGR_UNINSTALL=1     remove BIDS Manager, without asking
# =======================================================================
set -euo pipefail

INSTALLER_VERSION="2.0.3"
UV_VERSION="0.13.1"
UV_SHA256_x86_64_unknown_linux_gnu="eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220"
UV_SHA256_aarch64_unknown_linux_gnu="9041c73c99fdc5892cf128140beb354eca9231d502068cc36e6cb53a8e59ad14"

# 3.10 is what installer 1.x used and what every version pin was tested on.
# A newer Python brings newer libraries: on 3.11 and later brainchop gets
# tinygrad 0.12 or newer, untested here (BIDS Manager pins 0.11.0 from its
# next release). The update button moves an installation to a newer Python
# when a release needs one.
PYTHON_VERSION="${BIDSMGR_PYTHON:-3.10}"
PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f1-2)"
PACKAGE="${BIDSMGR_PACKAGE:-bids-manager}"
INSTALL_DIR="${BIDSMGR_INSTALL_DIR:-$HOME/BIDS-Manager}"
ASSUME_YES="${BIDSMGR_YES:-0}"
NO_SHORTCUTS="${BIDSMGR_NO_SHORTCUTS:-0}"

# The folder is removed whole on reinstall and uninstall, so it must be one
# of its own: never empty, relative, the home folder or the disk.
INSTALL_DIR="${INSTALL_DIR%/}"
case "$INSTALL_DIR" in
    /*/*) ;;
    *) echo "BIDSMGR_INSTALL_DIR must be an absolute path to a folder of its own."; exit 1 ;;
esac
if [[ "$INSTALL_DIR" == "${HOME%/}" ]]; then
    echo "BIDS Manager needs a folder of its own, not the home folder itself."; exit 1
fi

UV_DIR="$INSTALL_DIR/uv"
UV="$UV_DIR/uv"
ENV_DIR="$INSTALL_DIR/env"
ENV_PY="$ENV_DIR/bin/python"
MANIFEST="$INSTALL_DIR/installation.json"
LOG_FILE="$INSTALL_DIR/install.log"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
HICOLOR_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
ICON_DIR="$HICOLOR_DIR/256x256/apps"
ICON_FILE="$ICON_DIR/bids-manager.png"
MIN_DISK_MB=2500

# -- Output ---------------------------------------------------------------
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi
log_info()  { echo -e "${GREEN}[ok]${RESET} $*"; }
log_step()  { echo -e "${CYAN}${BOLD}[..]${RESET} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
log_error() { echo -e "${RED}[x]${RESET} $*" >&2; }
die()       { log_error "$*"; echo "    Log file: $LOG_FILE"; exit 1; }

# ask QUESTION DEFAULT: the answer, or DEFAULT when nobody can answer.
ask() {
    local reply=""
    if [[ "$ASSUME_YES" == "1" || ! -t 0 ]]; then
        echo "$2"
        return
    fi
    read -rp "$1" reply
    echo "${reply:-$2}"
}

# download URL FILE, with curl or wget, whichever this system has.
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 -O "$2" "$1"
    else
        return 1
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# version_at_least A B: A is B or newer, comparing major.minor ("3.13" "3.12").
version_at_least() {
    local a_major="${1%%.*}" b_major="${2%%.*}" a_rest="${1#*.}" b_rest="${2#*.}"
    local a_minor="${a_rest%%.*}" b_minor="${b_rest%%.*}"
    (( a_major > b_major || (a_major == b_major && a_minor >= b_minor) ))
}

# -- Isolation --------------------------------------------------------------
# Only this installer's uv, with only this installer's settings: variables
# set for another uv (an index, a cache, a Python preference) and any active
# environment must not reach it.
for name in $(compgen -e); do
    if [[ "$name" == UV_* ]]; then unset "$name"; fi
done
unset VIRTUAL_ENV CONDA_PREFIX PYTHONHOME PYTHONPATH 2>/dev/null || true
export UV_NO_CONFIG=1
export UV_CACHE_DIR="$INSTALL_DIR/cache"
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=automatic

# -- Platform ---------------------------------------------------------------
if [[ -n "${BIDSMGR_TARGET:-}" ]]; then
    TARGET="$BIDSMGR_TARGET"
else
    case "$(uname -m)" in
        x86_64|amd64)  TARGET="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) TARGET="aarch64-unknown-linux-gnu" ;;
        *) echo "This processor ($(uname -m)) is not supported: BIDS Manager needs x86_64 or ARM64."; exit 1 ;;
    esac
    if ldd --version 2>&1 | grep -qi musl || ls /lib/ld-musl-* >/dev/null 2>&1; then
        echo "This Linux uses the musl C library (Alpine, for example). The window"
        echo "toolkit BIDS Manager uses, PyQt6, is published only for Linux systems"
        echo "built on glibc (Ubuntu, Debian, Fedora, openSUSE, Arch...)."
        exit 1
    fi
fi
case "$TARGET" in
    x86_64-unknown-linux-gnu)  UV_SHA256="$UV_SHA256_x86_64_unknown_linux_gnu";  ARCH="x86_64" ;;
    aarch64-unknown-linux-gnu) UV_SHA256="$UV_SHA256_aarch64_unknown_linux_gnu"; ARCH="aarch64" ;;
    *) echo "Unknown build: $TARGET"; exit 1 ;;
esac
# The architecture is part of the request, so the Python and every package
# match the build chosen above.
PYTHON_REQUEST="cpython-${PYTHON_VERSION}-linux-${ARCH}-gnu"

for tool in tar gzip; do
    command -v "$tool" >/dev/null 2>&1 || { echo "This installer needs '$tool'."; exit 1; }
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "This installer needs curl or wget."; exit 1
fi

# -- Banner -----------------------------------------------------------------
clear 2>/dev/null || true
echo -e "${BOLD}"
cat << 'BANNER'

    +--------------------------------------------------+
    |                                                  |
    |          B I D S   M a n a g e r                 |
    |                                                  |
    |    Schema-driven BIDS converter and editor       |
    |        ANCP Lab, University of Oldenburg         |
    |                                                  |
    +--------------------------------------------------+

BANNER
echo -e "${RESET}"
echo -e "  Installer version:  ${BOLD}${INSTALLER_VERSION}${RESET}"
echo -e "  Platform:           Linux (${ARCH})"
echo -e "  Python:             ${PYTHON_VERSION}"
echo -e "  Install folder:     ${BOLD}${INSTALL_DIR}${RESET}"
echo ""

# -- Before anything is written ------------------------------------------------
mkdir -p "$(dirname "$INSTALL_DIR")"
AVAIL_MB=$(df -Pm "$(dirname "$INSTALL_DIR")" | awk 'NR==2 {print $4}')
if (( AVAIL_MB < MIN_DISK_MB )); then
    log_warn "Only ${AVAIL_MB} MB free; about ${MIN_DISK_MB} MB are needed."
fi
if ! download https://pypi.org/simple/pip/ /dev/null 2>/dev/null; then
    log_warn "pypi.org cannot be reached. Check the internet connection."
fi

MODE="install"
if [[ "${BIDSMGR_UNINSTALL:-0}" == "1" ]]; then
    MODE="uninstall"
elif [[ -f "$MANIFEST" || -d "$ENV_DIR" ]]; then
    echo -e "${YELLOW}  BIDS Manager is already installed in:${RESET}"
    echo "    $INSTALL_DIR"
    echo ""
    echo "  [u] Update: install the newest BIDS Manager (default)"
    echo "  [r] Reinstall: remove this installation and start again"
    echo "  [x] Uninstall: remove BIDS Manager from this computer"
    echo "  [a] Abort"
    echo ""
    while true; do
        choice="$(ask "  Your choice [U/r/x/a]: " u)"
        case "$choice" in
            [uU]) MODE="update"; break ;;
            [rR]) MODE="reinstall"; break ;;
            [xX]) MODE="uninstall"; break ;;
            [aA]) echo "  Nothing was changed."; exit 0 ;;
            *)    echo "  Please type u, r, x or a." ;;
        esac
    done
else
    echo -e "${BOLD}  The installer will:${RESET}"
    echo "    1. Download uv ${UV_VERSION} (about 20 MB) into the install folder"
    echo "    2. Download Python ${PYTHON_VERSION} with it (about 30 MB)"
    echo "    3. Install bids-manager and its dependencies (about 1.5 GB)"
    echo "    4. Add BIDS Manager to the application menu"
    echo "    No administrator password is needed."
    echo ""
    confirm="$(ask "  Continue? [Y/n] " y)"
    if [[ "$confirm" == [nN]* ]]; then
        echo "  Nothing was changed."
        exit 0
    fi
fi

# Removing a program while it runs would leave it to fail at random later.
if [[ "$MODE" == "reinstall" || "$MODE" == "uninstall" ]] && command -v pgrep >/dev/null 2>&1 \
    && pgrep -f "$ENV_DIR/" >/dev/null 2>&1; then
    echo "  BIDS Manager is running. Close it, then run the installer again."
    exit 1
fi

# Uninstall: everything this installer (or installer 1.x) put on the
# computer. Datasets and BIDS Manager's settings are not touched.
if [[ "$MODE" == "uninstall" ]]; then
    echo ""
    echo "  This removes BIDS Manager completely:"
    echo "    $INSTALL_DIR"
    echo "    and its menu entries, Desktop files and icon"
    echo "  Your datasets and your settings are not touched."
    if [[ "${BIDSMGR_UNINSTALL:-0}" != "1" ]]; then
        confirm="$(ask "  Remove it? [y/N] " n)"
        if [[ "$confirm" != [yY]* ]]; then
            echo "  Nothing was removed."
            exit 0
        fi
    fi
    for dir in "$APPS_DIR" "$DESKTOP_DIR" "$HOME/Desktop"; do
        rm -f "$dir/BIDS-Manager.desktop" "$dir/Uninstall_BIDS_Manager.desktop"
    done
    rm -f "$ICON_FILE"
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
    gtk-update-icon-cache "$HICOLOR_DIR" 2>/dev/null || true
    rm -rf "${INSTALL_DIR:?}"
    log_info "BIDS Manager has been removed."
    exit 0
fi

if [[ "$MODE" == "reinstall" ]]; then
    log_step "Removing the previous installation..."
    rm -rf "${INSTALL_DIR:?}"
fi

mkdir -p "$INSTALL_DIR"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "BIDS Manager installer $INSTALLER_VERSION ($MODE) started $(date)"
echo "Build $TARGET, Python $PYTHON_REQUEST, package $PACKAGE"
echo ""

TMP="$(mktemp -d "${TMPDIR:-/tmp}/bidsmgr-install.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# run_uv ARGS...: once as is, then once more with the system's certificates,
# which networks that inspect HTTPS (many universities) need.
run_uv() {
    if "$UV" "$@"; then
        return 0
    fi
    log_warn "Trying again with the system's certificates..."
    UV_SYSTEM_CERTS=1 "$UV" "$@"
}

# =======================================================================
# Step 1/5: uv
# =======================================================================
if [[ -x "$UV" && "$("$UV" --version 2>/dev/null | awk '{print $2}')" == "$UV_VERSION" ]]; then
    log_info "Step 1/5: uv ${UV_VERSION} is already here."
else
    log_step "Step 1/5: Downloading uv ${UV_VERSION}..."
    url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${TARGET}.tar.gz"
    download "$url" "$TMP/uv.tar.gz" || die "uv could not be downloaded from $url"
    if [[ "$(sha256_of "$TMP/uv.tar.gz")" != "$UV_SHA256" ]]; then
        die "The uv download does not match its published checksum, so it was not used."
    fi
    tar -xzf "$TMP/uv.tar.gz" -C "$TMP" || die "The uv archive could not be unpacked."
    mkdir -p "$UV_DIR"
    rm -f "$UV"
    mv "$TMP/uv-${TARGET}/uv" "$UV"
    chmod +x "$UV"
    "$UV" --version >/dev/null || die "uv does not run on this computer."
    log_info "uv ${UV_VERSION} installed in $UV_DIR"
fi

# =======================================================================
# Step 2/5: Python
# =======================================================================
log_step "Step 2/5: Python ${PYTHON_VERSION}..."
run_uv python install "$PYTHON_REQUEST" --no-bin \
    || die "Python ${PYTHON_VERSION} could not be installed."
log_info "Python ready: $("$UV" python find "$PYTHON_REQUEST")"

# =======================================================================
# Step 3/5: the environment
# =======================================================================
# Every call to the environment's Python uses -I: without it, Python looks
# in the current folder first, and run from inside a BIDS Manager checkout
# it would read that copy instead of the one just installed.
#
# Kept on update when it runs the requested Python for this build, or a
# newer one the update button moved it to because a release needed it
# (the record says so: moving it back would install an older BIDS
# Manager); built
# again otherwise (an installation from installer 1.x, another Python, a
# disk moved to another machine, a broken environment).
env_python="$("$ENV_PY" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
same_build=0
moved=0
if [[ -f "$MANIFEST" ]] && grep -q "\"target\": \"$TARGET\"" "$MANIFEST"; then
    same_build=1
    if grep -q '"moved_by_update": true' "$MANIFEST"; then moved=1; fi
fi
KEEP_MOVE=0
if [[ "$MODE" == "update" && "$same_build" == "1" && -n "$env_python" ]] \
    && { [[ "$env_python" == "$PYTHON_MINOR" ]] \
         || { [[ "$moved" == "1" ]] && version_at_least "$env_python" "$PYTHON_MINOR"; }; }; then
    if [[ "$env_python" != "$PYTHON_MINOR" ]]; then KEEP_MOVE=1; fi
    log_info "Step 3/5: The environment is kept (Python $env_python)."
else
    log_step "Step 3/5: Creating the environment..."
    # Installer 1.x unpacked a portable Python 3.10 into the install folder
    # itself; nothing uses it any more.
    if [[ -x "$INSTALL_DIR/bin/python3.10" ]]; then
        rm -rf "${INSTALL_DIR:?}/bin" "${INSTALL_DIR:?}/include" "${INSTALL_DIR:?}/lib" "${INSTALL_DIR:?}/share"
    fi
    # --seed adds pip, which the in-app updater of BIDS Manager 1.5 uses.
    "$UV" venv --clear --seed --prompt BIDS-Manager --python "$PYTHON_REQUEST" "$ENV_DIR" \
        || die "The environment could not be created."
    log_info "Environment ready in $ENV_DIR"
fi

# Pythons no environment uses any more (an earlier default, a move made by
# the update button), about 70 MB each.
in_use="$("$ENV_PY" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
for dir in "$UV_PYTHON_INSTALL_DIR"/cpython-*; do
    name="$(basename "$dir")"
    if [[ -n "$in_use" && "$name" =~ ^cpython-([0-9]+\.[0-9]+)\.[0-9]+[^-]*-(.+)$ \
          && "${BASH_REMATCH[1]}" != "$in_use" ]]; then
        "$UV" python uninstall "cpython-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}" >/dev/null 2>&1 \
            && log_info "Removed Python ${BASH_REMATCH[1]}, which nothing uses any more."
    fi
done

# =======================================================================
# Step 4/5: BIDS Manager
# =======================================================================
log_step "Step 4/5: Installing ${PACKAGE} (a few minutes the first time)..."
# --upgrade: an update installs the newest release, not the one already here.
# --compile-bytecode: what pip does, and what keeps the first start quick.
run_uv pip install --python "$ENV_PY" --upgrade --compile-bytecode "$PACKAGE" \
    || die "bids-manager could not be installed."
"$ENV_PY" -I -c "import bidsmgr, PyQt6.QtWidgets" \
    || die "bids-manager was installed but does not start."
BM_VERSION="$("$ENV_PY" -I -c 'from importlib.metadata import version; print(version("bids-manager"))')"
"$UV" cache prune >/dev/null 2>&1 || true
log_info "BIDS Manager ${BM_VERSION} installed."

# The record of this installation. BIDS Manager reads it to find the uv and
# the Python it was installed with.
BM_MANIFEST="$MANIFEST" BM_INSTALLER="$INSTALLER_VERSION" BM_TARGET="$TARGET" \
BM_UV="$UV" BM_UV_VERSION="$UV_VERSION" BM_PYTHON="$PYTHON_VERSION" \
BM_ENV="$ENV_DIR" BM_PACKAGE="$PACKAGE" BM_VERSION="$BM_VERSION" BM_KEEP_MOVE="$KEEP_MOVE" \
"$ENV_PY" -I - << 'PYEOF' || log_warn "installation.json could not be written."
import datetime, json, os, platform, sys
e = os.environ
path = e["BM_MANIFEST"]
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
try:
    with open(path, encoding="utf-8") as fh:
        previous = json.load(fh)
except (OSError, ValueError):
    previous = {}
first = previous.get("installed", now)
record = {
    "format": 1,
    "installed": first,
    "updated": now,
    "installer": {"version": e["BM_INSTALLER"], "os": "linux", "target": e["BM_TARGET"]},
    "uv": {"version": e["BM_UV_VERSION"], "executable": e["BM_UV"]},
    "uv_environment": {k: e[k] for k in (
        "UV_NO_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_MANAGED_PYTHON")},
    "python": {"requested": e["BM_PYTHON"], "version": platform.python_version(),
               "executable": sys.executable},
    "environment": e["BM_ENV"],
    "package": {"name": "bids-manager", "requested": e["BM_PACKAGE"],
                "version": e["BM_VERSION"]},
}
if e.get("BM_KEEP_MOVE") == "1":
    # A Python the update button moved to stays recorded as such.
    record["python"]["requested"] = previous["python"]["requested"]
    record["python"]["moved_by_update"] = True
with open(path, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2)
    fh.write("\n")
PYEOF

# Qt brings its own libraries but opens its window through the system's X11
# and EGL libraries. Name any that are missing: installing them is for the
# administrator, so this installer only says which.
QT_DIR="$("$ENV_PY" -I -c 'import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6"))' 2>/dev/null || true)"
if [[ -n "$QT_DIR" ]] && command -v ldd >/dev/null 2>&1; then
    missing=""
    for lib in "$QT_DIR/plugins/platforms/libqxcb.so" "$QT_DIR"/lib/libQt6XcbQpa.so.*; do
        [[ -f "$lib" ]] || continue
        missing="$missing $(LD_LIBRARY_PATH="$QT_DIR/lib" ldd "$lib" 2>/dev/null | awk '/not found/ {print $1}')"
    done
    missing="$(echo "$missing" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')"
    if [[ -n "$missing" ]]; then
        log_warn "The window cannot open until these system libraries are installed:"
        echo "      $missing"
        echo "    They need an administrator. On Ubuntu and Debian, for example:"
        echo "      sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1"
        echo "    On Fedora: sudo dnf install xcb-util-cursor libxkbcommon-x11"
        echo "    The command-line tools (bidsmgr-scan, bidsmgr-convert...) work without them."
    fi
fi

# =======================================================================
# Step 5/5: launcher, menu entry and uninstaller
# =======================================================================
RUN_SCRIPT="$INSTALL_DIR/run_BIDS_Manager.sh"
# The environment's bin folder goes first on PATH, as activating it would,
# so the programs installed with BIDS Manager are the ones found.
cat > "$RUN_SCRIPT" << LAUNCHER_EOF
#!/usr/bin/env bash
export PATH="$ENV_DIR/bin:\$PATH"
exec "$ENV_DIR/bin/bidsmgr" "\$@"
LAUNCHER_EOF
chmod +x "$RUN_SCRIPT"

UNINSTALL_SCRIPT="$INSTALL_DIR/uninstall_BIDS_Manager.sh"
cat > "$UNINSTALL_SCRIPT" << UNINSTALLER_EOF
#!/usr/bin/env bash
echo ""
echo "  BIDS Manager uninstaller"
echo ""
echo "  This removes BIDS Manager completely:"
echo "    $INSTALL_DIR"
echo "  Your datasets and your settings are not touched."
echo ""
read -rp "  Remove it? [y/N] " confirm
if [[ "\$confirm" != [yY]* ]]; then
    echo "  Nothing was removed."
    exit 0
fi
rm -f "$DESKTOP_DIR/BIDS-Manager.desktop" "$DESKTOP_DIR/Uninstall_BIDS_Manager.desktop"
rm -f "$APPS_DIR/BIDS-Manager.desktop" "$APPS_DIR/Uninstall_BIDS_Manager.desktop"
rm -f "$ICON_FILE"
gtk-update-icon-cache "$HICOLOR_DIR" 2>/dev/null || true
rm -rf "$INSTALL_DIR"
echo ""
echo "  BIDS Manager has been removed."
echo ""
UNINSTALLER_EOF
chmod +x "$UNINSTALL_SCRIPT"

make_desktop_entry() {
    local dest="$1" name="$2" comment="$3" exec_path="$4" icon="$5"
    cat > "$dest" << DEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$name
Comment=$comment
Exec="$exec_path"
Terminal=true
Icon=$icon
Categories=Science;Education;
DEOF
    chmod +x "$dest"
}

if [[ "$NO_SHORTCUTS" == "1" ]]; then
    log_info "Step 5/5: No menu entry or shortcuts (BIDSMGR_NO_SHORTCUTS=1)."
else
    log_step "Step 5/5: Adding BIDS Manager to the application menu..."
    DESKTOP_ICON="utilities-terminal"
    PKG_DIR="$("$ENV_PY" -I -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("bidsmgr").origin))' 2>/dev/null || true)"
    if [[ -n "$PKG_DIR" && -f "$PKG_DIR/gui/assets/macos/AppIcon256.png" ]]; then
        mkdir -p "$ICON_DIR"
        cp "$PKG_DIR/gui/assets/macos/AppIcon256.png" "$ICON_FILE"
        gtk-update-icon-cache --force --ignore-theme-index "$HICOLOR_DIR" 2>/dev/null || true
        DESKTOP_ICON="bids-manager"
    else
        log_warn "The app icon was not found; the menu entry shows a default icon."
    fi

    mkdir -p "$APPS_DIR"
    make_desktop_entry "$APPS_DIR/BIDS-Manager.desktop" \
        "BIDS Manager" "Schema-driven BIDS converter and editor" "$RUN_SCRIPT" "$DESKTOP_ICON"
    make_desktop_entry "$APPS_DIR/Uninstall_BIDS_Manager.desktop" \
        "Uninstall BIDS Manager" "Remove BIDS Manager" "$UNINSTALL_SCRIPT" "edit-delete"
    update-desktop-database "$APPS_DIR" 2>/dev/null || true

    if [[ -d "$DESKTOP_DIR" ]]; then
        cp "$APPS_DIR/BIDS-Manager.desktop" "$APPS_DIR/Uninstall_BIDS_Manager.desktop" "$DESKTOP_DIR/"
        gio set "$DESKTOP_DIR/BIDS-Manager.desktop" metadata::trusted true 2>/dev/null || true
        gio set "$DESKTOP_DIR/Uninstall_BIDS_Manager.desktop" metadata::trusted true 2>/dev/null || true
    fi
    log_info "BIDS Manager is in the application menu."
fi

# =======================================================================
# Done
# =======================================================================
echo ""
echo -e "${BOLD}=======================================================${RESET}"
echo -e "${GREEN}${BOLD}  BIDS Manager ${BM_VERSION} is installed.${RESET}"
echo -e "${BOLD}=======================================================${RESET}"
echo ""
echo -e "  ${BOLD}Start it:${RESET}"
echo -e "    from the application menu (${CYAN}BIDS Manager${RESET}), the Desktop,"
echo -e "    or with: $RUN_SCRIPT"
echo ""
echo -e "  ${BOLD}Update:${RESET}     run this installer again"
echo -e "  ${BOLD}Uninstall:${RESET}  Uninstall BIDS Manager, in the menu or on the Desktop"
echo -e "  ${BOLD}Log:${RESET}        $LOG_FILE"
echo ""
