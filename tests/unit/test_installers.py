"""The installers agree with each other, with their changelog and with the zip.

Three scripts in three languages pin the same uv, carry the same version and
must never ask for administrator rights. Nothing but a test notices when one
of them drifts, and the zip on the website is a copy that can silently fall
behind the sources it was built from.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALLERS = ROOT / "Installers"
MACOS = INSTALLERS / "MacOS" / "install_BIDS_Manager.command"
LINUX = INSTALLERS / "Linux" / "install_BIDS_Manager.sh"
WINDOWS = INSTALLERS / "Windows" / "install_BIDS_Manager.bat"
UNIX = [MACOS, LINUX]
ALL = [MACOS, LINUX, WINDOWS]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_installers", ROOT / "tools" / "build_installers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _installer_version(path: Path) -> str:
    pattern = (r"^\$InstallerVersion = '([^']+)'" if path == WINDOWS
               else r'^INSTALLER_VERSION="([^"]+)"')
    return re.search(pattern, _text(path), re.M).group(1)


def _uv_version(path: Path) -> str:
    pattern = r"^\$UvVersion = '([^']+)'" if path == WINDOWS else r'^UV_VERSION="([^"]+)"'
    return re.search(pattern, _text(path), re.M).group(1)


def _uv_checksums(path: Path) -> dict[str, str]:
    """Target triple -> SHA-256, as the script pins them."""
    text = _text(path)
    if path == WINDOWS:
        target = re.search(r"^\$UvTarget = '([^']+)'", text, re.M).group(1)
        return {target: re.search(r"^\$UvSha256 = '([^']+)'", text, re.M).group(1)}
    return {name.replace("_", "-").replace("x86-64", "x86_64"): value
            for name, value in re.findall(r'^UV_SHA256_(\w+)="([^"]+)"', text, re.M)}


def test_the_zip_holds_exactly_the_sources():
    builder = _builder()
    assert builder.contents(builder.ZIP) == builder.expected(), (
        "Installers.zip is out of date: run python tools/build_installers.py")


def test_the_zip_keeps_the_scripts_executable_and_carries_no_mac_metadata():
    names = _builder().contents()
    for name, (_data, mode) in names.items():
        assert "__MACOSX" not in name and "/._" not in name and ".DS_Store" not in name
        if name.endswith((".command", ".sh", ".bat")):
            assert mode == 0o755, name


def test_every_script_carries_the_changelog_version():
    newest = re.search(r"^## (\S+)", _text(INSTALLERS / "CHANGELOG.md"), re.M).group(1)
    assert {_installer_version(path) for path in ALL} == {newest}


def test_every_script_pins_the_same_uv_with_the_same_checksums():
    assert len({_uv_version(path) for path in ALL}) == 1
    pinned: dict[str, str] = {}
    for path in ALL:
        for target, digest in _uv_checksums(path).items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (path.name, target)
            assert pinned.setdefault(target, digest) == digest, target
    assert set(pinned) == {
        "aarch64-apple-darwin", "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
    }


@pytest.mark.parametrize("path", UNIX, ids=lambda p: p.name)
def test_unix_scripts_never_run_anything_as_administrator(path):
    for number, line in enumerate(_text(path).splitlines(), 1):
        code = line.split("#", 1)[0].strip()
        if code.startswith("echo"):
            continue  # advice printed for the user, not a command run
        assert not re.search(r"(^|[|;&(`]\s*)sudo\b", code), f"{path.name}:{number}"
        assert not re.search(r"\b(su|doas|pkexec)\s", code), f"{path.name}:{number}"


def test_the_windows_script_never_asks_for_elevation():
    text = _text(WINDOWS)
    assert not re.search(r"runas", text, re.I)
    assert "Start-Process" not in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_script_isolates_its_uv(path):
    text = _text(path)
    for variable in ("UV_NO_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_MANAGED_PYTHON"):
        assert variable in text, variable
    # Variables set for another uv are removed before this one runs.
    assert "UV_*" in text
    # The Python it downloads is never put on PATH or in the registry.
    assert "--no-bin" in text
    if path == WINDOWS:
        assert "--no-registry" in text
    # An update installs the newest release, not the one already there.
    assert "--upgrade" in text


def test_the_windows_script_is_ascii_with_crlf_line_endings():
    data = WINDOWS.read_bytes()
    data.decode("ascii")
    assert data.count(b"\n") == data.count(b"\r\n")
    # The batch part ends where the PowerShell block comment closes.
    assert data.startswith(b"<# :")


@pytest.mark.parametrize("path", UNIX, ids=lambda p: p.name)
def test_unix_scripts_use_lf_line_endings(path):
    assert b"\r" not in path.read_bytes()


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_call_to_the_new_environments_python_is_isolated(path):
    """Without -I, Python looks in the current folder first: an installer
    started from inside a BIDS Manager checkout checked, recorded and took
    the icon from that copy instead of the one it had just installed
    (2026-10-16, it recorded 1.5.0 over a fresh 1.4.1)."""
    text = _text(path)
    if path == WINDOWS:
        calls = re.findall(r"& \$EnvPy (\S+)", text) + re.findall(
            r"Invoke-Program \$EnvPy @\(([^,)]+)", text)
        assert calls and all(call in ("-I", "'-I'") for call in calls), calls
    else:
        calls = re.findall(r'"\$ENV_PY" (-\S*)', text)
        # "--python "$ENV_PY" --upgrade" is an argument to uv, not a call.
        calls = [call for call in calls if not call.startswith("--")]
        assert calls and all(call == "-I" for call in calls), calls


@pytest.mark.parametrize(
    ("have", "want", "kept"),
    [("3.12", "3.12", True), ("3.13", "3.12", True), ("3.11", "3.12", False),
     ("4.0", "3.12", True), ("3.12", "3.12.4", True), ("3.9", "3.12", False)],
)
def test_an_update_keeps_an_environment_on_the_requested_python_or_newer(have, want, kept):
    """BIDS Manager's own update can move an installation to a newer Python;
    re-running the installer must not move it back, which would install an
    older BIDS Manager."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash to run the installer's own comparison")
    text = _text(MACOS)
    function = re.search(r"^version_at_least\(\) \{.*?^\}", text, re.M | re.S).group(0)
    assert function in _text(LINUX), "the two Unix installers must compare alike"
    rc = subprocess.run([bash, "-c", f"{function}\nversion_at_least {have} {want}"]).returncode
    assert (rc == 0) is kept


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_installer_uses_python_3_10_by_default(path):
    """The version installer 1.x used and every pin was tested with. On 3.12,
    brainchop got tinygrad 0.14, and skull stripping failed from the app."""
    pattern = (r"^\$PythonVersion = Setting 'BIDSMGR_PYTHON' '([^']+)'" if path == WINDOWS
               else r'^PYTHON_VERSION="\$\{BIDSMGR_PYTHON:-([^}]+)\}"')
    assert re.search(pattern, _text(path), re.M).group(1) == "3.10"


def test_the_mac_app_starts_natively_on_apple_silicon():
    """A script launcher was started under Rosetta, so every program Python
    started ran as Intel and tinygrad's clang refused -mcpu=native."""
    text = _text(MACOS)
    assert "<key>LSRequiresNativeExecution</key>  <true/>" in text
    assert "<array><string>arm64</string></array>" in text
    assert "${ARCH_KEYS}" in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_launcher_puts_the_environments_programs_first(path):
    """As installer 1.x did by activating the environment."""
    text = _text(path)
    if path == WINDOWS:
        assert "('set \"PATH=' + (Batch-Path (Join-Path $EnvDir 'Scripts')) + ';%PATH%\"')" in text
    else:
        assert 'export PATH="$ENV_DIR/bin:\\$PATH"' in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_only_a_python_the_update_button_moved_to_is_kept_newer(path):
    """Installer 2.0.1 kept any newer Python, so a 3.12 installation could
    never return to the default. Only the update button's mark keeps one,
    and the installer carries the mark over when it rewrites the record."""
    text = _text(path)
    assert "moved_by_update" in text
    assert 'BM_KEEP_MOVE' in text
    assert 'record["python"]["moved_by_update"] = True' in text


def test_the_update_helper_marks_a_move_the_installers_look_for():
    helper = (ROOT / "bidsmgr" / "gui" / "_update_helper.py").read_text(encoding="utf-8")
    assert 'record["python"]["moved_by_update"] = True' in helper


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_every_installer_offers_to_uninstall_and_asks_first(path):
    text = _text(path)
    assert "[x] Uninstall: remove BIDS Manager from this computer" in text
    assert "BIDSMGR_UNINSTALL" in text
    # The confirmation defaults to No, so BIDSMGR_YES=1 (take every
    # default) can never uninstall anything.
    if path == WINDOWS:
        assert "$confirm = Ask '  Remove it? [y/N]' 'n'" in text
    else:
        assert 'confirm="$(ask "  Remove it? [y/N] " n)"' in text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_uninstalling_removes_every_launcher_the_installer_creates(path):
    """Whatever starts BIDS Manager must not outlive it: a launcher left
    behind points at a folder that is gone."""
    text = _text(path)
    block = text[text.index("Uninstall: everything this installer"):]
    if path == MACOS:
        for removed in ('rm -rf "$APP_BUNDLE"', "-u \"$APP_BUNDLE\"",
                        'rm -f "$DESKTOP_DIR/Uninstall_BIDS_Manager.command"'):
            assert removed in block, removed
    elif path == LINUX:
        for removed in ('"$APPS_DIR" "$DESKTOP_DIR" "$HOME/Desktop"', "BIDS-Manager.desktop",
                        "Uninstall_BIDS_Manager.desktop", 'rm -f "$ICON_FILE"'):
            assert removed in block, removed
    else:
        for removed in ("'BIDS-Manager.lnk'", "'Uninstall BIDS-Manager.lnk'",
                        "Remove-Item -LiteralPath $StartMenu -Recurse"):
            assert removed in block, removed
