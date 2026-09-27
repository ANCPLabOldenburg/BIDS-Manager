"""Falling back to the vendored dcm2niix when Windows kills the released one.

The released Windows dcm2niix cannot convert MR spectroscopy: it exhausts its
16,388,608-byte stack reserve and Windows terminates it with ``0xC00000FD`` and
an EMPTY stderr, so nothing is written and nothing is said. Every layer above
reads that as "this folder holds no DICOM images".

The retry is deliberately narrow. The wheel's binary is always tried first and
its answer always preferred, because it is the pinned build with the JPEG 2000
and JPEG-LS decoders; the vendored one is a narrow GCC build with those off.
Only the one exit code that means "killed before it could do anything" earns a
second attempt, so a genuine conversion failure is still reported as itself.

See bidsmgr/vendor/dcm2niix_win/PROVENANCE.md.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from bidsmgr.classifier import dcm2niix_bidsguess as bg
from bidsmgr.classifier.dcm2niix_bidsguess import (
    STACK_OVERFLOW_RC,
    run_dcm2niix,
    vendored_dcm2niix,
)


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def _recorder(codes):
    """A subprocess.run stand-in returning ``codes`` in order, recording argv."""
    calls: list[list[str]] = []
    seq = iter(codes)

    def fake(cmd, **kwargs):
        calls.append(list(cmd))
        return _Result(next(seq))

    return fake, calls


# ---------------------------------------------------------------------------
# When the fallback must NOT be reached
# ---------------------------------------------------------------------------


def test_success_is_returned_untouched(monkeypatch) -> None:
    fake, calls = _recorder([0])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    assert run_dcm2niix(["dcm2niix.exe", "-o", "out", "in"]).returncode == 0
    assert len(calls) == 1, "a successful run must not be repeated"


def test_an_ordinary_failure_is_not_retried(monkeypatch) -> None:
    """rc=2 is dcm2niix saying "I could not convert this", which is an answer.
    Retrying it would hide a real error behind a narrower binary."""
    fake, calls = _recorder([2])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    assert run_dcm2niix(["dcm2niix.exe", "in"]).returncode == 2
    assert len(calls) == 1


def test_no_retry_without_a_vendored_binary(monkeypatch) -> None:
    """Which is every non-Windows platform, where the bug does not exist."""
    fake, calls = _recorder([STACK_OVERFLOW_RC])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: None)
    assert run_dcm2niix(["dcm2niix", "in"]).returncode == STACK_OVERFLOW_RC
    assert len(calls) == 1


def test_the_fallback_is_not_retried_with_itself(monkeypatch, tmp_path) -> None:
    """If the vendored binary is the one that died, there is nothing left to
    try, and retrying it would loop."""
    vendored = tmp_path / "dcm2niix.exe"
    vendored.write_bytes(b"")
    fake, calls = _recorder([STACK_OVERFLOW_RC])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)
    assert run_dcm2niix([str(vendored), "in"]).returncode == STACK_OVERFLOW_RC
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# When it must
# ---------------------------------------------------------------------------


def test_a_stack_overflow_retries_with_the_vendored_build(
    monkeypatch, tmp_path,
) -> None:
    vendored = tmp_path / "vendored" / "dcm2niix.exe"
    vendored.parent.mkdir()
    vendored.write_bytes(b"")
    fake, calls = _recorder([STACK_OVERFLOW_RC, 0])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)

    result = run_dcm2niix(["C:\\wheel\\dcm2niix.exe", "-b", "o", "in"])

    assert result.returncode == 0
    assert len(calls) == 2
    assert calls[0][0] == "C:\\wheel\\dcm2niix.exe", "the wheel goes first"
    assert calls[1][0] == str(vendored), "the retry uses the vendored build"
    assert calls[0][1:] == calls[1][1:], "every other argument is unchanged"


def test_a_failing_retry_reports_the_retry(monkeypatch, tmp_path) -> None:
    """If the fallback also fails, the caller sees that, not the crash."""
    vendored = tmp_path / "v" / "dcm2niix.exe"
    vendored.parent.mkdir()
    vendored.write_bytes(b"")
    fake, _calls = _recorder([STACK_OVERFLOW_RC, 2])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)
    assert run_dcm2niix(["wheel.exe", "in"]).returncode == 2


def test_the_negative_form_of_the_code_is_recognised(monkeypatch, tmp_path) -> None:
    """Python can surface the status as a signed int; both spellings are the
    same Windows status and both must trigger the retry."""
    vendored = tmp_path / "v" / "dcm2niix.exe"
    vendored.parent.mkdir()
    vendored.write_bytes(b"")
    fake, calls = _recorder([STACK_OVERFLOW_RC - (1 << 32), 0])
    monkeypatch.setattr(bg.subprocess, "run", fake)
    monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)
    assert run_dcm2niix(["wheel.exe", "in"]).returncode == 0
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# The binary itself
# ---------------------------------------------------------------------------


def test_vendored_lookup_is_windows_only() -> None:
    if os.name == "nt":
        pytest.skip("Windows behaviour checked below")
    assert vendored_dcm2niix() is None


class TestItIsOnlyOfferedWhereItCanRun:
    """The vendored build is an x86-64 PE, so those are the only conditions
    under which it may be substituted.

    The arch gate matters because ``os.name == "nt"`` is equally true on
    Windows for ARM. Windows would run an x64 PE there under emulation, so the
    fallback would appear to work while being slower than the native binary
    the wheel already supplies, and on a path nobody has tested. Better to
    surface the real error.
    """

    def test_not_offered_off_windows(self, monkeypatch) -> None:
        monkeypatch.setattr(bg.os, "name", "posix", raising=False)
        assert vendored_dcm2niix() is None

    def test_not_offered_on_windows_arm(self, monkeypatch) -> None:
        monkeypatch.setattr(bg.os, "name", "nt", raising=False)
        monkeypatch.setattr(bg.platform, "machine", lambda: "ARM64")
        assert vendored_dcm2niix() is None

    @pytest.mark.parametrize("arch", ["AMD64", "x86_64", "amd64"])
    def test_offered_on_windows_x86_64(self, monkeypatch, arch) -> None:
        """Case-insensitively, since the spelling varies by interpreter."""
        if os.name != "nt":
            pytest.skip("the file only exists in a Windows checkout")
        monkeypatch.setattr(bg.platform, "machine", lambda: arch)
        assert vendored_dcm2niix() is not None

    def test_an_arm_process_gets_the_real_error_not_a_retry(
        self, monkeypatch,
    ) -> None:
        """End to end: the crash is reported rather than silently emulated."""
        monkeypatch.setattr(bg.os, "name", "nt", raising=False)
        monkeypatch.setattr(bg.platform, "machine", lambda: "ARM64")
        fake, calls = _recorder([STACK_OVERFLOW_RC])
        monkeypatch.setattr(bg.subprocess, "run", fake)
        assert run_dcm2niix(["dcm2niix.exe", "in"]).returncode == STACK_OVERFLOW_RC
        assert len(calls) == 1


class TestTheTwoBuildsTakeDifferentArguments:
    """The retry cannot simply reuse argv.

    ``_run_dcm2niix_sidecars`` hands the source folder to dcm2niix in Win32
    long-path form, because the released MSVC binary understands it and the
    user's tree can be nested past 260 characters. The vendored MinGW build
    does NOT parse that prefix: it reads it as a UNC path and answers

        rc=5  Error: Input folder invalid: \\30_svs_se

    Measured on the same folder: plain path gives rc=0 and
    ``BidsGuess=['mrs', '_svs']``; the prefixed one gives rc=5 and nothing.
    Passing argv through unchanged made the fallback produce no output at all,
    silently, which is precisely the failure it exists to remove.
    """

    def test_the_prefix_is_stripped_for_the_retry(self, monkeypatch, tmp_path) -> None:
        vendored = tmp_path / "v" / "dcm2niix.exe"
        vendored.parent.mkdir()
        vendored.write_bytes(b"")
        fake, calls = _recorder([STACK_OVERFLOW_RC, 0])
        monkeypatch.setattr(bg.subprocess, "run", fake)
        monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)

        run_dcm2niix([
            "wheel.exe", "-b", "o", "-o", "\\\\?\\C:\\out", "\\\\?\\C:\\in",
        ])

        assert calls[0][-1] == "\\\\?\\C:\\in", "the wheel keeps the prefix"
        assert calls[1][-1] == "C:\\in", "the retry must not"
        assert calls[1][calls[1].index("-o") + 1] == "C:\\out"
        assert calls[1][1:3] == ["-b", "o"], "flags pass through untouched"

    def test_a_unc_prefix_becomes_a_unc_path(self) -> None:
        assert bg._without_long_path_prefix(
            "\\\\?\\UNC\\server\\share\\study"
        ) == "\\\\server\\share\\study"

    def test_an_unprefixed_argument_is_untouched(self) -> None:
        for arg in ("-b", "o", "C:\\data\\study", "%s", ""):
            assert bg._without_long_path_prefix(arg) == arg

    def test_no_retry_when_stripping_would_exceed_max_path(
        self, monkeypatch, tmp_path,
    ) -> None:
        """If the path only fitted because of the prefix, the vendored build
        cannot convert it either. Report the crash rather than replace it with
        a second, more confusing failure."""
        vendored = tmp_path / "v" / "dcm2niix.exe"
        vendored.parent.mkdir()
        vendored.write_bytes(b"")
        fake, calls = _recorder([STACK_OVERFLOW_RC])
        monkeypatch.setattr(bg.subprocess, "run", fake)
        monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)

        # Joined with the HOST separator, because that is what the guard
        # tests for (``os.sep in a``) and what ``str(Path)`` produces on the
        # platform the fallback runs on. Spelled with backslashes only, this
        # passed on Windows and failed on macOS and Linux, where the argument
        # did not look like a path at all and the retry went ahead.
        deep = "\\\\?\\C:" + os.sep + os.sep.join(["d" * 40] * 8)
        result = run_dcm2niix(["wheel.exe", "-b", "o", deep])

        assert result.returncode == STACK_OVERFLOW_RC
        assert len(calls) == 1, "a retry that cannot work must not be made"


class TestTheRetryIsJudgedByWhatItMustOpen:
    """dcm2niix opens the FILES inside the folder it walks, so their length,
    not the folder's, decides whether the unprefixed retry can work.

    Built on a real tree rather than mocked, because the length of a real
    path is the whole question. Passes the same on every platform: the
    ceiling it enforces is Windows', but the arithmetic is not.
    """

    @staticmethod
    def _touch(path: Path) -> None:
        """Create ``path`` even past MAX_PATH.

        Through the long-path form, because on a Windows runner without
        long-path support a plain ``open`` of a 300-character path fails,
        and the test would then error on the very platform it protects. A
        no-op wrapper everywhere else.
        """
        from bidsmgr.util.paths import long_path

        with open(long_path(path), "wb"):
            pass

    def _deep_tree(self, root: Path, *, file_len: int) -> Path:
        folder = root / ("s" * 30)
        folder.mkdir(parents=True)
        self._touch(folder / ("d" * file_len + ".dcm"))
        return folder

    def _setup(self, monkeypatch, tmp_path):
        vendored = tmp_path / "v" / "dcm2niix.exe"
        vendored.parent.mkdir()
        vendored.write_bytes(b"")
        fake, calls = _recorder([STACK_OVERFLOW_RC, 0])
        monkeypatch.setattr(bg.subprocess, "run", fake)
        monkeypatch.setattr(bg, "vendored_dcm2niix", lambda: vendored)
        return calls

    def test_a_short_folder_of_long_files_is_not_retried(
        self, monkeypatch, tmp_path,
    ) -> None:
        """The case the old check let through: the FOLDER is well under the
        ceiling, and every file inside it is over."""
        calls = self._setup(monkeypatch, tmp_path)
        folder = self._deep_tree(tmp_path, file_len=250)
        assert len(str(folder)) < bg._MAX_PATH_BUDGET, "the folder itself is fine"
        longest = bg._longest_path_under(str(folder))
        assert longest > bg._MAX_FILE_PATH, "and its file is not"

        result = run_dcm2niix(["wheel.exe", "-o", str(tmp_path), str(folder)])

        assert result.returncode == STACK_OVERFLOW_RC, "the real crash is reported"
        assert len(calls) == 1, "a retry that can open nothing must not be made"

    def test_a_tree_that_fits_is_retried(self, monkeypatch, tmp_path) -> None:
        calls = self._setup(monkeypatch, tmp_path)
        folder = self._deep_tree(tmp_path, file_len=20)
        assert bg._longest_path_under(str(folder)) <= bg._MAX_FILE_PATH

        result = run_dcm2niix(["wheel.exe", "-o", str(tmp_path), str(folder)])

        assert len(calls) == 2
        assert result.returncode == 0

    def test_depth_is_bounded_as_dcm2niix_bounds_it(self, tmp_path) -> None:
        """A file below dcm2niix's own search depth is one it never opens,
        so it must not stop a retry either."""
        here = tmp_path
        for level in range(bg._DCM2NIIX_DEPTH + 2):
            here = here / f"l{level}"
        here.mkdir(parents=True)
        self._touch(here / ("x" * 200))
        measured = bg._longest_path_under(str(tmp_path))
        assert measured < len(str(here / ("x" * 200)))

    def test_an_unreadable_folder_is_not_an_error(self, tmp_path) -> None:
        assert bg._longest_path_under(str(tmp_path / "missing")) == len(
            str(tmp_path / "missing")
        )


class TestTheExitCodeIsNamed:
    """dcm2niix writes nothing to stderr when Windows kills it, so the
    number is all the log has to go on."""

    @pytest.mark.parametrize("rc", [0xC00000FD, 0xC00000FD - (1 << 32)])
    def test_the_measured_stack_overflow_is_recognised(self, rc) -> None:
        """The code measured on spectroscopy, whichever sign it arrives in."""
        assert "0xC00000FD" in bg._exit_hint(rc)
        assert "stack overflow" in bg._exit_hint(rc)

    def test_the_path_failure_is_a_different_code(self) -> None:
        assert "MAX_PATH" in bg._exit_hint(0xC0000409)
        assert "MAX_PATH" not in bg._exit_hint(0xC00000FD)

    def test_a_missing_dll_is_named(self) -> None:
        assert "DLL" in bg._exit_hint(0xC0000135)

    @pytest.mark.parametrize("rc,word", [(-11, "SIGSEGV"), (-6, "SIGABRT")])
    def test_posix_signals_survive_the_mask(self, rc, word) -> None:
        """Masking to 32 bits first would turn -11 into 4294967285."""
        assert word in bg._exit_hint(rc)

    @pytest.mark.parametrize("rc", [0, 1, 2])
    def test_an_ordinary_code_gets_no_note(self, rc) -> None:
        assert bg._exit_hint(rc) == ""


def test_posix_exit_codes_can_never_reach_the_fallback(monkeypatch) -> None:
    """The POSIX guarantee, checked on every platform.

    Two independent gates stand between a POSIX run and the vendored binary,
    and this pins both. A POSIX exit status is 0-255, or negative for a signal,
    so 0xC00000FD is unreachable there in the first place; and even if it were
    produced, ``vendored_dcm2niix`` answers ``None`` off Windows.
    """
    monkeypatch.setattr(bg.os, "name", "posix", raising=False)
    assert vendored_dcm2niix() is None

    for code in (0, 1, 2, 127, 255, -6, -9, -11):
        fake, calls = _recorder([code])
        monkeypatch.setattr(bg.subprocess, "run", fake)
        assert run_dcm2niix(["dcm2niix", "in"]).returncode == code
        assert len(calls) == 1, f"exit {code} must not be retried"


@pytest.mark.skipif(os.name != "nt", reason="the vendored build is Windows-only")
class TestTheVendoredBinary:
    def test_it_is_present_and_reserves_enough_stack(self) -> None:
        """The whole reason it exists. 16,388,608 is what fails."""
        import struct

        path = vendored_dcm2niix()
        assert path is not None and path.is_file()
        data = path.read_bytes()
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        reserve = struct.unpack_from("<Q", data, pe + 24 + 72)[0]
        assert reserve >= 16_777_216, (
            f"vendored dcm2niix reserves only {reserve:,} bytes; "
            "16,388,608 is the value that overflows on MR spectroscopy"
        )

    def test_it_runs(self) -> None:
        path = vendored_dcm2niix()
        proc = subprocess.run([str(path), "-h"], capture_output=True, text=True)
        assert "dcm2niiX version" in proc.stdout

    def test_it_ships_its_provenance(self) -> None:
        """A vendored binary without a written reason is one nobody can ever
        decide to remove."""
        doc = Path(vendored_dcm2niix()).with_name("PROVENANCE.md")
        assert doc.is_file()
        text = doc.read_text(encoding="utf-8")
        assert "0xC00000FD" in text
        assert "16,777,216" in text or "16777216" in text


class TestItIsPackagedWithWhatItOwes:
    """Runs on every platform, unlike ``TestTheVendoredBinary``.

    That class needs Windows to execute the binary, so on macOS and Linux CI
    nothing checked that a release carries the files the binary cannot do
    without. These are all things a wheel can silently drop.
    """

    DIR = Path(bg.__file__).resolve().parent.parent / "vendor" / "dcm2niix_win"

    def test_the_binary_ships_its_licence(self) -> None:
        """BSD 2-Clause, clause 2: a redistribution IN BINARY FORM must
        reproduce the notice. Shipping the executable without it is a licence
        breach, not an omission of style."""
        licence = self.DIR / "LICENSE"
        assert licence.is_file()
        text = licence.read_text(encoding="utf-8")
        assert "Chris Rorden" in text
        assert "Redistributions in binary form" in text

    def test_pyproject_ships_all_three(self) -> None:
        """package-data is an allowlist: a file present in the repo and absent
        here is left out of the wheel with no warning."""
        pyproject = (
            Path(bg.__file__).resolve().parents[2] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        line = next(
            ln for ln in pyproject.splitlines()
            if ln.lstrip().startswith('"bidsmgr.vendor.dcm2niix_win"')
        )
        for name in ("dcm2niix.exe", "PROVENANCE.md", "LICENSE"):
            assert f'"{name}"' in line, f"{name} would be left out of the wheel"

    def test_the_binary_needs_no_toolchain_dlls(self) -> None:
        """A MinGW build that links its runtime DYNAMICALLY runs on the machine
        it was built on, where the toolchain is on PATH, and nowhere else: it
        fails to start with STATUS_DLL_NOT_FOUND on a user's clean install.

        Reads the PE IMPORT TABLE, not the bytes. The string
        ``libgcc_s_dw2-1.dll`` IS in this binary, and must be: MinGW's startup
        code probes for it with LoadLibrary to register unwind frames and skips
        quietly when it is absent. That is not a dependency, and a substring
        search reports it as one.
        """
        imports = {n.lower() for n in _pe_imported_dlls(self.DIR / "dcm2niix.exe")}
        assert "kernel32.dll" in imports, "the parser found no imports at all"
        toolchain = {n for n in imports
                     if n.startswith(("libgcc", "libstdc", "libwinpthread"))}
        assert not toolchain, f"needs the MinGW runtime on PATH: {sorted(toolchain)}"
        # Everything left must be Windows' own: the kernel, advapi, and the
        # Universal CRT that ships with Windows 10+ and that Python needs too.
        foreign = {n for n in imports
                   if n not in ("kernel32.dll", "advapi32.dll")
                   and not n.startswith("api-ms-win-crt-")}
        assert not foreign, f"imports DLLs a clean Windows may lack: {sorted(foreign)}"


def _pe_imported_dlls(path: Path) -> list[str]:
    """The DLL names in a PE32+ file's import directory.

    Minimal and dependency-free on purpose: this runs on macOS and Linux CI,
    where no Windows tooling exists, and ``pefile`` is not a dependency.
    """
    import struct

    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0", "not a PE file"
    n_sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    assert struct.unpack_from("<H", data, opt)[0] == 0x20B, "not PE32+"
    # Data directory 1 is the import table; PE32+ puts the directories at +112.
    import_rva = struct.unpack_from("<I", data, opt + 112 + 8)[0]

    sections = []
    for i in range(n_sections):
        at = opt + opt_size + i * 40
        vsize, vaddr, rsize, raw = struct.unpack_from("<IIII", data, at + 8)
        sections.append((vaddr, max(vsize, rsize), raw))

    def offset(rva: int) -> int:
        for vaddr, size, raw in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raw
        raise ValueError(f"RVA {rva:#x} is in no section")

    names = []
    at = offset(import_rva)
    while True:
        name_rva = struct.unpack_from("<I", data, at + 12)[0]
        if name_rva == 0:
            break
        start = offset(name_rva)
        names.append(data[start:data.index(b"\0", start)].decode("ascii"))
        at += 20
    return names
