"""dcm2niix ``BidsGuess`` classifier — improvement_plan.md M1.

dcm2niix already classifies DICOM series via its built-in ``BidsGuess``
heuristics, which are kept up to date with the BIDS spec. We harvest that
classification by running dcm2niix in **sidecar-only** mode (``-b o``) once
per DICOM source folder and parsing the JSON sidecars. The result is the
highest-fidelity DICOM classifier available without writing our own rules.

This is the first feature to land in ``bidsmgr``: it validates the keystone
``schema/`` (because every BidsGuess output is schema-checked) and seeds
the per-row pipeline that the GUI inspector will surface later.

Reference: architecture.md §4.2 layer 1, ``../improvement_plan.md`` M1.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .. import schema
from ..inventory.types import InventoryRow
from ..util.paths import long_path, long_path_for_tree
from .types import Classification

log = logging.getLogger(__name__)

_SHORT_TO_LONG_CACHE: Optional[dict[str, str]] = None


def _short_to_long_entity_map() -> dict[str, str]:
    """Map BIDS short entity names (``"sub"``) to canonical keys (``"subject"``)."""
    global _SHORT_TO_LONG_CACHE
    if _SHORT_TO_LONG_CACHE is None:
        s = schema.get_schema()
        _SHORT_TO_LONG_CACHE = {
            str(s.objects.entities[k].get("name", k)): k
            for k in s.objects.entities.keys()
        }
    return _SHORT_TO_LONG_CACHE


def parse_bids_guess(guess: Sequence[str]) -> tuple[str, dict[str, str], str]:
    """Parse the ``BidsGuess`` field into ``(datatype, entities, suffix)``.

    dcm2niix encodes BidsGuess as ``[<datatype>, "_<key>-<value>_..._<suffix>"]``.
    For the ``"discard"`` datatype the second element is still parseable but
    callers should treat the row as skip-by-default.
    """

    if not guess or len(guess) < 2:
        raise ValueError(f"Malformed BidsGuess: {guess!r}")

    datatype = str(guess[0])
    tail = str(guess[1]).lstrip("_")
    parts = tail.split("_")
    if not parts:
        raise ValueError(f"BidsGuess has no suffix component: {guess!r}")

    suffix = parts[-1]
    short_to_long = _short_to_long_entity_map()
    entities: dict[str, str] = {}
    for chunk in parts[:-1]:
        if "-" not in chunk:
            continue
        short, _, value = chunk.partition("-")
        long_name = short_to_long.get(short, short)
        # dcm2niix encodes ``run-N`` from DICOM SeriesNumber, which is NOT the
        # BIDS-semantic run (BIDS run-N is for repeated acquisitions of the
        # same parameters within a session). Drop it here; the planner /
        # cli/scan re-derives runs cross-row after grouping.
        if long_name == "run":
            continue
        entities[long_name] = value

    return datatype, entities, suffix


def find_dcm2niix() -> Path:
    """Locate the dcm2niix executable.

    Prefers the binary shipped with the ``dcm2niix`` Python package (a pinned
    pip dependency), falling back to ``$PATH``.

    The package's own ``bin_path`` is built as ``<pkgdir>/dcm2niix`` with no
    extension on every platform, but its Windows wheel ships ``dcm2niix.exe``.
    So on Windows the packaged-binary branch never matches and the lookup
    depends entirely on the ``$PATH`` fallback finding the console-script shim
    the wheel drops in ``Scripts/``. That works from an activated venv and fails
    without one — a bundled or embedded interpreter, a desktop shortcut, or
    plain ``<venv>/Scripts/python.exe -m bidsmgr...`` — and the failure is a
    hard ``FileNotFoundError`` that skips the BidsGuess classifier and the probe
    conversion both. Hence the suffix sweep: find the binary the wheel actually
    shipped, rather than depending on how the process was launched.
    """

    try:
        import dcm2niix as _pkg  # type: ignore

        bin_path = Path(getattr(_pkg, "bin_path", "") or "")
        if bin_path.name:
            for candidate in _binary_candidates(bin_path):
                if candidate.is_file():
                    return candidate
    except ImportError:
        pass

    found = shutil.which("dcm2niix")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "dcm2niix executable not found. Install the ``dcm2niix`` pip package "
        "or place the binary on PATH."
    )


# What Windows returns when it kills a process for exhausting its stack:
# STATUS_STACK_OVERFLOW. Not to be confused with 0xC0000409,
# STATUS_STACK_BUFFER_OVERRUN, which is the path-length failure and has a
# different fix. dcm2niix writes NOTHING to stderr on the way out, so this
# number is the only evidence there is.
STACK_OVERFLOW_RC = 0xC00000FD


# The architectures the vendored binary can actually run as. It is an x86-64
# PE, built and tested as one. Windows on ARM would run it under x64 emulation,
# which we have not tested and which would be slower than the native build the
# wheel already supplies there, so an ARM process is left with the real error
# rather than handed a foreign binary that silently seems to work. A native
# ARM64 build could be added beside it later; see the PROVENANCE note.
_VENDORED_ARCHES = frozenset({"AMD64", "X86_64"})


def vendored_dcm2niix() -> Optional[Path]:
    """Our own Windows x86-64 build of dcm2niix, or ``None`` if not usable here.

    ``None`` on every platform but Windows, on Windows for any architecture but
    x86-64, and wherever the file is simply absent. Only ever a fallback. See
    :func:`run_dcm2niix`.
    """
    if os.name != "nt":
        return None
    if platform.machine().upper() not in _VENDORED_ARCHES:
        return None
    path = (
        Path(__file__).resolve().parent.parent
        / "vendor" / "dcm2niix_win" / "dcm2niix.exe"
    )
    return path if path.is_file() else None


def run_dcm2niix(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """Run dcm2niix, and retry with the vendored build if Windows kills it.

    The released Windows dcm2niix cannot convert MR spectroscopy. Its linker
    reserves 16,388,608 bytes of stack; a Siemens ``svs_se`` series needs more
    than the MSVC build fits in that, so Windows terminates the process with
    ``0xC00000FD`` and an empty stderr. Nothing is written and nothing is
    said, which is why it read as "this folder holds no DICOM images".

    Measured on that series, same machine, same input: the released binary
    fails; the same binary with its PE stack reserve patched to 16,777,216
    converts; a GCC build of the same source converts at either value. So it
    is the frames MSVC emits rather than the constant alone, and Windows fixes
    the reserve at link time and cannot grow it. macOS and Linux never see it.

    The wheel's binary is ALWAYS tried first and its result is always
    preferred: it is the pinned, JPEG2000- and JPEG-LS-capable build that
    every platform shares, and the vendored one is a narrow GCC build with
    those decoders off. The fallback is reached only on the one exit code that
    means "the process was killed before it could do anything", so a genuine
    conversion failure is still reported as itself.

    A no-op off Windows: :func:`vendored_dcm2niix` returns ``None`` there.
    """
    proc = subprocess.run(list(cmd), **kwargs)
    if (proc.returncode & 0xFFFFFFFF) != STACK_OVERFLOW_RC:
        return proc

    fallback = vendored_dcm2niix()
    if fallback is None:
        return proc
    try:
        same = Path(cmd[0]).resolve() == fallback.resolve()
    except OSError:
        same = False
    if same:
        return proc          # already the fallback; nothing left to try

    # The two builds do not accept the same arguments. The released MSVC
    # binary understands the Win32 long-path prefix and we rely on that in
    # _run_dcm2niix_sidecars; the vendored MinGW one does not parse it at all
    # and answers `rc=5, Input folder invalid: \\30_svs_se`. Handing the retry
    # the prefixed argv made the fallback produce nothing, silently, which is
    # the failure mode it exists to remove.
    args = [_without_long_path_prefix(a) for a in list(cmd)[1:]]
    too_long = [a for a in args if len(a) >= _MAX_PATH_BUDGET and os.sep in a]
    # The folder dcm2niix WALKS is the last argument, as its own CLI requires
    # and every caller here does. What decides whether the unprefixed retry
    # can open anything is the longest FILE under it, not the folder: a
    # 200-character folder of 90-character Siemens filenames passes a check on
    # the folder and then opens nothing, and the retry answers `rc=2, Unable
    # to find any DICOM images`, which reads as "not DICOM". This is the trap
    # util.paths.long_path_for_tree documents for the first attempt.
    walked = args[-1] if args else ""
    deepest = _longest_path_under(walked) if walked and os.path.isdir(walked) else 0
    if deepest > _MAX_FILE_PATH:
        log.warning(
            "dcm2niix was killed by Windows (0x%08X, stack exhausted) and the "
            "bundled build cannot be used here: it cannot open a path longer "
            "than %d characters, and the deepest file under %s is %d. Move "
            "the data somewhere shallower to convert MR spectroscopy.",
            STACK_OVERFLOW_RC, _MAX_FILE_PATH, walked[:80], deepest,
        )
        return proc
    if too_long:
        # Stripping the prefix would put it back over the ceiling, so the
        # retry cannot succeed either. Report the crash rather than replace it
        # with a second, more confusing failure.
        log.warning(
            "dcm2niix was killed by Windows (0x%08X, stack exhausted) and the "
            "bundled build cannot be used here: it does not accept long-path "
            "arguments, and %s is %d characters. Move the data somewhere "
            "shallower to convert MR spectroscopy.",
            STACK_OVERFLOW_RC, too_long[0][:80], len(too_long[0]),
        )
        return proc

    log.warning(
        "dcm2niix was killed by Windows (0x%08X, stack exhausted) running %s. "
        "This is the released build failing on MR spectroscopy. Retrying with "
        "the bundled build at %s.",
        STACK_OVERFLOW_RC, Path(cmd[0]).name, fallback,
    )
    retried = subprocess.run([str(fallback), *args], **kwargs)
    if retried.returncode == 0:
        log.info("the bundled dcm2niix converted what the released one could not")
    return retried


# Where the vendored binary stops being usable, since it cannot take the
# ``\\?\`` prefix that lifts the limit. Matches the threshold in
# ``util.paths.long_path``.
_MAX_PATH_BUDGET = 248

#: The longest FILE path the vendored build can open. MAX_PATH is 260
#: characters INCLUDING the terminating NUL, so 259 is the last that fits.
#: The 248 above is the directory limit, which leaves room for an 8.3 name.
_MAX_FILE_PATH = 259

#: How deep dcm2niix searches by default (``-d 5``), so how deep a file can
#: sit and still be one it will try to open.
_DCM2NIIX_DEPTH = 5


def _longest_path_under(folder: str, depth: int = _DCM2NIIX_DEPTH) -> int:
    """The longest path dcm2niix would have to open walking ``folder``.

    Measures names rather than opening files, so it works on a tree whose
    files are too long for this process to open either. Lengths are counted
    in the UNPREFIXED form, because that is what the retry hands dcm2niix;
    each folder is LISTED through ``long_path``, so a subfolder already past
    248 characters is still read on Windows instead of being skipped and
    leaving its files unmeasured. A folder that genuinely cannot be listed is
    skipped; its own path has already been counted.
    """
    longest = len(folder)
    stack = [(folder, 0)]
    while stack:
        here, level = stack.pop()
        try:
            with os.scandir(long_path(here)) as entries:
                for entry in entries:
                    full = os.path.join(here, entry.name)
                    longest = max(longest, len(full))
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        is_dir = False
                    if is_dir and level < depth:
                        stack.append((full, level + 1))
        except OSError:
            continue
    return longest


def _without_long_path_prefix(arg: str) -> str:
    """``arg`` with the Win32 long-path prefix removed, if it carries one.

    The vendored MinGW build cannot parse ``\\\\?\\``; it reads the argument as
    a UNC path and rejects it. Everything else is returned untouched, so
    flags and basenames pass through.
    """
    if arg.startswith("\\\\?\\UNC\\"):
        return "\\\\" + arg[len("\\\\?\\UNC\\"):]
    if arg.startswith("\\\\?\\"):
        return arg[len("\\\\?\\"):]
    return arg


def _binary_candidates(bin_path: Path) -> list[Path]:
    """``bin_path`` and the executable suffixes Windows spells it with.

    ``PATHEXT`` is what ``shutil.which`` consults, so honouring it here keeps
    the packaged-binary branch and the ``$PATH`` branch looking for the same
    set of names.
    """
    if os.name != "nt":
        return [bin_path]
    suffixes = [
        ext
        for ext in os.environ.get("PATHEXT", ".EXE;.BAT;.CMD;.COM").split(os.pathsep)
        if ext.strip()
    ]
    if ".EXE" not in {s.upper() for s in suffixes}:
        suffixes.insert(0, ".EXE")
    # The wheel ships a lowercase ``dcm2niix.exe`` and PATHEXT is conventionally
    # uppercase. Windows and macOS both match case-insensitively, so ``.EXE``
    # would resolve and we would return a path naming a file that exists under
    # no such spelling: right binary, wrong name in every log and error after
    # it. Try the spelling the wheel actually ships first.
    suffixes = [".exe"] + [s for s in suffixes if s.lower() != ".exe"]
    return [bin_path] + [bin_path.with_name(bin_path.name + s) for s in suffixes]


def _run_dcm2niix_sidecars(
    dicom_dir: Path,
    output_dir: Path,
    *,
    dcm2niix_bin: Optional[Path] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix to produce JSON sidecars only.

    Flags:

    * ``-b o``  : sidecar only (no NIfTI written)
    * ``-ba n`` : do not anonymize the sidecar (keeps SeriesInstanceUID)
    * ``-z n``  : no compression (irrelevant for sidecar-only)
    * ``-f %s`` : filename = SERIES NUMBER. **Not ``%j``.**

    ``%j`` is the SeriesInstanceUID, which is ~59 characters of unbounded
    identifier in a path component, and CROSS_PLATFORM_RULES 1.2 forbids
    exactly that: Windows caps a path at 260 characters and dcm2niix does not
    fail politely when handed a longer one. It dies with a stack buffer
    overrun and an EMPTY stderr, so every layer above reads "no sidecars" as
    "nothing to say" and the scan reports success having classified nothing.
    ``converter/backends/dcm2niix_direct`` already says "do not use ``%j``
    here" for the same reason; this call site was missed.

    The series number is short, and it is unique per series within a folder.
    It does not need to be unique beyond that: the join back to an inventory
    row reads ``SeriesInstanceUID`` from INSIDE the JSON, never from the
    filename. Where two pooled studies do collide, dcm2niix's default
    ``-w 2`` adds a suffix rather than overwriting, so no sidecar is lost.

    The two directories are prefixed by DIFFERENT helpers, and the difference
    matters. ``output_dir`` is a path we are about to write into, and we
    control the names in it, so its own length is the length that counts:
    :func:`~bidsmgr.util.paths.long_path`. ``dicom_dir`` is a path dcm2niix
    will WALK, and what has to fit under the ceiling is the files inside it,
    which we have not measured: :func:`~bidsmgr.util.paths.long_path_for_tree`.

    Shortening the output name to ``%s`` and prefixing both with ``long_path``
    was not enough on the reporting dataset, because its 200-character subject
    folder is under that helper's 248 threshold and was passed through
    unprefixed — while the ~90-character DICOM filenames inside put every file
    at 290. All three subjects still came back ``rc=2`` "Unable to find any
    DICOM images".
    """

    binary = str(dcm2niix_bin or find_dcm2niix())
    cmd = [
        binary,
        "-b", "o",
        "-ba", "n",
        "-z", "n",
        "-o", str(long_path(output_dir)),
        "-f", "%s",
        long_path_for_tree(dicom_dir),
    ]
    return run_dcm2niix(cmd, capture_output=True, text=True, timeout=timeout)


def _collect_sidecars(directory: Path) -> list[dict]:
    """Read every ``*.json`` in ``directory`` into memory."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read sidecar %s: %s", path, exc)
            continue
        data["_sidecar_path"] = str(path)
        out.append(data)
    return out


def canonicalise(datatype: str, suffix: str) -> tuple[str, str]:
    """Resolve a dcm2niix datatype / suffix to the schema's own spelling.

    dcm2niix does not always match the schema's case. PET is the case that
    matters: it emits ``["PET", "PET"]`` while the schema spells both ``pet``,
    so a case-sensitive comparison silently rejects every PET series.

    The resolve is deliberately **case-insensitive rather than lowercasing**:
    the schema spells plenty of suffixes in mixed case (``T1w``, ``FLAIR``,
    ``UNIT1``), so ``.lower()`` would break far more than it fixed. Anything
    with no case-insensitive match is returned untouched, and rejected by
    :func:`_validate_classification` as before.
    """
    if datatype == "discard":
        return datatype, suffix

    dt_map = {d.lower(): d for d in schema.list_datatypes()}
    dt = dt_map.get(datatype.lower(), datatype)
    if dt not in dt_map.values():
        return dt, suffix

    sfx_map = {s.lower(): s for s in schema.list_suffixes(dt)}
    return dt, sfx_map.get(suffix.lower(), suffix)


def looks_like_uid(value: object) -> bool:
    """Whether ``value`` is plausibly a DICOM UID.

    A UID is dot-separated numeric components, and in practice at least
    three of them. This exists because dcm2niix does not always write one.

    Measured 2026-09-26 on a Siemens study whose spectroscopy is stored under
    the STANDARD MR Spectroscopy Storage SOP class
    (``1.2.840.10008.5.1.4.1.1.4.2``): the sidecar's ``SeriesInstanceUID``
    came out as ``133347.357000``, which is the series TIME, not the UID. The
    image series in the same folder got a correct UID, and so did a second
    study whose spectroscopy uses the Siemens private CSA class
    (``1.3.12.2.1107.5.9.1``). So this is specific to that SOP class, and it
    is invisible until a join on the UID quietly matches nothing.
    """
    text = str(value or "")
    if not text:
        return False
    parts = text.split(".")
    return len(parts) >= 3 and all(part.isdigit() for part in parts)


def _validate_classification(datatype: str, suffix: str, entities: dict[str, str]) -> bool:
    """Return ``True`` if the schema accepts this (datatype, suffix, entities) tuple."""
    if datatype == "discard":
        # Not a real datatype — but the row should still produce a Classification
        # marked ``skip=True`` so the GUI can surface the recommendation.
        return True
    if datatype not in schema.list_datatypes():
        return False
    if suffix not in schema.list_suffixes(datatype):
        return False
    allowed = set(schema.allowed_entities(datatype, suffix))
    return all(ent in allowed or ent == "subject" for ent in entities)


def _exit_hint(returncode: int) -> str:
    """A human-readable note for an exit code worth recognising.

    dcm2niix writes nothing to stderr when Windows kills it, so without
    naming the code here the log says only that a number was not zero.

    Windows codes are compared after masking to 32 bits, the way
    :func:`run_dcm2niix` compares them, because the same NTSTATUS can reach
    Python as a large positive number or as its negative two's complement.
    POSIX signals are looked up BEFORE masking: they arrive negative, and
    masking ``-11`` would turn SIGSEGV into an unrecognisable 4294967285.
    """
    posix = {
        -11: " (SIGSEGV)",
        -6: " (SIGABRT)",
    }
    if returncode in posix:
        return posix[returncode]
    windows = {
        STACK_OVERFLOW_RC:
            " (0xC00000FD, a Windows stack overflow: the released dcm2niix "
            "dies this way on MR spectroscopy)",
        0xC0000409:
            " (0xC0000409, a Windows stack buffer overrun: this is what "
            "dcm2niix does when a path exceeds MAX_PATH)",
        0xC0000135:
            " (0xC0000135, a DLL the program needs was not found)",
        0xC0000005: " (0xC0000005, an access violation)",
    }
    return windows.get(returncode & 0xFFFFFFFF, "")


def classify_dicom_folder(
    dicom_dir: Path,
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> list[Classification]:
    """Run dcm2niix BidsGuess on ``dicom_dir`` and return classifications.

    Each :class:`InventoryRow` whose ``series_uid`` is present in the sidecar
    output produces one :class:`Classification`. Rows with no matching sidecar
    yield no classification (the caller decides whether to fall back to the
    next classifier in the chain).
    """

    rows = list(rows)
    rows_by_uid: dict[str, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        if r.series_uid:
            rows_by_uid[r.series_uid].append(r)

    # A second index, used only when the sidecar's UID is unusable. Keyed on
    # the series description, and ONLY where that description names exactly
    # one series in this folder: two runs of one protocol share a
    # description, and guessing between them would be worse than not
    # classifying either.
    by_description: dict[str, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        if r.series_description:
            by_description[r.series_description.strip()].append(r)
    unique_by_description = {
        desc: found for desc, found in by_description.items()
        if len({x.series_uid for x in found}) == 1
    }

    use_temp = workdir is None
    if use_temp:
        workdir_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(workdir_ctx.name)
    else:
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = _run_dcm2niix_sidecars(dicom_dir, out_dir, dcm2niix_bin=dcm2niix_bin)
        sidecars = _collect_sidecars(out_dir)
        # SAY SO when this produces nothing. The whole classifier chain reads
        # an empty result as "this classifier has no opinion", so a dcm2niix
        # that died leaves a scan that reports success with every MRI row
        # unclassified, and the user sees a feature that stopped working with
        # no error anywhere. That is the shape of every Windows defect in
        # CROSS_PLATFORM_RULES, and it is the reason this branch is loud.
        if rows and not sidecars:
            log.warning(
                "dcm2niix produced NO sidecars for %s, so none of its %d "
                "series could be classified. returncode=%s%s stderr=%r",
                dicom_dir, len(rows), proc.returncode,
                _exit_hint(proc.returncode),
                (proc.stderr or "")[-500:] or "(empty)",
            )
        elif proc.returncode != 0:
            log.warning(
                "dcm2niix returncode=%s for %s%s; stderr=%s",
                proc.returncode, dicom_dir, _exit_hint(proc.returncode),
                (proc.stderr or "")[-500:],
            )
    finally:
        if use_temp:
            workdir_ctx.cleanup()  # type: ignore[name-defined]

    out: list[Classification] = []
    for sidecar in sidecars:
        guess = sidecar.get("BidsGuess")
        if not guess:
            continue
        try:
            datatype, entities, suffix = parse_bids_guess(guess)
        except ValueError as exc:
            log.debug("could not parse BidsGuess %r: %s", guess, exc)
            continue
        # dcm2niix's casing is not always the schema's (it emits "PET"/"PET").
        datatype, suffix = canonicalise(datatype, suffix)

        uid = sidecar.get("SeriesInstanceUID")
        matching_rows = rows_by_uid.get(uid, [])
        if not matching_rows and not looks_like_uid(uid):
            # dcm2niix wrote something that is not a UID, so the join was
            # never going to land. Fall back to the series description, which
            # it reports correctly, and say so: a scan that silently
            # classified nothing is the symptom this repairs.
            description = str(sidecar.get("SeriesDescription") or "").strip()
            matching_rows = unique_by_description.get(description, [])
            if matching_rows:
                log.info(
                    "dcm2niix wrote %r as the SeriesInstanceUID for %r, which "
                    "is not a UID; matched on the series description instead",
                    uid, description,
                )
        if not matching_rows:
            log.debug("BidsGuess sidecar with no matching inventory row: uid=%s", uid)
            continue

        skip = (datatype == "discard")
        valid = _validate_classification(datatype, suffix, entities)
        if not valid:
            log.info(
                "BidsGuess output rejected by schema: datatype=%s suffix=%s entities=%s",
                datatype, suffix, entities,
            )
            continue

        rationale = f"dcm2niix BidsGuess: {list(guess)}"
        confidence = 0.0 if skip else 0.85

        for row in matching_rows:
            out.append(
                Classification(
                    row_id=row.row_id,
                    classifier="dcm2niix_bidsguess",
                    datatype=datatype,
                    suffix=suffix,
                    candidate_entities=dict(entities),
                    confidence=confidence,
                    rationale=rationale,
                    skip=skip,
                )
            )
    return out


def classify(
    rows: Iterable[InventoryRow],
    *,
    dcm2niix_bin: Optional[Path] = None,
) -> list[Classification]:
    """Top-level classifier entry point.

    Groups MRI rows by their containing folder (``row.source.parent`` if
    ``source`` is a file, ``row.source`` if it is a directory) and dispatches
    one dcm2niix invocation per folder.
    """

    rows = [r for r in rows if r.modality == "mri" and r.series_uid]
    if not rows:
        return []

    groups: dict[Path, list[InventoryRow]] = defaultdict(list)
    for r in rows:
        folder = r.source if r.source.is_dir() else r.source.parent
        groups[folder].append(r)

    out: list[Classification] = []
    for folder, group_rows in groups.items():
        out.extend(classify_dicom_folder(folder, group_rows, dcm2niix_bin=dcm2niix_bin))
    return out


__all__ = [
    "looks_like_uid",
    "classify",
    "classify_dicom_folder",
    "parse_bids_guess",
    "find_dcm2niix",
    "run_dcm2niix",
    "vendored_dcm2niix",
]
