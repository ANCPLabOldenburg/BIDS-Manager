"""Run a defacing engine on one image, and say what happened.

This module does not touch a dataset. It takes an image, produces a defaced
copy somewhere the caller chose, and returns a result. Deciding what to do with
those bytes, backing the original up, editing the sidecar, recording the
operation, is the caller's job, and keeping the split means the engine can be
tested without a dataset and the dataset logic can be tested without running a
binary.

The same shape ``editor/restructure.py`` uses: work out what would happen, then
apply it, in two pieces.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .engines import (
    BACKEND_BRAINCHOP,
    DEFAULT_ENGINE_ID,
    MASK,
    TEMPLATE,
    Engine,
    engine,
)
from .probe import read_dimensions

log = logging.getLogger(__name__)

# Generous. A large image on a slow disk with the Hellinger cost is the worst
# case, and a defacer that gives up early leaves a half-finished temp file and
# an unexplained failure. Measured on a 22.9 M-voxel T1w: 1.9 s.
DEFAULT_TIMEOUT_S = 600


class DefaceUnavailable(RuntimeError):
    """niimath is not installed, so defacing cannot run.

    Carries its own remedy, because the person who sees this is a user and not
    a packager.
    """


class DefaceFailed(RuntimeError):
    """The engine ran and did not produce a usable image."""


@dataclass(frozen=True)
class DefaceResult:
    """What one run produced."""

    source: Path
    output: Path
    engine_id: str
    seconds: float
    stdout: str = ""
    stderr: str = ""


def _binary_candidates(bin_path: Path) -> list[Path]:
    """``bin_path`` and the executable suffixes Windows spells it with.

    The same sweep :func:`bidsmgr.classifier.dcm2niix_bidsguess.find_dcm2niix`
    does, and for the same reason: these wheels build ``bin_path`` with no
    extension on every platform while the Windows wheel ships a ``.exe``, so
    the packaged-binary branch never matches on Windows unless it is asked
    about the suffixes too.
    """
    if os.name != "nt":
        return [bin_path]
    exts = [e for e in os.environ.get("PATHEXT", ".EXE").split(os.pathsep) if e]
    return [bin_path] + [bin_path.with_suffix(e.lower()) for e in exts]


def find_niimath() -> Path:
    """Locate the niimath executable.

    Prefers the binary inside the ``niimath`` pip package, falling back to
    ``$PATH`` for a system install. Raises :class:`DefaceUnavailable` rather
    than ``FileNotFoundError`` so callers can catch the one thing that means
    "offer this greyed out" without catching every missing file.
    """
    try:
        import niimath as _pkg  # type: ignore

        bin_path = Path(getattr(_pkg, "bin_path", "") or "")
        if bin_path.name:
            for candidate in _binary_candidates(bin_path):
                if candidate.is_file():
                    return candidate
    except ImportError:
        pass

    found = shutil.which("niimath")
    if found:
        return Path(found)

    raise DefaceUnavailable(
        "Defacing needs niimath, which is not installed. It ships with BIDS "
        "Manager on Python 3.10 to 3.13; there is no wheel for 3.14 yet, so "
        "on that version defacing needs a niimath binary on PATH."
    )


def available(engine_id: Optional[str] = None) -> bool:
    """True when this engine can run. For greying out a menu entry."""
    return unavailable_reason(engine_id) is None


def brainchop_present() -> bool:
    """True when brainchop is installed, WITHOUT importing it.

    ``find_spec`` rather than a try/import, and the difference is the whole
    reason the model runs in a subprocess: importing brainchop pulls in
    tinygrad, which sets up a GPU runtime, and having that in the GUI process
    is what crashed the application after a strip finished. A mere
    availability check must not be the thing that loads it.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("brainchop") is not None
    except (ImportError, ValueError):
        return False


def atlas_present(eng: Optional[Engine] = None) -> bool:
    """True when the template and the mask this engine needs both shipped."""
    mask = eng.mask if eng is not None else MASK
    return TEMPLATE.is_file() and Path(mask).is_file()


def unavailable_reason(engine_id: Optional[str] = None) -> Optional[str]:
    """Why this engine cannot run, or ``None``. For a tooltip.

    ``engine_id`` of ``None`` asks about defacing in general, which is what
    the menu entry needs.
    """
    eng = engine(engine_id) if engine_id else None

    if eng is not None and eng.backend == BACKEND_BRAINCHOP:
        if not brainchop_present():
            return (
                f"{eng.label} needs brainchop, which is not installed. It "
                "ships with BIDS Manager, so this means a broken or partial "
                "install. Reinstall, or use the atlas engine, which needs "
                "only niimath."
            )
        # brainchop shells out to niimath to conform the image, so it needs
        # the binary too even though it does not register against the atlas.
        try:
            find_niimath()
        except DefaceUnavailable as exc:
            return str(exc)
        return None

    try:
        find_niimath()
    except DefaceUnavailable as exc:
        return str(exc)
    if not atlas_present(eng):
        missing = "brain mask" if eng is not None and eng.is_strip else "atlas"
        return (
            f"The {missing} is missing from this installation, so there "
            "is nothing to register against. Reinstall bids-manager."
        )
    return None


def _strip_with_brainchop(
    source: Path, output: Path, eng: Engine,
) -> DefaceResult:
    """Skull-strip ``source`` with a brainchop model, in the image's own space.

    The model runs in a SEPARATE PROCESS (``bidsmgr.deface._mindgrab``). It
    is backed by tinygrad, which sets up a GPU runtime on first use, and doing
    that on a QThread crashed the application after the work had finished and
    the file was written. The parent never imports brainchop or tinygrad at
    all. Same treatment niimath already gets, for the same reason.

    brainchop conforms everything to a 256-cubed 1 mm grid before it runs, and
    writes its result in that grid. That is not good enough here: a
    skull-stripped scan whose voxels no longer line up with the original is
    not a derivative OF that scan, it is a different image, and nothing
    computed from the original applies to it. So the mask is brought back onto
    the source's own grid and applied there, leaving dimensions, affine and
    dtype exactly as they were.

    Voxels outside the brain are set to the image's own MINIMUM rather than to
    zero. Raw zero is not "nothing": with ``scl_inter`` set it displays as mid
    grey, and on a CT it is dense tissue. Using the minimum makes the
    background match the darkest voxel actually in the image, whatever the
    header says. That reasoning is BIDSvue's, and it is right.
    """
    import time

    started = time.monotonic()
    try:
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - both are hard deps
        raise DefaceFailed(f"skull stripping needs {exc.name}") from exc

    timeout = DEFAULT_TIMEOUT_S

    reason = unavailable_reason(eng.id)
    if reason:
        raise DefaceUnavailable(reason)

    # brainchop calls niimath BY NAME to conform the image, so the binary
    # inside our wheel has to be on the child's PATH.
    env = dict(os.environ)
    exe_dir = str(find_niimath().parent)
    if exe_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")

    tmp_mask = Path(tempfile.mkdtemp(prefix="bidsmgr-strip-")) / "mask.nii.gz"
    argv = [
        sys.executable, "-m", "bidsmgr.deface._mindgrab",
        str(source), str(tmp_mask), "--model", eng.model,
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(tmp_mask.parent, ignore_errors=True)
        raise DefaceFailed(
            f"{eng.label} took longer than {timeout} s on {source.name} and "
            "was stopped."
        ) from exc
    except OSError as exc:
        shutil.rmtree(tmp_mask.parent, ignore_errors=True)
        raise DefaceFailed(f"could not start {eng.label}: {exc}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        shutil.rmtree(tmp_mask.parent, ignore_errors=True)
        raise DefaceFailed(
            f"{eng.label} failed on {source.name}"
            + (f": {detail[-1]}" if detail else " with no message")
        )

    try:
        mask_img = nib.load(str(tmp_mask))
        mask_conf = np.asanyarray(mask_img.dataobj)
        conf_affine = np.asarray(mask_img.affine, dtype=float)

        if not mask_conf.any():
            raise DefaceFailed(
                f"{eng.label} found no brain in {source.name}. The result was "
                "discarded."
            )

        img = nib.load(str(source))
        # source voxel -> world -> conformed voxel. `affine_transform` maps
        # OUTPUT coordinates through the matrix to look them up in the input,
        # so this is the direction it wants. Nearest neighbour: it is a binary
        # mask, and interpolating one invents edges that are neither in nor
        # out.
        xfm = np.linalg.inv(conf_affine) @ img.affine
        keep = affine_transform(
            mask_conf.astype(np.float32),
            xfm[:3, :3], offset=xfm[:3, 3],
            output_shape=img.shape[:3], order=0, mode="constant", cval=0,
        ) > 0.5
    finally:
        shutil.rmtree(tmp_mask.parent, ignore_errors=True)

    data = np.asanyarray(img.dataobj)
    if data.ndim > 3:
        keep = keep[..., None]
    stripped = np.where(keep, data, data.min())

    out_img = nib.Nifti1Image(
        stripped.astype(data.dtype), img.affine, img.header,
    )
    out_img.set_data_dtype(data.dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output))

    return DefaceResult(
        source=source, output=output, engine_id=eng.id,
        seconds=time.monotonic() - started,
        stdout=f"kept {float(keep.mean()):.1%} of the voxels",
    )


def deface_to(
    source: Path,
    output: Path,
    *,
    engine_id: str = DEFAULT_ENGINE_ID,
    timeout: int = DEFAULT_TIMEOUT_S,
    binary: Optional[Path] = None,
) -> DefaceResult:
    """Deface ``source`` into ``output``. Neither is required to be in a dataset.

    ``output`` is overwritten if it exists. The caller owns it.
    """
    import time

    source = Path(source)
    output = Path(output)
    eng: Engine = engine(engine_id)

    if not source.is_file():
        raise DefaceFailed(f"{source} is not a file")
    if eng.backend == BACKEND_BRAINCHOP:
        return _strip_with_brainchop(source, output, eng)
    if not atlas_present(eng):
        raise DefaceFailed(
            "The defacing atlas is missing from this installation."
        )

    exe = Path(binary) if binary else find_niimath()
    argv = [str(exe)] + eng.argv(source, output)

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DefaceFailed(
            f"Defacing {source.name} took longer than {timeout} s and was "
            "stopped. The image may be unusually large, or the engine may be "
            "stuck; try the fast 'allineate' engine."
        ) from exc
    except OSError as exc:
        raise DefaceFailed(f"could not run niimath: {exc}") from exc
    elapsed = time.monotonic() - started

    # A non-zero code with empty output is a real failure mode for this family
    # of binaries, so the return code is reported even when there is nothing
    # to quote. See CROSS_PLATFORM_RULES 3.2.
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise DefaceFailed(
            f"niimath exited {proc.returncode} on {source.name}"
            + (f": {detail}" if detail else " with no message")
        )

    if not output.is_file() or output.stat().st_size == 0:
        raise DefaceFailed(
            f"niimath reported success but wrote no image for {source.name}"
        )

    # It claimed success and produced something; check the something is still
    # the same image. What "same" means depends on the engine, and getting this
    # wrong in the strict direction is how the first version of this rejected
    # every -robustfov run: that flag CROPS THE NECK, so changing the
    # dimensions is the entire point of it.
    try:
        before = read_dimensions(source)
        after = read_dimensions(output)
    except Exception:  # noqa: BLE001 - a probe failure is not a deface failure
        log.debug("could not re-probe %s after defacing", source)
    else:
        if eng.robustfov:
            # It may shrink, along any axis. It may not grow, and it may not
            # stop being one volume: either would mean something other than a
            # crop happened.
            grew = [
                (b, a) for b, a in zip(before.shape, after.shape) if a > b
            ]
            if grew or not after.is_3d:
                raise DefaceFailed(
                    f"{source.name} came back as {after.shape} from "
                    f"{before.shape}, which is not a crop. The result was "
                    "discarded. Report this."
                )
        elif before.shape != after.shape:
            raise DefaceFailed(
                f"{source.name} changed shape during defacing "
                f"({before.shape} -> {after.shape}), so the result was "
                "discarded. Report this."
            )

    return DefaceResult(
        source=source,
        output=output,
        engine_id=eng.id,
        seconds=elapsed,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def deface_to_temp(
    source: Path,
    *,
    engine_id: str = DEFAULT_ENGINE_ID,
    timeout: int = DEFAULT_TIMEOUT_S,
    binary: Optional[Path] = None,
    directory: Optional[Path] = None,
) -> DefaceResult:
    """Deface into a temporary file the caller must clean up.

    ``directory`` defaults to the system temp. Callers that will move the
    result into a dataset should pass the dataset's own staging directory, so
    the final move stays on one filesystem.
    """
    source = Path(source)
    suffix = ".nii.gz" if source.name.lower().endswith(".nii.gz") else ".nii"
    fd, tmp = tempfile.mkstemp(
        prefix="bidsmgr-deface-", suffix=suffix,
        dir=str(directory) if directory else None,
    )
    os.close(fd)
    # niimath writes the file itself; an empty placeholder would only be in
    # the way, and some builds refuse to overwrite.
    Path(tmp).unlink(missing_ok=True)
    try:
        return deface_to(
            source, Path(tmp), engine_id=engine_id, timeout=timeout,
            binary=binary,
        )
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DefaceFailed",
    "DefaceResult",
    "DefaceUnavailable",
    "atlas_present",
    "available",
    "deface_to",
    "deface_to_temp",
    "find_niimath",
    "unavailable_reason",
]
