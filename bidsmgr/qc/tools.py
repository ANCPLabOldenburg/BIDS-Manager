"""The established tools the quality check uses where they are better than
anything worth writing: brainchop's networks for the brain (mindgrab, any
modality) and the tissue classes (robust_tissue), niimath's affine
registration (AFNI's 3dAllineate, ``-allineate``).

Both ship with BIDS Manager (they are what defacing runs on), so they add
nothing to install. Measured on the tutorial and OpenNeuro images
(2026-10-07): mindgrab's brain against the template's brain registered by
niimath agreed at Dice 0.92 on a Philips T1w and on a TSE T2w slab, where the
numpy fallback's mask scored 0.69 and 0.95; the tissue model segments T1w,
T2w and FLAIR. Each runs in a SUBPROCESS: niimath is a binary, and brainchop
starts a GPU runtime that must not live on a ``QThread``.

When they cannot run (niimath has no wheel for Python 3.14 yet; a model is
downloaded the first time and the machine may be offline; a failure), the
check falls back to its own numpy masks and says so. ``BIDSMGR_QC_ENGINE=numpy``
forces the fallback.

Qt-free.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: The tissue model and how its labels map to our classes. The two brainchop
#: tissue models number them OPPOSITELY (robust_tissue: 1 grey, 2 white;
#: tissue_fast: 1 white, 2 grey), checked by intensity on a T1w.
TISSUE_MODEL = "robust_tissue"
TISSUE_LABELS = {"robust_tissue": {1: "gm", 2: "wm"}, "tissue_fast": {1: "wm", 2: "gm"}}
BRAIN_MODEL = "mindgrab"
#: Seconds a model run may take (the first run downloads the model).
TIMEOUT_S = 600


class ToolFailed(RuntimeError):
    pass


def _call(argv: list[str], *, timeout: float, env: Optional[dict] = None,
          cancel: Optional[Callable[[], bool]] = None) -> tuple[int, str, str]:
    """Run ``argv``; ``(returncode, stdout, stderr)``. Killed, and
    :class:`ToolFailed` raised, when it outlives ``timeout`` or ``cancel``
    says stop (a check of a file the user has already left)."""
    import time

    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env)
    except OSError as exc:
        raise ToolFailed(f"could not start {Path(argv[0]).name}: {exc}") from exc
    end = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.25)
            return proc.returncode, out or "", err or ""
        except subprocess.TimeoutExpired:
            stop = cancel is not None and cancel()
            if stop or time.monotonic() > end:
                proc.kill()
                proc.communicate()
                raise ToolFailed("cancelled" if stop else
                                 f"{Path(argv[0]).name} took longer than {timeout:.0f} s")


def unavailable_reason() -> Optional[str]:
    """Why the tools cannot run here, or None."""
    if os.environ.get("BIDSMGR_QC_ENGINE", "").lower() == "numpy":
        return "the numpy engine was asked for (BIDSMGR_QC_ENGINE=numpy)"
    from ..deface import run as DR

    try:
        DR.find_niimath()
    except Exception as exc:  # noqa: BLE001 - its message says what to do
        return str(exc)
    if not DR.brainchop_present():
        return "brainchop is not installed"
    return None


def _env() -> dict:
    """brainchop calls niimath BY NAME to conform the image, so the binary
    inside our wheel has to be on the child's PATH."""
    from ..deface.run import find_niimath

    env = dict(os.environ)
    exe_dir = str(find_niimath().parent)
    if exe_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
    return env


def _to_grid(path: Path, shape, affine: np.ndarray) -> np.ndarray:
    """A model's result (conformed grid) on the image's grid, nearest
    neighbour: labels, not intensities."""
    import nibabel as nib
    from scipy.ndimage import affine_transform

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    xfm = np.linalg.inv(np.asarray(img.affine, dtype=float)) @ np.asarray(affine, dtype=float)
    return affine_transform(data.astype(np.float32), xfm[:3, :3], offset=xfm[:3, 3],
                            output_shape=tuple(int(s) for s in shape[:3]), order=0,
                            mode="constant", cval=0.0).astype(np.uint8)


def run_models(source: Path, shape, affine: np.ndarray, models: list[str], *,
               timeout: float = TIMEOUT_S,
               cancel: Optional[Callable[[], bool]] = None) -> dict[str, np.ndarray]:
    """``{model: labels on the image's grid}`` for every model that ran.
    Raises :class:`ToolFailed` when none did."""
    out = Path(tempfile.mkdtemp(prefix="bidsmgr-qc-"))
    try:
        argv = [sys.executable, "-m", "bidsmgr.qc._models", str(source), str(out),
                "--models", *models]
        _rc, stdout, stderr = _call(argv, timeout=timeout, env=_env(), cancel=cancel)
        got = {}
        for m in models:
            f = out / f"{m}.nii.gz"
            if f.is_file():
                labels = _to_grid(f, shape, affine)
                if labels.any():
                    got[m] = labels
        if not got:
            detail = (stderr or stdout).strip().splitlines()
            raise ToolFailed("brainchop produced nothing"
                             + (f": {detail[-1]}" if detail else ""))
        return got
    finally:
        shutil.rmtree(out, ignore_errors=True)


def allineate(source: Path, base: Path, *, timeout: float = 120.0,
              cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """niimath's affine registration of ``source`` to ``base``: the 4 x 4
    taking BASE world mm to SOURCE world mm (``fixed_to_moving``), the
    convention ``qc.register.Registration.matrix`` uses."""
    from ..deface.run import find_niimath

    work = Path(tempfile.mkdtemp(prefix="bidsmgr-qc-reg-"))
    try:
        mat = work / "affine.json"
        argv = [str(find_niimath()), str(source), "-allineate", str(base), "-savemat",
                str(mat), str(work / "moved")]
        _rc, stdout, stderr = _call(argv, timeout=timeout, cancel=cancel)
        if not mat.is_file():
            detail = (stderr or stdout).strip().splitlines()
            raise ToolFailed("niimath -allineate wrote no matrix"
                             + (f": {detail[-1]}" if detail else ""))
        data = json.loads(mat.read_text(encoding="utf-8"))
        m = np.asarray(data["fixed_to_moving"], dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise ToolFailed("niimath -allineate wrote an unusable matrix")
        return m
    finally:
        shutil.rmtree(work, ignore_errors=True)


def allineate_brain(image: np.ndarray, affine: np.ndarray, brain: np.ndarray, key: str, *,
                    timeout: float = 120.0,
                    cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """:func:`allineate` of the image's brain (``image`` masked by ``brain``)
    to the template's brain (``key``: "T1w" or "T2w")."""
    import nibabel as nib

    work = Path(tempfile.mkdtemp(prefix="bidsmgr-qc-brain-"))
    try:
        source = work / "brain.nii.gz"
        masked = np.where(brain, np.asarray(image, dtype=np.float32), 0.0).astype(np.float32)
        nib.save(nib.Nifti1Image(masked, np.asarray(affine, dtype=float)), str(source))
        return allineate(source, _template_brain_file(key), timeout=timeout, cancel=cancel)
    finally:
        shutil.rmtree(work, ignore_errors=True)


@lru_cache(maxsize=4)
def _template_brain_file(key: str) -> Path:
    """The template's ``key`` image inside its brain, written once per
    process (and removed at exit)."""
    import atexit

    import nibabel as nib

    from .templates import template

    tpl = template()
    folder = Path(tempfile.mkdtemp(prefix="bidsmgr-qc-tpl-"))
    atexit.register(shutil.rmtree, folder, True)
    path = folder / f"{key}_brain.nii.gz"
    data = np.where(tpl["brain"] > 0.5, tpl[key], 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(data, tpl.affine), str(path))
    return path


def template_file(key: str) -> Path:
    """A bundled template image as a file (niimath reads files)."""
    from importlib import resources

    from .templates import FILES

    ref = resources.files("bidsmgr.qc.templates") / FILES[key]
    with resources.as_file(ref) as p:
        return Path(p)


__all__ = ["BRAIN_MODEL", "TISSUE_LABELS", "TISSUE_MODEL", "ToolFailed", "allineate",
           "allineate_brain", "run_models", "template_file", "unavailable_reason"]
