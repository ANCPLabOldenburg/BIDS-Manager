"""Synthetic images for the quality-check tests, with known defects planted.

The anatomical phantom is the bundled template's own T1w (or T2w) on its
2 mm grid, inside a field of view with air around it and Rician noise in
the air, so every measure has a head, a brain, tissues and real noise to
work on. The diffusion phantom is a sphere of "tissue" with a diagonal
fibre band and a CSF core, sampled on a known gradient table.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def anat_phantom(*, contrast: str = "T1w", noise: float = 0.02, pad: int = 12,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(data, affine)``: the template's head, padded with noisy air."""
    from bidsmgr.qc.templates import template

    tpl = template()
    head = tpl["head"] > 0.5
    img = np.where(head, np.maximum(tpl[contrast], 0.15), 0.0).astype(np.float64)
    img = np.pad(img, pad)
    affine = tpl.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ np.full(3, pad)
    rng = np.random.default_rng(seed)
    # Rician magnitude: |signal + complex noise|.
    real = img + rng.normal(0, noise, img.shape)
    imag = rng.normal(0, noise, img.shape)
    data = (np.sqrt(real ** 2 + imag ** 2) * 1000.0).astype(np.float32)
    return data, affine


def save(data: np.ndarray, affine: np.ndarray, path: Path, sidecar: dict | None = None) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    stem = path.name.split(".")[0]
    (path.parent / f"{stem}.json").write_text(json.dumps(sidecar or {}), encoding="utf-8")
    return path


def dataset(root: Path) -> Path:
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "phantom", "BIDSVersion": "1.10.0"}), encoding="utf-8")
    return root


def gradient_table(n_dirs: int = 30, n_b0: int = 3, b: float = 1000.0,
                   seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(bvals, bvecs n x 3)``: b0s spread through, directions evenly on a
    hemisphere (a Fibonacci spiral)."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - i / n_dirs)
    theta = np.pi * (1 + 5 ** 0.5) * i
    g = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    bvals, bvecs = [], []
    every = max(1, n_dirs // max(n_b0 - 1, 1))
    k = 0
    for j in range(n_dirs):
        if j % every == 0 and k < n_b0:
            bvals.append(0.0)
            bvecs.append([0.0, 0.0, 0.0])
            k += 1
        bvals.append(b)
        bvecs.append(g[j])
    while k < n_b0:
        bvals.append(0.0)
        bvecs.append([0.0, 0.0, 0.0])
        k += 1
    return np.asarray(bvals), np.asarray(bvecs)


def dwi_phantom(bvals: np.ndarray, bvecs: np.ndarray, *, shape=(40, 40, 30),
                noise: float = 15.0, seed: int = 2, radius: float = 14.0,
                band: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """``(data x, y, z, n, affine)``: a sphere of tissue (MD 0.8), a CSF core
    (MD 3) and a fibre band running diagonally in x-y (FA about 0.7), at
    2 mm, with Rician noise."""
    nx, ny, nz = shape
    x, y, z = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2,
                          np.arange(nz) - nz / 2, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + (z * 1.2) ** 2)
    brain = r < radius
    csf = r < 4
    band = brain & ~csf & (np.abs(x - y) < band) & (np.abs(z) < 1.5 * band)
    s0 = np.where(brain, 1000.0, 0.0)
    s0[csf] = 1500.0
    d_iso = np.where(csf, 3.0e-3, 0.8e-3)
    e = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    lam1, lam2 = 1.7e-3, 0.35e-3
    n = len(bvals)
    out = np.zeros(shape + (n,), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for k in range(n):
        g = bvecs[k]
        b = bvals[k]
        att = np.exp(-b * d_iso)
        proj = float(g @ e) ** 2
        d_band = lam2 + (lam1 - lam2) * proj
        att = np.where(band, np.exp(-b * d_band), att)
        sig = s0 * att
        real = sig + rng.normal(0, noise, shape)
        imag = rng.normal(0, noise, shape)
        out[..., k] = np.sqrt(real ** 2 + imag ** 2)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -np.asarray(shape) + 1.0
    return out, affine


def save_dwi(data, affine, bvals, bvecs, path: Path, sidecar: dict | None = None) -> Path:
    save(data, affine, path, sidecar)
    stem = path.name.split(".")[0]
    np.savetxt(path.parent / f"{stem}.bval", np.asarray(bvals)[None], fmt="%g")
    np.savetxt(path.parent / f"{stem}.bvec", np.asarray(bvecs).T, fmt="%.6f")
    return path
