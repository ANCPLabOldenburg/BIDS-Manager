"""Build the atlas BRAIN mask from the template we already ship.

The skull-strip atlas engine needs a brain mask in the same space as
``avg152T1.nii.gz``, and no such mask ships with anything we depend on. The
obvious sources are FSL's, whose licence does not clearly allow us to
redistribute it.

So it is derived here instead, from two things we already have and may
redistribute: the BSD-2-Clause template itself, and the BSD-2-Clause mindgrab
network. Run mindgrab on the template, then put the result back on the
template's own grid.

    python tools/make_brain_mask.py

Deterministic, so re-running reproduces the checksum recorded in
``bidsmgr/deface/atlas/PROVENANCE.md``. Not run at install time or at run
time: the output is committed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, binary_closing, binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bidsmgr.deface.engines import BRAIN_MASK, TEMPLATE  # noqa: E402
from bidsmgr.deface.run import find_niimath  # noqa: E402


def main() -> int:
    # brainchop shells out to niimath by name; ours lives inside its wheel.
    os.environ["PATH"] = (
        str(find_niimath().parent) + os.pathsep + os.environ.get("PATH", "")
    )
    import brainchop

    template = nib.load(str(TEMPLATE))
    vol = brainchop.load(str(TEMPLATE))
    out = brainchop.segment(vol, "mindgrab")

    # Write with BRAINCHOP'S writer and read back with nibabel. Its in-memory
    # `Volume.data` is indexed in the REVERSE axis order to the header it
    # ships with, and `save` is what transposes; pairing the two directly
    # produced a mask that was rotated with respect to the template, which
    # then cut every subject at an angle.
    with tempfile.TemporaryDirectory() as tmp:
        conf_path = Path(tmp) / "conformed_mask.nii.gz"
        brainchop.save(out, str(conf_path))
        conf_img = nib.load(str(conf_path))
        conf_data = np.asanyarray(conf_img.dataobj).astype(np.float32)
        conf_affine = np.asarray(conf_img.affine, dtype=float)

    print("mindgrab produced", conf_data.shape,
          "frac brain %.3f" % float((conf_data > 0).mean()))

    # Resample onto the template grid: template voxel -> world -> conformed
    # voxel, nearest neighbour because it is a binary mask.
    xfm = np.linalg.inv(conf_affine) @ template.affine
    resampled = affine_transform(
        conf_data,
        xfm[:3, :3], offset=xfm[:3, 3],
        output_shape=template.shape, order=0, mode="constant", cval=0,
    )
    mask = resampled > 0.5

    # Close the pinholes nearest-neighbour downsampling leaves behind. A hole
    # in a keep-mask punches a hole in the brain.
    mask = binary_fill_holes(binary_closing(mask, np.ones((3, 3, 3))))

    img = nib.Nifti1Image(mask.astype(np.uint8), template.affine,
                          template.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(BRAIN_MASK))

    vox_mm3 = float(np.abs(np.linalg.det(template.affine[:3, :3])))
    print("wrote", BRAIN_MASK)
    print("  shape       ", mask.shape, "matches template", mask.shape == template.shape)
    print("  brain voxels", int(mask.sum()))
    print("  volume       %.0f mL" % (mask.sum() * vox_mm3 / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
